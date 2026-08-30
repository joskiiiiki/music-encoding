import random

import torch as tc
import torch.nn.functional as F
import torchaudio as ta


class SpectrogramAugmenter:
    """
    Operates on mel-spectrograms of shape (B, 1, n_mels, T) — i.e. after
    self.mel / self.to_db, before the conv stack.
    """

    def __init__(
        self,
        rrc_scale_range: tuple[float, float] = (0.8, 1.0),
        rrc_ratio_range: tuple[float, float] = (0.85, 1.15),
        rrc_p: float = 0.6,
    ):
        self.rrc_scale_range = rrc_scale_range
        self.rrc_ratio_range = rrc_ratio_range
        self.rrc_p = rrc_p

    def random_resized_crop(self, spec: tc.Tensor, mask: tc.Tensor) -> tc.Tensor:
        B, C, H, W = spec.shape
        device = spec.device

        area_frac = tc.empty(B, device=device).uniform_(*self.rrc_scale_range)
        aspect = tc.empty(B, device=device).uniform_(*self.rrc_ratio_range)

        crop_h_frac = (area_frac / aspect).sqrt().clamp(max=1.0)
        crop_w_frac = (area_frac * aspect).sqrt().clamp(max=1.0)

        max_cx = 1.0 - crop_w_frac
        max_cy = 1.0 - crop_h_frac
        cx = (tc.rand(B, device=device) * 2 - 1) * max_cx
        cy = (tc.rand(B, device=device) * 2 - 1) * max_cy

        theta = tc.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = crop_w_frac
        theta[:, 1, 1] = crop_h_frac
        theta[:, 0, 2] = cx
        theta[:, 1, 2] = cy

        grid = F.affine_grid(theta, size=[B, C, H, W], align_corners=False)
        cropped = F.grid_sample(spec, grid, mode="bilinear", align_corners=False)

        return tc.where(mask.view(-1, 1, 1, 1), cropped, spec)

    def __call__(self, spec: tc.Tensor) -> tc.Tensor:
        B = spec.shape[0]
        mask = tc.rand(B, device=spec.device) < self.rrc_p
        return self.random_resized_crop(spec, mask)

class AudioAugmenter:
    def __init__(
        self,
        sr: int = 22050,
        noise_snr_range: tuple[float, float] = (5.0, 20.0),
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
        rir_t60s: tuple[float, ...] = (0.2, 0.4, 0.8),
    ):
        self.sr = sr
        self.noise_snr_range = noise_snr_range
        self.gain_db_range = gain_db_range
        # precompute a small bank of synthetic room impulse responses (cheap, reusable)
        self.rirs = [self._make_synthetic_rir(t60) for t60 in rir_t60s]

    def _make_synthetic_rir(self, t60: float) -> tc.Tensor:
        # exponential-decay noise burst — a crude but usable stand-in for a real room IR
        n = int(self.sr * t60)
        noise = tc.randn(n)
        decay = tc.exp(-tc.arange(n, dtype=tc.float32) / (self.sr * t60 / 6.9))
        rir = noise * decay
        return rir / rir.abs().max().clamp(min=1e-8)

    def add_noise(self, wav: tc.Tensor) -> tc.Tensor:
        snr_db = random.uniform(*self.noise_snr_range)
        noise = tc.randn_like(wav)
        sig_power = wav.pow(2).mean().clamp(min=1e-8)
        noise_power = noise.pow(2).mean().clamp(min=1e-8)
        factor = (sig_power / (noise_power * 10 ** (snr_db / 10))).sqrt()
        return wav + noise * factor

    def add_reverb(self, wav: tc.Tensor) -> tc.Tensor:
        rir = random.choice(self.rirs)
        orig_peak = wav.abs().max().clamp(min=1e-8)
        out = ta.functional.fftconvolve(wav, rir, mode="full")[: wav.shape[0]]
        return out / out.abs().max().clamp(min=1e-8) * orig_peak

    def gain(self, wav: tc.Tensor) -> tc.Tensor:
        db = random.uniform(*self.gain_db_range)
        return (wav * (10 ** (db / 20))).clamp(-1.0, 1.0)

    def lowpass(self, wav: tc.Tensor) -> tc.Tensor:
        # simulates phone-mic / lossy-compression frequency rolloff
        cutoff = random.uniform(2000, 8000)
        return ta.functional.lowpass_biquad(wav, self.sr, cutoff)

    def __call__(self, wav: tc.Tensor) -> tc.Tensor:
        # noise (always) -> lowpass (prob) -> gain (prob). Noise is unconditional
        # since it's a near-constant feature of real query audio, and applying it
        # every time avoids leaving clean/clean pairs that encourage collapse.
        # Reverb and pitch shift are excluded by default; Mixup runs separately in
        # the training loop since it needs cross-sample (batch) visibility.
        wav = self.add_noise(wav)
        # if random.random() < 0.4:
        #     wav = self.add_reverb(wav)
        if random.random() < 0.3:
            wav = self.lowpass(wav)
        if random.random() < 0.5:
            wav = self.gain(wav)
        return wav


class Mixup:
    """
    In-batch sample mixing (domain-agnostic linear interpolation between two
    batch members). Operates on a *batch* of tensors — any shape with a leading
    batch dim, e.g. (B, N) waveforms or (B, 1, n_mels, T) spectrograms — not a
    single sample, since it needs visibility into other samples in the batch.
    Applied in the training loop (after moving to device), independently to each
    view, so each gets a different partner. The dataset now returns spectrograms,
    so Mixup blends spectrograms.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        target_weight_min: float = 0.6,
        p: float = 0.5,
    ):
        self.alpha = alpha
        self.target_weight_min = target_weight_min
        self.p = p
        self.beta = tc.distributions.Beta(alpha, alpha)

    def __call__(self, wav: tc.Tensor) -> tc.Tensor:
        B = wav.shape[0]
        device = wav.device
        if B < 2:
            return wav

        # partner index j != i for every sample: sample uniformly from [0, B-2],
        # then shift any value >= i up by one, mapping [0, B-2] bijectively onto
        # {0..B-1}\{i}. Guaranteed self-pairing-free, no permutation bookkeeping.
        j = tc.randint(0, B - 1, (B,), device=device)
        j = j + (j >= tc.arange(B, device=device)).to(j.dtype)
        partners = wav[j]

        # weight of the ORIGINAL (target) signal; keep it dominant
        weight = self.beta.sample((B,)).to(device).clamp(min=self.target_weight_min)
        # only mix a per-sample random subset; the rest keep weight 1.0 (unchanged)
        apply = tc.rand(B, device=device) < self.p
        # (B,) broadcastable over any trailing shape — (B, N) waveforms or
        # (B, 1, n_mels, T) spectrograms
        weight = tc.where(apply, weight, tc.ones_like(weight)).view(
            B, *([1] * (wav.dim() - 1))
        )

        return weight * wav + (1.0 - weight) * partners
