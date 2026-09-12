import random

import torch as tc
import torch.nn.functional as F
import torchaudio as ta

# Safety floor for round-tripping dB log-mel through linear magnitude.
_DB_MIN = -120.0


def _to_amp(db: tc.Tensor) -> tc.Tensor:
    """log-mel dB -> linear magnitude (10**(db/20)), floored to avoid zero."""
    return tc.pow(10.0, (db * 0.05).clamp(min=_DB_MIN * 0.05))


def _to_db(amp: tc.Tensor) -> tc.Tensor:
    """linear magnitude -> log-mel dB (20*log10), floored to avoid -inf."""
    return 20.0 * amp.clamp(min=1e-12).log10()


class SpectrogramAugmenter:
    """Content distortion + RRC on log-mel spectrograms of shape (B, 1, n_mels, T).

    This is the single home of augmentation now. The waveform-domain
    AudioAugmenter is out of the training path — mel-mode data has no waveform,
    and the distortion set is equivalent when applied to the spectrogram — so the
    training-pipeline distortions live here:

        spectral noise  (always, SNR 5-20 dB)
        spectral reverb (p=0.4, exp-decay time-smear in the magnitude domain)
        lowpass         (p=0.3, rolloff above a random mel band)
        gain            (p=0.5, +/- 6 dB level offset)
        RRC             (p=0.6, random-resized crop)

    Order mirrors the old waveform chain (distort content first, then geometry),
    and every stage is applied per-sample via a mask so the class works on both a
    single (1, 1, n_mels, T) window from the dataset and a collated batch. Noise
    is unconditional (same rationale as before: it is near-constant in real query
    audio, and clean/clean pairs encourage embedding collapse).
    """

    def __init__(
        self,
        rrc_scale_range: tuple[float, float] = (0.8, 1.0),
        rrc_ratio_range: tuple[float, float] = (0.85, 1.15),
        rrc_p: float = 0.6,
        # --- content distortions (ported from AudioAugmenter) ---
        noise_p: float = 1.0,
        noise_snr_range: tuple[float, float] = (5.0, 20.0),
        reverb_p: float = 0.4,
        # RIR decay lengths in frames: ~0.2/0.4/0.8 s at the ~44-47 fps mel rates
        rir_lengths_frames: tuple[int, ...] = (9, 19, 38),
        lowpass_p: float = 0.3,
        # fraction of mel bands below which no rolloff; above it, a ramp down
        lowpass_cutoff_range: tuple[float, float] = (0.45, 0.8),
        rolloff_db_range: tuple[float, float] = (15.0, 30.0),
        gain_p: float = 0.5,
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
    ):
        self.rrc_scale_range = rrc_scale_range
        self.rrc_ratio_range = rrc_ratio_range
        self.rrc_p = rrc_p
        self.noise_p = noise_p
        self.noise_snr_range = noise_snr_range
        self.reverb_p = reverb_p
        self.rir_lengths_frames = rir_lengths_frames
        self.lowpass_p = lowpass_p
        self.lowpass_cutoff_range = lowpass_cutoff_range
        self.rolloff_db_range = rolloff_db_range
        self.gain_p = gain_p
        self.gain_db_range = gain_db_range

    # --- stochastic helpers ---------------------------------------------------

    @staticmethod
    def _mask(B: int, device: tc.device, p: float) -> tc.Tensor:
        return tc.rand(B, device=device) < p

    # --- content distortions (log-mel dB domain) ------------------------------

    def _add_noise(self, db: tc.Tensor, mask: tc.Tensor) -> tc.Tensor:
        # additive COMPLEX-GAUSSIAN noise in the linear-magnitude domain, i.e.
        # power addition: out_amp = |amp + z| for complex z. Because the noise
        # power is independent of the signal, |amp + z|^2 = amp^2 + |z|^2 (in
        # expectation), which is always >= amp — quiet bins rise toward a noise
        # floor, loud bins barely move, and nothing can go negative, so a dB
        # log never sees a -inf/-240 clamp spike. This is the spectrogram
        # equivalent of waveform hiss/channel noise.
        #
        # Noise power is set SNR dB below each window's 95th-PERCENTILE bin
        # amplitude (a robust "loud-content" level), NOT its global peak: a 5 s
        # music window's peak is almost always a single percussive transient —
        # measured, only ~10% of bins are within 20 dB of it — so anchoring the
        # hiss to the peak planted the floor above ~90% of the content, and two
        # augmented views of the same window shared almost no signal (the model
        # could not learn any invariance; on_diag stayed pinned near D). The p95
        # level is what the legacy waveform AudioAugmenter's mean-power
        # reference was approximating; a window mean is unusable here because
        # it is floor-dominated (far quieter than the audible content).
        if not mask.any():
            return db
        B = db.shape[0]
        amp = _to_amp(db)  # >= 10^-6 (i.e. -120 dB) by construction
        loud = amp.flatten(1).quantile(0.95, dim=1).clamp(min=1e-12).view(B, 1, 1, 1)
        snr = tc.empty(B, device=db.device).uniform_(*self.noise_snr_range).view(B, 1, 1, 1)
        # real std of the noise field: loud * 10^(-snr/20) / sqrt(2) per
        # quadrature component, so total noise power = loud^2 * 10^(-snr/10).
        r = loud * tc.pow(10.0, -snr / 20.0) * (2.0 ** -0.5)
        a = tc.randn_like(amp)
        b = tc.randn_like(amp)
        out_amp = ((amp + r * a).pow(2) + (r * b).pow(2)).sqrt().clamp(min=1e-9)
        out = _to_db(out_amp)
        return tc.where(mask.view(-1, 1, 1, 1), out, db)

    def _add_reverb(self, db: tc.Tensor, mask: tc.Tensor) -> tc.Tensor:
        # temporal smearing in the magnitude domain: convolve each mel row along
        # time with a unit-energy exponential-decay "room" kernel (length = one of
        # rir_lengths_frames). Cheap FFT conv; a crude stand-in for a real RIR's
        # tail that still teaches time-blur invariance.
        if not mask.any():
            return db
        B, _, H, T = db.shape
        kernels = []
        for _ in range(B):
            L = random.choice(self.rir_lengths_frames)
            L = max(1, min(L, T - 1))
            k = tc.exp(-tc.arange(L, dtype=tc.float32) / (L / 6.9))
            kernels.append(k / k.sum().clamp(min=1e-8))
        max_len = max(len(k) for k in kernels)
        ks = tc.zeros(B, max_len, device=db.device)
        for i, k in enumerate(kernels):
            ks[i, : len(k)] = k

        amp = _to_amp(db).reshape(B, H, T)
        fft_n = 1 << (T + max_len - 1 - 1).bit_length()  # next pow2 >= conv_len
        out = tc.fft.irfft(
            tc.fft.rfft(amp, n=fft_n, dim=-1) * tc.fft.rfft(ks, n=fft_n, dim=-1).unsqueeze(1),
            n=fft_n,
            dim=-1,
        )[..., :T]
        out_db = _to_db(out.reshape(B, 1, H, T))
        return tc.where(mask.view(-1, 1, 1, 1), out_db, db)

    def _lowpass(self, db: tc.Tensor, mask: tc.Tensor) -> tc.Tensor:
        # phone-mic / lossy-codec high-frequency rolloff: above a random mel band,
        # attenuate with a dB ramp reaching `rolloff` dB at the top band.
        if not mask.any():
            return db
        B, _, H, _ = db.shape
        # mel-row index in the CHANNEL-agnostic position (B, 1, H, 1) so `db -
        # above*rolloff` attenuates the frequency axis only — the channel dim
        # must stay size 1 for the batch broadcast, not collapse into H.
        k = tc.arange(H, device=db.device).float().view(1, 1, H, 1)
        cutoff = tc.empty(B, device=db.device).uniform_(*self.lowpass_cutoff_range).view(B, 1, 1, 1)
        rolloff = tc.empty(B, device=db.device).uniform_(*self.rolloff_db_range).view(B, 1, 1, 1)
        above = ((k - cutoff * (H - 1)) / ((1.0 - cutoff) * (H - 1)).clamp(min=1.0)).clamp(min=0.0)
        out = db - above * rolloff
        return tc.where(mask.view(-1, 1, 1, 1), out, db)

    def _gain(self, db: tc.Tensor, mask: tc.Tensor) -> tc.Tensor:
        # scalar level offset (+/- dB) — the spectrogram equivalent of volume gain
        if not mask.any():
            return db
        B = db.shape[0]
        offset = tc.empty(B, device=db.device).uniform_(*self.gain_db_range).view(B, 1, 1, 1)
        out = db + offset
        return tc.where(mask.view(-1, 1, 1, 1), out, db)

    # --- geometry (RRC) -------------------------------------------------------

    def random_resized_crop(self, spec: tc.Tensor, mask: tc.Tensor) -> tc.Tensor:
        if not mask.any():
            return spec
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
        dtype = spec.dtype
        spec = spec.float()
        # guard non-finite bins (some consumers emit -inf for zero mel energy)
        spec = tc.nan_to_num(spec, nan=_DB_MIN, posinf=120.0, neginf=_DB_MIN)
        B = spec.shape[0]
        device = spec.device

        if self.noise_p > 0:
            spec = self._add_noise(spec, self._mask(B, device, self.noise_p))
        if self.reverb_p > 0:
            spec = self._add_reverb(spec, self._mask(B, device, self.reverb_p))
        if self.lowpass_p > 0:
            spec = self._lowpass(spec, self._mask(B, device, self.lowpass_p))
        if self.gain_p > 0:
            spec = self._gain(spec, self._mask(B, device, self.gain_p))
        if self.rrc_p > 0:
            spec = self.random_resized_crop(spec, self._mask(B, device, self.rrc_p))
        return spec.to(dtype)


class AudioAugmenter:
    """LEGACY waveform-domain augmenter.

    Retained only as a manual QA / listening tool (test_augmentation.py) and a
    reference implementation. It is NOT part of the training pipeline: training
    consumes spectrograms, so all distortions are applied spectrogram-side by
    SpectrogramAugmenter. Its distortion set mirrors that class (noise, reverb,
    lowpass, gain).
    """

    def __init__(
        self,
        sr: int = 22050,
        noise_snr_range: tuple[float, float] = (5.0, 20.0),
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
        rir_t60s: tuple[float, ...] = (0.2, 0.4, 0.8),
        reverb_p: float = 0.4,
        pitch_p: float = 0.0,  # OFF by default — see pitch_shift's cost note
        pitch_range_semitones: tuple[float, float] = (-2.0, 2.0),
    ):
        self.sr = sr
        self.noise_snr_range = noise_snr_range
        self.gain_db_range = gain_db_range
        self.reverb_p = reverb_p
        self.pitch_p = pitch_p
        self.pitch_range = pitch_range_semitones
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

    def pitch_shift(self, wav: tc.Tensor) -> tc.Tensor:
        # quarter-tone resolution (24 bins/octave) so small shifts stay musical;
        # returns wav unchanged when the sampled shift rounds to 0 steps.
        # COST: torchaudio.functional.pitch_shift takes ~2.3 s per 2 s clip on
        # this CPU/ROCm build (~100x every other stage) — wired to self.pitch_p
        # but defaults to OFF so it can't stall per-sample augmentation.
        bins_per_octave = 24  # each "step" is a quarter-tone, not a semitone
        shift_semitones = random.uniform(*self.pitch_range)
        n_steps = round(shift_semitones * (bins_per_octave / 12))
        if n_steps == 0:
            return wav
        return ta.functional.pitch_shift(
            wav.unsqueeze(0), self.sr, n_steps, bins_per_octave=bins_per_octave
        ).squeeze(0)

    def gain(self, wav: tc.Tensor) -> tc.Tensor:
        db = random.uniform(*self.gain_db_range)
        return (wav * (10 ** (db / 20))).clamp(-1.0, 1.0)

    def lowpass(self, wav: tc.Tensor) -> tc.Tensor:
        # simulates phone-mic / lossy-compression frequency rolloff
        cutoff = random.uniform(2000, 8000)
        return ta.functional.lowpass_biquad(wav, self.sr, cutoff)

    def __call__(self, wav: tc.Tensor) -> tc.Tensor:
        wav = self.add_noise(wav)
        if random.random() < self.reverb_p:
            wav = self.add_reverb(wav)
        if random.random() < self.pitch_p:
            wav = self.pitch_shift(wav)
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
