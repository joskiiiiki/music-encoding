from torch import nn
import time
import random
import torch as tc
import torchaudio as ta


class AudioAugmenter:
    def __init__(
        self,
        sr: int = 22050,
        noise_snr_range: tuple[float, float] = (5.0, 20.0),
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
        pitch_range_semitones: tuple[float, float] = (-2.0, 2.0),
        rir_t60s: tuple[float, ...] = (0.2, 0.4, 0.8),
    ):
        self.sr = sr
        self.noise_snr_range = noise_snr_range
        self.gain_db_range = gain_db_range
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

    def gain(self, wav: tc.Tensor) -> tc.Tensor:
        db = random.uniform(*self.gain_db_range)
        return (wav * (10 ** (db / 20))).clamp(-1.0, 1.0)

    def pitch_shift(self, wav: tc.Tensor) -> tc.Tensor:
        bins_per_octave = 24  # each "step" is now a quarter-tone, not a semitone
        shift_semitones = random.uniform(*self.pitch_range)
        n_steps = round(shift_semitones * (bins_per_octave / 12))
        if n_steps == 0:
            return wav
        return ta.functional.pitch_shift(
            wav.unsqueeze(0), self.sr, n_steps, bins_per_octave=bins_per_octave
        ).squeeze(0)

    def lowpass(self, wav: tc.Tensor) -> tc.Tensor:
        # simulates phone-mic / lossy-compression frequency rolloff
        cutoff = random.uniform(2000, 8000)
        return ta.functional.lowpass_biquad(wav, self.sr, cutoff)

    def __call__(self, wav: tc.Tensor) -> tc.Tensor:
        # apply a random subset — not every augmentation every time, so the
        # network sees a genuine distribution of distortion combinations
        if random.random() < 0.7:
            wav = self.add_noise(wav)
        if random.random() < 0.4:
            wav = self.add_reverb(wav)
        # if random.random() < 0.25:
        #     wav = self.pitch_shift(wav)
        # skip for now since it doesnt fit use case
        if random.random() < 0.3:
            wav = self.lowpass(wav)
        if random.random() < 0.5:
            wav = self.gain(wav)
        return wav


class BatchAudioAugmenter:
    def __init__(
        self,
        sr: int = 22050,
        noise_snr_range: tuple[float, float] = (5.0, 20.0),
        gain_db_range: tuple[float, float] = (-6.0, 6.0),
        rir_t60s: tuple[float, ...] = (0.15, 0.3),
        noise_p: float = 0.7,
        reverb_p: float = 0.4,
        lowpass_p: float = 0.3,
        gain_p: float = 0.5,
        device: str = "cuda",
    ):
        self.sr = sr
        self.noise_snr_range = noise_snr_range
        self.gain_db_range = gain_db_range
        self.noise_p = noise_p
        self.reverb_p = reverb_p
        self.lowpass_p = lowpass_p
        self.gain_p = gain_p
        self.device = device
        # precompute RIR bank as a single stacked tensor, padded to the longest kernel
        rirs = [self._make_synthetic_rir(t60) for t60 in rir_t60s]
        max_len = max(r.shape[0] for r in rirs)
        self.rirs = tc.stack(
            [nn.functional.pad(r, (0, max_len - r.shape[0])) for r in rirs]
        ).to(device)  # (n_rirs, max_rir_len)

    def _make_synthetic_rir(self, t60: float) -> tc.Tensor:
        n = int(self.sr * t60)
        noise = tc.randn(n)
        decay = tc.exp(-tc.arange(n, dtype=tc.float32) / (self.sr * t60 / 6.9))
        rir = noise * decay
        return rir / rir.abs().max().clamp(min=1e-8)

    def add_noise(self, wav: tc.Tensor, mask: tc.Tensor) -> tc.Tensor:
        B = wav.shape[0]
        snr_db = tc.empty(B, device=wav.device).uniform_(*self.noise_snr_range)
        noise = tc.randn_like(wav)
        sig_power = wav.pow(2).mean(dim=-1).clamp(min=1e-8)
        noise_power = noise.pow(2).mean(dim=-1).clamp(min=1e-8)
        factor = (sig_power / (noise_power * 10 ** (snr_db / 10))).sqrt().unsqueeze(-1)
        out = wav + noise * factor
        return tc.where(mask.unsqueeze(-1), out, wav)

    def add_reverb(self, wav: tc.Tensor, mask: tc.Tensor) -> tc.Tensor:
        B, N = wav.shape
        rir_idx = tc.randint(0, self.rirs.shape[0], (B,), device=wav.device)
        rir = self.rirs[rir_idx]  # (B, max_rir_len)
        orig_peak = wav.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)

        # batched FFT convolution: pad both to a common conv length, multiply in freq domain
        conv_len = N + rir.shape[-1] - 1
        fft_n = 1 << (conv_len - 1).bit_length()  # next power of 2 for efficiency
        wav_f = tc.fft.rfft(wav, n=fft_n)
        rir_f = tc.fft.rfft(rir, n=fft_n)
        out = tc.fft.irfft(wav_f * rir_f, n=fft_n)[:, :N]

        out_peak = out.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        out = out / out_peak * orig_peak
        return tc.where(mask.unsqueeze(-1), out, wav)

    def lowpass(self, wav: tc.Tensor, mask: tc.Tensor) -> tc.Tensor:
        # simple single-pole lowpass per-sample cutoff, vectorized across batch
        B = wav.shape[0]
        cutoff = tc.empty(B, device=wav.device).uniform_(2000, 8000)
        alpha = tc.exp(-2 * tc.pi * cutoff / self.sr)  # (B,)
        out = tc.zeros_like(wav)
        out[:, 0] = wav[:, 0]
        # sequential recurrence — see note below on cost
        for t in range(1, wav.shape[-1]):
            out[:, t] = alpha * out[:, t - 1] + (1 - alpha) * wav[:, t]
        return tc.where(mask.unsqueeze(-1), out, wav)

    def gain(self, wav: tc.Tensor, mask: tc.Tensor) -> tc.Tensor:
        B = wav.shape[0]
        db = tc.empty(B, device=wav.device).uniform_(*self.gain_db_range)
        factor = (10 ** (db / 20)).unsqueeze(-1)
        out = (wav * factor).clamp(-1.0, 1.0)
        return tc.where(mask.unsqueeze(-1), out, wav)

    def __call__(self, wav: tc.Tensor) -> tc.Tensor:
        B = wav.shape[0]
        device = wav.device
        wav = self.add_noise(wav, tc.rand(B, device=device) < self.noise_p)
        wav = self.add_reverb(wav, tc.rand(B, device=device) < self.reverb_p)
        wav = self.lowpass(wav, tc.rand(B, device=device) < self.lowpass_p)
        wav = self.gain(wav, tc.rand(B, device=device) < self.gain_p)
        return wav
