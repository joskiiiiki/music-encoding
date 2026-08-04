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
        padded = tc.nn.functional.pad(wav, (rir.shape[0] - 1, 0))
        out = tc.nn.functional.conv1d(
            padded.view(1, 1, -1), rir.flip(0).view(1, 1, -1)
        ).view(-1)[: wav.shape[0]]
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
