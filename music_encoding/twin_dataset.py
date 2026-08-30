import time
import os
import pathlib
import random

import datasets
import torch as tc
import torchaudio as ta
from torch import nn
from torchcodec.decoders import AudioDecoder

from music_encoding.augmenter import AudioAugmenter, SpectrogramAugmenter

DEFAULT_TGT_SR = 22050
DEFAULT_WIN_LEN_S = 5
DEFAULT_MIN_OFF_S = 2
DEFAULT_MIN_WINDOW_RMS = 0.03  # re-draw windows quieter than this (near-silence)
MAX_SILENCE_TRIES = 12


def random_window(
    wav: tc.Tensor,
    win_length: int = DEFAULT_TGT_SR * DEFAULT_WIN_LEN_S,
    min_offset: int = DEFAULT_MIN_OFF_S * DEFAULT_TGT_SR,
    exclude: int | None = None,
) -> tuple[tc.Tensor, int]:
    n = wav.shape[-1]

    if n <= win_length:
        pad: int = win_length - n
        return nn.functional.pad(wav, (0, pad)), 0

    start: int = 0

    for _ in range(20):
        start = random.randint(0, n - win_length)

        if exclude is None or abs(start - exclude) >= min_offset:
            return wav[start : start + win_length], start

    return wav[start : start + win_length], start

def load(ds: datasets.Dataset, idx: int) -> AudioDecoder:
    return ds[idx]["audio"]

def resample(wav: tc.Tensor, src_sr: int, tgt_sr: int) -> tc.Tensor:
    if wav.ndim > 1:
        wav = wav.mean(dim=0)
    if src_sr == tgt_sr:
        return wav

    return ta.functional.resample(wav, src_sr, tgt_sr)

def load_resampled_cached(ds, idx, tgt_sr, cache_dir: pathlib.Path):
    cache_path = cache_dir / f"{idx}.pt"
    if cache_path.exists():
        return tc.load(cache_path)
    decoder = ds[idx]["audio"]
    samples = decoder.get_all_samples()
    wav = resample(samples.data, samples.sample_rate, tgt_sr)  # reuse the shared helper
    tc.save(wav, cache_path)
    return wav

class LogMelSpectrogram:
    """Waveform -> (B, 1, n_mels, T) log-mel spectrogram.

    The waveform-to-spectrogram step lives here (NOT inside the model) so it can
    be shared by the pair dataset — which appends SpectrogramAugmenter (RRC) on
    top — and the eval scripts, giving every consumer identical mel settings.
    SiameseEncoderBT.forward receives an already-computed spectrogram.
    """

    def __init__(
        self,
        sr: int = DEFAULT_TGT_SR,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 64,
    ) -> None:
        self.mel = ta.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.to_db = ta.transforms.AmplitudeToDB()
        # the transform modules start on CPU; the dataset runs them there, but
        # the eval scripts batch windows onto the GPU. Track the last device we
        # placed them on so __call__ can follow whichever device the input is on
        # (STFT fails if input and window live on different devices).
        self._device: tc.device | None = None

    def __call__(self, wav: tc.Tensor) -> tc.Tensor:
        # always emit a 4-D (B, 1, n_mels, T) batch: accept both a single
        # (N,) window (as produced by the dataset) and a (B, N) batch.
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        dev = wav.device
        if self._device != dev:
            self.mel = self.mel.to(dev)
            self.to_db = self.to_db.to(dev)
            self._device = dev
        return self.to_db(self.mel(wav)).unsqueeze(1)


class FMAPairDataset(tc.utils.data.Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        window_sec: float = DEFAULT_WIN_LEN_S,
        target_sr: int = DEFAULT_TGT_SR,
        min_offset_sec: float = DEFAULT_MIN_OFF_S,
        cache_dir: str | os.PathLike | None = None,
        augmenter: AudioAugmenter | None = None,
        spectrogram_augmenter: SpectrogramAugmenter | None = None,
        min_window_rms: float = DEFAULT_MIN_WINDOW_RMS,
    ) -> None:
        self.ds = dataset
        self.win_length = int(target_sr * window_sec)
        self.min_offset = int(min_offset_sec * target_sr)
        self.target_sr = target_sr
        self.augmenter = augmenter
        self.spectrogram_augmenter = spectrogram_augmenter
        self.min_window_rms = min_window_rms
        self.mel = LogMelSpectrogram(sr=target_sr)
        self.cache_dir = None if cache_dir is None else pathlib.Path(cache_dir)

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # drop tracks that are globally near-silent: their windows are all quiet
        # (the per-window gate below can't rescue them), and their embeddings
        # collapse to one degenerate point that pollutes the learned space
        self._valid_tracks: list[int] | None = None
        if self.min_window_rms > 0:
            valid = [
                i
                for i in range(len(dataset))
                if float(self._window_rms(self._load_resampled(i))) >= self.min_window_rms
            ]
            self._valid_tracks = valid
            print(
                f"[FMAPairDataset] keeping {len(valid)}/{len(dataset)} tracks "
                f"(full-track RMS >= {self.min_window_rms})",
                flush=True,
            )

    def __len__(self):
        return len(self._valid_tracks) if self._valid_tracks is not None else len(self.ds)

    @staticmethod
    def _window_rms(wav: tc.Tensor) -> tc.Tensor:
        return tc.sqrt((wav.float() ** 2).mean())

    def _rand_window(
        self, wav: tc.Tensor, exclude: int | None = None
    ) -> tuple[tc.Tensor, int]:
        # reject near-silent windows so the encoder never learns a degenerate
        # "quiet" embedding point (quiet/sparse tracks collapse to one vector);
        # re-draw up to MAX_SILENCE_TRIES, then accept whatever we last drew.
        if self.min_window_rms <= 0:
            return random_window(wav, self.win_length, self.min_offset, exclude)
        for _ in range(MAX_SILENCE_TRIES):
            win, start = random_window(wav, self.win_length, self.min_offset, exclude)
            if float(self._window_rms(win)) >= self.min_window_rms:
                return win, start
        return win, start  # track is (nearly) all-silent; take the last draw

    def _load_resampled(self, idx: int) -> tc.Tensor:
        if self.cache_dir:
            return load_resampled_cached(self.ds, idx, self.target_sr, self.cache_dir)

        decoder: AudioDecoder = load(self.ds, idx)
        samples = decoder.get_all_samples()
        wav: tc.Tensor = samples.data  # (num_channels, num_samples), float32
        sr: int = samples.sample_rate

        return resample(wav, src_sr=sr, tgt_sr=self.target_sr)

        

    def __getitem__(self, index: int) -> tuple[tc.Tensor, tc.Tensor]:
        if self._valid_tracks is not None:
            index = self._valid_tracks[index]
        wav = self._load_resampled(index)
        win_a, start_a = self._rand_window(wav)
        win_b, _ = self._rand_window(wav, exclude=start_a)
        # waveform-domain augmentation (per-sample, CPU)
        if self.augmenter is not None:
            win_a = self.augmenter(win_a)
            win_b = self.augmenter(win_b)
        # waveform -> log-mel spectrogram, then spectrogram-domain augmentation (RRC)
        spec_a = self.mel(win_a)
        spec_b = self.mel(win_b)
        if self.spectrogram_augmenter is not None:
            spec_a = self.spectrogram_augmenter(spec_a)
            spec_b = self.spectrogram_augmenter(spec_b)
        return spec_a, spec_b


def collate_pairs(
    batch: list[tuple[tc.Tensor, tc.Tensor]],
) -> tuple[tc.Tensor, tc.Tensor]:
    # each item is already a 4-D (1, 1, n_mels, T) spectrogram — concatenate
    # along the batch dim rather than stacking, so the batch stays 4-D
    # (B, 1, n_mels, T) for the conv stack.
    a, b = zip(*batch)
    return tc.cat(a, dim=0), tc.cat(b, dim=0)
