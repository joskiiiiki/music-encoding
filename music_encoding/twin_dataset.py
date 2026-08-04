import time
import os
import pathlib
import random

import datasets
import torch as tc
import torchaudio as ta
from torch import nn
from torchcodec.decoders import AudioDecoder

from music_encoding.augmenter import AudioAugmenter

DEFAULT_TGT_SR = 22050
DEFAULT_WIN_LEN_S = 5
DEFAULT_MIN_OFF_S = 2


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

class FMAPairDataset(tc.utils.data.Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        window_sec: float = DEFAULT_WIN_LEN_S,
        target_sr: int = DEFAULT_TGT_SR,
        min_offset_sec: float = DEFAULT_MIN_OFF_S,
        cache_dir: str | os.PathLike | None = None,
        augmenter: AudioAugmenter | None = None,
    ) -> None:
        self.ds = dataset
        self.win_length = int(target_sr * window_sec)
        self.min_offset = int(min_offset_sec * target_sr)
        self.target_sr = target_sr
        self.augmenter = augmenter
        self.cache_dir = None if cache_dir is None else pathlib.Path(cache_dir)

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.ds)

    def _rand_window(
        self, wav: tc.Tensor, exclude: int | None = None
    ) -> tuple[tc.Tensor, int]:
        return random_window(wav, self.win_length, self.min_offset, exclude)

    def _load_resampled(self, idx: int) -> tc.Tensor:
        if self.cache_dir:
            return load_resampled_cached(self.ds, idx, self.target_sr, self.cache_dir)

        decoder: AudioDecoder = load(self.ds, idx)
        samples = decoder.get_all_samples()
        wav: tc.Tensor = samples.data  # (num_channels, num_samples), float32
        sr: int = samples.sample_rate

        return resample(wav, src_sr=sr, tgt_sr=self.target_sr)

        

    def __getitem__(self, index: int) -> tuple[tc.Tensor, tc.Tensor]:
        wav = self._load_resampled(index)
        win_a, start_a = self._rand_window(wav)
        win_b, _ = self._rand_window(wav, exclude=start_a)
        if self.augmenter is not None:
            return self.augmenter(win_a), self.augmenter(win_b)
        return win_a, win_b


def collate_pairs(
    batch: list[tuple[tc.Tensor, tc.Tensor]],
) -> tuple[tc.Tensor, tc.Tensor]:
    a, b = zip(*batch)
    return tc.stack(a), tc.stack(b)
