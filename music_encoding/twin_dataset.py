import time
import os
import pathlib
import random

import datasets
import numpy as np
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

# MTG-Jamendo melspec .npy files are log-mel dB spectrograms clipped at a -90 dB
# power floor. A fully-silent window therefore sits at ~-90 dB; the per-window
# gate below re-draws anything whose mean is below DEFAULT_MIN_WINDOW_DB.
MEL_FLOOR_DB = -90.0
DEFAULT_MIN_WINDOW_DB = -85.0


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
        # waveform -> log-mel spectrogram, then spectrogram-domain augmentation
        # (content distortions + RRC)
        spec_a = self.mel(win_a)
        spec_b = self.mel(win_b)
        if self.spectrogram_augmenter is not None:
            spec_a = self.spectrogram_augmenter(spec_a)
            spec_b = self.spectrogram_augmenter(spec_b)
        return spec_a, spec_b


# --- precomputed log-mel input mode (MTG-Jamendo "melspecs" .npy) ------------
#
# FMAPairDataset decodes waveform -> LogMelSpectrogram. MelSpecPairDataset reads
# already-computed full-track (n_mels, T) log-mel spectrograms straight off disk,
# so there is no audio decode and no conversion. Time-stability positives are two
# random ~win_sec windows of the SAME track's spectrogram. SpectrogramAugmenter
# now carries the whole distortion set (noise / reverb / lowpass / gain) plus RRC
# (see augmenter.py), all applied on the spectrogram — so no waveform is needed
# and both data paths distort identically.
#
# Layout (verified over the corpus): <root>/<id % 100:02d>/<id>.npy, one float32
# (96, T) array per track. The frame rate is a constant ~46.875 fps
# (T == round(seconds * 46.875) for tracks of known duration), so the waveform
# window sizes (5 s / 2 s) map to frame counts at that rate. The native 96 mel
# bands are merged down to n_mels (default 64, matching the waveform/eval
# LogMelSpectrogram) in linear-power space per window, before the augmentation
# chain, so the model sees the same frequency resolution as the 64-mel audio path
# (see MelSpecWindowSource._merge_mels).
#
# MelSpecWindowSource owns all of that windowing/merge logic and is shared by the
# training dataset and the eval / QA scripts (build_chroma, test_similar_pairs,
# test_augmentation), so the two cannot drift on window geometry, short-track
# padding, the silence gate or the band merge.
MEL_FPS = 46.875


def enumerate_mel_spec_ids(root: str | os.PathLike) -> list[int]:
    """Track ids present under a melspec root laid out as <id%100:02d>/<id>.npy."""
    ids: list[int] = []
    for p in pathlib.Path(root).glob("*/*.npy"):
        try:
            ids.append(int(p.stem))
        except ValueError:
            continue
    return sorted(ids)


def mel_spec_path(root: str | os.PathLike, track_id: int) -> pathlib.Path:
    """<root>/<id % 100:02d>/<id>.npy — the layout MTG's download uses."""
    return pathlib.Path(root) / f"{track_id % 100:02d}" / f"{track_id}.npy"


def random_spec_window(
    spec: np.ndarray,
    win_frames: int,
    min_offset_frames: int,
    exclude: int | None = None,
) -> tuple[np.ndarray, int]:
    """Column-windowed draw from a (n_mels, T) spectrogram, mirroring random_window.

    Draws a contiguous T-window of win_frames columns; keeps it only if it starts
    at least min_offset_frames away from `exclude` (the other view's start), else
    re-draws (up to 20 tries, then accepts the last draw — same fallback as the
    waveform path). Tracks shorter than a window are padded at the -90 dB floor
    (the dB equivalent of padding silence into a short waveform).
    """
    n = spec.shape[1]
    if n <= win_frames:
        pad = win_frames - n
        out = np.full((spec.shape[0], win_frames), MEL_FLOOR_DB, dtype=spec.dtype)
        out[:, :n] = spec
        return out, 0

    start = 0
    for _ in range(20):
        start = random.randint(0, n - win_frames)
        if exclude is None or abs(start - exclude) >= min_offset_frames:
            return spec[:, start : start + win_frames], start
    return spec[:, start : start + win_frames], start


class MelSpecWindowSource:
    """Random-window access to precomputed MTG-Jamendo log-mel .npy spectrograms.

    The single home of "how is a window cut out of a stored spectrogram": which
    track ids exist, where each one lives, how many frames a window spans at
    MEL_FPS, how short tracks are padded, the near-silence gate, and how the native
    96 mel bands are merged to the model's n_mels. MelSpecPairDataset (training)
    and the eval / QA scripts (build_chroma, test_similar_pairs, test_augmentation)
    each build one, so they cannot drift on window geometry or frequency
    resolution — the same rationale that keeps waveform->mel in LogMelSpectrogram.

    `random_window` returns a ready-to-stack (1, 1, n_mels, win_frames) tensor: the
    silence gate is applied to the NATIVE (96-band) window, then the bands are
    merged — the order the training path has always used.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        window_sec: float = DEFAULT_WIN_LEN_S,
        min_offset_sec: float = DEFAULT_MIN_OFF_S,
        n_mels: int = 64,
        min_window_db: float | None = DEFAULT_MIN_WINDOW_DB,
    ) -> None:
        self.root = pathlib.Path(root)
        self.ids = enumerate_mel_spec_ids(self.root)
        if not self.ids:
            raise FileNotFoundError(f"no <dir>/<id>.npy spectrograms under {self.root}")
        self.win_frames = max(1, round(window_sec * MEL_FPS))
        self.min_offset_frames = max(1, round(min_offset_sec * MEL_FPS))
        # model-input mel bands: the .npy are native (96, T); the default merges
        # them down to 64 to match the waveform path / audio evals
        # (LogMelSpectrogram n_mels=64). n_mels >= source rows is a no-op (keeps
        # the native 96).
        self.n_mels = n_mels
        self.min_window_db = min_window_db
        mel_note = f"-> {n_mels} bands" if n_mels < 96 else "native (96) bands"
        print(
            f"[MelSpecWindowSource] {len(self.ids)} tracks under {self.root}, "
            f"window={window_sec}s ({self.win_frames} frames @ {MEL_FPS} fps), "
            f"min-offset={min_offset_sec}s ({self.min_offset_frames} frames), "
            f"mel {mel_note}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.ids)

    def path(self, track_id: int) -> pathlib.Path:
        """<root>/<id % 100:02d>/<id>.npy — see `mel_spec_path`."""
        return mel_spec_path(self.root, track_id)

    def load(self, track_id: int) -> np.ndarray:
        """The track's full native (n_mels_native, T) float32 log-mel, memmapped."""
        return np.load(self.path(track_id), mmap_mode="r")

    def random_window(
        self, spec: np.ndarray, exclude: int | None = None
    ) -> tuple[tc.Tensor, int]:
        """A random window as (1, 1, n_mels, win_frames), plus its start frame."""
        win, start = self._draw(spec, exclude)
        return self._merge_mels(self._to_tensor(win), self.n_mels), start

    def _draw(
        self, spec: np.ndarray, exclude: int | None = None
    ) -> tuple[np.ndarray, int]:
        # reject near-silent windows so nothing downstream learns a degenerate
        # "quiet" embedding point; re-draw up to MAX_SILENCE_TRIES, then accept
        # whatever we last drew.
        if self.min_window_db is None:
            return random_spec_window(
                spec, self.win_frames, self.min_offset_frames, exclude
            )
        for _ in range(MAX_SILENCE_TRIES):
            win, start = random_spec_window(
                spec, self.win_frames, self.min_offset_frames, exclude
            )
            if float(win.mean()) >= self.min_window_db:
                return win, start
        return win, start  # track is (nearly) all-silent; take the last draw

    @staticmethod
    def _to_tensor(win: np.ndarray) -> tc.Tensor:
        # copy off the memmap before it is released, then emit (1, 1, n_mels, T)
        arr = np.ascontiguousarray(win).copy()
        return tc.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _merge_mels(spec: tc.Tensor, n_mels: int) -> tc.Tensor:
        """Merge the mel-frequency axis H -> n_mels on a (1, 1, H, T) log-mel.

        Done in linear-power space: combining mel bands is an energy sum, and
        averaging the dB values would under-estimate it (a quiet band next to a
        loud one blends to a mid value instead of ~the loud one). Bilinear
        resample 96 -> 64 is an approximation of the exact 64-band filterbank the
        waveform/eval path computes, but keeps the model resolution-agnostic and
        cuts the conv activations ~1/3. Returns the spec unchanged if H <= n_mels.
        """
        H = spec.shape[2]
        if H <= n_mels:
            return spec
        amp = tc.pow(10.0, spec.float() / 10.0)  # log-mel dB -> linear power
        amp = nn.functional.interpolate(
            amp, size=(n_mels, spec.shape[3]), mode="bilinear", align_corners=False
        )
        db = 10.0 * tc.log10(amp.clamp(min=1e-12))  # power -> dB (floor ~ -120)
        return db.to(spec.dtype)


class MelSpecPairDataset(tc.utils.data.Dataset):
    """Barlow-Twins pair dataset over precomputed log-mel spectrograms.

    Time-stability positives are two random ~window_sec windows of the SAME
    track's spectrogram, forced at least min_offset_sec apart, each independently
    distorted by `spectrogram_augmenter` (see augmenter.py) — no waveform is
    involved anywhere in this path, so it matches the waveform path's distortion
    exactly without decoding anything.

    window_sec / min_offset_sec are given in seconds and converted at MEL_FPS.
    Silence handling is a per-window mean-dB gate (min_window_db, default -85 dB):
    near-empty windows are re-drawn. Pass min_window_db below MEL_FLOOR_DB (or
    None) to disable. The windowing/merge itself lives in MelSpecWindowSource,
    shared with the eval and QA scripts.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        window_sec: float = DEFAULT_WIN_LEN_S,
        min_offset_sec: float = DEFAULT_MIN_OFF_S,
        spectrogram_augmenter: SpectrogramAugmenter | None = None,
        min_window_db: float | None = DEFAULT_MIN_WINDOW_DB,
        n_mels: int = 64,
    ) -> None:
        self.source = MelSpecWindowSource(
            root,
            window_sec=window_sec,
            min_offset_sec=min_offset_sec,
            n_mels=n_mels,
            min_window_db=min_window_db,
        )
        self.spectrogram_augmenter = spectrogram_augmenter
        # passthroughs: these used to be set directly on the dataset
        self.root = self.source.root
        self.ids = self.source.ids
        self.win_frames = self.source.win_frames
        self.min_offset_frames = self.source.min_offset_frames
        self.n_mels = self.source.n_mels
        self.min_window_db = self.source.min_window_db

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[tc.Tensor, tc.Tensor]:
        spec = self.source.load(self.ids[index])  # (96, T) float32, memmapped
        # both windows are merged to n_mels BEFORE augmentation, so distortions
        # act at the resolution the model (and the 64-mel eval path) actually see
        spec_a, start_a = self.source.random_window(spec)
        spec_b, _ = self.source.random_window(spec, exclude=start_a)
        del spec  # both windows are already copied into their own tensors
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
