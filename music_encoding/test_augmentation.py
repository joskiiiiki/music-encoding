"""Manual QA for the augmentation chain.

Two modes, because the pipeline has two augmenters and only one of them is live:

  --mel-root    (default path) dumps a grid PNG of the LIVE `SpectrogramAugmenter`:
                a clean window followed by several independent augmented draws of
                that same window. This is the transform training actually applies
                (see augmenter.py), so it is the one worth eyeballing — and it needs
                no decoded audio, which is why it works on this machine.
  --mtg-data    LEGACY: writes 20 augmented .wav files from the waveform-domain
                `AudioAugmenter`, kept only as a listening reference. Needs the MTG
                audio unpacked.

Both modes also print numbers, since a spectrogram is hard to judge by eye alone:
mean dB and 5th-percentile dB (clean vs augmented — noise should lift the floor
without burying the loud content), and the correlation between two augmented views
of the SAME window. That last one is the "two-view share": it is ~1 when the
augmentation preserves the shared signal and collapses toward 0 when it destroys
it, which is exactly how the peak-anchored-noise bug was found (see instructions.md
§8).

Usage:
    python -m music_encoding.test_augmentation --mel-root ~/mtg_jamendo
    python -m music_encoding.test_augmentation --mtg-data <data> --audio-root <audio>
"""

import argparse
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torchaudio as ta

from music_encoding.augmenter import AudioAugmenter, SpectrogramAugmenter
from music_encoding.mtg import MTGJamendoBase
from music_encoding.twin_dataset import (
    DEFAULT_MIN_WINDOW_DB,
    MEL_FLOOR_DB,
    MelSpecWindowSource,
    resample,
)

N = 20          # legacy waveform mode: how many augmented wavs to write
N_ROWS = 8      # mel mode: tracks (rows) in the grid
N_DRAWS = 3     # mel mode: augmented draws per row, alongside the clean window


def _corr(x: tc.Tensor, y: tc.Tensor) -> float:
    """Pearson correlation between two flattened windows."""
    x = x.flatten().float()
    y = y.flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp(min=1e-8)
    return float((x * y).sum() / denom)


def mel_mode(args) -> None:
    aug = SpectrogramAugmenter()
    source = MelSpecWindowSource(
        args.mel_root, min_window_db=args.min_window_db
    )
    if args.n > len(source):
        print(f"[main] only {len(source)} tracks available; using that many")
    n = min(args.n, len(source))
    random.seed(args.seed)

    fig, axes = plt.subplots(
        n, N_DRAWS + 1, figsize=(4 * (N_DRAWS + 1), 2.1 * n), squeeze=False
    )
    print("[main] two-view share is corr(aug_i, aug_j) on the same clean window")
    for row, t in enumerate(range(n)):
        spec = source.load(source.ids[t])
        clean, _ = source.random_window(spec)
        del spec
        draws = [aug(clean) for _ in range(N_DRAWS)]
        vmax = float(clean.max())
        panels = [("clean", clean)]
        panels += [(f"aug {i + 1}", d) for i, d in enumerate(draws)]
        for col, (label, panel) in enumerate(panels):
            ax = axes[row][col]
            ax.imshow(
                panel[0, 0].numpy(),
                aspect="auto",
                origin="lower",
                vmin=MEL_FLOOR_DB,
                vmax=vmax,
                cmap="magma",
            )
            ax.set_title(f"{label} (track {source.ids[t]})", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        flat_clean = clean.flatten().float()
        flat_aug = tc.stack([d.flatten().float() for d in draws]).flatten()
        shares = [
            _corr(draws[i], draws[j])
            for i in range(len(draws))
            for j in range(i + 1, len(draws))
        ]
        mean_clean = float(flat_clean.mean())
        mean_aug = float(flat_aug.mean())
        p5_clean = float(tc.quantile(flat_clean, 0.05))
        p5_aug = float(tc.quantile(flat_aug, 0.05))
        print(
            f"[main] track {source.ids[t]:>9}: "
            f"mean {mean_clean:7.1f} -> {mean_aug:7.1f} dB | "
            f"p5 {p5_clean:7.1f} -> {p5_aug:7.1f} dB | "
            f"two-view share {np.mean(shares):.3f}"
        )

    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    plt.close(fig)
    print(f"[main] wrote {args.out}")


def waveform_mode(args) -> None:
    ds = MTGJamendoBase(args.mtg_data, audio_root=args.audio_root)
    augmenter = AudioAugmenter()
    for i in range(min(N, len(ds))):
        samples = ds[i]["audio"].get_all_samples()
        wav = resample(samples.data, samples.sample_rate, 22050)
        wav_augmented = augmenter(wav)
        ta.save(f"test_aug_{i}.wav", wav_augmented.unsqueeze(0), 22050)
    print(f"[main] wrote test_aug_0..{min(N, len(ds)) - 1}.wav")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="dump augmented examples for manual QA"
    )
    parser.add_argument(
        "--mel-root", default=None, type=pathlib.Path,
        help="dir of precomputed MTG-Jamendo log-mel .npy — renders the live "
             "SpectrogramAugmenter as a grid PNG (no audio needed)",
    )
    parser.add_argument(
        "--mtg-data", default=None, type=pathlib.Path,
        help="LEGACY waveform mode: MTG-Jamendo data dir; writes augmented .wavs "
             "from the waveform AudioAugmenter",
    )
    parser.add_argument(
        "--audio-root", default=None, type=pathlib.Path,
        help="dir where MTG audio unpacked (defaults to --mtg-data); legacy mode only",
    )
    parser.add_argument(
        "--n", type=int, default=N_ROWS, help=f"rows in the grid (default: {N_ROWS})",
    )
    parser.add_argument(
        "--out", default="test_aug_spectrograms.png",
        help="output PNG (default: test_aug_spectrograms.png)",
    )
    parser.add_argument(
        "--min-window-db", default=DEFAULT_MIN_WINDOW_DB, type=float,
        help="re-draw mel-mode windows quieter than this mean dB, matching "
             f"training's gate (default {DEFAULT_MIN_WINDOW_DB}); pass < -90 to "
             "disable so even near-silent windows are shown",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="window-draw seed (default: 0)",
    )
    args = parser.parse_args()

    if (args.mel_root is None) == (args.mtg_data is None):
        parser.error(
            "set exactly one of --mel-root (spectrogram grid, the live augmenter) "
            "or --mtg-data (legacy waveform .wav dump)"
        )

    if args.mel_root is not None:
        mel_mode(args)
    else:
        waveform_mode(args)


if __name__ == "__main__":
    main()
