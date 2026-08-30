"""Dump 20 augmented wavs for manual listening (QA for AudioAugmenter output).

Usage:
    python -m music_encoding.test_augmentation --mtg-data <data> [--audio-root <audio>]
"""

import argparse
import pathlib

import torch as tc
import torchaudio as ta

from music_encoding.augmenter import AudioAugmenter
from music_encoding.mtg import MTGJamendoBase
from music_encoding.twin_dataset import resample

N = 20


def main() -> None:
    parser = argparse.ArgumentParser(
        description="dump 20 augmented wavs for manual listening"
    )
    parser.add_argument(
        "--mtg-data", required=True, type=pathlib.Path,
        help="MTG-Jamendo data dir (contains autotagging.tsv, raw.meta.tsv)",
    )
    parser.add_argument(
        "--audio-root", default=None, type=pathlib.Path,
        help="dir where MTG audio unpacked (defaults to --mtg-data)",
    )
    args = parser.parse_args()

    ds = MTGJamendoBase(args.mtg_data, audio_root=args.audio_root)
    augmenter = AudioAugmenter()
    for i in range(min(N, len(ds))):
        samples = ds[i]["audio"].get_all_samples()
        wav = resample(samples.data, samples.sample_rate, 22050)
        wav_augmented = augmenter(wav)
        ta.save(f"test_aug_{i}.wav", wav_augmented.unsqueeze(0), 22050)
    print(f"[main] wrote test_aug_0..{min(N, len(ds)) - 1}.wav")


if __name__ == "__main__":
    main()
