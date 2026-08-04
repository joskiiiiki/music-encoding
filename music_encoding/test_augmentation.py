import pathlib
from music_encoding.twin_dataset import load_resampled_cached, resample
import torch as tc
import datasets
from music_encoding.augmenter import AudioAugmenter

from torchcodec.decoders import AudioDecoder
import torchaudio as ta

ds = datasets.load_dataset("benjamin-paine/free-music-archive-small")["train"]
for i in range(20):
    decoder: AudioDecoder  = ds[0]["audio"]
    samples = decoder.get_all_samples()
    wav: tc.Tensor = samples.data  # (num_channels, num_samples), float32
    sr: int = samples.sample_rate

    wav =resample(wav, src_sr=sr, tgt_sr=22050)

    augmenter = AudioAugmenter()

    wav_augmented = augmenter(wav)

    ta.save(f"test_aug_{i}.wav", wav_augmented, 22050)


