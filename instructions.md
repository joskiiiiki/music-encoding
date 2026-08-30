# Augmentation & Hyperparameter Changes

Context: Barlow Twins audio encoder (`music_encoding`) for a Shazam-style
fingerprinting model. We reviewed the original Barlow Twins paper
(arXiv:2103.03230) and Audio Barlow Twins (arXiv:2209.14345), which adapts BT
to audio and found RRC (random-resized-crop on the spectrogram) to be the
single most effective augmentation, batch sizes of 64-512 to be optimal
(unlike vision's need for huge batches), and projector dimensionality to
*saturate* around ~1024-2048 for audio (unlike vision, where it kept helping
up to 16k+). The changes below apply those findings, informed by our own
prior experiments (disjoint-window-only got 42-46% genre k-NN; our first
waveform-augmented run underperformed at ~31%, likely due to a weak/narrow
augmentation set — noise+reverb+lowpass+gain — rather than "augmentation
doesn't help").

## 1. `music_encoding/augmenter.py` — `AudioAugmenter` (waveform-domain, CPU, per-sample)

This runs inside `FMAPairDataset.__getitem__`, NOT batched on GPU — we tested
a GPU-batched version and it was measured to be *slower* than the CPU
per-sample version for our workload, so keep this on CPU.

- **Remove `pitch_shift`** entirely (already done — confirm it's gone, not
  just unused).
- **Comment out `add_reverb`** in `__call__` (already done by hand — confirm
  the FFT-convolution version of `add_reverb` stays in the class for later
  re-enabling/ablation, just not called by default).
- **Make `add_noise` unconditional** — remove its probability gate. Keep the
  SNR itself randomized within `noise_snr_range` (already implemented) so
  difficulty still varies, but noise should be applied to every sample, every
  call, since it's a near-constant feature of real-world query audio and
  removing the gate also avoids leaving "clean vs clean" pairs in training
  (which we suspect contributed to embedding-space collapse in earlier runs).
- **Add a new `Mixup` transform** to this file (in-batch waveform mixing, NOT
  a separate dataset load):
  - Sample a partner waveform from elsewhere in the *same batch* (shuffle
    indices, avoid self-pairing).
  - Blend with a weight sampled from `Beta(alpha, alpha)` (default
    `alpha=0.4`), clamped so the target signal's weight is never below
    `target_weight_min` (default `0.6`) — the original clip must stay
    dominant.
  - Apply with probability `p` (default `0.5`).
  - This needs to operate on a *batch* of waveforms (shape `(B, N)`), not a
    single sample — it can't live in `__getitem__` the way the other
    transforms do, since it needs visibility into other samples in the
    batch. Call it explicitly in the training loop, right after
    `wav_a, wav_b = wav_a.to(device), wav_b.to(device)` and before other
    per-sample-batched steps, applied independently to `wav_a` and `wav_b`
    (each gets a different random partner).

Resulting default `__call__` behavior for the per-sample CPU augmenter:
noise (always) → lowpass (probabilistic) → gain (probabilistic). Reverb and
pitch shift excluded by default; Mixup applied separately in the training
loop, not inside this class's `__call__`.

## 2. `music_encoding/augmenter.py` — new `SpectrogramAugmenter` class

Add alongside `AudioAugmenter`. Operates on mel-spectrograms of shape
`(B, 1, n_mels, T)` — i.e. AFTER `self.mel`/`self.to_db` inside the model,
BEFORE the conv stack. This is the RRC (random-resized-crop) implementation,
fully batched via `affine_grid` + `grid_sample` (no per-sample Python loop).

- Constructor args: `rrc_scale_range=(0.8, 1.0)`, `rrc_ratio_range=(0.85, 1.15)`,
  `rrc_p=0.6`. Keep the scale range gentler than typical vision RRC (which
  often crops to 8% of the image) — we need fine spectral/pitch detail
  preserved for track-identity discrimination, not just coarse invariance.
  Tune empirically later; start conservative.
- `random_resized_crop(spec, mask)`: per-sample random crop box (area +
  aspect ratio sampled independently), built as an affine transform matrix,
  applied via `F.affine_grid` + `F.grid_sample` (bilinear, `align_corners=False`
  on both calls — must match).
- `__call__(spec)`: builds a per-sample boolean mask from `rrc_p`, applies
  `random_resized_crop`, done. Only RRC — do not bundle in freq/time masking
  (SpecAugment) unless explicitly asked for later; keep this class scoped to
  just the crop-and-resize transform.

## 3. `music_encoding/model.py` — wire `SpectrogramAugmenter` into `SiameseEncoderBT`

- Add an optional `spec_augmenter: SpectrogramAugmenter | None = None`
  constructor arg, stored on `self`.
- In `forward`, right after computing `x = self.to_db(self.mel(wav)).unsqueeze(1)`
  and before `self.conv(x)`, apply the augmenter only during training:
  ```python
  if self.training and self.spec_augmenter is not None:
      x = self.spec_augmenter(x)
  ```
- Keep mel-spectrogram computation inside the model's `forward` (do NOT move
  it into the `Dataset` or training loop) — every consumer of this model
  (training loop, k-NN eval script, stability test script) needs identical
  mel-spectrogram settings, and keeping it inside the model is what
  guarantees that consistency. This was a deliberate decision — do not
  "simplify" by moving spectrogram computation elsewhere without flagging it.

## 4. `music_encoding/train.py` — wire in `Mixup` and `SpectrogramAugmenter`

- Instantiate `Mixup()` and `SpectrogramAugmenter()` alongside the existing
  model/optimizer/scheduler setup.
- In the batch loop, after moving `wav_a`, `wav_b` to device, apply `Mixup`
  independently to each:
  ```python
  wav_a = mixup(wav_a)
  wav_b = mixup(wav_b)
  ```
- Pass the `SpectrogramAugmenter` instance into `SiameseEncoderBT(...,
  spec_augmenter=spec_augmenter)` at construction time.

## 5. Hyperparameters — `proj_dims` and `lambd`

- **Do NOT push `proj_dims` to 4096 or 8192.** Audio Barlow Twins found
  performance saturates around an output size of ~1024-2048 for audio
  (unlike vision, which kept improving up to 16k+), with considerable
  degradation observed at 16,384. Keep `proj_dims=2048` (current value) or
  try dropping to `1024` as a comparison — do not go higher.
- **Recalibrate `lambd` into the paper-supported range `0.002 < λ < 0.05`**,
  rather than the strict `λ ≈ 1/D` heuristic used previously. Try
  `lambd=2e-3` to `5e-3` as a starting point at `proj_dims=2048` (this is
  a meaningfully different value from the previously-used `4.9e-4`, which
  is just below the paper's stated preferred range).

## 6. Batch size

- Keep batch size in the `64-512` range. Audio Barlow Twins found this range
  optimal for audio and specifically noted degradation *outside* it — audio
  does not benefit from the very large batches vision-domain BT prefers.
  Current batch size of 512 is fine; do not push higher without a specific
  reason.

## 7. After implementing — re-run evaluation

Once the above changes are in place, run a fresh training run (resume from
a disjoint-only checkpoint if warm-starting, or from scratch — try both if
time allows) and re-run BOTH evals to check for improvement:

1. Genre k-NN eval (existing script) — compare against baseline results:
   disjoint-only ≈ 42-46%, first augmented attempt (noise+reverb+lowpass+gain,
   no RRC/Mixup) ≈ 31%.
2. Intra/inter-track stability eval (existing script) — compare against
   baseline: intra-track mean ≈ 0.98-0.99, inter-track mean ≈ 0.81-0.87
   (lower inter-track mean with wider std = less embedding-space collapse,
   which is what we're trying to improve for the fingerprinting use case).

Track results in a simple table (checkpoint config → both eval numbers) so
we can compare against the runs already logged in this conversation.
