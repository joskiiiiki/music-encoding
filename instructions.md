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

## 8. 2026-09-09 — noise reference bug (peak → p95) + Mixup dropped (loss wasn't decreasing)

A fresh 64-mel run flatlined (loss ~1100, `on_diag` pinned ~950/1024 across
16 epochs → LR auto-cut by ReduceLROnPlateau). Root cause, found by ablation on
real windows + short fresh-model runs (240 steps, fp16):

- **`SpectrogramAugmenter._add_noise` anchored its SNR 5–20 dB below each
  window's GLOBAL PEAK.** That was a porting regression: the legacy waveform
  `AudioAugmenter.add_noise` references waveform MEAN POWER. A 5 s music
  window's peak is a single percussive transient; only ~10% of spectrogram bins
  sit within 20 dB of it, so the "hiss" floor landed above ~90% of the content.
  Two augmented views of the SAME window shared almost no signal
  (`corr(aug,aug)` ≈ 0.07 vs 0.75 fixed), so the invariance target was
  unlearnable — `on_diag` could never descend.
- Ablation result (on_diag final @240): no-aug different-windows **88** (learns);
  same-window double-aug with peak noise ~900 (fails), with p95 noise **101**
  (learns). Confirms the noise reference was the primary blocker.
- **Fix applied: reference noise to each window's 95th-percentile per-bin
  amplitude** (robust "loud-content" level; a window mean is unusable — it's
  floor-dominated). Noise stays unconditional, `noise_snr_range=(5,20)`
  unchanged. Measured two-view share 0.15 → ~0.75.
- **`Mixup` also dropped from the default train loop.** Decomposition showed
  every simpler objective learns (A 88, D 101, D2 same+aug+mixup 178, C2
  diff+mixup 258) but the full train compound — different windows, each
  augmented — is the hard case (C1 no-mixup 518 slow, C with-mixup 802 flat);
  Mixup tips it over. `train()` keeps `mixup=` as an experiment hook only.
- Next: rerun a fresh short run with these fixes; confirm `on_diag` descends.
  If the diff-window + augmented compound is STILL too slow to learn, the chain
  itself (reverb/lowpass/gain/RRC probabilities/magnitudes) is the remaining
  knob — the compound's two-view ceiling is ~0.75 even with noise fixed.

## 9. 2026-09-11 — it was never the chain. `lambd` was 10–25× too high.

§8 ended by pointing at the augmentation chain as "the remaining knob". A direct
ablation says otherwise, and the real cause is a hyperparameter misconfiguration.

**The config bug.** `train.py`'s CLI default is `--lambd 5e-2`, which overrides
both `BarlowTwinsLoss`'s own default (`2e-2`) and everything the docs claim
(CLAUDE.md says `2e-2` in one place and "currently `3e-3`" in another; §5 above
recommends `2e-3`–`5e-3`). `setup_local_train.sh` passes no `--lambd`, so every
mel-mode run so far trained at **5e-2 — the very top of the BT paper's supported
range**. The e42 log decomposes exactly as `388.7 + 0.05 × 4757 = 626.6`, i.e.
`off_diag` was **38% of the objective** while the mean off-diagonal correlation was
already only |c| ≈ 0.067. A third of the gradient was being spent on an
already-solved term instead of on the invariance, which is still far from solved
(mean diagonal correlation ≈ 0.38 at e42, vs ≈ 0.71 for the no-aug case).

**The ablation** (128 real window pairs, drawn ONCE and reused by every config so a
difference is the config's doing and not window-sampling variance; 240 steps, fresh
`SiameseEncoderBT(proj_dims=1024)`, fp16, lr 1e-4; `on_diag` alongside the
D-normalised `off_diag_norm`, because `on_diag` alone is degenerate — dropping
`lambd` to ~0 drives it down by collapsing the embedding):

| `lambd` | `on_diag` | `off_diag_norm` |
|---|---|---|
| 5e-2 (what we ran) | 83 | 0.00400 |
| 2e-2 | 19 | 0.00584 |
| 1e-2 | 5 | 0.00690 |
| **5e-3** | **12** | **0.00867** |
| 2e-3 | 2 | 0.01157 |

| chain variant (@ 5e-2) | `on_diag` |
|---|---|
| current | 97 |
| soft-content | 80 |
| soft-all | 77 |
| noise+rrc | 78 |
| noise-only (no content distortions, no RRC) | 73 |
| control: same window, full chain | 138 |

**Conclusion: the chain is NOT the bottleneck.** Deleting every content distortion
*and* RRC buys ~25%; fixing `lambd` buys up to 40×. `off_diag_norm` never exceeds
0.0116 even at `lambd=2e-3` (mean |c| ≤ 0.11), so the invariance is not being bought
by collapsing the embedding. **No `augmenter.py` change is justified** — the chain
stays as it is.

Read absolute `on_diag` here only as a *relative* signal: the fixed 128-pair set
overfits hard, so these numbers sit far below real-run levels (447 at e16/e42).

### Correction — the harness pointed the right way but `on_diag` is the wrong target

I first concluded "choose `lambd = 5e-3`" from this table alone. **Real runs say
otherwise, and the correction matters more than the original finding.**

Built a direct retrieval metric (for a sampled window, what fraction of its top-k
nearest neighbours are the SAME track — the objective this model actually exists for;
now `music_encoding/test_retrieval.py`). Real runs at **matched 17 epochs**:

| `lambd` (17 ep) | `on_diag` | retrieval top-1 | collapsed /1000 | genre `single` |
|---|---|---|---|---|
| 5e-2 | 447 | 0.3453 | **50** | 0.410 |
| 2e-2 | 334 | 0.3520 | 209 | 0.436 |
| 5e-3 | 228 | 0.3553 | 70 | 0.438 |
| *(5e-2 @ 42 ep, e42)* | *389* | *0.435* | *42* | *0.425* |

**`lambd` is a wash, and `on_diag` is not a valid proxy.** Retrieval is flat across
all three `lambd` values (0.345–0.355, within sampling noise) — the 0.435 I first
attributed to `5e-2` was entirely **epochs** (42 vs 17), which is why the earlier
draft of this section ran the wrong comparison. What actually moves: genre accuracy
improves slightly as `lambd` falls (0.410 → 0.436 → 0.438), collapse prefers high
`lambd` (50 vs 209/70), and `on_diag` falls steeply either way.

So `on_diag` can be driven down ~2× by a change with **no retrieval benefit at all**.
Mechanism: `off_diag` (redundancy reduction) is what *spreads* the embedding cloud.
Lowering `lambd` removes that pressure, the cloud contracts, and the two views of a
track move closer — exactly what `on_diag` measures — while different tracks crowd
together. Use `on_diag` only for its original job (spotting a config whose views
share too little signal to learn at all, i.e. pinned near ~800–1000), and judge real
changes on retrieval + collapse.

**Decision: keep `lambd = 5e-2`.** It is the best on collapse (the axis this project
explicitly wants to reduce) and the worst on genre by only ~0.03, i.e. ~2 standard
errors on 1409 clean-label tracks. The doc inconsistency (CLAUDE.md saying `2e-2` and
`3e-3` in different places, §5 recommending `2e-3`–`5e-3`) is a *doc* problem: those
recommendations were reasoned about for `on_diag`, which we now know is the wrong
target. CLAUDE.md's `lambd` note is updated accordingly.

**The real remaining lever is the objective, not any hyperparameter.** Barlow Twins
pushes two views of the same track together and decorrelates dimensions, but never
pushes *different tracks apart* — redundancy reduction is not instance
discrimination. That is why raw inter-track similarity sits at 0.80–0.86 and
retrieval top-1 stalls at ~0.44. A contrastive/InfoNCE term (or VICReg-style
variance+invariance) is the next thing to try; the augmentation chain (§8's
"suspected culprit"), the mel resolution (`n_mels` 96 measured equal to 64) and
`lambd` are all now ruled out for it.

Note also: whitening barely helps retrieval (0.4353 → 0.4400), unlike the intra/inter
diagnostic where it transformed the gap — so the weak retrieval is *not* poor
conditioning.

## 10. 2026-09-11 — inter-sample repulsion (InfoNCE) added on top of BT

**The diagnosis, measured.** `test_retrieval.py` (new) asks the question the model
exists to answer — for a 5 s window, is its nearest neighbour its own track? On e42
that is **0.4353** (chance 2.4e-05), and whitening barely moves it (0.4400). Two free
diagnostics say *why*:

| diagnostic (4,000 sampled windows) | value |
|---|---|
| pairwise cosine between random *different* windows | **0.833** |
| uniformity of the learned space | −0.630 |
| uniformity of an isotropic Gaussian on the same sphere | −3.937 |
| **gap (more negative = more spread; this is +3.3 the wrong way)** | **+3.308** |

The space is far more crowded than random. BT has invariance (`on_diag`) and
*dimension* decorrelation (`off_diag`) but **no term that spreads samples apart**.

**Why not VICReg.** Its variance + covariance terms amount to a whitening objective,
and whitening the windows post-hoc barely moved retrieval. Nuance: post-hoc whitening
is a linear map on frozen features whereas a VICReg variance term changes what the
encoder learns, so this weakens the case rather than settling it — but the direct fix
for cos≈0.83 between different tracks is repulsion, not better conditioning.

**Why not supervised contrastive on the genre tags.** It would work, and labels exist
for all 32,783 tracks, but it makes the genre k-NN eval measure label leakage instead
of learned representation — disqualifying for a self-supervised project.

**Implementation** — additive and default-off, so no earlier number changes:
`InfoNCELoss` in `model.py` (NT-Xent; `BarlowTwinsLoss` untouched), wired into
`train.py` behind `--nce-weight` (default **0.0** = plain BT) and `--nce-temp` (0.1).
An `nce` column was added to the train CSV and the epoch log prints the term's
**share of the total loss**. Two properties were checked because they are what could
silently break it: **no false negatives** (one window-pair per track per epoch, so a
batch never holds a second window of the same track — a future multi-crop change
would need a mask), and the `(arange + B) % 2B` positive mapping (verified as
`[384, 385, 767, 0, 1, 383]` at B=384). A unit check confirmed the loss is
≈ln(2B−1)=6.64 at random init, 0.03 for ideal pairs, symmetric, and differentiable.

**The scale trap.** BT totals ~1,100 at init / ~640 trained; InfoNCE is
~ln(2B−1) ≈ 6.6 at init. A useful weight is therefore **O(30), not a small
fraction** — weighted like an ordinary loss coefficient it would vanish. First
attempt: `--nce-weight 30` (≈20% of the total at init), `--nce-temp 0.1`.

**The risk being watched.** Instance discrimination is documented to learn
*instance-level* features that cluster **less** semantically, so retrieval may improve
while genre k-NN drops. Hence a modest weight on top of BT rather than replacing it,
and verifying both metrics — not retrieval alone.

**Batch size.** Peak VRAM at 64 mel bands / fp16, with real train steps:
B=256 3.97 GB, B=320 4.94 GB, **B=384 5.90 GB** (0.75 GB driver-free), B=448
6.87 GB. So 384 fits at 64 bands — the recorded B=384 OOM was at **96** bands (1.5×
the activations).

**Also fixed while running this:** `train.py`'s per-batch and per-epoch prints had no
`flush=True`, so with stdout redirected to a log they sat in the 8 KB block buffer —
making a killed run indistinguishable from a hung one. Both now flush. (The
per-epoch CSV already flushed.) Long ROCm runs on this box keep dying at epoch 0 with
no traceback, so checkpoint-per-epoch + `-r` resume is the only safe pattern.

### Results — attempt 1: NCE on `z`, weight 30, temp 0.1, batch 384, 17 epochs

Matched against **e16** (17 epochs, batch 256, plain BT), same corpus, same windows:

| metric | e16 (no NCE) | NCE on `z` | |
|---|---|---|---|
| retrieval top-1 **raw** | 0.3453 | 0.3847 | +0.039 — within the ±0.03 sampling noise |
| retrieval top-1 **whitened** | 0.3620 | **0.3680** | **+0.006 — nothing** |
| genre `primary` / `any-tag` / `single` | 0.302 / 0.416 / 0.410 | 0.309 / 0.427 / **0.426** | slight gain |
| raw intra → inter | 0.9293 → 0.7956 | 0.9468 → **0.8498** | raw inter **worse** |
| raw gap | 0.1337 | **0.0970** | narrower |
| whitened gap | 0.3860 | 0.3952 | ~flat |
| collapsed per 1000 | **50** | **167** | **much worse** |

**Verdict: it does not work at this weight.** The decisive tell is that the
**whitened** retrieval — the informative space — did not move, so the raw gain is
cancelled by raw inter-track similarity rising just as much. And the extra collapse
is *less* explained by genuine repertoire similarity than before (same-artist lift
falls **167× → 89×**, and classical's genre lift 4.4× → 2.7×), i.e. it is crowding
rather than real duplicates.

Two useful things it did establish:
- **The term is live and doing something.** `off_diag_norm` sat ~22% lower for the
  whole run (0.00468 vs 0.00603 at e16) — InfoNCE spreads the embedding across
  dimensions, so it supplies *implicit* redundancy reduction. Its share of the loss
  held steady at 17.6–17.9%, so weight 30 is well-calibrated to the ~20% target.
- **It did not change the invariance.** `on_diag` tracked the no-NCE run almost
  exactly (429.4 vs 433.7 at e16). My first-epoch read that NCE *improved* `on_diag`
  (795.9 vs 816.4) was noise plus a batch-size confound — worth recording as a
  reminder not to draw conclusions from epoch 0.

### Attempt 2: repulsion on `emb` instead of `z`

Mechanistic hypothesis for the null result: NCE acts on `z` (1024-d projector, what
BT shapes), but **retrieval is measured on `emb` (128-d)** — the space `build_chroma`
stores. BT only shapes `emb` *indirectly*, through `z`, so the repulsion may never
reach the representation being evaluated. Added `--nce-on {z,emb}` (default `z`) and
re-running at the same weight/temperature; the loss L2-normalizes its inputs, so an
unnormalized 128-d `emb` is handled correctly (verified scale-invariant).

*(Attempt-2 results appended once evaluated.)*

### Results — attempt 2: NCE on `emb`, weight 30, temp 0.1, batch 384, 17 epochs

| metric | e16 (no NCE) | NCE on `z` | **NCE on `emb`** |
|---|---|---|---|
| retrieval top-1 **raw** | 0.3453 | 0.3847 | 0.3507 |
| retrieval top-1 **whitened** | 0.3620 | 0.3680 | **0.3567** |
| genre `primary` / `any-tag` / `single` | 0.302 / 0.416 / 0.410 | 0.309 / 0.427 / 0.426 | 0.306 / 0.424 / 0.415 |
| raw intra → inter | 0.9293 → 0.7956 | 0.9468 → 0.8498 | **0.7814 → 0.4177** |
| raw gap | 0.1337 | 0.0970 | **0.3636** |
| **whitened gap** | **0.3860** | 0.3952 | **0.3848** |
| collapsed per 1000 | 50 | 167 | **0** (none above 0.99) |

**Repelling `emb` reshapes the raw space dramatically** — inter 0.7956 → 0.4177, std 0.063 → 0.199, zero near-duplicate pairs — so `--nce-on` genuinely matters, and repelling the 1024-d projector (attempt 1) barely touches the 128-d space that is actually measured.

**But it is a learned reparameterization, not new information.** The decisive rows are the last two: the **whitened gap is unchanged** (0.3848 vs e16's 0.3860), and e16's *whitened* metrics (gap 0.3860, retrieval 0.3620) already match nce_emb's *raw* ones (0.3636, 0.3507). NCE on `emb` bakes in approximately what post-hoc whitening does — and the project already has `tracks_whitened` for exactly that, for free. The raw crowding is a removable common mode; removing it does not improve discrimination, which is why retrieval does not move.

**Conclusion: NCE is off for the final run.** Judged on retrieval — the operational metric — it gives nothing at weight 30, in either space. The intra/inter histogram is conditioning-sensitive (whitening transforms it), so the spectacular raw numbers above are a *proxy* win, the same trap as optimising `on_diag` in §9. Recorded as explored-and-rejected rather than carried forward.

**Caveat, stated honestly:** 17 epochs is not converged (`on_diag` ~430 here vs ~340 at e85), so this is "no effect measurable at 17 epochs", not a proof that no weight/space/duration would help. If repulsion is revisited, the things to try are a much larger weight (30 → 100, where the term would dominate rather than share), a lower temperature for harder negatives, and hard-negative mining (in-batch negatives are random tracks, i.e. easy).

The 100-epoch run therefore uses **plain BT, batch 384, λ=5e-2, plus cosine LR** — the epochs lever is the one confirmed to move retrieval (0.345 @ 17 ep → 0.435 @ 42 → 0.525 @ 86), and cosine fixes the schedule that measurement showed to be inert (43 epochs at a constant 1e-4, §2.7).

## 11. 2026-09-11 — the lever is optimizer STEPS, not epochs (and batch 384 costs steps)

The 100-epoch run at batch 384 with cosine LR came out **worse than the 86-epoch
batch-256 run on every headline metric**, despite a *lower* BT loss (589.7 vs ~640)
and the lowest `off_diag_norm` of any run (0.00435). Another objective/metric
mismatch — and this one has a clean, quantitative explanation.

| run | batch | epochs | **steps** | retrieval top-1 |
|---|---|---|---|---|
| e16 | 256 | 17 | 2,176 | 0.3453 |
| e42 | 256 | 42 | 5,376 | 0.4353 |
| final100c | **384** | **100** | **8,500** | 0.4540 |
| e85 | 256 | 86 | **11,008** | **0.5247** |
| cont150 (resumed e85) | 256 | 150 | ~19,200 | *pending* |

Retrieval tracks **steps** almost monotonically. Batch 384 yields 85 batches/epoch
versus 128 at batch 256, so 100 epochs at 384 is *fewer* steps than 86 epochs at 256
— which is the whole regression. Batch size here is not just a memory knob: **at
fixed epochs it sets how much optimisation happens.** (The literature's "BT saturates
at 64–512" bounds what one *step* can learn, not how many steps you get.)

Consequences:

- **Prefer batch 256** — it maximises steps/epoch on this corpus and still fits
  comfortably (measured 3.97 GB at 64 bands). Reserving 384 for when VRAM forces it
  would leave quality on the table.
- **Do not anneal at this stage.** The 8,500-step point sits *below* the step-count
  trend, so cosine LR cost something on top of the step deficit: with the loss still
  descending ~−1/epoch there is no plateau for annealing to exploit, and decaying the
  LR just makes the remaining steps shorter. Keep the plateau schedule at 1e-4 and
  accumulate steps instead.
  - Correction to an overgeneralization: the plateau scheduler is not *permanently*
    inert, only far too slow to matter. In the 300-epoch continuation it held 1e-4
    until **e295**, where it finally fired once (1e-4 → 1e-5). Within any normal-length
    run it therefore provides no annealing — which is the point — but "provably never
    fires" (as §11 first put it) was wrong.
- **The way to improve the model is more steps**, i.e. more epochs at batch 256 —
  which is exactly what `cont150` does (resuming e85 rather than restarting, so the
  11k steps already paid for are kept).

Full comparison of the two 100-ish-epoch configs, same corpus and window protocol:

| metric | e85 (256, 86 ep) | final100c (384, 100 ep, cosine) |
|---|---|---|
| retrieval top-1 raw / whitened | **0.5247 / 0.5320** | 0.4540 / 0.4900 |
| genre `primary` / `any-tag` / `single` | **0.341 / 0.461 / 0.450** | 0.326 / 0.449 / 0.436 |
| whitened gap | **0.4886** | 0.4767 |
| same-artist / same-genre edges | 0.015 / 0.113 | 0.015 / 0.111 |
| collapsed per 1000 | **71** | 200 |

## 12. 2026-09-12 — the 1000-epoch push: steps keep paying

Ran e298 → e1000 (128,000 total steps) to test whether the step lever continues.
**It does**, and it was still paying at the end, so both saturation calls I made
along the way were wrong.

| steps (k) | 2.2 | 5.4 | 11 | 19.2 | 38 | 57 | 78 | 99 | **128** |
|---|---|---|---|---|---|---|---|---|---|
| raw top-1 | 0.3453 | 0.4353 | 0.5247 | 0.5787 | 0.6280 | 0.6387 | 0.6500 | 0.6573 | **0.6733** |
| whitened top-1 | — | — | 0.5320 | 0.5873 | 0.6387 | 0.6493 | 0.6587 | 0.6600 | **0.6653** |
| genre `single` | 0.410 | 0.425 | 0.450 | 0.459 | 0.463 | 0.459 | 0.468 | 0.467 | **0.469** |

Return diminishes ~5× from the steepest region but never reached zero. **FINAL MODEL:
`checkpoint_e1000.pt`** — retrieval 0.6733 raw / 0.6653 whitened, top-10 0.7933, genre
`single` 0.469 / `any-tag` 0.491, whitened gap 0.5690, collapsed 17.1% (149× same-artist
lift, i.e. mostly genuine).

### Two wrong saturation calls — the methodological lesson

- At **57k steps** one +0.011 interval looked like flattening; the next interval
  repeated +0.011 *exactly*. A linear tail, not a plateau.
- At **99k steps** whitened gained only +0.001 and genre was flat, so I projected
  ~+0.005 raw / ~0 whitened for the last 29k steps and said stopping would cost
  nothing. The actual result was **+0.016 raw / +0.005 whitened**, and whitened
  *resumed* climbing.

With a metric this noisy (±0.03 at 1500 queries) on a slowly-turning curve, **one
interval cannot establish saturation.** Require the marginal gain to collapse across
several consecutive points — and even then, expect it may recover.

### What the unattended push cost in engineering

Six attempts, ~14 h wall clock, **1 GPU fault auto-recovered, 0 manual
interventions**. But three supervisor defects surfaced, and **all three were invisible
in the log** — unattended runs fail silently:

1. **Missing `PYTHONPATH`** — the supervisor `cd`s into the run dir (so checkpoints
   cannot clobber the repo's), which takes the repo off `sys.path`; every attempt died
   with `ModuleNotFoundError`. Fixed + a preflight that aborts loudly.
2. **LR silently decaying** — the plateau scheduler re-fired to 1e-5 within ~20 epochs
   of a `--reset-lr`, so the run looked healthy while crawling. Fixed with
   `--lr-schedule constant` (a no-op `LambdaLR`, so it keeps the same
   `step()`/`state_dict()` interface).
3. **The reap killed the supervisor itself** — `kill -9 -$pgid` targeted its *own*
   group, because `setsid` does not always leave the child in a new process group. It
   died right after logging `attempt exited rc=0`, silently ending the push at e771.
   Fixed with an `own_pgid` guard, which then correctly refused to reap twice more
   (including on the final attempt) — so the guard was load-bearing, not defensive.

Two more near-misses: **disk** (700 epochs × 32 MB = 22.4 GB against 26 GB free — freed
40 GB and added a pruner), and a **silent build failure** (the GPU fault killed
`build_chroma` mid-embed and the evals then ran against a stub DB with their errors
filtered away; `verify_ckpt.sh` now aborts with `BUILD FAILED` if the build yields
<1000 tracks).

Also worth recording: **never pattern-match a string that appears in the command doing
the matching.** Three separate incidents — two monitors whose `pgrep` matched their own
command line and so could never detect a crash, and a cleanup `pkill` that killed my
own shell.
