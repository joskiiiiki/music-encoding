# music-encoding

Self-supervised **Barlow Twins** audio encoder for a Shazam-style music fingerprinting model. The learned embedding of a song must be **distortion-stable** (robust to noise, EQ, compression, reverb…) and **time-stable** (any 5-second window of the same track maps near its other windows).

Everything lives in the `music_encoding` package (note the **underscore**). The top-level `music-encoding/` directory (dash) is a stale duplicate and was deleted in the working tree — ignore it.

## How it works

Barlow Twins: two views of the same track are pushed through a shared encoder, then a projection head; the loss makes the cross-correlation of the two projected views approximate the identity matrix (invariance on the diagonal, redundancy-reduction off the diagonal).

Pipeline per training step — the dataset takes in the waveform and returns **augmented spectrogram pairs** (the model never sees raw audio):

```
track  → 2 non-overlapping random 5s windows  →  AudioAugmenter (waveform, CPU)
      → LogMelSpectrogram  →  SpectrogramAugmenter (RRC)     [all inside FMAPairDataset]
      → conv stack → embed (128-d) → projector (→ 2048-d) → BarlowTwinsLoss
```

`Mixup` (in-batch, needs batch visibility so it can't live in `__getitem__`) blends the collated spectrogram batch in the training loop, right after moving to device.

Time stability comes from pairing two different windows of the *same* track as positives; distortion stability comes from the augmentation applied to each window.

## Repository layout

| Path | Purpose |
|---|---|
| `music_encoding/model.py` | `SiameseEncoderBT` (pure encoder: conv → embed → projector — **no mel**) + `BarlowTwinsLoss` |
| `music_encoding/augmenter.py` | `AudioAugmenter` (waveform), `SpectrogramAugmenter` (RRC), `Mixup` (in-batch) |
| `music_encoding/twin_dataset.py` | `LogMelSpectrogram` (shared wav→spec), `FMAPairDataset` (paired augmented spectrograms of one track), caching, resampling, silence gating |
| `music_encoding/mtg.py` | `MTGJamendoBase` — loader for the official MTG-Jamendo TSVs + downloaded audio |
| `music_encoding/train.py` | Training loop + CLI |
| `music_encoding/db.py` | Shared chroma DB helpers (`load_all`, collection names) |
| `music_encoding/build_chroma.py` | Embeds all tracks (pooled raw/whitened + per-window raw) into a persistent chroma vec DB |
| `music_encoding/test_knn.py` | Genre k-NN eval from `tracks_raw` in the chroma DB |
| `music_encoding/test_similarity_same_song.py` | Intra/inter-track similarity eval from `windows_raw` (embedding collapse), with post-hoc whitening diagnostic |
| `music_encoding/test_network.py` | k-NN track-similarity network from `tracks_whitened`, colored by metadata (genre/artist/…) |
| `music_encoding/test_similar_pairs.py` | Saves the top-N most-similar track pairs as `.wav`s for manual QA (ranks from `tracks_whitened`) |
| `music_encoding/test_augmentation.py` | Dumps 20 augmented `.wav`s for manual listening |
| `cached_mtg/` | Resampled full-track waveforms as `.pt` keyed by dataset index (MTG-Jamendo ~55.6k tracks) |
| `chroma_db/` | Persistent chroma vec DB: `tracks_raw`, `tracks_whitened` (pooled), `windows_raw` (per-window) |
| `checkpoints/` | `checkpoint_e{epoch}.pt` per epoch |
| `train_log_*.csv` / `logs/` | Training metrics (`epoch,lr,loss,on_diag,off_diag`) |
| `instructions.md` | Notes on the augmentation & hyperparameter changes (read before touching augmenters) |
| `plot_log.py` | Plots a train log CSV → `log.png` |

## Key files

### `model.py` — `SiameseEncoderBT`
- Pure encoder: takes an *already-computed, already-augmented* log-mel spectrogram — there is **zero** mel/spectrogram code in the model. Constructor is just `SiameseEncoderBT(embed_dims=128, proj_dims=...)` (no `sr`/`n_mels`).
- Conv stack: 3× `(Conv2d 3×3 + BN + SiLU + MaxPool2d(2))`, channels 32→64→128, then `AdaptiveAvgPool2d(1)`.
- `embed`: 128-d (`embed_dims`). `projector`: `Linear→BN→SiLU→Linear→BN→SiLU→Linear`, size `proj_dims`.
- `forward(spec)` runs conv stack onward, returns `(emb, z)`.
- `forward_pair(spec_a, spec_b)` concatenates and splits to share weights.
- `BarlowTwinsLoss(lambd)` returns `(loss, on_diag, off_diag)`; loss = `on_diag + lambd * off_diag`.

### `twin_dataset.py` — `LogMelSpectrogram` + `FMAPairDataset`
- `LogMelSpectrogram`: the waveform→log-mel step (`sr=22050`, `n_fft=1024`, `hop=256`, `n_mels=64` → `AmplitudeToDB` → `unsqueeze(1)`). **The shared home of the mel settings** — the dataset and both eval scripts construct it, so all consumers agree on mel params now that the model no longer owns them. Accepts a single `(N,)` window or a `(B, N)` batch; always emits 4-D `(B,1,n_mels,T)`.
- `FMAPairDataset` wraps a dataset-like source (the MTG-Jamendo `MTGJamendoBase`, or anything yielding `ds[idx]["audio"]`). `__len__` = number of tracks.
- `__getitem__` loads the full track (resampled to 22050 Hz), draws window A at random, window B at random **≥ 2 s away** (`min_offset`) from A, then chains the two augmenters per window: `AudioAugmenter` (waveform, CPU) → `LogMelSpectrogram` → `SpectrogramAugmenter` (RRC). Returns `(spec_a, spec_b)`, each `(1,1,n_mels,T)`.
- **Silence handling** (`min_window_rms`, default 0.03): near-silent audio collapses to one degenerate embedding that pollutes the learned space, so (a) tracks whose *full-track* RMS is below the threshold are dropped entirely (built once at init; dataset length reflects this), and (b) windows drawn below the threshold are re-drawn up to `MAX_SILENCE_TRIES` (12). Set `min_window_rms=0` to disable. Caveat: whole-window RMS misses *partly*-silent windows (e.g. a vocal track that is 60% silence but RMS above threshold) — those can still collapse.
- `collate_pairs` uses `torch.cat` (not `stack`) along dim 0 so the batch stays 4-D `(B,1,n_mels,T)`.
- Audio decoded via `torchcodec.decoders.AudioDecoder` (`ds[idx]["audio"].get_all_samples()`); stereo is mean-downmixed.
- `cache_dir` caches resampled full-track waveforms as `{idx}.pt` (reuse `cached_mtg/`).

### `augmenter.py`
- **`AudioAugmenter`** — waveform-domain, CPU, per-sample (runs inside `__getitem__`). Do **not** move it to GPU-batched: a GPU-batched version was measured *slower*. Default `__call__`: `add_noise` (always, SNR 5–20 dB) → `lowpass` (p=0.3, cutoff 2–8 kHz) → `gain` (p=0.5, ±6 dB). Noise is unconditional because it's near-constant in real query audio and leaving "clean vs clean" pairs is suspected to cause embedding-space collapse. Reverb (`add_reverb`, FFT-conv with synthetic RIRs) and pitch shift are implemented but commented out — keep them for future ablations.
- **`SpectrogramAugmenter`** — random-resized-crop on the log-mel spectrogram, applied inside `__getitem__` (per-sample, after `LogMelSpectrogram`). `F.affine_grid` + `F.grid_sample` (both `align_corners=False`). Defaults: scale `(0.8,1.0)`, ratio `(0.85,1.15)`, `p=0.6`. Scoped to RRC only — don't bundle SpecAugment unless asked. `FMAPairDataset` is train-only, so RRC is effectively gated to training.
- **`Mixup`** — in-batch mixing (domain-agnostic linear interpolation), any shape with a leading batch dim — it now blends the collated **spectrograms** `(B,1,n_mels,T)`, not waveforms, since the dataset already consumed the waveforms. Needs cross-sample visibility so it can't live in `__getitem__`. Beta(0.4) weights clamped so the original stays ≥ 0.6, applied per-sample with `p=0.5`. Call explicitly in the train loop on `spec_a` and `spec_b` independently.

### `train.py`
- CLI: `-r/--resume <checkpoint.pt>`, `-e/--epochs` (100), `--workers` (4), `--batch-size` (128), `--grad-accum` (1), `--prefetch` (2), `--mtg-data <data>`, `--audio-root <dir>`, `--cache-dir` (default `cached_mtg`), `--min-window-rms` (0.03, silence gate; 0 disables), `--lambd` (2e-2). Effective batch = `--batch-size × --grad-accum` (e.g. `--batch-size 128 --grad-accum 6` = 768) — lets a small GPU simulate large-batch updates.
- `FMAPairDataset` is constructed with both augmenters (`augmenter=AudioAugmenter()`, `spectrogram_augmenter=SpectrogramAugmenter()`) — the dataloader now yields `(spec_a, spec_b)`.
- Model built as `SiameseEncoderBT(proj_dims=2048)`. AdamW `lr=1e-4`, `weight_decay=0.05`, `ReduceLROnPlateau(factor=0.1, patience=10)`. Loss `BarlowTwinsLoss(lambd=args["lambd"])`. Forward + loss under `autocast(bfloat16)`.
- **`torch.compile` is dropped for now** — compiled kernels raised `HSA_STATUS_ERROR_EXCEPTION` hardware faults on the ROCm dev GPU. Re-enable for CUDA/RunPod runs, where it's reliable.
- Order per micro-batch: move specs to device → `Mixup` on each side (spectrogram-domain) → `zero_grad` at window start → `forward_pair` → `(loss/grad_accum).backward()` → `optimizer.step()` every `grad_accum` micro-batches (trailing partial window flushed at epoch end). Logged loss is the unscaled micro-batch loss, so CSV values stay comparable to non-accumulated runs.
- Saves `checkpoints/checkpoint_e{epoch}.pt` (model/optimizer/scheduler state + loss) every epoch. Resume sets `start_epoch = checkpoint epoch + 1`.
- Logs a CSV `train_log_{timestamp}.csv` (header `epoch,lr,loss,on_diag,off_diag`), opened once and appended + flushed each epoch so it survives a kill.

## Training

```bash
# from repo root, inside the nix dev shell (direnv loads it)
python -m music_encoding.train --mtg-data <data> --audio-root <audio> -e 300
python -m music_encoding.train -r checkpoints/checkpoint_e99.pt --mtg-data <data> --audio-root <audio> -e 100
```

Uses CUDA/ROCm (bfloat16 autocast). Data is the official **MTG-Jamendo** dataset — see below.

## MTG-Jamendo

The whole pipeline (training, chroma DB, QA audio) reads the official **MTG-Jamendo** dataset (github.com/MTG/mtg-jamendo-dataset, ~55.6k full tracks, 44.1 kHz) via `music_encoding/mtg.py` (`MTGJamendoBase`). It parses `data/autotagging.tsv` (tracks + `category---tag` entries → primary genre/instrument/mood) and `data/raw.meta.tsv` (titles, artist/album names, release date), and decodes the downloaded MP3s relative to `--audio-root`.

**Download the audio** (large!):

```bash
# from a clone of https://github.com/MTG/mtg-jamendo-dataset (needs gdown, in the flake)
python3 scripts/download/download.py --dataset raw_30s --type audio-low <dir> --unpack --remove
```

`raw_30s` audio is 508 GB (`audio` 320 kbps) or 156 GB (`audio-low` mono). Point `--audio-root` at the unpacked dir; `--mtg-data` at the repo's `data/` dir. Every entry point takes `--mtg-data <data>` (required) and `--audio-root <dir>` (defaults to `--mtg-data`); caches default to `cached_mtg/`. `chroma_db` is rebuilt with MTG embeddings (the old FMA DB was a build artifact).

DB metadata keys come from MTG: `title`, `artist`, `album`, `genre`, `instrument`, `mood_theme`, `track_id`, `released` (year). No `listens`.

## Evaluation

`build_chroma.py` embeds the whole corpus (5 seeded windows/track, mean-pooled) into a **persistent chroma vec DB** (`chroma_db/`, cosine space) with three collections: `tracks_raw` and `tracks_whitened` (one pooled vector per track) and `windows_raw` (the 5 per-window vectors). All four eval scripts read embeddings **from the DB** — they take no checkpoint and run no model forwards, so re-running an eval is ~seconds. `build_chroma` runs inside the nix shell (chroma lives in the flake env, not `pyproject.toml`).

```bash
python -m music_encoding.build_chroma checkpoints/checkpoint_e99.pt --mtg-data <data> --audio-root <audio>   # populate chroma_db/ once
python -m music_encoding.test_knn
python -m music_encoding.test_similarity_same_song
python -m music_encoding.test_network --color genres
python -m music_encoding.test_similar_pairs --mtg-data <data> --audio-root <audio>   # top similar pairs as wavs for listening
python -m music_encoding.test_augmentation --mtg-data <data> --audio-root <audio>   # writes test_aug_*.wav for listening
```

Every eval accepts `--db-dir` (default `chroma_db`). `build_chroma` and `test_similar_pairs` additionally need `--mtg-data`/`--audio-root`.

- **k-NN genre accuracy** (`test_knn.py`): loads pooled embeddings + genres from the DB (default `tracks_whitened`, which scores higher for genre), seeded 80/20 track split, cosine-sim top-50, majority genre vote. Prints random baseline (= 1/n_genres). Flags: `--k` (50), `--seed` (42), `--space` (whitened|raw, default whitened).
- **Stability / collapse** (`test_similarity_same_song.py`): loads the per-window raw embeddings from `windows_raw`, groups to `(N, W, D)`, computes within-track pairwise cosine sim (**intra**) and between-random-tracks sim (**inter**), plots histograms, prints means/stds and the gap. Also applies a post-hoc ZCA whitening and replots — a big separation gain means the raw space was just poorly conditioned, not that the model failed.
- **Similarity network** (`test_network.py`): k-NN graph over sampled tracks from `tracks_whitened` — edges = top-`k` inter-track sims — force-directed layout, nodes colored by a metadata column (`--color` in {genres, artist, album, instrument, mood_theme, released}); prints same-artist / same-genre edge fractions against random-pairing baselines. Flags: `--n-tracks` (400), `--color`, `--interactive`.
- **Similar pairs QA** (`test_similar_pairs.py`): ranks different-track pairs from `tracks_whitened` (ranking space via `--space` whitened|raw), saves each pair's full tracks (A + 1 s gap + B, 44.1 kHz) as a `.wav` plus a `pairs.txt` manifest; skips collapsed/near-duplicate pairs (near-silent tracks collapse to one embedding). Audio is decoded from the MTG download (`--mtg-data`, `--audio-root`). Flags: `--n-tracks` (1000), `--pairs` (10), `--out-dir` (similar_pairs), `--max-raw-sim` (0.98).

### Reference baselines (from previous runs)
| Config | k-NN genre acc | Intra mean | Inter mean |
|---|---|---|---|
| Disjoint windows only (no augmentation) | 42–46% | ≈0.98–0.99 | ≈0.81–0.87 |
| First augmented run (noise+reverb+lowpass+gain, no RRC/Mixup) | ≈31% | — | — |

Desired direction for the fingerprinting use case: **lower inter-track mean with wider std** (less embedding-space collapse) while keeping intra-track high. Track new results as (checkpoint config → both eval numbers).

## Decisions that matter (do not "simplify")

- **Keep the waveform→spectrogram step in `LogMelSpectrogram` (`twin_dataset.py`)** — train loop and evals construct it identically, so all consumers agree on mel settings. Do not re-introduce mel computation inside the model.
- **Keep the augmentation chain inside the dataset** — both `AudioAugmenter` and `SpectrogramAugmenter` run per-sample in `__getitem__` (CPU). The GPU-batched `AudioAugmenter` was measured *slower*; RRC per-sample (instead of the old GPU-batched call) trades a bit of throughput for the clean separation the user asked for — if training slows noticeably, revisit batching RRC rather than moving mel back into the model.
- **`proj_dims` ≤ 2048.** Audio Barlow Twins (arXiv:2209.14345) found performance saturates at ~1024–2048 for audio (unlike vision's 16k+) and degrades at 16k. The default arg in `model.py` is still `8192` — a trap. `train.py` and both evals pass 2048 explicitly; keep them in sync with whatever the checkpoint used.
- **`lambd` in the paper-supported range 0.002–0.05** (currently `3e-3`), not the old strict `≈1/D` heuristic (`4.9e-4` was just below the range).
- **Batch size 64–512** — audio BT does *not* benefit from the huge batches vision wants. Current default 128 is fine.
- RRC scale range is intentionally gentler than vision RRC (which often crops to 8%) — fine spectral/pitch detail must survive for track-identity discrimination.

## Environment

- Nix flake dev shell (`flake.nix`, ROCm-enabled torch/torchvision/torchaudio, plus datasets/transformers/matplotlib/pillow). Loaded via `direnv` (`.envrc` → `use flake`). Ruff for linting (config in `pyproject.toml`, `line-length=88`).
- Python ≥ 3.10. Note: `torchcodec` (used for audio decoding) is **not** listed in `pyproject.toml` dependencies — it's currently just present in the shell.
- `git status` is a work-in-progress tree against `HEAD`: `instructions.md` untracked, `music-encoding/` dash dir deleted, and the input pipeline refactored (mel moved out of the model into the dataset chain) — `model.py`, `augmenter.py`, `twin_dataset.py`, `train.py`, and both eval scripts are modified.
