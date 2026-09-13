# music-encoding

Self-supervised **Barlow Twins** audio encoder for a Shazam-style music fingerprinting model. The learned embedding of a song must be **distortion-stable** (robust to noise, EQ, compression, reverb…) and **time-stable** (any 5-second window of the same track maps near its other windows).

Everything lives in the `music_encoding` package (note the **underscore**). The top-level `music-encoding/` directory (dash) is a stale duplicate and was deleted in the working tree — ignore it.

## How it works

Barlow Twins: two views of the same track are pushed through a shared encoder, then a projection head; the loss makes the cross-correlation of the two projected views approximate the identity matrix (invariance on the diagonal, redundancy-reduction off the diagonal).

Pipeline per training step — the dataset takes in the waveform and returns **augmented spectrogram pairs** (the model never sees raw audio):

```
track  → 2 non-overlapping random 5s windows  →  LogMelSpectrogram
      → SpectrogramAugmenter (noise/reverb/lowpass/gain + RRC)   [inside FMAPairDataset]
      → conv stack → embed (128-d) → projector (→ 2048-d) → BarlowTwinsLoss
```

Mixup (in-batch spectral blending) exists in `augmenter.py` but is **off by default** — with the current positives (different windows + augmentation) it measurably worsens the time-stability objective, so `train()` takes it only as an experiment hook (see `train.py`).

Time stability comes from pairing two different windows of the *same* track as positives; distortion stability comes from the augmentation applied to each window.

## Repository layout

| Path | Purpose |
|---|---|
| `music_encoding/model.py` | `SiameseEncoderBT` (pure encoder: conv → embed → projector — **no mel**) + `BarlowTwinsLoss` |
| `music_encoding/augmenter.py` | `SpectrogramAugmenter` (content distortions + RRC), `AudioAugmenter` (legacy waveform, QA only), `Mixup` (in-batch) |
| `music_encoding/twin_dataset.py` | `LogMelSpectrogram` (shared wav→spec), `MelSpecWindowSource` (shared `.npy`→window), `FMAPairDataset` (audio path), `MelSpecPairDataset` (mel path), caching, resampling, silence gating |
| `music_encoding/mtg.py` | `MTGJamendoBase` — loader for the official MTG-Jamendo TSVs + downloaded audio; `row_by_num` joins the melspec `.npy` stems to TSV rows |
| `music_encoding/train.py` | Training loop + CLI |
| `music_encoding/db.py` | Shared chroma DB helpers (`load_all`, collection names) |
| `music_encoding/build_chroma.py` | Embeds all tracks (pooled raw/whitened + per-window raw) into a persistent chroma vec DB; `--mel-root` embeds from spectrograms, the audio path decodes |
| `music_encoding/test_knn.py` | Genre k-NN eval from `tracks_raw` in the chroma DB |
| `music_encoding/test_similarity_same_song.py` | Intra/inter-track similarity eval from `windows_raw` (embedding collapse), with post-hoc whitening diagnostic |
| `music_encoding/test_retrieval.py` | Same-track retrieval eval from `windows_raw` — the metric that matches the fingerprinting objective (a window's top-k neighbours should be its own track) |
| `music_encoding/test_network.py` | k-NN track-similarity network from `tracks_whitened`, colored by metadata (genre/artist/…) |
| `music_encoding/test_similar_pairs.py` | Top-N most-similar track pairs for manual QA (ranks from `tracks_whitened`): spectrogram PNGs with `--mel-root`, listenable `.wav`s with `--mtg-data` |
| `music_encoding/test_augmentation.py` | Augmentation QA: a grid PNG of the live `SpectrogramAugmenter` with `--mel-root`, 20 augmented `.wav`s from the legacy waveform augmenter with `--mtg-data` |
| `music_encoding/compare_songs.py` | Case study: two song names → their 30 s Deezer/iTunes previews → the model's cosine similarity, with the corpus baseline needed to read it |
| `cached_mtg/` | Resampled full-track waveforms as `.pt` keyed by dataset index (MTG-Jamendo ~55.6k tracks) |
| `chroma_db/` | Persistent chroma vec DB: `tracks_raw`, `tracks_whitened` (pooled), `windows_raw` (per-window) |
| `checkpoints/` | `checkpoint_e{epoch}.pt` per epoch |
| `train_log_*.csv` / `logs/` | Training metrics (`epoch,lr,loss,on_diag,off_diag`) |
| `instructions.md` | Notes on the augmentation & hyperparameter changes (read before touching augmenters) |
| `plot_log.py` | Plots a train log CSV → `log.png` |
| `webapp/` | **The web app** — browse/search the corpus, play a track, walk its similarity graph. Its own `README.md`; `webapp/api` (FastAPI, reads exported artifacts) + `webapp/ui` (SvelteKit). See the section below |

## Key files

### `model.py` — `SiameseEncoderBT`
- Pure encoder: takes an *already-computed, already-augmented* log-mel spectrogram — there is **zero** mel/spectrogram code in the model. Constructor is just `SiameseEncoderBT(embed_dims=128, proj_dims=...)` (no `sr`/`n_mels`).
- Conv stack: **4×** `(Conv2d 3×3 + BN + SiLU + MaxPool2d(2))`, channels 1→32→64→128→128, then `AdaptiveAvgPool2d(2)` + `Flatten` → **512** features. (Earlier notes here said 3 blocks ending in `AdaptiveAvgPool2d(1)`; the stack was refactored to drop an early-feature concat and add a 4th block — read `model.py`, not this line, if they disagree.)
- `embed`: `Linear(512→2·embed_dims) → SiLU → Linear(2·embed_dims→embed_dims) → SiLU → Linear(embed_dims→embed_dims)`, so `embed_dims` (128) is what `build_chroma` stores in the DB. `projector`: `Linear→BN→SiLU→Linear→BN→SiLU→Linear`, size `proj_dims` (index 6 = the last Linear, which is what `build_chroma` reads for `proj_dims`).
- `forward(spec)` runs conv stack onward, returns `(emb, z)`.
- `forward_pair(spec_a, spec_b)` concatenates and splits to share weights.
- `BarlowTwinsLoss(lambd)` returns `(loss, on_diag, off_diag)`; loss = `on_diag + lambd * off_diag`. Its **own default is `2e-2`** — note `train.py`'s CLI default (`5e-2`) overrides it, which is the misconfiguration diagnosed in §9 of `instructions.md`.

### `twin_dataset.py` — `LogMelSpectrogram` + `FMAPairDataset`
- `LogMelSpectrogram`: the waveform→log-mel step (`sr=22050`, `n_fft=1024`, `hop=256`, `n_mels=64` → `AmplitudeToDB` → `unsqueeze(1)`). **The shared home of the mel settings** — the dataset and both eval scripts construct it, so all consumers agree on mel params now that the model no longer owns them. Accepts a single `(N,)` window or a `(B, N)` batch; always emits 4-D `(B,1,n_mels,T)`.
- `FMAPairDataset` wraps a dataset-like source (the MTG-Jamendo `MTGJamendoBase`, or anything yielding `ds[idx]["audio"]`). `__len__` = number of tracks.
- `__getitem__` loads the full track (resampled to 22050 Hz), draws window A at random, window B at random **≥ 2 s away** (`min_offset`) from A, then `LogMelSpectrogram` → `SpectrogramAugmenter` (content distortions + RRC). The waveform `AudioAugmenter` is not part of training — all distortion happens on the spectrogram. Returns `(spec_a, spec_b)`, each `(1,1,n_mels,T)`.
- **Silence handling** (`min_window_rms`, default 0.03): near-silent audio collapses to one degenerate embedding that pollutes the learned space, so (a) tracks whose *full-track* RMS is below the threshold are dropped entirely (built once at init; dataset length reflects this), and (b) windows drawn below the threshold are re-drawn up to `MAX_SILENCE_TRIES` (12). Set `min_window_rms=0` to disable. Caveat: whole-window RMS misses *partly*-silent windows (e.g. a vocal track that is 60% silence but RMS above threshold) — those can still collapse.
- `collate_pairs` uses `torch.cat` (not `stack`) along dim 0 so the batch stays 4-D `(B,1,n_mels,T)`.
- Audio decoded via `torchcodec.decoders.AudioDecoder` (`ds[idx]["audio"].get_all_samples()`); stereo is mean-downmixed.
- `cache_dir` caches resampled full-track waveforms as `{idx}.pt` (reuse `cached_mtg/`).

### `augmenter.py`
- **`AudioAugmenter`** — LEGACY waveform-domain augmenter (noise/reverb/lowpass/gain on the waveform). Retained only as a QA/listening tool (`test_augmentation.py`) and reference; it is **not** in the training pipeline, because training consumes spectrograms and the distortion set now lives in `SpectrogramAugmenter`. Do **not** GPU-batch a waveform augmenter (measured slower). Pitch shift (`pitch_shift`) exists but is off by default (`pitch_p=0.0`): `torchaudio.functional.pitch_shift` costs ~2.3 s per 2 s clip on this CPU/ROCm build (~100× every other stage).
- **`SpectrogramAugmenter`** — the single home of training augmentation, applied per-sample inside `__getitem__` after `LogMelSpectrogram` (waveform path) or directly on the loaded window (mel path), so both paths distort identically. Chain, in log-mel dB (noise/reverb round-trip through linear magnitude): spectral noise (always, SNR 5–20 dB *below each window's 95th-percentile loud-content level* — anchoring to the *peak* was a porting bug from the waveform `AudioAugmenter`'s mean-power reference and buried ~90% of a wide-dynamic-range window, leaving two views of the same content nearly independent and the invariance unlearnable; complex-Gaussian power addition, so the quiet floor rises toward a noise floor while loud content is untouched — never clamps to −inf) → spectral reverb (p=0.4, exp-decay time-smear, kernel lengths ≈0.2/0.4/0.8 s) → lowpass (p=0.3, rolloff above a random mel band, 15–30 dB) → gain (p=0.5, ±6 dB) → RRC (`F.affine_grid`/`F.grid_sample`, both `align_corners=False`; scale `(0.8,1.0)`, ratio `(0.85,1.15)`, p=0.6). Every stage is per-sample masked so it works on single windows and collated batches. Noise stays unconditional (anti-collapse rationale unchanged). SpecAugment (masking) still deliberately not bundled.
- **`Mixup`** — in-batch mixing (domain-agnostic linear interpolation), any shape with a leading batch dim — it blends the collated **spectrograms** `(B,1,n_mels,T)`, not waveforms, since the dataset already consumed the waveforms. Needs cross-sample visibility so it can't live in `__getitem__`. Beta(0.4) weights clamped so the original stays ≥ 0.6, applied per-sample with `p=0.5`. **Now OFF in the default pipeline** — with the different-window + augmented positives it measurably worsens the objective (ablation: diff-windows+aug `on_diag` 802 with Mixup vs 518 without). `train()` accepts it as an experiment hook only; `train.py` passes `mixup=None`.

### `train.py`
- CLI: `-r/--resume <checkpoint.pt>`, `-e/--epochs` (100), `--workers` (4), `--batch-size` (128), `--prefetch` (2), `--mtg-data <data>`, `--audio-root <dir>`, `--cache-dir` (default `cached_mtg`), `--min-window-rms` (0.03, silence gate; 0 disables), `--n-mels` (64; mel path only, merges native 96 mel bands down), `--lambd` (**`5e-2` — too high; see the `lambd` decision below and §9 of `instructions.md`**), `--autocast {fp16,bf16,fp32}` (default **bf16**; see below).
⚠️ `train.py` writes `checkpoints/` and `train_log_*.csv` **relative to the cwd**, so a fresh run started from the repo root overwrites `checkpoint_e0…` in place. Run experiments from a scratch cwd (e.g. `~/mel_runs/<tag>/`) unless you mean to replace the existing checkpoints. The DataLoader batch IS the BT batch — there is deliberately **no gradient accumulation** (see below).
- `FMAPairDataset` is constructed with only `spectrogram_augmenter=SpectrogramAugmenter()` (no waveform augmenter) — the dataloader yields `(spec_a, spec_b)`. `MelSpecPairDataset` (mel path) uses the same `SpectrogramAugmenter`.
- Model built as `SiameseEncoderBT(proj_dims=1024)` (`build_chroma` reads proj_dims from the checkpoint's `projector.6.weight`, so evals adapt to whatever D a run used). AdamW `lr=1e-4`, `weight_decay=0.05`, `ReduceLROnPlateau(factor=0.1, patience=10)`. Loss `BarlowTwinsLoss(lambd=args["lambd"])`. Forward under autocast gated by `--autocast {fp16,bf16,fp32}`: **bf16** on CUDA/Ampere+ & ROCm CDNA; **fp16** on consumer RDNA (RX 6xxx/7xxx) — RDNA has *native* fp16 (~2× fp32 speed, half the activation memory), while bf16 is *emulated* there ~2.4× *slower* than fp32 (measured: fp16 @ batch 256 ≈ 311 ms/step vs fp32 @ 160 ≈ 695 ms/step). `fp32` = autocast disabled. `BarlowTwinsLoss` upcasts `z_a/z_b` to fp32 internally, so the cross-correlation statistic is never accumulated in fp16/bf16. Backward runs outside the autocast block, so grads accumulate in fp32 (no GradScaler needed).
- **`torch.compile` is dropped for now** — compiled kernels raised `HSA_STATUS_ERROR_EXCEPTION` hardware faults on the ROCm dev GPU. Re-enable for CUDA/RunPod runs, where it's reliable.
- Per batch: move specs to device → `zero_grad` → `forward_pair` → `backward` → `optimizer.step()` (Mixup is off by default; pass it into `train()` only as an experiment). **No gradient accumulation, by design**: the BT loss is a batch-normalized statistic (projector BN mean/std over the batch + a cross-correlation that couples all samples), so it does not decompose into independent per-sample terms — summing micro-batch gradients would *not* equal one larger batch. Want more samples? Raise `--batch-size` directly (BT saturates 64–512 anyway).
- Saves `checkpoints/checkpoint_e{epoch}.pt` (model/optimizer/scheduler state + loss) every epoch. Resume sets `start_epoch = checkpoint epoch + 1`.
- Logs a CSV `train_log_{timestamp}.csv` (header `epoch,lr,loss,on_diag,off_diag`), opened once and appended + flushed each epoch so it survives a kill.

## Training

> ⚠️ **`~/mtg_jamendo` was deleted on 2026-09-13** to free disk for the web app's audio
> download (135 GB; the disk was at 99%). It was the precomputed-mel corpus that
> `setup_local_train.sh` and every `--mel-root` path below read, so **mel-mode training and
> `build_chroma --mel-root` do not work until it is re-downloaded** (MTG's
> `raw_30s/melspecs`, 229 GB). The audio fetched into `~/mtg` does **not** substitute: mel
> mode uses MTG's front end (96 mel @ 24 kHz, `norm='slaney'`), while the waveform path in
> `twin_dataset.py` computes 64 mel @ 22.05 kHz, so regenerating the corpus from audio gives
> ~0.987 cosine to the originals, not a bit-identical corpus. In exchange, the web app has
> exact audio for ~98% of the corpus instead of fuzzy-matched third-party previews.

```bash
# from repo root, inside the nix dev shell (direnv loads it)
python -m music_encoding.train --mtg-data <data> --audio-root <audio> -e 300
python -m music_encoding.train -r checkpoints/checkpoint_e99.pt --mtg-data <data> --audio-root <audio> -e 100
```

Uses CUDA/ROCm (autocast default bf16; `setup_local_train.sh` uses **fp16** + batch **256** on the local RDNA card — see the `--autocast` note under `train.py`). Data is the official **MTG-Jamendo** dataset — see below.

**Precomputed-mel mode (no decode, no conversion).** ⚠️ **The `.npy` corpus this section
describes was deleted on 2026-09-13 — see the warning at the top of `## Training`; the
paths below are what it *was*, and the mode needs a re-download to run again.** The
dataset can instead be the official MTG-Jamendo log-mel `.npy` download (`~/mtg_jamendo/`,
`<id % 100:02d>/<id>.npy`, 32,783 × `(96, T)` float32 @ 46.875 fps). Pass
`--mel-root` in place of `--mtg-data`/`--audio-root` and the pipeline skips audio
decode + `LogMelSpectrogram` entirely — all augmentation happens on the
spectrogram:

```bash
python -m music_encoding.train --mel-root /home/johannes/mtg_jamendo -e 300
# one-shot launcher (resumable, logs to train.log):
#   nohup bash setup_local_train.sh > train.log 2>&1 &
```

`MelSpecPairDataset` (`twin_dataset.py`) draws two random ≥2 s-apart ~5 s windows
straight from each spectrogram (234 frames @ 46.875 fps), then merges the native
96 mel bands down to `--n-mels` (default **64**, matching the 64-mel waveform /
audio-eval `LogMelSpectrogram`) in linear-power space (`_merge_mels`), then
`SpectrogramAugmenter` applies the whole distortion set — noise / reverb /
lowpass / gain (ported from the old waveform `AudioAugmenter`, now spectrogram-
domain) — plus RRC. Model input becomes `(1, 64, T)`; n_mels is not baked into
the model (conv + adaptive pool are resolution-agnostic), so checkpoints are
interchangeable across resolutions — pass `--n-mels 96` to keep the native 96
bands, or to resume a 96-mel checkpoint line. Windows below `--min-window-db`
(−85 dB default) are re-drawn.

All of that windowing/merge/gate logic lives in **`MelSpecWindowSource`**, which
`build_chroma` and both QA scripts also build — same reason waveform→mel lives in
`LogMelSpectrogram`: eval must cut and merge windows exactly the way training did.
**The evaluation path has been ported to mel mode**, so a mel-mode checkpoint can be
scored: `build_chroma --mel-root` needs no audio, and the three DB-only evals are
unchanged. Pass `build_chroma` the same `--n-mels` you trained with — nothing can
validate it for you, because a resolution-agnostic model will happily run and
silently return wrong numbers for the wrong value.

Uses CUDA/ROCm (autocast default bf16; `setup_local_train.sh` uses **fp16** + batch **256** on the local RDNA card — see the `--autocast` note under `train.py`). Data is the official **MTG-Jamendo** dataset — see below.

## MTG-Jamendo

The whole pipeline (training, chroma DB, QA audio) reads the official **MTG-Jamendo** dataset (github.com/MTG/mtg-jamendo-dataset, ~55.6k full tracks, 44.1 kHz) via `music_encoding/mtg.py` (`MTGJamendoBase`). It parses `data/autotagging.tsv` (tracks + `category---tag` entries → primary genre/instrument/mood) and `data/raw.meta.tsv` (titles, artist/album names, release date), and decodes the downloaded MP3s relative to `--audio-root`.

**Two id spaces, and the join between them.** A TSV row is `TRACK_ID = track_0000214`, while the number `214` appears in the `PATH` column as `<num % 100:02d>/<num>.mp3` — and the melspec `.npy` files are `<num>.npy` under `<num % 100:02d>/`. So a `.npy` stem is a **track number, not a TRACK_ID**; `track_num()` converts, and `row_by_num(num)` looks metadata up by it. `row(idx)` remains positional for the audio path, and `track_nums` bridges positions → numbers.

`data/autotagging.tsv` here is a **symlink to a curated subset** (`raw_30s_cleantags_50artists.tsv`, 55,609 rows), which omits 65 of the 32,783 melspec tracks. `MTGJamendoBase` therefore reads `raw_30s.tsv` as a fallback for exactly those missing numbers, so every melspec track gets full metadata instead of a blank row — without it those 65 would either be dropped from the eval corpus (making it a subset of the trained corpus) or land in the k-NN vote with an empty genre.

**Download the audio** (large!):

```bash
# from a clone of https://github.com/MTG/mtg-jamendo-dataset (needs gdown, in the flake)
python3 scripts/download/download.py --dataset raw_30s --type audio-low <dir> --unpack --remove
```

`raw_30s` audio is 508 GB (`audio` 320 kbps) or 156 GB (`audio-low` mono). Point `--audio-root` at the unpacked dir; `--mtg-data` at the repo's `data/` dir. Every entry point takes `--mtg-data <data>` (required) and `--audio-root <dir>` (defaults to `--mtg-data`); caches default to `cached_mtg/`.

⚠️ The `chroma_db/` currently on disk is **not** MTG: it holds 7,916 **FMA-era** tracks (its metadata still carries the FMA `listens` key and FMA genres like `Hip-Hop`), so those genre k-NN / intra-inter numbers describe a different corpus. On this machine the MTG audio is still `.tar` only, so the audio path cannot rebuild it — use mel-mode `build_chroma` to replace it.

DB metadata keys come from MTG: `title`, `artist`, `album`, `genre`, `instrument`, `mood_theme`, `track_id`, `track_num`, `released` (year), plus `idx` (the DB position the `track_<idx>` ids refer to). No `listens`.

### MTG's mel front end — now identified (needed to embed arbitrary audio)

The model never sees raw audio, so embedding a downloaded preview (see
`compare_songs.py`) requires reproducing the front end that made the `.npy` files.
MTG's STFT parameters were undocumented; they were measured by using the audio tarball
(`~/mtg/raw_30s_audio-low-00.tar`), whose tracks we ALSO have `.npy` for, as ground
truth — embedding the same track from audio and from `.npy` and comparing:

    sr=24000, n_fft=1024, hop=512, n_mels=96
    norm='slaney', mel_scale='slaney'      (i.e. librosa's defaults, 46.875 fps)
    -> 0.987 mean cosine to the .npy-derived embedding already in chroma_db

`norm='slaney'` is the piece to get right — torchaudio's `MelSpectrogram` defaults to
`norm=None`, which costs 0.047 (0.9475 instead of 0.9872). The 48k/2048/1024 variants
reach 0.977. So an audio-derived embedding carries ~1.3% error, which is the precision
to quote for anything built on downloaded audio. Two notes: the tarball files are
**full tracks, not 30 s previews** despite the `raw_30s` name, and they align with the
`.npy` at frame offset 0.

Also: the whitening in `tracks_whitened` is **not stored**, but it is reproducible
from `tracks_raw` alone (ZCA is a deterministic function of the corpus; verified to a
max diff of 1e-5), so a query embedding can be whitened consistently by re-deriving it.

## Evaluation

`build_chroma.py` embeds the whole corpus (5 seeded windows/track, mean-pooled) into a **persistent chroma vec DB** (`chroma_db/`, cosine space) with three collections: `tracks_raw` and `tracks_whitened` (one pooled vector per track) and `windows_raw` (the 5 per-window vectors). All four eval scripts read embeddings **from the DB** — they take no checkpoint and run no model forwards, so re-running an eval is ~seconds. `build_chroma` runs inside the nix shell (chroma lives in the flake env, not `pyproject.toml`).

```bash
# mel mode — the training path; pass the same --n-mels you trained with
python -m music_encoding.build_chroma checkpoints/checkpoint_e99.pt \
    --mel-root ~/mtg_jamendo --mtg-data <data> --n-mels 64
# audio mode (needs the MTG audio unpacked)
python -m music_encoding.build_chroma checkpoints/checkpoint_e99.pt --mtg-data <data> --audio-root <audio>
python -m music_encoding.test_knn
python -m music_encoding.test_similarity_same_song
python -m music_encoding.test_retrieval            # the fingerprinting metric
python -m music_encoding.test_network --color genres
python -m music_encoding.test_similar_pairs --mel-root ~/mtg_jamendo   # top pairs as spectrogram PNGs
python -m music_encoding.test_augmentation --mel-root ~/mtg_jamendo   # augmentation grid PNG
```

Every eval accepts `--db-dir` (default `chroma_db`). `build_chroma` always needs `--mtg-data` (in mel mode for **metadata only** — the join is by track number, since a `.npy` stem is a number like `214`, not a `track_0000214`); `--audio-root`/`--cache-dir` are audio-mode only. `test_similar_pairs` and `test_augmentation` take `--mel-root` for PNG output or `--mtg-data`/`--audio-root` for audio output.

**`build_chroma` upserts, it does not reset.** Two consequences, both observed: a run covering fewer tracks than the previous one (e.g. `--limit`) leaves the surplus **vectors** in place, and `upsert` **merges metadata dicts rather than replacing them**, so a key the previous DB had and the new one doesn't survives on every row it overwrote. Rebuilding over the old FMA DB left the FMA `listens` key on exactly rows 0–7915 (the old DB's extent) while every shared key took the correct MTG value — harmless for the evals, which read only shared keys, but it means a *rebuild in place* is not byte-identical to a fresh DB. To get a pristine DB, delete the three collections first (or point `--db-dir` at a new dir). Measured: ~6 min and a flat ~1.6 GB peak RSS for 3k tracks (the RSS is fixed runtime overhead, not data-proportional); the full 32,783-track rebuild finishes in ~7 min and is 365 MB on disk.

- **k-NN genre accuracy** (`test_knn.py`): loads pooled embeddings + genres from the DB (default `tracks_whitened`, which scores higher for genre), seeded 80/20 track split, cosine-sim top-50, majority genre vote. Prints random baseline (= 1/n_genres). Flags: `--k` (50), `--seed` (42), `--space` (whitened|raw, default whitened).
- **Stability / collapse** (`test_similarity_same_song.py`): loads the per-window raw embeddings from `windows_raw`, groups to `(N, W, D)`, computes within-track pairwise cosine sim (**intra**) and between-random-tracks sim (**inter**), plots histograms, prints means/stds and the gap. Also applies a post-hoc ZCA whitening and replots — a big separation gain means the raw space was just poorly conditioned, not that the model failed.
- **Same-track retrieval** (`test_retrieval.py`): for sampled windows in `windows_raw`, the fraction whose top-k neighbours include the SAME track, raw and whitened. This is the metric that matches the fingerprinting objective (a genre label is only a proxy), and it is the one to judge model changes on. Chance is ≈2.4e-05; a working fingerprint should approach 1.0 at top-1, and the current checkpoint sits at **≈0.41–0.44** (1500 queries; it has ±0.03 sampling noise, so compare like-for-like query counts). Whitening barely moves it (0.435 → 0.440) *unlike* the intra/inter gap — which is the evidence that the weakness is the **objective**, not the geometry: BT never pushes different tracks apart. Flags: `--queries` (1500), `--k` (10), `--seed` (42).
- **Similarity network** (`test_network.py`): k-NN graph over sampled tracks from `tracks_whitened` — edges = top-`k` inter-track sims — force-directed layout, nodes colored by a metadata column (`--color` in {genres, artist, album, instrument, mood_theme, released}); prints same-artist / same-genre edge fractions against random-pairing baselines. Flags: `--n-tracks` (400), `--color`, `--interactive`.
- **Similar pairs QA** (`test_similar_pairs.py`): ranks different-track pairs by similarity (ranking space via `--space` whitened|raw) and renders each pair plus a `pairs.txt` manifest. Ranks the **whole corpus by default**, a query block at a time, because materialising the n² similarity matrix would be 4.3 GB and a sample would not surface the corpus's real top pairs (a 1,000-track sample is 3% of it). Three exclusions, each count printed so the QA set's composition is visible: **collapsed** tracks (raw cosine ≥ `--collapse-sim` to some other track), **near-duplicate audio** (raw ≥ `--max-raw-sim`), and **same-artist** pairs unless `--allow-same-artist` — the same act twice dominates the extreme top, so excluding it is what shows whether the model finds ordinary-similarity related music. With `--mel-root` each pair becomes a 2-row PNG (full log-mel of A above B, shared dB scale) — the same representation the model sees. With `--mtg-data`/`--audio-root` each pair becomes a `.wav` (A + 1 s gap + B, 44.1 kHz), which is the better check but needs the audio unpacked; note audio mode needs a DB built from the *same* source, since a row's `idx` is a position in whichever corpus built it.
  - ⚠️ Two defaults were recalibrated for the e1000 model and are **not** the old values: `--max-raw-sim` is **0.995** (was 0.98 — the model's genuine cross-track pairs now sit at 0.97+, so 0.98 no longer separated duplicates from real similarity and let same-artist near-duplicates through), and `--collapse-sim` is **0.999** (a 0.99 threshold flags **374 per 1000**, a third of the corpus, because the raw space is crowded — mean inter-similarity 0.844 and every track is compared against 32,782 others, so the *median* track's best match is already 0.9888; 0.995 flags 48, 0.999 flags 1.0).
  - `--n-tracks` is **0 = whole corpus** (was 1000). Caveat: the collapse count scales with how many tracks you compare against, so a sample reports fewer (0/1000 at `--n-tracks 1000` vs 33 at full corpus) — only compare it at a fixed `--n-tracks`.
  - Flags: `--n-tracks` (0), `--pairs` (10), `--out-dir` (similar_pairs), `--space`, `--max-raw-sim` (0.995), `--collapse-sim` (0.999), `--allow-same-artist`.
- **Augmentation QA** (`test_augmentation.py`): with `--mel-root`, a grid PNG of the live `SpectrogramAugmenter` — a clean window beside 3 independent augmented draws of it — plus per-row mean dB, 5th-percentile dB and the **two-view share** `corr(aug_i, aug_j)`. Share is ~1 when the augmentation preserves the shared signal and collapses toward 0 when it destroys it; that is the number that exposed the peak-anchored-noise bug (§8 of `instructions.md`), so it is the quickest way to sanity-check a chain change. With `--mtg-data`, the legacy waveform `AudioAugmenter` wav dump.

### The genre label is NOT single-valued — read accuracy carefully
An MTG-Jamendo track carries **2–9 genre tags** (82% of the corpus does; mean ≈2.8), and the TSV lists them in **alphabetical order**. So `genre` = `tags[0]` is "the alphabetically first genre", not a primary one — `track_0000946` is `ambient, chillout, downtempo, easylistening, electronic, lounge` and the old metric called it **ambient** because 'a' < 'c'. A neighbor sharing a genuine genre but not the first letter was scored wrong, which deflates accuracy on ~80% of the test set.

`test_knn` therefore reports three scorings, each with its chance level:
| scoring | meaning | e42 value | chance |
|---|---|---|---|
| `primary` | prediction == `tags[0]` — the historical metric | 0.314 | 0.008 (121 labels) |
| `any-tag` | prediction ∈ the track's **full** tag set | **0.435** | 0.098 |
| `single` | `primary` on tracks with exactly ONE tag (clean labels) | **0.425** | 0.008 |

`single` is the closest thing to a like-for-like number across corpora, because it removes the ambiguity that differs between label sets. `any-tag`/`single` need the `genres` key that `build_chroma` writes; a DB built before that key only supports `primary` (and says so). To retro-fit an existing DB without a GPU rebuild, `collection.update(ids=..., metadatas=...)` patches metadata in place and preserves embeddings.

### Reference baselines (from previous runs)
⚠️ These are **FMA-era and not comparable** to current numbers: the old `chroma_db` held 7,916 FMA tracks (its rows all carried the FMA `listens` key and FMA genres like `Hip-Hop`), i.e. a ~16-label problem, whereas the MTG mel corpus is 32,783 tracks over 121 primary labels. Against chance the FMA runs sat at ~5.8× (0.360/0.0625) while the current MTG model is at 39–53×. Git cannot settle the provenance of the two rows below — the table and the FMA references arrive in one squashed commit (`9c6ebed`) whose own text *claimed* the DB had already been rebuilt to MTG when it had not.

| Config | k-NN genre acc | Intra mean | Inter mean |
|---|---|---|---|
| Disjoint windows only (no augmentation) | 42–46% | ≈0.98–0.99 | ≈0.81–0.87 |
| First augmented run (noise+reverb+lowpass+gain, no RRC/Mixup) | ≈31% | — | — |
| MTG mel corpus, e42 (p95-noise, no Mixup, n_mels 64) | 0.314 primary / 0.435 any-tag / 0.425 single | 0.9451 | 0.8335 |
| **MTG mel, `checkpoint_e1000` (1000 ep, 128k steps) — current best** | **0.359 / 0.491 / 0.469** | **0.9519** | 0.8442 |

**Current best model: `~/mel_runs/cont1000/checkpoints/checkpoint_e1000.pt`** (batch
256, plain BT, λ=5e-2, constant LR, n_mels 64, 1000 epochs = 128k steps). Retrieval
top-1 **0.6733** raw / **0.6653** whitened (vs 2.4e-05 chance), top-10 0.7933,
whitened gap **0.5690**, collapsed 17.1% with a 149× same-artist lift. Its DB is
promoted into `chroma_db/`, so every eval runs with no arguments. Retrieval vs steps:
2.2k 0.345 · 5.4k 0.435 · 11k 0.525 · 19.2k 0.579 · 38k 0.628 · 57k 0.639 · 78k 0.650
· 99k 0.657 · **128k 0.673** — still improving at the end, so more steps would likely
help further (`instructions.md` §12).

Desired direction for the fingerprinting use case: **lower inter-track mean with wider std** (less embedding-space collapse) while keeping intra-track high. Track new results as (checkpoint config → both eval numbers).

## Web app (`webapp/`)

A local SvelteKit + FastAPI app over the same embeddings: search the corpus, play a track, walk
its similarity graph. `webapp/README.md` has the run instructions; the load-bearing facts:

- **Strictly separated from the model.** `webapp/api/scripts/export_index.py` is the *only*
  thing that imports `music_encoding` (plus `chromadb`, for `load_all`). It writes
  `webapp/data/catalog.sqlite` (metadata + FTS5 index + facet tables) and `vectors.npz`
  (whitened + raw, unit-normalised, ordered by `idx`). After that the API needs **numpy and
  sqlite only** — no torch, no chromadb, no checkpoint, no GPU.
- **Similarity is exact, not approximate.** The whole matrix is 32,783×128 float32 (17 MB per
  space), so `X @ X[i]` is ~2 ms and returns the true top-k. `verify_export.py` checks it
  reproduces chroma's own neighbours (top-1 40/40, mean top-10 overlap 0.997) — that is the test
  that would catch a transposed or misordered export, which nothing else would.
- ⚠️ **`col.get(ids=[...])` returns rows sorted by id, NOT in the requested order.** Zipping its
  output against the request list silently pairs every seed with the wrong embedding. This
  produced a wrong number that briefly went into this file: an enrichment of 1.15×/1.9×, where
  the truth (keyed by id, and reproduced independently through chroma's HNSW) is **3.96× genre /
  138× same-artist** at k=6. The 138× agrees with `test_retrieval.py`'s independent 149× lift.
- **The graph's enrichment compares against the corpus, not a within-graph label shuffle.** A
  permutation null is right for `test_network.py`, which *samples* the corpus and then derives
  edges; it is wrong for a seed-centred ego-graph, because the node set is chosen *by* similarity
  to the seed, so shuffling labels inside it leaves the enrichment in place. Measured: tracks 0-7
  are all David TMX and 61% of their top-20 neighbours share the artist, yet the within-graph lift
  read **0.8×** — below chance. `webapp/api/app/enrichment.py` explains this at length.
- **Audio comes from MTG, not from previews.** `raw_30s/audio-low` is published as one tar per
  `track_num % 100` bucket; our corpus spans 59 of them and the buckets are a *representative*
  slice (bucket 00's genre shares match the corpus within 1.4 pp). `fetch_mtg_audio.py` pulls what
  fits on disk (the full set is **98.5 GB** against ~44 GB free, so it stops at a free-space floor
  and is re-runnable), and `local_audio_index.py` records each member's archive + byte offset so
  the API streams by range and **never unpacks** — the tars are the only disk cost. This is the
  audio the embeddings were actually built from.
- ⚠️ **A local row must name its archive** (`audio.tar` + `audio.offset`). Serving an offset
  against the wrong tar does not raise — it streams plausible bytes from the middle of another
  file. That happened live while adding a second archive, because the running API still had the
  old single-tar path; `serve()` now refuses when the named archive is missing, and the indexer
  drops rows whose archive is gone.
- ⚠️ **Previews must never accept a title-only match.** The resolver used to take the best-scoring
  candidate even when no candidate's artist matched, which put **2,700 wrong songs** in the cache
  (`Both` → `Beth Crowley`, `Alexander Blu` → `Monty Alexander`) — and those were inside the
  headline "6,518 playable (20%)" figure. An artist match is now required (accents folded; 111
  rows were false negatives for that reason), the wrong rows were purged, and playable *fell* to
  4,275, which is the honest direction. Previews are a fallback for buckets whose audio is not
  downloaded; they are a different recording, so the player labels them and offers "not it?".
- ⚠️ **Deezer reports its rate limit as HTTP 200** with `{"error": {"message": "Quota limit
  exceeded"}}`. Reading only `data` cannot distinguish that from a genuine miss, and a miss is
  cached forever: a first warm run at 10 workers recorded 12,512 tracks as permanently
  unavailable at an 8% hit rate where a random sample gives ~28%. Those rows were purged. A
  refusal is now retried and never written, `provider="none"` requires *every* provider to have
  answered, and one shared limiter caps lookups at 4/s (`WEBAPP_PREVIEW_RATE`).
- `.gitignore` anchors the vendored `/lib/` rule — unanchored it also matched SvelteKit's
  `src/lib/`, which would have silently dropped the frontend's library directory from git.

## Decisions that matter (do not "simplify")

- **Keep the waveform→spectrogram step in `LogMelSpectrogram` (`twin_dataset.py`)** — train loop and evals construct it identically, so all consumers agree on mel settings. Do not re-introduce mel computation inside the model.
- **Keep the augmentation chain inside the dataset, on the spectrogram** — `SpectrogramAugmenter` (content distortions + RRC) runs per-sample in `__getitem__` (CPU). GPU-batched augmentation was measured *slower*; per-sample trades a bit of throughput for the clean separation the user asked for — if training slows noticeably, revisit batching rather than moving mel back into the model.
- **`proj_dims` ≤ 2048.** Audio Barlow Twins (arXiv:2209.14345) found performance saturates at ~1024–2048 for audio (unlike vision's 16k+) and degrades at 16k. The default arg in `model.py` is still `8192` — a trap. `train.py` and both evals pass 2048 explicitly; keep them in sync with whatever the checkpoint used.
- **`lambd = 5e-2` — keep it, and don't "fix" it down.** The docs here used to claim `2e-2` and `3e-3` in different places while `train.py`'s CLI default (and `setup_local_train.sh`, which passes no `--lambd`) ran `5e-2`; §9 of `instructions.md` chased that inconsistency. At **matched epochs** the three values are a wash: retrieval top-1 0.345/0.352/0.355 (5e-2/2e-2/5e-3), genre `single` 0.410/0.436/0.438, collapsed per 1000 50/209/70. Keep `5e-2` — best on collapse, worst on genre by only ~0.03.
  - **`on_diag` is a misleading proxy.** It falls ~2× (447 → 228) as `lambd` drops, with *no* retrieval benefit — because `off_diag` (redundancy reduction) is what spreads the cloud, so less of it contracts the embedding while the two views of a track move closer. Use `on_diag` only to spot a config whose views share too little signal to learn at all (pinned near ~800–1000 ≈ D). **Judge real changes on `test_retrieval` + the collapse count.**
  - **The bottleneck is the objective, not a hyperparameter.** Barlow Twins pulls two views of one track together and decorrelates dimensions, but never pushes *different tracks apart* — so raw inter-similarity sits at 0.80–0.86 and retrieval top-1 stalls near 0.41–0.44 (whitening barely helps: 0.435 → 0.440, unlike the intra/inter gap). A contrastive/InfoNCE term is the next real lever. Ruled out for retrieval: the augmentation chain, `n_mels` (96 measured equal to 64), and `lambd`. Epochs *do* help (17 ep → 0.35, 42 ep → 0.435).
- **Batch size 64–512** — audio BT does *not* benefit from the huge batches vision wants. Local default is 256 (fp16); 128–256 is fine.
  - **But batch size is not only a memory knob: at fixed epochs it sets how much optimisation happens.** 32,783 tracks ÷ batch = batches/epoch (128 at batch 256, **85 at 384**), so 100 epochs at 384 is *fewer* optimizer steps than 86 epochs at 256 — which is exactly why a 100-epoch batch-384 run came out **worse** than an 86-epoch batch-256 run on every metric (retrieval 0.454 vs 0.525, collapsed 200 vs 71 per 1000) despite a lower BT loss. **Prefer batch 256**, and measure progress in *steps* (~2.2k → 0.345 retrieval, 5.4k → 0.435, 11k → 0.525). See `instructions.md` §11.
  - **Do not add LR annealing while the loss is still descending.** Cosine decay was tried and cost quality on top of the step deficit: with no plateau to exploit, shortening the remaining steps only removes progress.
  - The plateau scheduler (`ReduceLROnPlateau(factor=0.1, patience=10)`) is *effectively* inert for a normal-length run — measured: it held `lr` at 1e-4 for 43 epochs, and in a 300-epoch run it fired **exactly once, at e295**. It does not never fire (I overgeneralized that at first); it just fires far too late to matter, because the ~−1/epoch descent keeps beating its 1e-4 *relative* threshold. Keep it at 1e-4 and accumulate steps.
- RRC scale range is intentionally gentler than vision RRC (which often crops to 8%) — fine spectral/pitch detail must survive for track-identity discrimination.

## MCP servers (`memory` + `filesystem`)

Both ship in the nix dev shell (`flake.nix`: `pkgs.mcp-server-memory`, `pkgs.mcp-server-filesystem`) and are registered project-wide in `.mcp.json`, which calls them by **bare name** — so they only spawn when Claude Code itself was launched inside the dev shell (direnv). Outside it the names aren't on `PATH` and both servers fail to start. After editing `.mcp.json`, restart Claude Code and approve the project servers.

### `memory` — persistent knowledge graph
`MEMORY_FILE_PATH=.claude/memory.json` (gitignored), absolute — the server resolves a relative path against its own read-only nix store dir, and it `writeFile`s without `mkdir`, so `.claude/` must keep existing. The graph is repo-scoped and survives across sessions.

Use it for the **accumulating experimental record** this file is a bad home for:
- every eval result as `(checkpoint config → k-NN genre acc, intra mean, inter mean)`; the baselines table above is the *seed*, not the destination
- ablation outcomes together with the config that produced them (the Mixup, `autocast`, `lambd`, `n_mels` numbers quoted throughout this file)
- run observations that aren't repo state: per-epoch loss shape, why a run was killed, the measured OOM ceiling per batch size

Model runs as entities (`run:<timestamp>`, `checkpoint:e<N>`), attach numbers as observations, and link them (`evaluated_on`, `ablation_of`, `resumed_from`). Query it before re-deriving a measurement an earlier session already took; write to it when a run finishes instead of leaving the number in scrollback. Do **not** mirror this file into it — architecture, file roles and the "do not simplify" decisions stay here, where they load every session.

### `filesystem` — allowlisted read/write outside the working dir
Roots: the repo, `~/mtg_jamendo` (the 32,783 precomputed `.npy` log-mels training reads via `--mel-root`), plus `~/mtg-jamendo-dataset/data` (MTG TSVs) and `~/mtg` (audio root). Nothing outside those is reachable — to reach another path, extend `args` in `.mcp.json` rather than shelling around the allowlist.

It earns its place where the built-in tools are awkward: `train_log_*.csv` and `checkpoints/` listings in large dirs, a `.npy` header or `.pt` size without loading it, and inspecting the `~/mtg` download — which is **incomplete** (a complete `raw_30s_audio-low-00.tar`, a partial `-01.tar`, no unpacked mp3s), so the decode path still cannot run against it.

> Distinct from Claude Code's own file-based memory under `~/.claude/projects/…/memory/`: that is per-session scratch, `.claude/memory.json` is the project's shared graph.

## Environment

- Nix flake dev shell (`flake.nix`, ROCm-enabled torch/torchvision/torchaudio, plus datasets/transformers/matplotlib/pillow, `chromadb`, `pyvis`, `gdown`, and the two MCP servers). Loaded via `direnv` (`.envrc` → `use flake`). Ruff for linting (config in `pyproject.toml`, `line-length=88`).
- Python ≥ 3.10. Note: `torchcodec` (used for audio decoding) is **not** listed in `pyproject.toml` dependencies — it's currently just present in the shell.
- `git status` is a work-in-progress tree against `HEAD`: `instructions.md` untracked, `music-encoding/` dash dir deleted, and the input pipeline refactored (mel moved out of the model into the dataset chain) — `model.py`, `augmenter.py`, `twin_dataset.py`, `train.py`, and both eval scripts are modified.

### ⚠️ Host RAM is the constraint on this box — check it before every long run

15 GB total, **no swap**. Two traps compound, and together they make runs die at
**epoch 0 with no traceback** (exit code **137** = SIGKILL, i.e. the OOM killer):

1. **DataLoader workers leak on death.** Each holds **~670 MB** RSS (it imports
   torch + ROCm). When a run dies its workers are **not** reaped — they reparent to
   the session and sit idle forever. Observed: **21 leaked workers holding 11.1 GB**.
   Because each failed attempt leaks another ~5 GB (8 workers), the box spirals — and
   the *symptom* then looks like "runs randomly die at epoch 0", not like OOM.
2. **`--workers 0` was broken** until recently (the DataLoader got `prefetch_factor`
   unconditionally, which `num_workers=0` rejects) — so the obvious escape hatch
   crashed instead of helping. Now fixed.

**Before launching:** `free -g` (want ≥ 8 GB available) and reap dead runs' workers.
An idle leaked worker is identifiable as a `python … multiprocessing.forkserver …`
process whose parent is not a live `train.py`; kill those **by explicit PID**
(verified list), not by pattern. Confirm with `free -g` afterwards and
`df -h /dev/shm` (leaked workers also leave tmpfs behind — 1.2 GB was observed).

Other consequences of the same pressure: Python 3.14 made `forkserver` the default
multiprocessing start method on Linux, so worker spawn pickles the dataset through a
pipe — under memory pressure that surfaces as `_pickle.UnpicklingError: pickle data
was truncated` from inside `forkserver.py`, which is a *symptom* of the host being
out of memory, not a data problem.
