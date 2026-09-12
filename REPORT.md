# music-encoding — evaluation port, metric corrections, and lever investigation

**Date:** 2026-09-11
**Scope:** make mel-mode checkpoints scorable, establish whether the model is
actually worse than the FMA-era numbers suggested, and find a real improvement
lever. Long run done to 86/100 epochs (crashed on a GPU fault; e85 verified).

---

## 1. TL;DR

1. **Mel-mode checkpoints could not be evaluated at all.** `build_chroma` only
   worked by decoding MTG audio, which exists on this machine only as `.tar`
   archives. Fixed: `--mel-root` embeds straight from the precomputed `.npy`
   spectrograms. The three DB-only evals needed no changes.
2. **The model was never meaningfully "worse than FMA".** The comparison was
   invalid on two counts: the historical baselines came from a **different corpus
   with ~8× fewer labels**, and the genre metric was scoring against the
   **alphabetically-first** genre tag. On clean labels the model scores **0.410–0.438**,
   which sits inside the historical 42–46% band.
3. **Four of the five levers the project's notes pointed at are dead ends**, measured:
   the augmentation chain, `n_mels`, `lambd`, and — after implementing and testing it —
   the contrastive **objective** term. The confirmed lever is **optimizer steps**.
4. **A direct retrieval metric was added, and the model now scores 0.579 on it**
   (`test_retrieval.py`: for a 5 s window, is its nearest neighbour its own track?
   chance is 2.4e-05). It rose monotonically with optimizer steps — 0.345 at 2.2k
   steps → 0.525 at 11k → **0.579 at 19.2k** — while the whitened separation went
   0.386 → **0.518** and collapsed tracks fell to **4.3%**, of which 206× more than
   chance are the same artist (i.e. genuine near-duplicates, not crowding).
5. **Batch size is not only a memory knob.** A 100-epoch run at batch 384 scored
   *worse* than an 86-epoch run at batch 256 on every metric, because 384 gives 85
   batches/epoch against 128 — fewer optimizer steps at fixed epochs. And LR
   annealing made it worse still, since the loss was never flat enough for annealing
   to exploit. Both corrections are recorded.

---

## 2. Issues faced

### 2.1 The blocker: no way to score a mel-mode run

Training reads precomputed log-mel `.npy` files (`~/mtg_jamendo`, 32,783 tracks).
Every eval path decoded audio. This machine has no unpacked audio (`~/mtg` holds
`raw_30s_audio-low-*.tar` only; there is no `cached_mtg/`). So a mel-mode run's
*quality* was unknowable — the train CSV (`on_diag`/loss) was the only feedback,
which says whether the loss descends but not whether the embedding is any good.

### 2.2 The `.npy` ↔ metadata join was documented wrong

The project memory stated "the `.npy` ids *are* MTG track ids". They are not: a
`.npy` stem is a **bare integer** (`214.npy`) while a TSV `TRACK_ID` is
`track_0000214`. The number also appears in the TSV `PATH` column as
`<num%100:02d>/<num>.mp3`, matching the `.npy` directory layout. The join is
`int(TRACK_ID.split("_")[1])`. Verified: 32,718 of 32,783 mels joined by number;
the rest are explained by 2.3.

### 2.3 `autotagging.tsv` is a symlink to a curated subset

It points at `raw_30s_cleantags_50artists.tsv` (55,609 rows), which **omits 65 of
the 32,783 melspec tracks**. Naively joining would have either dropped those 65
(making the eval corpus a strict subset of the trained corpus) or given them blank
metadata (an empty `genre` entering the k-NN majority vote). Fixed by falling back
to the full `raw_30s.tsv` for exactly the missing numbers — all 32,783 now resolve
with complete metadata.

### 2.4 The existing `chroma_db/` was FMA-era, not MTG

CLAUDE.md claimed "*chroma_db is rebuilt with MTG embeddings (the old FMA DB was a
build artifact)*". On inspection the DB held **7,916 FMA tracks** — every row
carried the FMA `listens` key and FMA genres (`Hip-Hop`, `Pop`), not MTG's
(`punkrock`, `gospel`). The docs asserted a rebuild that had not happened. Git
cannot settle the provenance of the baseline table either: the table and every FMA
reference arrive together in one squashed commit (`9c6ebed`) whose own text carried
that same false claim. **Consequence: the historical baselines are unusable for
comparison, and the old DB is now overwritten, so it cannot be re-measured.**

### 2.5 Chroma's `upsert` merges metadata instead of replacing it

Rebuilding over the FMA DB left the FMA `listens` key on exactly rows 0–7915 (the
old DB's extent) while every *shared* key took the correct MTG value. Harmless for
the evals (which read only shared keys) but it means a rebuild-in-place is not
identical to a fresh DB. Separately, a run covering fewer tracks than the last
(e.g. `--limit`) leaves surplus *vectors* behind. Both documented; partial runs
should use a scratch `--db-dir`.

### 2.6 Memory: several GB of avoidable Python floats

`build_chroma` materialised per-window embeddings with `.tolist()` and `db.load_all`
accumulated each page as Python floats. At 163,915 × 128 that is ~670 MB of boxed
floats on a box with ~3 GB free. Both now pass float32 numpy arrays (verified chroma
accepts a 2-D ndarray and preserves values). Measured peak RSS is **flat at
~1.6 GB** from 100 → 3,000 tracks, i.e. fixed runtime overhead, not
data-proportional.

### 2.7 Operational hazards

- **`train.py` writes `checkpoints/` and `train_log_*.csv` relative to the cwd**, so
  a fresh run started from the repo root silently overwrites `checkpoint_e0…`. A
  17-epoch ablation would have destroyed e0–e16. Experiments now run from
  `~/mel_runs/<tag>/`. (Chose this over adding a flag, to avoid touching the
  training path.)
- **Runs kept dying at epoch 0 — it was the OOM killer, and it was self-inflicted.**
  I first attributed this to external teardown on the grounds that "GPU and RAM were
  clean". **That was wrong.** `echo $?` gave **137** (SIGKILL) with *0 GB available*
  and no swap. The cause is a leak: every DataLoader worker holds **~670 MB** RSS
  (each imports torch + ROCm), and when a run dies its workers are **not** reaped —
  they reparent and sit idle forever. Measured on this box: **21 leaked workers
  holding 11.1 GB of RSS**, plus 1.2 GB of leaked `/dev/shm` and 1,207 stale entries
  in it, against 15 GB total and **no swap**. So each failed attempt consumed another
  ~5 GB, and the machine spiralled: after two deaths, every subsequent run was
  OOM-killed at its first batch. Two generations were visible at once — one 1h56m old
  (the crashed 100-epoch run) and one 26m old (a killed NCE attempt).
  **Prevention: check `free -g` and reap dead runs' workers before launching**, and
  keep `--workers` low (it multiplies the leak by 0.67 GB per worker per run). Note
  `--workers 0` is currently unusable without the `prefetch_factor` fix applied
  below, which would otherwise be the clean escape hatch.
- **`--workers 0` was broken** (now fixed): the DataLoader was constructed with
  `prefetch_factor=args["prefetch"]` unconditionally, and `DataLoader` rejects that
  when `num_workers=0`.
- **The 100-epoch run crashed at 86/100** (17:57) with a ROCm hardware fault:
  `Memory access fault by GPU node-1 … Page not present or supervisor privilege`,
  followed by a failed GPU coredump. This is the *same fault class* CLAUDE.md already
  documents for compiled kernels on this RX 6650 XT — it is not a code or data
  problem, and it leaves orphaned dataloader workers behind. Checkpoints `e0–e85`
  survived. **Assume long ROCm runs will fault; checkpoint every epoch and resume
  with `-r`.** The evaluation here was done on e85 rather than resuming the last 14.
- **A completion monitor was blind to that crash.** Its liveness check was
  `pgrep -f "music_encoding.train"` — and the monitor's *own* shell command line
  contains that literal string, so it matched itself and reported "alive"
  indefinitely. Watch the **log** (epoch count advancing, plus crash signatures like
  `Memory access fault`/`Traceback`) rather than process presence; it also catches a
  hung process, which `pgrep` cannot.
- **The LR schedule is inert.** Over 43 epochs the reference run's `lr` never left
  1e-4: `ReduceLROnPlateau(patience=10, threshold=1e-4 rel)` never fires, because
  the loss keeps inching down ~−1 epoch⁻¹ on a loss of ~627. So the model was never
  annealed. Annealing is the next hyperparameter lever after the objective.

---

## 3. Corrections made — including to my own work

### 3.1 My `lambd` conclusion was confounded by epochs

I found `train.py`'s `--lambd` default (5e-2) contradicts the docs (which say `2e-2`
in one place and `3e-3` in another, while `instructions.md` §5 recommends
`2e-3`–`5e-3`), and that at e42 `off_diag` was **38% of the objective** while the
mean off-diagonal correlation was already only |c| ≈ 0.067.

A harness sweep then showed `on_diag` falling 83 → 2 as `lambd` dropped, and I
concluded the `5e-2` CLI default was a real misconfiguration. **That conclusion was
wrong**, for two reasons I found only by testing properly:

- I had compared `lambd`=5e-2 **at 42 epochs** against the other values **at 17
  epochs**. At *matched* 17 epochs all three are identical on retrieval.
- I was optimising `on_diag`, which turns out not to be a valid proxy (3.2).

### 3.2 `on_diag` is not a valid proxy for the objective

`on_diag` falls **447 → 334 → 228** across `lambd` = 5e-2 → 2e-2 → 5e-3, with **no
retrieval benefit whatsoever** (0.3453 / 0.3520 / 0.3553 — within sampling noise).
Mechanism: `off_diag` (redundancy reduction) is what *spreads* the embedding cloud.
Lowering `lambd` removes that pressure, the cloud contracts, and the two views of a
track move closer — which is precisely what `on_diag` measures — while different
tracks crowd together.

`on_diag` remains useful for its original purpose: spotting a config whose two views
share too little signal to learn at all (pinned near ~800–1000 ≈ D). **Real changes
should be judged on the retrieval metric and the collapse count.**

I corrected `instructions.md` §9 and CLAUDE.md rather than leaving the wrong
version standing.

### 3.3 The genre metric was measuring the alphabetically-first tag

**82% of the corpus carries 2–9 genre tags, and the TSV stores them in alphabetical
order.** So `tags["genre"][0]` — which `test_knn` had always scored against — means
"the alphabetically first genre", not a primary one. `track_0000946` is
`ambient, chillout, downtempo, easylistening, electronic, lounge` and was labelled
**ambient** because 'a' < 'c'. A neighbour sharing a genuine genre under a different
first letter was scored wrong, deflating accuracy on ~80% of the test set.

`test_knn` now reports three scorings with their chance levels (see §5). This is the
single change that most altered the headline number — and it moved it *up*.

### 3.4 `db.py` never actually sorted by window

`db.load_all`'s docstring promises ordering by `(track idx, window)`, but it parsed
the window index as `id.split("_")[3]` for ids shaped `track_<idx>_w<k>`, whose split
is `['track', '<idx>', 'w<k>']` — only 3 elements, so the branch never ran and window
was always 0. The sort was effectively idx-only. Fixed; **verified to produce
identical eval numbers** on the e42 DB (chroma happened to return windows in
insertion order), so no recorded number is invalidated.

### 3.5 Stale documentation corrected

- CLAUDE.md described the conv stack as **3** blocks ending in `AdaptiveAvgPool2d(1)`;
  `model.py` has **4** blocks ending in `AdaptiveAvgPool2d(2)` → 512 features. An
  earlier refactor dropped an early-feature concat and added a block without
  updating the doc.
- CLAUDE.md's `lambd` note quoted three different values across two places; now one
  documented value plus the reasoning.
- CLAUDE.md's `chroma_db` claim (2.4) is corrected.
- Memory's `.npy`-id claim (2.2) is corrected.
- `test_network.py` referenced a `listens` metadata key MTG does not have.

### 3.6 Bugs in my own throwaway tooling

- The chain-ablation harness fed **CPU** tensors to a CUDA model. Under autocast this
  raises a misleading `Input type (float) and bias type (c10::Half)` rather than a
  device error, which cost several debugging rounds. The repo code was never
  affected (`train.py` does `spec.to(device)`).
- A monitor's `pgrep` pattern was malformed, producing a false "process died" alarm.
- The retrieval script parsed the window index with the same wrong split as 3.4.

---

## 4. What was built

| Change | Why |
|---|---|
| `build_chroma --mel-root` (+ `--n-mels`, `--min-window-db`) | Mel-mode checkpoints could not be scored at all; this is the original blocker |
| `MelSpecWindowSource` (`twin_dataset.py`) | Window cut / silence gate / mel merge shared by training **and** eval, so they cannot drift. Verified **bit-identical** to the pre-refactor dataset over 8 configs (both `n_mels`, plain and augmented) — a byte-for-byte A/B against the old class reconstructed verbatim |
| `MTGJamendoBase.row_by_num` / `track_num` / `raw_30s.tsv` fallback | Joins `.npy` stems to TSV rows by number; complete metadata for all 32,783 |
| `test_knn` three scorings + `genres`/`n_genres` in DB metadata | The label-ambiguity fix (§3.3) |
| **`test_retrieval.py`** (new) | The metric that matches the fingerprinting objective (§6) |
| `test_similar_pairs --mel-root`, `test_augmentation --mel-root` | The two audio-blocked QA scripts now work as spectrogram PNGs |
| `db.py` paged float32 + window-sort fix | Memory (§2.6) + correctness (§3.4) |

---

## 5. Current results

All numbers below are on the **same corpus and window protocol** (32,783 MTG
mel tracks, 5 seeded windows/track, 163,915 window vectors, `n_mels` 64 unless
noted, `random.seed(42)`), so they are mutually comparable. Genre: 121 primary
labels, chance 0.008; `single` = clean-label test tracks only (n=1409), chance 0.008.

| config | genre `primary` | genre `any-tag` | genre `single` | retrieval top-1 | coll. /1000 | raw intra→inter |
|---|---|---|---|---|---|---|
| e16, λ=5e-2 (17 ep) | 0.302 | 0.416 | 0.410 | 0.3453 | 50 | 0.9293 → 0.7956 |
| e42, λ=5e-2 (42 ep) | 0.314 | 0.435 | 0.425 | 0.4353 | **42** | 0.9451 → 0.8335 |
| λ=5e-3 (17 ep) | 0.311 | 0.434 | 0.438 | 0.3553 | 70 | 0.9397 → 0.8071 |
| λ=2e-2 (17 ep) | 0.313 | 0.429 | 0.436 | 0.3520 | 209 | 0.9526 → 0.8572 |
| `n_mels`=96 (17 ep) | 0.305 | 0.421 | 0.420 | — | 319¹ | 0.9491 → 0.8610 |
| e85, λ=5e-2 (86 ep, B=256) | 0.341 | 0.461 | 0.450 | 0.5247 | 71² | 0.9513 → 0.8445 |
| final100c (100 ep, B=384, cosine LR) | 0.326 | 0.449 | 0.436 | 0.4540 | 200² | 0.9573 → 0.8661 |
| cont150 (150 ep, B=256) | 0.344 | 0.468 | 0.459 | 0.5787 | 43² | 0.9519 → 0.8442 |
| e298 (298 ep, B=256) | 0.347 | 0.473 | 0.463 | 0.6280 | 81² | 0.9672 → 0.8961 |
| **e621 (621 ep, B=256) — BEST** | **0.352** | **0.483** | **0.468** | **0.6500** | 223² | — |

Steps are the effective lever, so read the retrieval column against step count rather
than epochs — the full measured curve:

| steps (k) | 2.2 | 5.4 | 11 | 19.2 | 38 | 57 | 78 | 99 | **128** |
|---|---|---|---|---|---|---|---|---|---|
| **raw top-1** | 0.3453 | 0.4353 | 0.5247 | 0.5787 | 0.6280 | 0.6387 | 0.6500 | 0.6573 | **0.6733** |
| **whitened top-1** | — | — | 0.5320 | 0.5873 | 0.6387 | 0.6493 | 0.6587 | 0.6600 | **0.6653** |
| genre `single` | 0.410 | 0.425 | 0.450 | 0.459 | 0.463 | 0.459 | 0.468 | 0.467 | **0.469** |
| genre `any-tag` | 0.416 | — | — | 0.468 | 0.473 | 0.478 | 0.483 | 0.483 | **0.491** |

**The 1000-epoch result — the final model:**

| metric | value |
|---|---|
| retrieval top-1 raw / whitened | **0.6733 / 0.6653** |
| retrieval top-10 raw | **0.7933** |
| genre `primary` / `any-tag` / `single` | 0.359 / **0.491** / **0.469** |
| whitened gap | **0.5690** |
| collapsed per 1000 | **171** (149× same-artist lift — mostly genuine) |
| network same-artist / same-genre edges | 0.019 / 0.117 (vs 0.004 / 0.052 random) |

**Two saturation calls were made and both were wrong**, which is itself the finding —
recorded so it isn't repeated:

- At **57k steps** a single +0.011 interval looked like flattening; the next interval
  repeated +0.011 exactly (a linear tail, not a plateau).
- At **99k steps** the whitened metric gained only +0.001 and genre was flat, so the
  remaining steps were projected at ~+0.005 raw / ~0 whitened. The actual final
  stretch delivered **+0.016 raw and +0.005 whitened**, and the whitened metric
  resumed climbing after its 99k pause.

The lesson: with a metric this noisy (±0.03 at 1500 queries) and a curve this slowly
turning, a single interval is not enough to declare saturation — two consecutive
intervals of the *same* marginal gain indicate a tail; genuine flattening needs the
gain to collapse across several points, and even then the resumption at 100k→128k
shows the curve can recover. **Compute kept paying to 128k steps**, which vindicates
the push; the honest summary is that the return diminished roughly 5× from the
steepest region but never reached zero.

¹ measured over a 3,000-track sample (958 collapsed). ² over 3,000 (213 collapsed);
every other row is over 1,000, so compare per-1,000 rates, not raw counts.

**Collapse counts are noisy**, so read them as indicative: `off_diag` varied ~29%
between two runs of the *identical* config (6,181 vs 4,806 at the same epoch), and
single-run collapse spanned 42–213 per 1,000. This is why the λ decision leaned on
retrieval (1,500 sampled queries, ±0.03 noise) rather than collapse.

### e85 (86 epochs) — every metric improved except the collapse count

Doubling the epochs from 42 → 86 moved **every** semantic and separation metric:

| metric | e42 (42 ep) | e85 (86 ep) |
|---|---|---|
| retrieval top-1 (raw) | 0.4353 | **0.5247** (+20% rel.) |
| retrieval top-10 (raw) | 0.5820 | **0.6580** |
| genre `primary` / `any-tag` / `single` | 0.314 / 0.435 / 0.425 | **0.341 / 0.461 / 0.450** |
| whitened gap | 0.4410 | **0.4886** |
| same-artist / same-genre edges | 0.011 / 0.101 | **0.015 / 0.113** |
| collapsed per 1,000 | **42** | 71 |

The one regression is the collapse count — and it is **mostly genuine musical
similarity, not dimensional collapse**. Of the 213 collapsed tracks in the 3,000
sample: **29.1% have their nearest neighbour as the same artist** against a 0.174%
random-pair baseline (~170× enrichment), 15.0% the same album. The effect is
concentrated by style — **classical is 8.4% of the corpus but 37.1% of collapsed
tracks (4.4× lift)** while electronic is *depleted* (12.9% → 3.3%, 0.3×) — and the
inspection examples are all sparse solo-piano / new-age material (`Galdson`,
`Roger Subirana Mata`, Chopin via `Kranto studijos`, `Ronny Matthes`). Sparse
acoustically-homogeneous windows genuinely do sound alike. Note also that this
threshold is measured in **raw** space, whose crowding whitening largely removes
(whitened gap 0.489), so it is not the informative-space picture.

This reframes the earlier λ result too: the 4.2% → 7.1% "regression" I flagged as a
trade against the epoch gain is not a model-quality loss. It does mean the collapse
count is a poor stand-alone acceptance test — pair it with the same-artist lift.

### Targets

| Target | Result |
|---|---|
| retrieval top-1 up from 0.435 | **met** — 0.5247 |
| genre `single` ≥ 0.50 | not met — 0.450 (was 0.425) |
| genre `any-tag` ≥ 0.50 | not met — 0.461 (was 0.435) |
| network ≥ 3× artist, ≥ 2× genre | **met** — 3.75× / 2.17× |
| collapsed < 3% | not met — 7.1%, but style-concentrated (above) |

Whitened gap (intra−inter) by config: e16 0.386, **e42 0.441**, λ=5e-3 0.379,
λ=2e-2 0.417, `n_mels`=96 0.394.

### Historical baselines (NOT comparable — kept only to show why)

| Config | k-NN genre | Corpus / labels |
|---|---|---|
| Disjoint windows only | 42–46% | FMA-era, ~16 labels |
| First augmented run | ≈31% | FMA-era, ~16 labels |
| Old FMA DB | 0.360 whitened | 7,916 FMA tracks |

Against chance the FMA runs sat at ~5.8× (0.360 / 0.0625); the current MTG model is
at **39–53×** (0.314 / 0.008, and 0.410–0.438 on clean labels). The apparent
regression was an artefact of an ~8× larger label space plus the `tags[0]` bug.

---

## 6. The finding that matters: the objective is the bottleneck

`test_retrieval.py` asks the question the model exists to answer: *for a 5 s window,
does its nearest neighbour belong to the same track?*

| k | raw | whitened |
|---|---|---|
| 1 | 0.4353 | 0.4400 |
| 3 | 0.5140 | 0.5260 |
| 5 | 0.5427 | 0.5620 |
| 10 | 0.5820 | 0.6093 |

Chance is 2.4e-05 — so the model is ~18,000× above chance, but for a Shazam-style
fingerprint top-1 should approach **1.0**, and **whitening barely moves it** (unlike
the intra/inter gap, which whitening transforms). That combination is diagnostic:
the weakness is **not** poor conditioning and **not** any hyperparameter tested.

It is structural. Barlow Twins (i) pulls two views of the *same* track together and
(ii) decorrelates *dimensions*. It never pushes *different tracks apart*. Redundancy
reduction is not instance discrimination — which is why raw inter-track similarity
sits at 0.80–0.86 while genre k-NN looks respectable.

**Levers ruled out, with evidence:**

| Lever | Verdict |
|---|---|
| Augmentation chain | Removing **every** content distortion *and* RRC moves `on_diag` only 97 → 73 (~25%); λ moves it 40×. The chain is not the constraint. `instructions.md` §8 had named it "the remaining knob" — that is now refuted, and **no `augmenter.py` change was made** |
| `n_mels` 96 | Equal to 64 on genre (0.305 vs 0.302), **worse** on raw collapse (0.8610 inter vs 0.7956). The bilinear 96→64 merge costs nothing |
| `lambd` | A wash on retrieval at matched epochs; a trade elsewhere (lower λ: +0.03 genre, worse collapse) |
| **Optimizer steps** | **The confirmed lever.** Retrieval tracks steps almost monotonically (see below) |

### The objective term was tried, and rejected on evidence

A contrastive/InfoNCE term was implemented (`InfoNCELoss`, default off) and tested at
two targets (`--nce-on z|emb`), 17 epochs each at weight 30 (a deliberately large
weight — BT totals ~1,100 while InfoNCE is ~ln(2B−1) ≈ 6.6, so a normal-sized
coefficient would vanish; the term held a steady ~18% of the loss):

| | e16 (no NCE) | NCE on `z` | NCE on `emb` |
|---|---|---|---|
| retrieval raw / whitened | 0.3453 / 0.3620 | 0.3847 / 0.3680 | 0.3507 / 0.3567 |
| raw intra → inter | 0.9293 → 0.7956 | 0.9468 → 0.8498 | **0.7814 → 0.4177** |
| **whitened gap** | **0.3860** | 0.3952 | **0.3848** |
| collapsed /1000 | 50 | 167 | **0** |

Repelling `emb` reshapes the raw space spectacularly — inter 0.7956 → 0.4177, std
0.063 → 0.199, zero near-duplicate pairs. **But it is a learned reparameterization,
not new information**: the whitened gap is unchanged (0.3848 vs 0.3860), and e16's
*whitened* numbers already match nce_emb's *raw* ones. It approximates what post-hoc
whitening does — and `tracks_whitened` already provides that, for free. Judged on
retrieval it gives nothing in either space, so the dramatic raw numbers are a **proxy
win** — the same trap as optimizing `on_diag`. Explored, documented, not carried
forward. (Caveat: 17 epochs is unconverged, so this is "no measurable effect", not a
refutation; the untried directions are a much larger weight, a lower temperature, and
hard-negative mining, since in-batch negatives are random tracks and therefore easy.)

### What actually improved the model: optimizer steps, not epochs

The 100-epoch run at **batch 384** came out *worse* than the 86-epoch batch-256 run on
every headline metric — despite a lower BT loss and the lowest `off_diag_norm` of any
run. The cause is arithmetic: 32,783 tracks ÷ batch = batches/epoch (**128 at batch
256, 85 at 384**), so 100 epochs at 384 is *fewer* optimizer steps than 86 epochs at
256. Batch size is not only a memory knob — **at fixed epochs it decides how much
optimisation happens**. Retrieval against steps:

| run | batch | epochs | steps | retrieval top-1 |
|---|---|---|---|---|
| e16 | 256 | 17 | 2,176 | 0.3453 |
| e42 | 256 | 42 | 5,376 | 0.4353 |
| final100c | 384 | 100 | 8,500 | 0.4540 |
| e85 | 256 | 86 | 11,008 | 0.5247 |
| **cont150** | 256 | 150 | ~19,200 | **0.5787** |

Cosine LR annealing was **also net-negative** at this stage (my call, and it was
wrong): the 8,500-step point sits *below* the step trend. With the loss still
descending ~−1/epoch there is no plateau for annealing to exploit, so decaying the LR
only shortens the remaining steps. Keep the (inert) plateau schedule and accumulate
steps — which is what the final run does by resuming rather than restarting, keeping
the steps already paid for.

**Final configuration: plain Barlow Twins, batch 256, λ=5e-2, plateau LR (1e-4),
n_mels 64, ~19,200 steps (`~/mel_runs/cont150/checkpoints/checkpoint_e150.pt`).**

---

## 7. Reproducing

```bash
# score a mel-mode checkpoint (the port this work added)
nix develop . --command python -m music_encoding.build_chroma \
    <ckpt>.pt --mel-root ~/mtg_jamendo \
    --mtg-data ~/mtg-jamendo-dataset/data --n-mels 64
nix develop . --command python -m music_encoding.test_knn
nix develop . --command python -m music_encoding.test_similarity_same_song
nix develop . --command python -m music_encoding.test_retrieval
nix develop . --command python -m music_encoding.test_network --color genres
```

`--n-mels` **must match the training run**: the conv stack is resolution-agnostic, so
a mismatched value runs fine and silently returns wrong numbers. Nothing can validate
it for you.

**Do not launch a training run from the repo root** unless you intend to replace
`checkpoints/` (§2.7).

---

## 8. Status

- **Complete and verified:** the mel-mode eval port, the metadata join, the three
  metric/labelling corrections, the lever ablations (chain / `n_mels` / λ), the
  retrieval metric, and all documentation.
- **FINAL MODEL: `~/mel_runs/cont1000/checkpoints/checkpoint_e1000.pt`** — 1000
  epochs, batch 256, plain BT, λ=5e-2, constant LR, **128,000 optimizer steps**.
  Retrieval **0.6733** raw / **0.6653** whitened, top-10 0.7933, genre `single`
  **0.469**, `any-tag` **0.491**, whitened gap 0.5690, collapsed 17.1% (149×
  same-artist). Its DB is promoted to `chroma_db/`, so every eval runs with no
  arguments.
- **The 1000-epoch push completed**: e298 → e1000 (702 epochs added) in 6 supervised
  attempts over ~14 h of wall clock. 1 GPU fault, recovered automatically in 20 s;
  **0 manual interventions after launch**. Nothing leaked (RAM back to 12 GB free).
- **Explored and rejected, documented:** the contrastive objective term (§6), the
  augmentation chain, `n_mels` 96, λ tuning, batch 384 (costs steps), and cosine LR
  annealing. The confirmed lever is **optimizer steps** — and the curve shows it was
  still paying at 128k steps, so more compute would very likely help further, just
  with diminishing returns.
- **Still short of the objective:** retrieval 0.673 is well above chance (2.4e-05)
  and roughly double the day's starting point (0.345), but far from the ~1.0 a
  fingerprint product needs. Genre plateaus near the label-noise ceiling
  (`single` 0.469, and ~20% of the corpus has no reliable single genre). Untried
  directions, in rough order of promise: **more steps** (the proven lever), `embed_dims`
  128 → 256, hard-negative mining (in-batch negatives are random tracks, so easy), and
  a larger contrastive weight.
