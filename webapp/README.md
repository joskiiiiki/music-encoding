# MTG-Jamendo similarity explorer

A local web app over the embeddings in `music_encoding/`: search 32,783 MTG-Jamendo tracks by
artist/title with genre/instrument/mood filters, play one, and explore its cosine-similar
neighbours as a graph you can walk (click a node to recentre).

```
webapp/
├── data/            generated artifacts (gitignored) -- catalog.sqlite, vectors.npz, audio.sqlite
├── api/             FastAPI backend: reads the artifacts, serves JSON + audio
│   ├── app/         paths, index (search + exact k-NN + graph), enrichment, previews, audio
│   └── scripts/     export_index, verify_export, check_api, local_audio_index,
│                    warm_previews, measure_enrichment
└── ui/              SvelteKit + Tailwind + shadcn-svelte frontend
```

The backend never imports `torch`, `chromadb`, or the `music_encoding` package at request time.
One script (`api/scripts/export_index.py`) reads the model side; everything after that is numpy
and sqlite.

## Setup

Two dev shells, on purpose: `.#web` is light (no ROCm torch) and evaluates fast, while the
default shell is the one with `chromadb`, which only the exporter needs.

```bash
# 1. Export the index from chroma_db  (default shell -- needs chromadb)
nix develop --command python webapp/api/scripts/export_index.py

#    ...and check it reproduces chroma's own neighbours
nix develop --command python webapp/api/scripts/verify_export.py

# 2. Index the local audio (optional but recommended: 586 full tracks)
nix develop .#web --command python webapp/api/scripts/local_audio_index.py

# 3. Pre-resolve previews so coverage is known upfront (optional, ~2.2 h measured, resumable)
nix develop .#web --command python webapp/api/scripts/warm_previews.py --workers 6
```

`export_index.py` writes `catalog.sqlite` (metadata + FTS5 index + facet tables) and
`vectors.npz` (whitened + raw, unit-normalised, ordered by `idx`). Re-run it after training a
new checkpoint; the preview cache lives in a **separate** `audio.sqlite` so a re-export can
never wipe hours of warming.

## Running

```bash
# terminal 1 -- API on :8000
cd webapp/api && nix develop ../..#web --command uvicorn app.main:app --reload --port 8000

# terminal 2 -- UI on :5173 (proxies /api to the API)
cd webapp/ui && nix develop ../..#web --command pnpm dev
```

Then open <http://localhost:5173>. Proxying `/api` keeps the browser on one origin, so there is
no CORS and the `<audio>` element can issue Range requests against the same host.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `WEBAPP_CHROMA_DB` | main checkout's `chroma_db/` | source DB for the exporter |
| `WEBAPP_DATA_DIR` | `webapp/data` | where the artifacts live |
| `MTG_DATA_DIR` | `~/mtg-jamendo-dataset/data` | the MTG TSVs (durations) |
| `MTG_AUDIO_TAR` | `~/mtg/raw_30s_audio-low-00.tar` | local audio slice |
| `WEBAPP_API` | `http://127.0.0.1:8000` | proxy target for the UI dev server |
| `WEBAPP_PREVIEW_RATE` | `4` | preview lookups per second, shared across all warm workers |

Paths default sensibly and the API derives the chroma path from `git rev-parse
--git-common-dir`, so it finds the real `chroma_db/` even when the app runs from a linked git
worktree (where that gitignored directory does not exist).

## Verifying

```bash
nix develop .#web --command python webapp/api/scripts/check_api.py       # search, facets, k-NN, graph, ranges, FTS
nix develop .#web --command python webapp/api/scripts/check_previews.py  # resolver outcomes + rate limiter, offline
nix develop .#web --command python webapp/api/scripts/measure_enrichment.py  # the numbers the UI cites
cd webapp/ui && nix develop ../..#web --command pnpm check               # svelte-check
nix develop .#web --command ruff check webapp/api
```

`check_api.py` asserts on results rather than exit codes: that a text search finds the artist,
that facet counts ignore their own dimension's filter, that edge weights equal the cosine of
their endpoints, and that the graph is deterministic.

## What the model actually looks like through this app

Measured on the promoted `checkpoint_e1000`, whitened space, 300 random seeds
(`measure_enrichment.py`):

| k | genre-tag overlap vs chance | same-artist vs chance |
|---|---|---|
| 6 | 3.96× (84% raw overlap) | 138× |
| 10 | 3.68× | 114× |
| 25 | 3.29× | 79× |

So neighbours are strongly artist-organised and moderately genre-organised. The **enrichment
panel** on every track page reports these against the corpus, because a force-directed picture
looks structured whatever the input — `CLAUDE.md` documents that a k-NN plot renders a
perfectly genre-encoded embedding as the same hairball as isotropic noise. Note that a single
seed can legitimately score zero (an atypical track with no siblings among its 20 neighbours);
the panel says so rather than showing an error.

## Audio: read this before trusting playback

Two sources, and neither covers everything:

* **Local** — 586 of 32,783 tracks (1.8%) exist as mp3s inside
  `~/mtg/raw_30s_audio-low-00.tar`. These are served by seeking into the archive using an index
  of member byte offsets, so nothing is unpacked. This is the *actual* audio the embedding came
  from.
* **Preview** — resolved from artist + title via Deezer (iTunes as an optional second pass). MTG-
  Jamendo is royalty-free and largely absent from those services. A completed full pass resolved
  **5,918 tracks (18% of the 32,197 that are not local)**; a 40-track random sample had suggested
  28%, which is a reminder of how wide the interval is at n=40. In total **6,518 of 32,783 tracks
  (20%) are playable** and 26,265 are confirmed unavailable. Deezer is the default for a measured
  reason — see the pre-warm note below. A preview is also **a different recording** of the song
  matched by fuzzy text search, so the player labels it and offers "not it?" to re-resolve.

Playback is therefore deliberately optional everywhere: unplayable rows are greyed rather than
failing, and "Only tracks with audio" filters browse and graph to what is playable. Only URLs
are cached, never audio, matching both providers' terms.

**Quota errors must never be cached as misses.** Deezer reports its rate limit as **HTTP 200**
with `{"error": {"message": "Quota limit exceeded"}}` — a resolver that reads only `data` cannot
tell that from "this track has no preview". A first warm run at 10 workers did exactly that:
it recorded 12,512 tracks as permanently unavailable, showed an **8%** hit rate where the corpus
yields far more, and the wrongly-recorded tracks stayed misses after the quota recovered. Those
rows were deleted and the run redone. Now a refusal is retried and never written, and
`provider="none"` requires *every* provider to have answered cleanly. `check_previews.py` is the
regression test.

**The pre-warm is Deezer-only by default, measured rather than assumed.** Across the first full
pass Deezer produced 1,135 hits and iTunes 14 — about 1% of coverage for ~72% of the request
volume, because every Deezer miss falls through to a second query, and iTunes is the tighter
quota. With iTunes in the loop, 59 of every 100 tracks came back as refusals and the job ran at
**0.6 tracks/s with a 14-hour ETA**; Deezer alone runs **4.0/s with zero refusals** (~2 h for the
corpus). `--provider itunes --refresh` runs the tail separately if wanted. One shared limiter
caps lookups at 4 req/s (`WEBAPP_PREVIEW_RATE`), because the quota is per-IP and a wider worker
pool only trips it sooner.

## Notes for anyone changing this

* Similarity is **exact** brute-force cosine over the exported matrix (`X @ X[i]`, ~2 ms for
  32,783×128), not an approximate index. `verify_export.py` checks it reproduces chroma's
  answers, which is what would catch a transposed or misordered export.
* `col.get(ids=[...])` in chromadb returns rows **sorted by id, not in the requested order**.
  Zipping its output against the request list silently pairs every seed with the wrong
  embedding — this produced a wrong published measurement once (it reported 1.15× genre
  enrichment where the truth is 3.96×). Key by id.
* The graph's enrichment compares against the **corpus**, not a within-graph label shuffle.
  The node set is chosen *by* similarity to the seed, so shuffling labels inside it leaves the
  enrichment in place and reports a lift of ~1 even when every neighbour is the same artist.
* `.gitignore` anchors the vendored `/lib/` rule, because unanchored it also matched
  SvelteKit's `src/lib/`. Repo-wide `*.png` and `*.csv` rules mean frontend assets are SVG.
* **`app.css` defines `data-horizontal:` and `data-vertical:` as custom Tailwind variants.**
  shadcn-svelte's generated components style orientation with those variants (slider,
  scroll-area, separator, tabs), but the installed bits-ui publishes the orientation as
  `data-orientation="horizontal" | "vertical"`. The short attributes never appear in the
  DOM, so those variants never matched — and the failure is silent: a Slider drew its thumb
  with **no track**. Defining the two variants in terms of the attribute that exists fixes
  all four components at once, and unlike editing the component files it survives a re-run
  of `shadcn-svelte add`.
* `ScrollArea`'s own scrollbar thumb does not mount in this app: bits-ui only renders it
  when its measurement reports overflow, and it reports none (a 14-row facet list inside
  `h-56` is visibly clipped, yet `--bits-scroll-area-thumb-height` comes out at 222px in a
  224px viewport). The panes scroll correctly and the bar you see is the platform's own.
  Styling it with `::-webkit-scrollbar` was tried and removed — this Chromium uses overlay
  scrollbars, which ignore that CSS entirely, so the rule was inert.
