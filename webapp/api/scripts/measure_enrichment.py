#!/usr/bin/env python3
"""Measure how organised the neighbourhoods actually are, from the exported index.

    python webapp/api/scripts/measure_enrichment.py

Needs neither chromadb nor torch -- it reads ``webapp/data/`` only, so it runs in
either dev shell.

This exists because the web app's headline is "similar songs", and the honest answer
for this model is "modestly". The similarity graph shows a seed and its neighbours;
without a baseline, a force-directed picture invites the reader to see structure that
may not be there (`CLAUDE.md` is explicit that a k-NN plot at default sampling "says
nothing about the model"). The numbers printed here are the reference the UI's
enrichment panel is checked against.

Chance is **prevalence-weighted**, not uniform: genre tags are heavily skewed
(`electronic` alone is ~29% of the corpus), so dividing by the tag count would
understate chance and overstate the lift by an order of magnitude. For a seed with tags
S, the expected overlap is ``sum((count[t]-1)/(N-1) for t in S)``.

Reference values for the promoted e1000 checkpoint (300 random seeds, whitened space):

    k=6   genre 3.96x   same-artist 138x
    k=10  genre 3.68x   same-artist 114x
    k=25  genre 3.29x   same-artist  79x

Genre tags are ~3.3-4.0x chance with 84% raw overlap at k=6, and same-artist is two
orders of magnitude above chance (22% of the 6 nearest neighbours share the artist).
The same-artist figure is consistent with the 149x lift `test_retrieval.py` reports
for the same checkpoint, which is a useful corroboration -- the artist-dominance
`CLAUDE.md` documents is real and strong, not marginal.

An earlier version of this measurement reported 1.15x/1.9x. That was a bug, not a
different population: it fetched seed embeddings with ``col.get(ids=[...])``, which
returns rows **sorted by id rather than in the requested order**, so every seed was
paired with another seed's neighbours. The numbers above were reproduced independently
through chroma's HNSW query with ids keyed by string, and agree to three decimals.

Clustering is also strongly *local*: seeded from a prolific artist, the same-artist
share of the top-10 is 0.29-0.66 (Soldiah Beez 0.66, Zeffon 0.57, Plastic3 0.50,
Anitek 0.44, Podington Bear 0.38, MFYM 0.29) -- higher than at corpus-wide k=10
because these artists have hundreds of tracks to find.
"""

from __future__ import annotations

import argparse
import collections
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> webapp/api
from app.paths import catalog_path, vectors_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--space", default="whitened", choices=["whitened", "raw"])
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{catalog_path()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    tracks = {r["idx"]: r for r in con.execute("SELECT * FROM tracks")}
    tags: dict[int, set[str]] = {}
    for idx, tag in con.execute("SELECT idx, tag FROM track_tags"):
        tags.setdefault(idx, set()).add(tag)
    n = len(tracks)
    if n == 0:
        raise SystemExit("catalog is empty -- run export_index.py")

    artist = {i: (r["artist"] or "") for i, r in tracks.items()}
    tag_count = collections.Counter(t for s in tags.values() for t in s)
    artist_count = collections.Counter(a for a in artist.values() if a)

    W = np.load(vectors_path())[args.space]
    rng = np.random.default_rng(1)
    seeds = rng.choice(n, size=min(args.seeds, n), replace=False)
    S = W @ W[seeds].T
    S[seeds, np.arange(len(seeds))] = -np.inf  # a track is its own best neighbour

    print(f"{n} tracks, {len(tag_count)} genre tags, {len(artist_count)} artists")
    print(f"space={args.space}, {len(seeds)} random seeds")
    print(
        "top tags by prevalence:",
        ", ".join(f"{t}={v / n:.0%}" for t, v in tag_count.most_common(5)),
    )
    print()

    for k in (6, 10, 25):
        top = np.argsort(-S, axis=0)[:k]
        obs = exp = same_artist = expected_artist = 0.0
        for j, seed in enumerate(seeds):
            neighbours = top[:, j]
            seed_tags = tags.get(int(seed), set())
            obs += sum(len(seed_tags & tags.get(int(m), set())) for m in neighbours)
            exp += sum((tag_count[t] - 1) / (n - 1) for t in seed_tags) * k
            a = artist[int(seed)]
            if a:
                same_artist += sum(1 for m in neighbours if artist[int(m)] == a)
                expected_artist += (artist_count[a] - 1) / (n - 1) * k
        slots = k * len(seeds)
        genre_lift = obs / exp
        artist_lift = same_artist / expected_artist
        print(
            f"k={k:2d}: genre-tag overlap {obs / slots:.4f} vs chance {exp / slots:.4f}"
            f" -> {genre_lift:5.2f}x | same-artist {same_artist / slots:.5f} vs "
            f"chance {expected_artist / slots:.5f} -> {artist_lift:5.2f}x"
        )

    print("\nper-artist top-10 same-artist share (the dense regions):")
    for name, _ in artist_count.most_common(6):
        members = np.array([i for i, a in artist.items() if a == name][:60])
        if len(members) < 5:
            continue
        this = W @ W[members].T
        np.fill_diagonal(this, -np.inf)
        top = np.argsort(-this, axis=0)[:10]
        same = sum(
            1 for j in range(len(members)) for x in top[:, j] if artist[int(x)] == name
        )
        print(
            f"  {name:18s} n={len(members):3d}  "
            f"same-artist in top-10: {same / (len(members) * 10):.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
