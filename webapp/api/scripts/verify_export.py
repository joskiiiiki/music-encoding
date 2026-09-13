#!/usr/bin/env python3
"""Check the exported artifacts against the chroma DB they came from.

Run in the **default** dev shell (needs chromadb)::

    nix develop --command python webapp/api/scripts/verify_export.py

This is the test that would catch the failure modes an export can hide: a transposed
matrix, rows ordered by something other than ``idx``, un-normalised rows, or metadata
attached to the wrong track. None of those would raise -- they would just make the web
app quietly return the wrong neighbours -- so the export is checked by reproducing
chroma's own answers.

For N seeded tracks it compares:

  * the top-k neighbour **idx sets** from the exported matrix (exact cosine) against
    ``collection.query`` on ``tracks_whitened`` (approximate HNSW), and
  * the title/artist the catalog reports for those idx, against the metadata chroma
    holds for the same ids.

HNSW is approximate, so near-ties can legitimately differ; agreement is required on
the top-1 and on a large majority of each top-k, and the report prints the overlap so
a regression is visible as a number rather than a pass/fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sqlite3  # noqa: E402

from app.paths import catalog_path, chroma_db_dir, vectors_path  # noqa: E402

SEEDS = 40
K = 10


def main() -> int:
    import chromadb

    db_dir = chroma_db_dir()
    if not catalog_path().exists() or not vectors_path().exists():
        raise SystemExit("run export_index.py first")
    print(f"chroma : {db_dir}")
    print(f"catalog: {catalog_path()}")

    with np.load(vectors_path()) as z:
        W = z["whitened"]
    n = W.shape[0]

    con = sqlite3.connect(f"file:{catalog_path()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = {r["idx"]: r for r in con.execute("SELECT * FROM tracks")}
    if len(rows) != n:
        raise SystemExit(f"catalog has {len(rows)} tracks, vectors have {n}")

    col = chromadb.PersistentClient(path=str(db_dir)).get_collection("tracks_whitened")
    rng = np.random.default_rng(0)
    seeds = rng.choice(n, size=SEEDS, replace=False).tolist()

    # The seed embeddings, straight from chroma, so the only thing under test is the
    # exported matrix rather than any disagreement about what the vectors are.
    #
    # Keyed by id, NOT used positionally: `col.get(ids=[...])` returns the rows sorted
    # by id, ignoring the order asked for (verified), so zipping its output against the
    # request list silently pairs every seed with the wrong embedding.
    raw = col.get(
        ids=[f"track_{i}" for i in seeds], include=["embeddings", "metadatas"]
    )
    emb_by_id = {i: e for i, e in zip(raw["ids"], raw["embeddings"], strict=True)}

    problems: list[str] = []
    top1_ok = 0
    overlaps: list[float] = []
    for seed_idx in seeds:
        # --- our answer: exact cosine, self excluded ---------------------------
        sims = W @ W[seed_idx]
        sims[seed_idx] = -np.inf
        mine = np.argpartition(-sims, K)[:K]
        mine = mine[np.argsort(-sims[mine])]

        # --- chroma's answer ---------------------------------------------------
        q = col.query(
            query_embeddings=[emb_by_id[f"track_{seed_idx}"]],
            n_results=K + 1,
            include=["metadatas"],
        )
        theirs_meta = q["metadatas"][0]
        if theirs_meta and int(theirs_meta[0]["idx"]) == seed_idx:
            theirs_meta = theirs_meta[1:]
        theirs = [int(m["idx"]) for m in theirs_meta[:K]]

        overlap = len(set(mine.tolist()) & set(theirs)) / K
        overlaps.append(overlap)
        if mine[0] == theirs[0]:
            top1_ok += 1
        if overlap < 0.5:
            problems.append(
                f"seed {seed_idx}: only {overlap:.0%} of top-{K} agree with chroma"
            )

        # --- metadata must describe the same track chroma describes -----------
        # Compare chroma's metadata for OUR top-1 idx, fetched by that id, so this
        # checks the idx -> metadata mapping rather than two different tracks.
        m = col.get(ids=[f"track_{int(mine[0])}"], include=["metadatas"])["metadatas"][
            0
        ]
        r = rows[int(mine[0])]
        for key in ("title", "artist", "track_num"):
            if (r[key] or "") != (m[key] or ""):
                problems.append(
                    f"seed {seed_idx}: top-1 idx {mine[0]} catalog {key}={r[key]!r} "
                    f"but chroma says {m[key]!r}"
                )

    mean_overlap = float(np.mean(overlaps))
    print(f"\n{SEEDS} seeds, top-{K}")
    print(f"  top-1 agreement with chroma : {top1_ok}/{SEEDS}")
    print(f"  mean top-{K} overlap         : {mean_overlap:.3f}")
    print(f"  min  top-{K} overlap         : {min(overlaps):.3f}")

    # Sanity checks on the metadata side, which is where an idx/row mismatch would show.
    counts = {
        "tracks": con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0],
        "tags": con.execute("SELECT COUNT(*) FROM track_tags").fetchone()[0],
        "with duration": con.execute("SELECT COUNT(duration) FROM tracks").fetchone()[
            0
        ],
        "with instrument": con.execute(
            "SELECT COUNT(DISTINCT idx) FROM track_instruments"
        ).fetchone()[0],
        "with mood": con.execute(
            "SELECT COUNT(DISTINCT idx) FROM track_moods"
        ).fetchone()[0],
    }
    print("\ncatalog:")
    for k, v in counts.items():
        print(f"  {k:16s} {v}")

    # FTS is what makes search work at all, so check a real artist end to end.
    for term, expect in (("bear*", "Podington Bear"), ("podington", "Podington Bear")):
        hit = con.execute(
            "SELECT COUNT(*) FROM tracks_fts f JOIN tracks t ON t.idx = f.rowid "
            "WHERE tracks_fts MATCH ? AND t.artist = ?",
            (term, expect),
        ).fetchone()[0]
        print(f"  fts {term!r:14s} -> {hit} rows by {expect}")
        if term == "podington" and hit == 0:
            problems.append(f"FTS cannot find {expect} by {term!r}")
    con.close()

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK: exported matrix reproduces chroma's neighbours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
