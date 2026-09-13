#!/usr/bin/env python3
"""Export ``chroma_db`` into the web app's self-contained artifacts.

Run in the **default** dev shell, which is the one that has chromadb::

    nix develop --command python webapp/api/scripts/export_index.py

Writes, into ``webapp/data/`` (gitignored):

    catalog.sqlite   metadata, tags, FTS5 index, facet vocabularies
    vectors.npz      whitened + raw embeddings, row-normalised, ordered by idx

``audio.sqlite`` (the audio-availability / preview cache) is deliberately **not**
written here -- see :func:`app.paths.audio_db_path`.

After this runs, the API needs only numpy + sqlite: no chromadb, no torch, no GPU.

Why export rather than query chroma live
----------------------------------------
Three measured reasons:

1. ``col.get(include=["metadatas"])`` with no limit dies with ``too many SQL
   variables`` at 32,783 rows, so any consumer needs its own pagination anyway.
2. chroma has **no** full-text index -- ``embedding_fulltext_search`` is empty
   because ``build_chroma`` passes no ``documents=`` -- so "search by artist/title"
   needs an index we build ourselves.
3. Brute-force cosine over the exported matrix is **~2 ms** and *exact*, where
   chroma's HNSW answer is approximate.

The result is also the separation the app wants: the web app reads an artifact and
never imports the training stack.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# This script is the one place the web app reaches into the model package, so it has
# to import `music_encoding` -- which lives at the repo root, not next to the script.
# Python puts the *script's* directory on sys.path, not the cwd, so add the root
# explicitly; otherwise the export only works when invoked from one particular place.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> webapp/api

from app.paths import (  # noqa: E402
    catalog_path,
    chroma_db_dir,
    data_dir,
    mtg_data_dir,
    vectors_path,
)

from music_encoding.db import (  # noqa: E402
    COLLECTION_RAW,
    COLLECTION_WHITENED,
    load_all,
)

SCHEMA = """
CREATE TABLE tracks (
    idx        INTEGER PRIMARY KEY,
    track_id   TEXT,
    track_num  INTEGER,
    title      TEXT,
    artist     TEXT,
    album      TEXT,
    genre      TEXT,      -- tags[0]: alphabetically first genre, NOT a primary one
    n_genres   INTEGER,
    released   INTEGER,   -- year, NULL when the TSV has none
    duration   REAL       -- full-track seconds, NULL when unavailable
);

-- One row per (track, tag). Child tables rather than delimited columns so that
-- filtering and facet counts are a single indexed GROUP BY.
CREATE TABLE track_tags (
    idx INTEGER, tag TEXT, PRIMARY KEY (idx, tag)
) WITHOUT ROWID;
CREATE TABLE track_instruments (
    idx INTEGER, tag TEXT, PRIMARY KEY (idx, tag)
) WITHOUT ROWID;
CREATE TABLE track_moods (
    idx INTEGER, tag TEXT, PRIMARY KEY (idx, tag)
) WITHOUT ROWID;
CREATE INDEX track_tags_tag        ON track_tags (tag);
CREATE INDEX track_instruments_tag ON track_instruments (tag);
CREATE INDEX track_moods_tag       ON track_moods (tag);
CREATE INDEX tracks_artist         ON tracks (artist);
CREATE INDEX tracks_released       ON tracks (released);

-- External-content FTS5: shares storage with `tracks` and is (re)built from it.
CREATE VIRTUAL TABLE tracks_fts USING fts5(
    title, artist, album,
    content='tracks', content_rowid='idx',
    tokenize="unicode61 remove_diacritics 2"
);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
"""


def _split(value: str | None) -> list[str]:
    """Split a delimited metadata value into tags.

    ``genres`` is ``|``-joined by ``build_chroma.build_metadata``; ``instrument`` and
    ``mood_theme`` hold a single tag. Both separators are accepted defensively, and
    an empty/absent value yields no tags (chroma omits the key entirely rather than
    writing None, so absent is the common case for ~46% of tracks).
    """
    if not value:
        return []
    out = []
    for chunk in value.replace("|", ",").split(","):
        tag = chunk.strip()
        if tag:
            out.append(tag)
    return out


def _matrix_by_idx(metas: list[dict], embs: np.ndarray, what: str) -> np.ndarray:
    """Place rows into an array indexed by ``meta["idx"]``.

    Never assumes the collection came back in order -- ``idx`` is the app's contract,
    so rows are placed by their own ``idx`` and the result is verified to be dense.
    """
    idx = np.array([int(m["idx"]) for m in metas], dtype=np.int64)
    if len(set(idx.tolist())) != len(idx):
        raise SystemExit(f"{what}: duplicate idx values in metadata")
    n = int(idx.max()) + 1
    if n != len(idx):
        raise SystemExit(f"{what}: idx is not dense (max {idx.max()}, {len(idx)} rows)")
    out = np.zeros((n, embs.shape[1]), dtype=np.float32)
    out[idx] = embs
    return out


def _load_durations(track_nums: list[int]) -> dict[int, float]:
    """Full-track durations, which chroma metadata does not carry.

    ``duration`` lives only in the MTG TSVs (e.g. track 214 -> 124.6 s). A missing
    TSV directory is not fatal: durations are a nice-to-have for the UI, so the
    export proceeds and says so.
    """
    data_dir_ = mtg_data_dir()
    if not (data_dir_ / "autotagging.tsv").exists():
        print(f"  ! {data_dir_}/autotagging.tsv not found -- skipping durations")
        return {}
    from music_encoding.mtg import MTGJamendoBase

    ds = MTGJamendoBase(str(data_dir_))
    out: dict[int, float] = {}
    for num in track_nums:
        row = ds.row_by_num(num)
        if row and row.get("duration"):
            out[num] = float(row["duration"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-dir", type=Path, default=None, help="source chroma DB")
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="defaults to webapp/data"
    )
    ap.add_argument(
        "--force", action="store_true", help="overwrite an existing catalog"
    )
    args = ap.parse_args()

    db_dir = args.db_dir or chroma_db_dir()
    if args.out_dir:
        # paths.py reads the env var, so honour the flag by exporting alongside it.
        import os

        os.environ["WEBAPP_DATA_DIR"] = str(args.out_dir)
    out_catalog = catalog_path()
    out_vectors = vectors_path()

    print(f"source DB      : {db_dir}")
    print(f"destination    : {data_dir()}")
    if out_catalog.exists() and not args.force:
        raise SystemExit(
            f"{out_catalog} exists -- pass --force to overwrite "
            "(the preview cache in audio.sqlite is unaffected either way)"
        )
    out_catalog.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("loading collections from chroma ...")
    w_ids, w_embs, w_metas = load_all(str(db_dir), COLLECTION_WHITENED)
    r_ids, r_embs, r_metas = load_all(str(db_dir), COLLECTION_RAW)
    if w_ids != r_ids:
        raise SystemExit("raw/whitened id lists differ -- rebuild the DB")
    n, dim = w_embs.shape
    print(f"  {n} tracks x {dim}-d  ({time.time() - t0:.1f}s)")

    W = _matrix_by_idx(w_metas, w_embs, "whitened")
    R = _matrix_by_idx(r_metas, r_embs, "raw")

    # Row-normalise so cosine similarity is a plain dot product. chroma is configured
    # with hnsw:space=cosine, so this matches what the eval scripts compute.
    def unit_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
        return matrix / norms

    W, R = unit_rows(W), unit_rows(R)

    track_nums = [int(m.get("track_num", -1)) for m in w_metas]
    print("loading durations from the MTG TSVs ...")
    durations = _load_durations(track_nums)
    print(f"  {len(durations)}/{n} tracks have a duration")

    print("writing catalog.sqlite ...")
    if out_catalog.exists():
        out_catalog.unlink()
    con = sqlite3.connect(out_catalog)
    con.executescript(SCHEMA)

    rows, tags, insts, moods = [], [], [], []
    for m in w_metas:
        i = int(m["idx"])
        num = int(m.get("track_num", -1))
        rows.append(
            (
                i,
                m.get("track_id", ""),
                num,
                m.get("title", ""),
                m.get("artist", ""),
                m.get("album", ""),
                m.get("genre", ""),
                int(m.get("n_genres", 0) or 0),
                int(m["released"]) if m.get("released") else None,
                durations.get(num),
            )
        )
        tags += [(i, t) for t in _split(m.get("genres"))]
        insts += [(i, t) for t in _split(m.get("instrument"))]
        moods += [(i, t) for t in _split(m.get("mood_theme"))]

    con.executemany("INSERT INTO tracks VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    con.executemany("INSERT INTO track_tags VALUES (?,?)", tags)
    con.executemany("INSERT INTO track_instruments VALUES (?,?)", insts)
    con.executemany("INSERT INTO track_moods VALUES (?,?)", moods)
    # External-content FTS5 is populated from its content table by 'rebuild'.
    con.execute("INSERT INTO tracks_fts(tracks_fts) VALUES('rebuild')")
    con.executemany(
        "INSERT INTO meta VALUES (?,?)",
        [
            ("built_at", datetime.now(UTC).isoformat(timespec="seconds")),
            ("n_tracks", str(n)),
            ("dim", str(dim)),
            ("source_db", str(db_dir)),
            ("collections", f"{COLLECTION_WHITENED},{COLLECTION_RAW}"),
            ("duration_coverage", str(len(durations))),
        ],
    )
    con.commit()
    con.execute("PRAGMA optimize")
    con.close()

    print("writing vectors.npz ...")
    np.savez(out_vectors, whitened=W, raw=R, idx=np.arange(n, dtype=np.int64))

    # ---- verify the artifacts, not the exit code -------------------------------
    print("\nverifying artifacts ...")
    problems = []
    con = sqlite3.connect(f"file:{out_catalog}?mode=ro", uri=True)
    got = con.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    if got != n:
        problems.append(f"tracks table has {got} rows, expected {n}")
    for table, label in (
        ("track_tags", "tags"),
        ("track_instruments", "instruments"),
        ("track_moods", "moods"),
    ):
        cnt = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {label:12s} {cnt:>7d}")
        if cnt == 0:
            problems.append(f"{table} is empty")
    sample = con.execute(
        "SELECT t.artist FROM tracks_fts f JOIN tracks t ON t.idx = f.rowid "
        "WHERE tracks_fts MATCH ? LIMIT 1",
        ('"bear"',),
    ).fetchone()
    print(f"  fts 'bear'   -> {sample[0] if sample else 'NO MATCH'}")
    if not sample:
        problems.append("FTS5 search for 'bear' returned nothing")
    d214 = con.execute("SELECT duration FROM tracks WHERE idx = 0").fetchone()[0]
    have_dur = con.execute("SELECT COUNT(duration) FROM tracks").fetchone()[0]
    print(f"  durations    {have_dur}/{n}   (first track: {d214})")
    con.close()

    with np.load(out_vectors) as z:
        for key in ("whitened", "raw"):
            a = z[key]
            print(f"  {key:9s} {a.shape} {a.dtype}")
            if a.shape != (n, dim):
                problems.append(f"{key} has shape {a.shape}, expected {(n, dim)}")
            if not np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-4):
                problems.append(f"{key} rows are not unit-norm")
    print(f"  catalog.sqlite {out_catalog.stat().st_size / 1e6:.1f} MB")
    print(f"  vectors.npz    {out_vectors.stat().st_size / 1e6:.1f} MB")

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"\nOK: exported {n} tracks in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
