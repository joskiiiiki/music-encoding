"""Where a track's audio comes from, and whether we have looked yet.

One table drives both audio sources:

* ``local``  -- an mp3 sealed inside the MTG tarball; ``path``/``offset``/``size`` let
  the API stream it by byte range without unpacking 1.6 GB.
* ``deezer`` / ``itunes`` -- a 30 s preview URL resolved from artist + title. Only the
  URL is stored; both providers' terms restrict caching the audio itself, and
  ``compare_songs.py`` documents the same constraint.
* ``none``   -- a resolution attempt was made and found nothing.

That last row is not a detail: it is what makes ``warm_previews.py`` resumable. A miss
has to be *recorded as a miss*, otherwise every re-run re-queries the ~72% of the corpus
that is absent from both services, which is most of the wall-clock time.

The distinction between "no row" and ``source='none'`` is also what the UI needs: no row
means "play and we'll find out", ``none`` means "there is nothing to play".
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .paths import audio_db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS audio (
    idx            INTEGER PRIMARY KEY,
    source         TEXT NOT NULL,   -- local | deezer | itunes | none
    path           TEXT,            -- tar member name, for source='local'
    offset         INTEGER,         -- byte offset of the member's data
    size           INTEGER,         -- member size in bytes
    url            TEXT,            -- preview URL, for the network sources
    provider       TEXT,
    matched_artist TEXT,            -- what the provider thought we asked for
    matched_title  TEXT,
    artist_match   INTEGER,         -- 1 when the provider's artist == the MTG artist
    fetched_at     REAL
);
CREATE INDEX IF NOT EXISTS audio_source ON audio (source);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or audio_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # WAL so the API can read while a long warm job writes; the timeout covers the
    # brief moments where a writer holds the lock.
    con = sqlite3.connect(p, timeout=30.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    con.commit()


def get(con: sqlite3.Connection, idx: int) -> dict | None:
    row = con.execute("SELECT * FROM audio WHERE idx = ?", (idx,)).fetchone()
    return dict(row) if row else None


def get_many(con: sqlite3.Connection, idxs: list[int]) -> dict[int, dict]:
    """Availability for a page of results, in one query rather than N."""
    if not idxs:
        return {}
    out: dict[int, dict] = {}
    # SQLite's variable limit makes a chunked query necessary for large pages.
    for start in range(0, len(idxs), 500):
        chunk = idxs[start : start + 500]
        marks = ",".join("?" * len(chunk))
        for row in con.execute(f"SELECT * FROM audio WHERE idx IN ({marks})", chunk):
            out[row["idx"]] = dict(row)
    return out


def put(con: sqlite3.Connection, idx: int, **fields) -> None:
    """Upsert availability. Unspecified columns keep their existing values."""
    fields.setdefault("fetched_at", time.time())
    fields["idx"] = idx
    cols = ", ".join(fields)
    marks = ", ".join("?" * len(fields))
    updates = ", ".join(f"{c} = excluded.{c}" for c in fields if c != "idx")
    con.execute(
        f"INSERT INTO audio ({cols}) VALUES ({marks}) "
        f"ON CONFLICT(idx) DO UPDATE SET {updates}",
        list(fields.values()),
    )


def kind(record: dict | None) -> str:
    """Collapse a row into what the UI shows: local | preview | none | unknown."""
    if not record:
        return "unknown"
    source = record.get("source")
    if source == "local":
        return "local"
    if source in ("deezer", "itunes"):
        return "preview"
    return "none"


def stats(con: sqlite3.Connection, total_tracks: int) -> dict:
    counts = {
        row["source"]: row["n"]
        for row in con.execute(
            "SELECT source, COUNT(*) AS n FROM audio GROUP BY source"
        )
    }
    local = counts.get("local", 0)
    deezer = counts.get("deezer", 0)
    itunes = counts.get("itunes", 0)
    none = counts.get("none", 0)
    attempted = local + deezer + itunes + none
    return {
        "total_tracks": total_tracks,
        "local": local,
        "deezer": deezer,
        "itunes": itunes,
        "playable": local + deezer + itunes,
        "attempted_misses": none,
        "unresolved": max(0, total_tracks - attempted),
        "attempted": attempted,
    }
