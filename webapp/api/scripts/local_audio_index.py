#!/usr/bin/env python3
"""Index the local MTG audio so it can be streamed without unpacking the archives.

    nix develop .#web --command python scripts/local_audio_index.py

Reads every ``raw_30s_audio-low-*.tar`` in the audio directory (see
``fetch_mtg_audio.py`` for how they get there) and records, per track, **which archive
holds it and at what byte offset**. The API then serves a track by seeking into that
archive, so no copy of the audio is ever unpacked -- the tars are the only disk cost.

Member names are ``<num % 100:02d>/<num>.low.mp3``, so the join key is ``track_num`` --
*not* the ``idx`` position, which is an artefact of whichever run built the DB.

Run it after fetching more buckets: it is idempotent, and it also **drops local rows
whose archive is no longer on disk**, so "playable" in the UI never claims audio that
cannot actually be streamed.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> webapp/api

from app import audio_store  # noqa: E402
from app.paths import catalog_path, mtg_audio_dir  # noqa: E402

BLOCK = 512
SUFFIX = ".low.mp3"
TAR_GLOB = "raw_30s_audio-low-*.tar"


def iter_members(path: Path):
    """Yield ``(name, data_offset, size)`` for every regular file in a tar.

    Handles GNU long names (typeflag ``L``), where the real filename lives in the
    preceding block's data. Tar pads every member to 512 bytes, so the next header sits
    at ``data_offset + ceil(size / 512) * 512``.
    """
    pending_name: str | None = None
    with open(path, "rb") as handle:
        offset = 0
        while True:
            handle.seek(offset)
            header = handle.read(BLOCK)
            if len(header) < BLOCK or header == b"\0" * BLOCK:
                return  # truncated or end-of-archive marker
            raw_name = header[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
            raw_size = header[124:136].split(b"\0", 1)[0].strip()
            typeflag = header[156:157]
            try:
                size = int(raw_size, 8) if raw_size else 0
            except ValueError:
                size = 0
            data_offset = offset + BLOCK

            if typeflag == b"L":
                handle.seek(data_offset)
                pending_name = (
                    handle.read(size).split(b"\0", 1)[0].decode("utf-8", "replace")
                )
            elif typeflag in (b"0", b"\0", b"") and size:
                yield (pending_name or raw_name), data_offset, size
                pending_name = None

            offset = data_offset + ((size + BLOCK - 1) // BLOCK) * BLOCK


def looks_like_mp3(path: Path, offset: int) -> bool:
    """True if the bytes at ``offset`` start an MPEG audio frame or an ID3 tag."""
    with open(path, "rb") as handle:
        handle.seek(offset)
        head = handle.read(3)
    if head[:3] == b"ID3":
        return True
    return len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, default=None)
    args = ap.parse_args()
    audio_dir = args.audio_dir or mtg_audio_dir()

    if not catalog_path().exists():
        raise SystemExit(f"{catalog_path()} missing -- run export_index.py first")
    tars = sorted(audio_dir.glob(TAR_GLOB))
    if not tars:
        raise SystemExit(
            f"no {TAR_GLOB} in {audio_dir}\n"
            "Fetch them with scripts/fetch_mtg_audio.py (see webapp/README.md). Local "
            "audio is optional: without it only preview-backed tracks are playable."
        )

    cat = sqlite3.connect(f"file:{catalog_path()}?mode=ro", uri=True)
    idx_by_num = {
        num: idx for num, idx in cat.execute("SELECT track_num, idx FROM tracks")
    }
    cat.close()

    con = audio_store.connect()
    audio_store.ensure_schema(con)
    print(f"audio dir: {audio_dir}")
    print(f"archives : {len(tars)}\n")

    total = 0
    present_tars = {tar.name for tar in tars}
    for tar in tars:
        matched, unmatched, samples = 0, [], []
        for name, offset, size in iter_members(tar):
            base = Path(name).name
            if not base.endswith(SUFFIX):
                unmatched.append(name)
                continue
            try:
                num = int(base[: -len(SUFFIX)])
            except ValueError:
                unmatched.append(name)
                continue
            idx = idx_by_num.get(num)
            if idx is None:
                unmatched.append(name)
                continue
            audio_store.put(
                con,
                idx,
                source="local",
                tar=tar.name,
                path=name,
                offset=offset,
                size=size,
            )
            matched += 1
            if len(samples) < 3:
                samples.append((name, offset))
        con.commit()
        total += matched

        # Verify the offsets, not just the row count: a wrong offset would not raise, it
        # would stream plausible bytes from the middle of another file.
        bad = [n for n, o in samples if not looks_like_mp3(tar, o)]
        note = "" if not bad else f"  !! {len(bad)} bad offset(s): {bad[:2]}"
        print(
            f"  {tar.name}: {matched} tracks indexed  "
            f"({tar.stat().st_size / 1e9:.2f} GB){note}",
            flush=True,
        )
        if bad:
            print(f"\nFAILED: {tar.name} offsets do not point at mp3 data")
            return 1

    # A track that used to be preview-backed and is now on disk keeps its old preview
    # columns, because `put()` only writes the fields it is given. Exact audio wins, so
    # clear the leftovers rather than leave a row claiming both sources: if the archive
    # is ever deleted the row goes with it and the track falls back to on-demand
    # resolution, which is the behaviour we want anyway.
    cleared = con.execute(
        "UPDATE audio SET url = NULL, provider = NULL, matched_artist = NULL, "
        "matched_title = NULL, artist_match = NULL "
        "WHERE source = 'local' AND url IS NOT NULL"
    ).rowcount

    # A row can outlive its archive (deleted to reclaim disk), and `serve()` refuses to
    # stream from a missing tar -- so drop those rows rather than let the UI promise
    # audio that 404s.
    stale = [
        row[0]
        for row in con.execute(
            "SELECT idx FROM audio WHERE source = 'local' AND (tar IS NULL "
            f"OR tar NOT IN ({','.join('?' * len(present_tars))}))",
            sorted(present_tars),
        )
    ]
    for idx in stale:
        con.execute("DELETE FROM audio WHERE idx = ?", (idx,))
    con.commit()

    counts = dict(con.execute("SELECT source, COUNT(*) FROM audio GROUP BY source"))
    con.close()

    print(f"\n{total} local tracks indexed across {len(tars)} archives")
    if cleared:
        print(f"cleared stale preview fields on {cleared} now-local tracks")
    if stale:
        print(f"dropped {len(stale)} rows whose archive is gone")
    print(f"availability now: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
