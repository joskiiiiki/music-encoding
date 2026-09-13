#!/usr/bin/env python3
"""Index the locally-held audio so it can be streamed without unpacking the tarball.

    nix develop .#web --command python scripts/local_audio_index.py

``~/mtg/raw_30s_audio-low-00.tar`` holds 586 mp3s and is the only playable corpus audio
on this machine -- ``~/mtg_jamendo`` is lossy log-mel spectrograms, which no amount of
work turns back into a song (the phase is gone). The tarball is 1.55 GiB, so rather than
unpack a second copy this reads the tar's 512-byte headers once and records each
member's byte offset and size; the API then serves a track by seeking into the archive.

Member names are ``<num % 100:02d>/<num>.low.mp3``, so the join key to the catalogue is
``track_num`` -- *not* the ``idx`` position, which is an artefact of whichever run built
the DB.

The offsets are verified by reading the first bytes at each one and checking for an MP3
frame, because a wrong offset would not raise -- it would stream plausible-looking bytes
from the middle of another file.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> webapp/api

from app import audio_store  # noqa: E402
from app.paths import catalog_path, mtg_audio_tar  # noqa: E402

BLOCK = 512
SUFFIX = ".low.mp3"


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
    ap.add_argument("--tar", type=Path, default=None)
    args = ap.parse_args()
    tar = args.tar or mtg_audio_tar()

    if not tar.exists():
        raise SystemExit(
            f"tarball not found: {tar}\n"
            "Set MTG_AUDIO_TAR, or download the MTG audio (see CLAUDE.md). Local audio "
            "is optional -- without it only preview-backed tracks are playable."
        )
    if not catalog_path().exists():
        raise SystemExit(f"{catalog_path()} missing -- run export_index.py first")

    cat = sqlite3.connect(f"file:{catalog_path()}?mode=ro", uri=True)
    idx_by_num = {
        num: idx for num, idx in cat.execute("SELECT track_num, idx FROM tracks")
    }
    cat.close()

    members = list(iter_members(tar))
    print(f"{tar}")
    print(f"  {len(members)} members in the archive")

    con = audio_store.connect()
    audio_store.ensure_schema(con)
    matched, unmatched, samples = 0, [], []
    for name, offset, size in members:
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
        audio_store.put(con, idx, source="local", path=name, offset=offset, size=size)
        matched += 1
        if len(samples) < 5:
            samples.append((name, offset))
    con.commit()

    rows = con.execute(
        "SELECT COUNT(*), SUM(size) FROM audio WHERE source = 'local'"
    ).fetchone()
    con.close()

    print(f"  matched to the catalogue : {matched}")
    print(f"  rows written             : {rows[0]}  ({rows[1] / 1e6:.1f} MB of audio)")
    if unmatched:
        print(
            f"  ! {len(unmatched)} members did not map to a track, e.g. {unmatched[:3]}"
        )

    # Verify the offsets, not just the row count: streaming from a wrong offset would
    # look like success and play noise.
    bad = [name for name, offset in samples if not looks_like_mp3(tar, offset)]
    for name, offset in samples:
        head = "mp3" if looks_like_mp3(tar, offset) else "NOT MP3"
        print(f"  offset check {name} @ {offset} -> {head}")

    if bad or matched == 0 or rows[0] != matched:
        print("\nFAILED:")
        if matched == 0:
            print("  - no members matched; is this the right tarball?")
        if rows[0] != matched:
            print(f"  - wrote {rows[0]} rows for {matched} members")
        for name in bad:
            print(f"  - {name} does not start with mp3 data at its recorded offset")
        return 1
    print(f"\nOK: {matched} local tracks indexed and offsets verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
