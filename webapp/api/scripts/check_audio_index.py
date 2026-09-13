#!/usr/bin/env python3
"""Verify the local audio index against the archives on disk.

    nix develop .#web --command python scripts/check_audio_index.py

Worth running after any fetch or re-index, because a wrong offset does not raise: it
streams plausible bytes from the middle of another file. That happened once already --
an offset belonging to archive 01 was served out of archive 00, when the API still had
the old single-tar path -- so this checks the artifact rather than trusting the indexer.

Two independent checks:

1. every local row: the bytes at its recorded offset must start an MP3 (ID3 tag or frame
   sync), and the member must lie inside its archive;
2. a spread of archives re-walked header-by-header and compared row-for-row against the
   database, which catches a desynchronised walk that happened to land on plausible
   offsets.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> webapp/api

from app.paths import audio_db_path, mtg_audio_dir  # noqa: E402
from scripts.local_audio_index import iter_members  # noqa: E402


def starts_mp3(head: bytes) -> bool:
    if head[:3] == b"ID3":
        return True
    return len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, default=None)
    ap.add_argument("--rewalk", type=int, default=3, help="archives to re-walk fully")
    args = ap.parse_args()
    audio_dir = args.audio_dir or mtg_audio_dir()

    con = sqlite3.connect(f"file:{audio_db_path()}?mode=ro", uri=True)
    rows = list(
        con.execute(
            "SELECT idx, tar, path, offset, size FROM audio WHERE source='local'"
        )
    )
    if not rows:
        print("no local audio rows -- run fetch_mtg_audio.py then local_audio_index.py")
        return 0
    print(f"local rows to verify: {len(rows):,}")

    sizes = {tar.name: tar.stat().st_size for tar in audio_dir.glob("*.tar")}
    handles: dict[str, object] = {}
    bad_header: list = []
    bad_bounds: list = []
    for idx, tar, path, offset, size in rows:
        if tar not in sizes:
            bad_bounds.append((idx, tar, "archive missing"))
            continue
        if offset + size > sizes[tar]:
            bad_bounds.append((idx, path, "member runs past end of archive"))
            continue
        handle = handles.get(tar)
        if handle is None:
            handle = handles[tar] = open(audio_dir / tar, "rb")
        handle.seek(offset)
        if not starts_mp3(handle.read(3)):
            bad_header.append((idx, tar, path))
    for handle in handles.values():
        handle.close()

    print(f"  offsets not starting an MP3 frame : {len(bad_header)}")
    print(f"  members outside their archive     : {len(bad_bounds)}")
    for problem in (bad_header + bad_bounds)[:5]:
        print("   ", problem)

    by_key = {(tar, path): (offset, size) for _, tar, path, offset, size in rows}
    mismatch = checked = 0
    spread = sorted(sizes)
    sample = [
        spread[i] for i in range(0, len(spread), max(1, len(spread) // args.rewalk))
    ]
    for name in sample[: args.rewalk]:
        members = 0
        for member, offset, size in iter_members(audio_dir / name):
            key = (name, member)
            if not Path(member).name.endswith(".low.mp3") or key not in by_key:
                continue
            members += 1
            checked += 1
            if by_key[key] != (offset, size):
                mismatch += 1
                if mismatch <= 3:
                    print(
                        f"   MISMATCH {name} {member}: "
                        f"db={by_key[key]} walk={(offset, size)}"
                    )
        print(f"  {name}: {members} members re-walked and matched")

    ok = not bad_header and not bad_bounds and mismatch == 0
    print(f"\nre-walked {checked:,} members, {mismatch} mismatches")
    print("VERIFIED: every local row streams its own track" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
