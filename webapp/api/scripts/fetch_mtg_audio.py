#!/usr/bin/env python3
"""Fetch the corpus's own audio from MTG-Jamendo.

    nix develop .#web --command python scripts/fetch_mtg_audio.py --dest ~/mtg

Why this instead of the Deezer/iTunes previews the app also supports: this is the
**actual audio the embeddings were built from**. Previews are a different recording
found by fuzzy match — 2,741 of the completed pass's 5,918 "hits" were a different
artist, i.e. title-only coincidences. Exact audio also removes the quotas, the
throttling and the ~20% ceiling.

Layout: MTG ships `raw_30s/audio-low` as 100 tar archives, `raw_30s_audio-low-NN.tar`,
where `NN == track_num % 100`. The melspec corpus spans 59 of those buckets, and the
buckets are a *representative* slice (bucket 00's genre shares match the corpus within
1.4pp), so taking the first N buckets is a fair N/59 of the corpus rather than a biased
one. Archives are **not unpacked** — `local_audio_index.py` records each member's byte
offset and the API streams by range, so the tar is the only disk cost.

Downloads are resumable, sha256-verified against MTG's published list, and stop before
filling the disk.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> webapp/api

from app.paths import catalog_path  # noqa: E402

BASE = (
    "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/raw_30s_audio-low-%02d.tar"
)
SHA_LIST = Path.home() / "mtg/data/download/raw_30s_audio-low_sha256_tars.txt"


def buckets_needed() -> list[int]:
    """Which archives our corpus actually needs: one per distinct track_num % 100."""
    con = sqlite3.connect(f"file:{catalog_path()}?mode=ro", uri=True)
    try:
        return sorted(
            {num % 100 for (num,) in con.execute("SELECT track_num FROM tracks")}
        )
    finally:
        con.close()


def expected_hashes() -> dict[str, str]:
    if not SHA_LIST.exists():
        return {}
    out = {}
    for line in SHA_LIST.read_text().splitlines():
        digest, _, name = line.partition(" ")
        if name:
            out[name.strip()] = digest.strip()
    return out


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def fetch(
    bucket: int, dest: Path, expect: str | None, tries: int = 3
) -> tuple[str, str]:
    """Download one archive, verifying the checksum. Returns (status, detail)."""
    name = f"raw_30s_audio-low-{bucket:02d}.tar"
    target = dest / name
    url = BASE % bucket

    if target.exists():
        if expect and sha256(target) == expect:
            return "present", f"{target.stat().st_size / 1e9:.2f} GB, checksum ok"
        target.unlink()  # incomplete or corrupt: start over rather than trust it

    for attempt in range(1, tries + 1):
        partial = dest / f"{name}.part"
        # -C - resumes a partial file; MTG's mirror advertises byte ranges.
        result = subprocess.run(
            ["curl", "-sS", "-L", "--fail", "-C", "-", "-o", str(partial), url],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()[-1:] or ["curl failed"]
            if attempt == tries:
                return "failed", detail[0][:120]
            time.sleep(5 * attempt)
            continue

        if expect:
            got = sha256(partial)
            if got != expect:
                partial.unlink(missing_ok=True)
                if attempt == tries:
                    return "checksum", f"expected {expect[:12]}…, got {got[:12]}…"
                time.sleep(5 * attempt)
                continue
        partial.rename(target)
        return "fetched", f"{target.stat().st_size / 1e9:.2f} GB, checksum ok"
    return "failed", "exhausted retries"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path.home() / "mtg")
    ap.add_argument(
        "--floor-gb",
        type=float,
        default=8.0,
        help="stop before free space drops below this (the box has no swap and a full "
        "root is unpleasant to recover)",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = as many as fit")
    ap.add_argument("--only", type=int, default=None, help="fetch a single bucket")
    args = ap.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    hashes = expected_hashes()
    wanted = [args.only] if args.only is not None else buckets_needed()

    # Only the old gdown-style temp names are junk. `*.part` files are *deliberately*
    # kept: `fetch()` downloads to `<name>.part` and curl's `-C -` resumes it, so
    # deleting them here would throw away a partially-downloaded archive and make a
    # killed run start that bucket from zero.
    for stale in sorted(args.dest.glob("*.tarc*")):
        print(
            f"removing stale temp: {stale.name} ({stale.stat().st_size / 1e6:.0f} MB)"
        )
        stale.unlink()
    for part in sorted(args.dest.glob("*.part")):
        already = part.stat().st_size / 1e9
        print(f"resuming partial: {part.name} ({already:.2f} GB already)")

    free = shutil.disk_usage(args.dest).free
    print(f"destination : {args.dest}")
    print(f"free space  : {free / 1e9:.1f} GB (floor {args.floor_gb:.1f} GB)")
    print(f"buckets     : {len(wanted)} needed\n")

    fetched = failed = skipped = 0
    bytes_added = 0
    for bucket in wanted:
        free = shutil.disk_usage(args.dest).free
        if free < args.floor_gb * 1e9:
            print(
                f"\nSTOPPING: {free / 1e9:.1f} GB free is below the "
                f"{args.floor_gb} GB floor"
            )
            break
        if args.limit and fetched >= args.limit:
            print(f"\nstopping after --limit {args.limit}")
            break

        name = f"raw_30s_audio-low-{bucket:02d}.tar"
        before = free
        status, detail = fetch(bucket, args.dest, hashes.get(name))
        if status == "fetched":
            fetched += 1
            bytes_added += before - shutil.disk_usage(args.dest).free
        elif status == "present":
            skipped += 1
            print(f"  {name}: already present — {detail}", flush=True)
            continue
        else:
            failed += 1
        print(
            f"  {name}: {status} — {detail}   "
            f"[{shutil.disk_usage(args.dest).free / 1e9:.1f} GB free]",
            flush=True,
        )

    print(
        f"\ndone: {fetched} fetched, {skipped} already present, {failed} failed "
        f"({bytes_added / 1e9:.1f} GB added)"
    )
    print(f"free now: {shutil.disk_usage(args.dest).free / 1e9:.1f} GB")
    if bytes_added > 0:
        print("\nnext: index them so the API can serve them --")
        print("  nix develop .#web --command python scripts/local_audio_index.py")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
