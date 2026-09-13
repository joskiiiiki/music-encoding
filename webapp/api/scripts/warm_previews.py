#!/usr/bin/env python3
"""Pre-resolve preview URLs for the corpus, so audio coverage is known upfront.

    nix develop .#web --command python scripts/warm_previews.py --workers 8

MTG-Jamendo is a royalty-free corpus and much of it is absent from Deezer and iTunes: a
40-track random sample suggested ~28%, and the first full pass measures the real figure.
Knowing it ahead of time is the point of this script -- it makes the app's "playable
only" filter complete instead of filling in as you click.

**Deezer only, by default, and that is a measured choice.** Across the first full pass
Deezer produced 1,135 hits and iTunes 14 -- about 1% of coverage for ~72% of the request
volume, because every Deezer miss falls through to a second query. iTunes is also the
tighter quota, so with it in the loop 59 of 100 tracks came back as refusals and the job
ran at **0.6 tracks/s with a 14-hour ETA**; Deezer alone runs **4.0/s with zero
refusals** (~2.2 h for the corpus). Run the tail as a separate optional pass if you want
``--provider itunes --refresh``.

Design notes, all of them about not losing hours of work:

* **Resumable.** Only tracks with no row in ``audio.sqlite`` are attempted, and a
  *miss is recorded as a miss* (``source='none'``), so a re-run skips it. Without that,
  every re-run would re-query the ~72% nothing has, which is most of the wall clock.
* **Errors are not misses.** A timeout or a 429 writes nothing, so the next run retries
  it. Caching a transient failure as a permanent hole is the failure mode this avoids.
* **Parallel with backpressure.** Workers do network I/O only; the main thread does
  every write, so SQLite never sees concurrent writers. The request rate is set by one
  shared limiter in ``previews.py``, not by the worker count -- a burst test did 33/s
  happily, but a sustained run at 10 workers tripped Deezer's per-IP quota within a few
  thousand tracks.
* **Only URLs are stored**, never audio -- both providers' terms restrict caching the
  audio itself, and ``compare_songs.py`` documents the same constraint.

Run it in the background; the app works fine while it runs and coverage grows under it.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> webapp/api

from app import audio_store, previews  # noqa: E402
from app.paths import catalog_path  # noqa: E402


def fetch(row: tuple[int, str, str], provider: str, retries: int):
    """Resolve one track, retrying only on a refusal.

    A refusal (quota, timeout, 429) is never written to the cache, so an exhausted retry
    costs time rather than correctness -- the next run picks it up.
    """
    idx, artist, title = row
    result = previews.resolve(artist, title, provider)
    for _ in range(retries):
        if result.get("provider") != "error":
            break
        result = previews.resolve(artist, title, provider)
    return idx, result


def main() -> int:
    ap = argparse.ArgumentParser()
    # Workers hide latency; the request RATE is set by one shared limiter in previews.py
    # (WEBAPP_PREVIEW_RATE, default 4/s), because the providers' quota is per-IP and a
    # wide pool just trips it sooner.
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="0 = everything pending")
    ap.add_argument(
        "--chunk", type=int, default=200, help="submissions in flight per batch"
    )
    ap.add_argument(
        "--provider",
        default="deezer",
        choices=["auto", "deezer", "itunes"],
        help="deezer (default, ~99%% of hits, fastest) | auto (adds iTunes, 6x slower)",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=1,
        help="extra attempts per refusal; refusals are not cached, so a rerun retries",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="also re-attempt recorded misses (never touches local audio)",
    )
    args = ap.parse_args()

    if not catalog_path().exists():
        raise SystemExit(f"{catalog_path()} missing -- run export_index.py first")

    cat = sqlite3.connect(f"file:{catalog_path()}?mode=ro", uri=True)
    tracks = list(cat.execute("SELECT idx, artist, title FROM tracks ORDER BY idx"))
    cat.close()

    con = audio_store.connect()
    audio_store.ensure_schema(con)
    if args.refresh:
        attempted = {
            row[0]
            for row in con.execute("SELECT idx FROM audio WHERE source != 'local'")
        }
    else:
        attempted = {row[0] for row in con.execute("SELECT idx FROM audio")}

    everything = [
        (idx, artist or "", title or "")
        for idx, artist, title in tracks
        if idx not in attempted
    ]
    total = len(tracks)
    pending = everything[: args.limit] if args.limit else everything

    print(f"corpus        : {total} tracks")
    print(f"already known : {total - len(everything)}")
    print(
        f"to resolve    : {len(pending)}"
        f"{f' (of {len(everything)} pending, --limit)' if args.limit else ''}"
        f"  (provider={args.provider}, {args.workers} workers)"
    )
    if not pending:
        print("nothing to do")
        con.close()
        return 0

    hits = misses = errors = 0
    started = time.time()
    try:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(pending), args.chunk):
                batch = pending[start : start + args.chunk]
                for idx, result in pool.map(
                    lambda row: fetch(row, args.provider, args.retries), batch
                ):
                    if result.get("url"):
                        audio_store.put(
                            con,
                            idx,
                            source=result["provider"],
                            url=result["url"],
                            provider=result["provider"],
                            matched_artist=result.get("matched_artist"),
                            matched_title=result.get("matched_title"),
                            artist_match=int(bool(result.get("artist_match"))),
                        )
                        hits += 1
                    elif result.get("provider") == "none":
                        audio_store.put(
                            con,
                            idx,
                            source="none",
                            provider=",".join(result.get("tried", [])),
                        )
                        misses += 1
                    else:
                        # Transport error: write nothing so a later run retries.
                        errors += 1
                con.commit()

                done = hits + misses + errors
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0
                remaining = (len(pending) - done) / rate if rate else 0
                print(
                    f"  {done:6d}/{len(pending)}  hits {hits}  misses {misses}  "
                    f"errors {errors}  {rate:5.1f}/s  eta {remaining / 60:5.1f} min",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\ninterrupted -- progress is committed and this run is resumable")

    con.commit()
    stats = audio_store.stats(con, total)
    con.close()

    print("\ncoverage:")
    print(
        f"  local {stats['local']}  deezer {stats['deezer']}  "
        f"itunes {stats['itunes']}  none {stats['attempted_misses']}"
    )
    print(
        f"  playable {stats['playable']}/{total} "
        f"({stats['playable'] / total:.0%})  unresolved {stats['unresolved']}"
    )
    if errors:
        print(f"  {errors} transport errors were not cached -- re-run to retry them")
    outcome = "committed" if hits else "nothing new"
    print(f"\nOK: {outcome} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
