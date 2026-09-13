#!/usr/bin/env python3
"""Offline tests for the preview resolver's three-way outcome.

    nix develop .#web --command python scripts/check_previews.py

No network: `urlopen` is stubbed. This is the regression test for a bug that cost a full
warm run -- Deezer reports its quota limit as **HTTP 200** with
``{"error": {"message": "Quota limit exceeded"}}``, and a resolver reading only `data`
cannot tell that from "this track has no preview". The difference matters because a miss
is cached permanently: the polluted run recorded 11,774 such rows, showed an 8% hit rate
where a random sample of the same corpus gives ~28%, and the wrongly-recorded tracks did
not resolve even after the quota recovered.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> webapp/api

from app import previews  # noqa: E402

problems: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    print(f"  {status} {label}{f' -- {detail}' if detail else ''}")
    if not condition:
        problems.append(label)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def stub(payload, status: int = 200):
    """Make urlopen return `payload` as JSON, for one call."""

    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps(payload).encode())

    return mock.patch("urllib.request.urlopen", fake_urlopen)


def main() -> int:
    print("deezer payloads")
    with stub(
        {"error": {"type": "Exception", "message": "Quota limit exceeded", "code": 4}}
    ):
        hit, error = previews.search_deezer("Anitek", "anything")
    check(
        "quota error is reported, not swallowed",
        hit is None and error is not None,
        str(error),
    )
    check(
        "quota message is surfaced", error is not None and "Quota" in error, str(error)
    )

    with stub({"data": [], "total": 0}):
        hit, error = previews.search_deezer("Nobody", "Nothing")
    check("an empty result is a genuine miss", hit is None and error is None)

    # The regression that put 2,741 wrong songs in the cache: a candidate matching on
    # title alone was accepted. It must now be rejected and reported as a miss.
    wrong_artist = {
        "preview": "https://example.invalid/w.mp3",
        "title": "Monster",
        "artist": {"name": "Beth Crowley"},
    }
    with stub({"data": [wrong_artist]}):
        hit, error = previews.search_deezer("Both", "Monster")
    check(
        "a title-only candidate is rejected, not cached as a hit",
        hit is None and error is None,
        f"got hit={bool(hit)}",
    )
    with stub({"data": [wrong_artist]}):
        hit, error = previews.search_deezer("Beth Crowley", "Monster")
    check("the same candidate is accepted when the artist matches", bool(hit))

    # Accents must not defeat the comparison: 111 cached rows were false negatives.
    accented = {
        "preview": "https://example.invalid/a.mp3",
        "title": "otra forma de ahogo",
        "artist": {"name": "Mi Rara Colección"},
    }
    with stub({"data": [accented]}):
        hit, error = previews.search_deezer("Mi Rara Coleccion", "otra forma")
    check("an accent-only artist difference still matches", bool(hit))

    track = {
        "preview": "https://example.invalid/p.mp3",
        "title": "Sry",
        "artist": {"name": "Podington Bear"},
    }
    with stub({"data": [track], "total": 1}):
        hit, error = previews.search_deezer("Podington Bear", "Sry")
    check(
        "a hit carries url, artist and title",
        bool(hit)
        and hit["url"] == track["preview"]
        and hit["matched_artist"] == "Podington Bear",
    )
    check("artist match is recorded", bool(hit) and hit["artist_match"] is True)

    print("\nresolve() outcomes")
    with (
        stub({"error": {"message": "Quota limit exceeded", "code": 4}}),
        stub({"error": {"message": "Quota limit exceeded", "code": 4}}),
    ):
        result = previews.resolve("Anitek", "")
    check(
        "provider='error' when every provider refuses",
        result["provider"] == "error",
        result["provider"],
    )
    check("errors are listed for the log", bool(result.get("errors")))

    with stub({"data": []}), stub({"results": []}):
        result = previews.resolve("Nobody", "Nothing")
    check(
        "provider='none' when providers answer but have nothing",
        result["provider"] == "none",
        result["provider"],
    )

    with (
        mock.patch.object(previews, "search_deezer", return_value=(None, "http 429")),
        mock.patch.object(previews, "search_itunes", return_value=(None, None)),
    ):
        result = previews.resolve("Somebody", "Something")
    check(
        "a refusal on one provider is NOT a miss, even if the other answered cleanly",
        result["provider"] == "error",
        result["provider"],
    )

    with (
        mock.patch.object(previews, "search_deezer", return_value=(None, None)),
        mock.patch.object(previews, "search_itunes", return_value=(None, None)),
    ):
        result = previews.resolve("Somebody", "Something")
    check(
        "'none' requires every provider to have answered", result["provider"] == "none"
    )

    check(
        "empty query is a miss, not an error",
        previews.resolve("", "")["provider"] == "none",
    )

    print("\nrate limiter")

    class Clock:
        """A virtual monotonic clock, so pacing is asserted rather than timed."""

        def __init__(self) -> None:
            self.now = 1000.0

        def __call__(self) -> float:
            return self.now

        def advance(self, seconds: float) -> None:
            self.now += seconds

    def sleeps_for(limiter, calls: int) -> list[float]:
        clock = Clock()
        with (
            mock.patch("time.monotonic", clock),
            mock.patch("time.sleep", side_effect=clock.advance) as slept,
        ):
            for _ in range(calls):
                limiter.wait()
        return [call.args[0] for call in slept.call_args_list]

    # The first call is free; each subsequent one waits out the interval.
    check(
        "a 1/s limiter paces one interval per extra call",
        sleeps_for(previews._RateLimiter(1.0), 3) == [1.0, 1.0],
    )
    check(
        "a 4/s limiter paces at 0.25s",
        sleeps_for(previews._RateLimiter(4.0), 3) == [0.25, 0.25],
    )
    check("a single call never waits", sleeps_for(previews._RateLimiter(1.0), 1) == [])
    check(
        "a rate of 0 disables throttling", sleeps_for(previews._RateLimiter(0), 3) == []
    )

    limiter = previews._RateLimiter(1.0)
    clock = Clock()
    with (
        mock.patch("time.monotonic", clock),
        mock.patch("time.sleep", side_effect=clock.advance),
    ):
        limiter.wait()
        limiter.backoff(30.0)
        clock_before = clock.now
        sleeps = []
        with mock.patch("time.sleep", side_effect=lambda s: sleeps.append(s)):
            limiter.wait()
    check(
        "backoff pushes the whole pool back, not just one thread",
        len(sleeps) == 1 and sleeps[0] > 25,
        f"slept {sleeps} after backoff at {clock_before}",
    )

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nOK: resolver outcomes behave")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
