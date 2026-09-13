"""Resolve ``artist + title`` to a playable 30 s preview URL.

Ported from ``music_encoding/compare_songs.py`` (``resolve_deezer``,
``resolve_itunes``).
That module solves the same problem, but it imports torch at module level because it
also *embeds* the preview -- importing it here would pull the training stack into the
API process to make an HTTP call. The resolver is a few dozen lines of urllib, so it is
reimplemented; the provider order and endpoints match the original.

Two things worth being explicit about, because they shape what the UI may claim:

* **A preview is not the corpus track.** It is a different recording of the same song
  (usually the hook), found by fuzzy text match. ``CASE_STUDY.md`` makes the same point
  for the case-study flow. The UI labels the source and shows what was matched.
* **Only URLs are stored, never audio.** Both providers' terms restrict caching the
  audio, and ``compare_songs`` documents the same constraint. The API streams through.

Coverage is genuinely partial (~28% of the corpus on a 40-track sample, measured): MTG-
Jamendo is royalty-free and mostly absent from these services. That is a property of the
corpus, not something to fix here.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "music-encoding-webapp/1.0"
TIMEOUT = 10

# Stripped before comparing artist names, so "Podington Bear" matches "podington bear"
# and decorative suffixes do not defeat the comparison.
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def _norm(text: str | None) -> str:
    """Case-, punctuation- and **accent**-insensitive form used for artist comparison.

    Folding accents matters: without it "Mi Rara Colección" does not match "Mi Rara
    Coleccion", and 111 cached hits were scored as artist mismatches for that reason
    alone rather than because the act differed.
    """
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return _NON_WORD.sub(" ", stripped.lower()).strip()


class _RateLimiter:
    """One shared request budget across all threads.

    Deezer documents roughly 50 requests per 5 s per IP, and a short burst test happily
    did 33 req/s -- but a *sustained* run at 10 workers blew through the quota within a
    few thousand tracks. Since every thread shares this limiter, `--workers` controls
    latency-hiding while the request rate stays fixed and predictable.
    """

    def __init__(self, per_second: float) -> None:
        self.interval = 1.0 / per_second if per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if sleep_for:
            time.sleep(sleep_for)

    def backoff(self, seconds: float) -> None:
        """Push the whole pool back after a quota refusal, not just one thread."""
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


# 4 req/s: comfortably under the documented limit, ~2 h for the whole corpus with
# iTunes as fallback. Correct beats fast here -- a wrong "no audio" is permanent.
_limiter = _RateLimiter(float(os.environ.get("WEBAPP_PREVIEW_RATE", "4")))


def _get_json(url: str) -> tuple[dict | None, str | None]:
    """Fetch and parse, returning ``(payload, error)``.

    The ``error`` return distinguishes "the provider has no such track" from "the
    provider is refusing to answer", and that difference is load-bearing twice over: a
    miss is cached permanently, and a refusal must be retried later rather than
    recorded.
    """
    _limiter.wait()
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            _limiter.backoff(5.0)
        return None, f"http {exc.code}"
    except (urllib.error.URLError, TimeoutError) as exc:
        return None, f"network {type(exc).__name__}"
    except ValueError:
        return None, "bad json"

    # Deezer reports its quota limit as HTTP 200 with {"error": {"message": "Quota
    # limit exceeded", "code": 4}}. Reading only `data` cannot tell that from a genuine
    # miss, so a throttled warm run silently writes thousands of permanent "no audio"
    # rows for tracks that do have previews -- observed exactly that: an 8% hit rate
    # over 13k tracks where a random sample suggests ~28%, and the recorded misses did
    # not re-resolve.
    if isinstance(payload, dict) and payload.get("error"):
        detail = payload["error"]
        message = detail.get("message") if isinstance(detail, dict) else str(detail)
        # Only a *rate limit* should slow the pool down. Backing off for every error key
        # makes a provider that is merely unhappy about one query stall the whole run --
        # measured at 59 refusals per 100 tracks and a 14-hour ETA.
        if any(word in str(message).lower() for word in ("quota", "limit", "rate")):
            _limiter.backoff(5.0)
        return None, f"provider error: {message}"
    return payload, None


def _match_quality(
    artist: str, title: str, cand_artist: str, cand_title: str
) -> tuple[int, int]:
    """Rank candidates: (artist matches, title tokens in common)."""
    a_hit = 1 if _norm(artist) and _norm(artist) == _norm(cand_artist) else 0
    want = set(_norm(title).split())
    got = set(_norm(cand_title).split())
    return a_hit, len(want & got)


def search_deezer(
    artist: str, title: str, limit: int = 5
) -> tuple[dict | None, str | None]:
    """Deezer search. Candidates are ranked before choosing, not just taken first.

    Taking ``limit=1`` (as the case-study script does) yields a result whose artist
    often has nothing to do with the query; fetching a handful and preferring an
    artist-name match raised the trustworthy-hit rate noticeably in testing.

    Returns ``(hit_or_None, error_or_None)``. The distinction matters: "no such track"
    is a permanent answer worth caching, while a timeout is not.
    """
    query = urllib.parse.urlencode({"q": f"{artist} {title}".strip(), "limit": limit})
    data, error = _get_json(f"https://api.deezer.com/search?{query}")
    if error:
        return None, error
    candidates = (data or {}).get("data") or []
    best, best_rank = None, None
    for cand in candidates:
        if not cand.get("preview"):
            continue
        rank = _match_quality(
            artist, title, (cand.get("artist") or {}).get("name"), cand.get("title")
        )
        # The artist MUST match. Accepting the best title-only candidate is how 2,741
        # cached rows came to point at a different act's song -- corpus artist "Both"
        # matched to "Beth Crowley", "Alexander Blu" to "Monty Alexander". A wrong song
        # is worse than no song, because the badge tells the user it is the track.
        if not rank[0]:
            continue
        if best_rank is None or rank > best_rank:
            best, best_rank = cand, rank
    if not best:
        return None, None
    return {
        "provider": "deezer",
        "url": best["preview"],
        "matched_artist": (best.get("artist") or {}).get("name") or "",
        "matched_title": best.get("title") or "",
        "artist_match": True,
    }, None


def search_itunes(
    artist: str, title: str, limit: int = 5
) -> tuple[dict | None, str | None]:
    query = urllib.parse.urlencode(
        {"term": f"{artist} {title}".strip(), "entity": "song", "limit": limit}
    )
    data, error = _get_json(f"https://itunes.apple.com/search?{query}")
    if error:
        return None, error
    candidates = (data or {}).get("results") or []
    best, best_rank = None, None
    for cand in candidates:
        if not cand.get("previewUrl"):
            continue
        rank = _match_quality(
            artist, title, cand.get("artistName"), cand.get("trackName")
        )
        if not rank[0]:  # same rule as Deezer: the artist must match
            continue
        if best_rank is None or rank > best_rank:
            best, best_rank = cand, rank
    if not best:
        return None, None
    return {
        "provider": "itunes",
        "url": best["previewUrl"],
        "matched_artist": best.get("artistName") or "",
        "matched_title": best.get("trackName") or "",
        "artist_match": True,
    }, None


def resolve(artist: str, title: str, provider: str = "auto") -> dict:
    """Find a preview.

    Three outcomes, and the difference between the last two decides whether a long warm
    run is resumable:

    * ``provider="deezer"|"itunes"`` -- a hit, with the URL and what it matched.
    * ``provider="none"`` -- both providers answered, neither has the track. Permanent;
      cache it and never ask again (~72% of the corpus).
    * ``provider="error"`` -- every request failed (timeout, 429, bad JSON). **Not** a
      miss: caching this as "none" would silently convert a transient network problem
      into a permanent hole in the corpus, which is exactly the kind of silent failure
      this repo's rules warn about.
    """
    if not artist and not title:
        return {
            "provider": "none",
            "url": None,
            "reason": "track has no artist or title",
        }
    order = {
        "auto": ("deezer", "itunes"),
        "deezer": ("deezer",),
        "itunes": ("itunes",),
    }.get(provider, ("deezer", "itunes"))
    tried: list[str] = []
    errors: list[str] = []
    for name in order:
        tried.append(name)
        search = search_deezer if name == "deezer" else search_itunes
        hit, error = search(artist, title)
        if hit:
            return hit
        if error:
            errors.append(f"{name}: {error}")
    if errors:
        # ANY refusal means we cannot claim the track has no preview: we have not heard
        # from every provider. Requiring *all* of them to refuse would be wrong -- a
        # throttled Deezer plus an honest "nothing" from iTunes would be cached as a
        # permanent miss for a track Deezer may well have.
        return {"provider": "error", "url": None, "tried": tried, "errors": errors}
    return {"provider": "none", "url": None, "tried": tried}


def open_stream(url: str, range_header: str | None = None):
    """Open a preview URL for proxying, forwarding a Range request if given.

    Proxying rather than redirecting keeps the browser on one origin (no CORS, no
    mixed-content downgrade) and lets the API be the only thing that talks to the CDN.
    """
    headers = {"User-Agent": USER_AGENT}
    if range_header:
        headers["Range"] = range_header
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=TIMEOUT)
