"""Serve a track's audio: byte ranges out of the MTG tarball, or a proxied preview.

Two sources, resolved in this order:

1. **local** -- one of 586 mp3s sealed inside ``raw_30s_audio-low-00.tar``. The tarball
   is *not* unpacked: ``local_audio_index.py`` records each member's byte offset and
   size, so a track is served by seeking into the archive. That avoids both a 1.6 GB
   copy and the ~30 s a sequential ``tar -xOf`` scan would cost per request.
2. **preview** -- a Deezer/iTunes 30 s clip, resolved on first play and cached as a URL.

Ranges are supported for both, because seeking is the first thing anyone does with an
audio player, and a 30 s clip that cannot be scrubbed feels broken.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from starlette.responses import StreamingResponse

from . import audio_store, previews
from .paths import mtg_audio_dir

CHUNK = 64 * 1024
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Parse a single-range ``bytes=`` header into inclusive ``(start, end)``.

    Returns ``None`` for an absent or unparseable header (meaning: send the whole
    thing). Multi-range requests are deliberately not supported -- no audio player
    issues them, and the multipart/byteranges response is a lot of machinery for
    nothing.
    """
    if not header:
        return None
    match = _RANGE_RE.fullmatch(header.strip())
    if not match:
        return None
    start_raw, end_raw = match.groups()
    if not start_raw and not end_raw:
        return None
    if not start_raw:  # suffix range: last N bytes
        length = min(int(end_raw), size)
        return size - length, size - 1
    start = int(start_raw)
    end = int(end_raw) if end_raw else size - 1
    end = min(end, size - 1)
    if start > end or start >= size:
        return None
    return start, end


def _chunks(handle, length: int) -> Iterator[bytes]:
    remaining = length
    while remaining > 0:
        chunk = handle.read(min(CHUNK, remaining))
        if not chunk:
            break
        remaining -= len(chunk)
        yield chunk


def _range_headers(start: int, end: int, size: int, partial: bool) -> dict[str, str]:
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        # Local audio is a static file behind an immutable corpus index, so let the
        # browser keep it; a preview URL may rotate and is re-resolved on failure.
        "Cache-Control": "public, max-age=3600",
    }
    if partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return headers


def local_stream(
    record: dict, tar_path, range_header: str | None = None
) -> StreamingResponse:
    """Stream one member out of its archive by byte range, without unpacking it."""
    size = int(record["size"])
    offset = int(record["offset"])
    rng = parse_range(range_header, size)
    start, end = rng if rng else (0, size - 1)
    length = end - start + 1

    def generate() -> Iterator[bytes]:
        with open(tar_path, "rb") as handle:
            handle.seek(offset + start)
            yield from _chunks(handle, length)

    return StreamingResponse(
        generate(),
        status_code=206 if rng else 200,
        media_type="audio/mpeg",
        headers=_range_headers(start, end, size, rng is not None),
    )


def preview_stream(url: str, range_header: str | None = None) -> StreamingResponse:
    """Proxy a provider's preview, forwarding Range so seeking still works.

    Streaming through the API rather than redirecting keeps the browser on a single
    origin -- no CORS, and no downgrade from the app's https-free localhost to the
    CDN's scheme.
    """
    upstream = previews.open_stream(url, range_header)
    status = getattr(upstream, "status", 200)
    headers = {}
    for name in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"):
        value = upstream.headers.get(name)
        if value:
            headers[name] = value
    headers.setdefault("Accept-Ranges", "bytes")

    def generate() -> Iterator[bytes]:
        try:
            while True:
                chunk = upstream.read(CHUNK)
                if not chunk:
                    break
                yield chunk
        finally:
            upstream.close()

    return StreamingResponse(
        generate(),
        status_code=status if status in (200, 206) else 200,
        media_type=headers.pop("Content-Type", "audio/mpeg"),
        headers=headers,
    )


def resolve_and_store(
    idx: int, artist: str, title: str, provider: str = "auto"
) -> dict:
    """Resolve a preview and **record the outcome, including the miss**.

    Recording misses is what makes the pre-warm resumable: without a row, every re-run
    re-queries the ~72% of the corpus that neither provider has.
    """
    result = previews.resolve(artist, title, provider)
    con = audio_store.connect()
    try:
        audio_store.ensure_schema(con)
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
        elif result.get("provider") == "none":
            audio_store.put(
                con,
                idx,
                source="none",
                provider=",".join(result.get("tried", [])) or provider,
            )
        # provider == "error": deliberately write nothing. Recording a transient failure
        # as a permanent miss is how a flaky network becomes a hole in the corpus.
        con.commit()
    finally:
        con.close()
    return result


def serve(idx: int, artist: str, title: str, range_header: str | None = None):
    """Pick a source, resolve on demand if needed, and stream it.

    Returns ``None`` when nothing is playable, so the route can answer 404 with a
    message that matches the UI's "no audio available" state.
    """
    con = audio_store.connect()
    try:
        audio_store.ensure_schema(con)
        record = audio_store.get(con, idx)
    finally:
        con.close()

    kind = audio_store.kind(record)
    if kind == "unknown":
        result = resolve_and_store(idx, artist, title)
        if not result.get("url"):
            return None
        con = audio_store.connect()
        try:
            record = audio_store.get(con, idx)
        finally:
            con.close()
        kind = audio_store.kind(record)
    if kind == "local":
        # The row names the archive holding it. A row can outlive its tar (the archive
        # was deleted to reclaim disk, or the DB was indexed before it was fetched), and
        # serving the offset blindly would stream bytes out of the middle of another
        # file -- so a missing archive means "not playable", not a guess.
        tar_name = record.get("tar")
        tar_path = mtg_audio_dir() / tar_name if tar_name else None
        if not tar_path or not tar_path.exists():
            return None
        return local_stream(record, tar_path, range_header)
    if kind == "preview":
        try:
            return preview_stream(record["url"], range_header)
        except Exception:
            # Preview URLs rotate. Re-resolve once, then serve the fresh one; a second
            # failure is a real miss and is recorded as such.
            result = resolve_and_store(idx, artist, title)
            if result.get("url") and result["url"] != record.get("url"):
                return preview_stream(result["url"], range_header)
            return None
    return None
