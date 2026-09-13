"""Search Deezer/iTunes and find their songs' neighbours *in the corpus*.

The point of the app is the embedding space; this lets an outside song be dropped into
it. A query is resolved to a 30-second preview, embedded by the torch sidecar
(``webapp/embedder``) using MTG's own mel front end, and ranked against the exported
matrix with the same cosine the rest of the app uses.

Two measured decisions
----------------------
**Ranking is in the RAW space.** Whitening a *corpus* embedding is right -- it spreads
the cloud and the eval suite prefers it -- but whitening a *query* destroys it. ZCA
amplifies the low-variance directions: the exported covariance has a 50x eigenvalue
spread and the
clip at 1e-4 means up to 100x amplification, so the query's own ~1% embedding error is
magnified into near-orthogonality. Measured on three tracks whose audio we hold, ranked
against their own corpus embeddings::

    cos(query, corpus embedding)   raw 0.987-0.994   whitened -0.007-0.264

So the raw space is used here regardless of the ``space`` the corpus views default to.

**A query is a different recording.** The provider serves a preview, usually the hook,
a song that may be a remix or a live take; ``compare_songs`` measured ~0.987 between the
audio and .npy paths *for the same recording*, and a preview is not that. The response
carries a note so the UI can say so rather than implying corpus-grade precision.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np

from . import previews
from .index import get_catalog
from .paths import embedder_url

# Embedding a preview takes the sidecar a few hundred ms (measured 300-570 ms for a
# full track, less for a 30 s clip); the download adds a little.
EMBED_TIMEOUT = 60


class EmbedderUnavailable(RuntimeError):
    """The torch sidecar is not running, or refused the request."""


def embed_url(url: str) -> dict:
    """Ask the sidecar for the embedding of an audio URL."""
    body = json.dumps({"url": url}).encode()
    request = urllib.request.Request(
        f"{embedder_url().rstrip('/')}/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=EMBED_TIMEOUT) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:200]
        raise EmbedderUnavailable(f"embedder returned {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise EmbedderUnavailable(
            f"embedder not reachable at {embedder_url()} -- start it with "
            "`nix develop --command python webapp/embedder/service.py`"
        ) from exc


def search(query: str, provider: str = "auto", limit: int = 8) -> dict:
    """Provider search for the search box."""
    if not query.strip():
        return {"query": query, "provider": provider, "results": []}
    return {
        "query": query,
        "provider": provider,
        "results": previews.browse(query, provider=provider, limit=limit),
    }


def similar(
    preview_url: str,
    provider: str = "deezer",
    external_id: str | None = None,
    title: str = "",
    artist: str = "",
    k: int = 20,
) -> dict:
    """Embed a provider preview and rank the corpus against it.

    The preview URL comes from a search result the client already has, so no
    re-resolution is normally needed -- unless it has expired, in which case the
    provider is asked again by ``artist title``.
    """
    payload = None
    for attempt in (preview_url, None):
        if attempt is None:
            # Expired URL: re-resolve by text before giving up.
            hit = previews.resolve(artist, title, provider)
            attempt = hit.get("url")
            if not attempt:
                break
        try:
            payload = embed_url(attempt)
            break
        except EmbedderUnavailable:
            raise
        except Exception:  # noqa: BLE001 - a dead URL should fall through to the retry
            continue
    if payload is None:
        raise EmbedderUnavailable(
            "could not embed that preview: the URL may have expired and re-resolving "
            "found nothing"
        )

    query = np.asarray(payload["vector"], dtype=np.float32)
    norm = float(np.linalg.norm(query)) or 1.0
    query = query / norm

    catalog = get_catalog()
    # Raw, never whitened -- see the module docstring.
    sims = catalog.space("raw") @ query
    k = max(1, min(k, catalog.n - 1))
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]

    return {
        "query": {
            "provider": provider,
            "id": external_id,
            "title": title,
            "artist": artist,
            "preview_url": payload.get("source"),
            "audio_seconds": payload.get("audio_seconds"),
            "embed_ms": payload.get("elapsed_ms"),
        },
        "space": "raw",
        "note": (
            "Embedded from a provider preview — a different recording of the song, so "
            "these neighbours are noisier than the ones a corpus track gets."
        ),
        "items": [catalog.serialize(int(i), score=float(sims[i])) for i in top],
    }
