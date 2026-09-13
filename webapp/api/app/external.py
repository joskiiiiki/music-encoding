"""Search Deezer/iTunes and find their songs' neighbours *in the corpus*.

The point of the app is the embedding space; this lets an outside song be dropped into
it. A query is resolved to a 30-second preview, embedded by the torch sidecar
(``webapp/embedder``) using MTG's own mel front end, and ranked against the exported
matrix with the same cosine the rest of the app uses.

Two measured decisions
----------------------
**Ranking is in the MEAN-CENTRED space.** Not whitened, and not raw either -- the three
were measured against a query whose right answer is known.

The embeddings are dominated by a shared component: every vector is unit-norm and
``||mean|| = 0.971``, so 97% of each embedding is a direction they all agree on and the
per-track signal is a residual of norm ~0.23. Consequences, measured:

* **Raw** cosines are compressed into 0.94-1.0 (mean pairwise cosine **0.943**), so a
  ranking is decided in the fourth decimal. On a Deezer preview of Podington Bear's
  "Dry Air" the top 10 spanned just 0.008 of cosine.
* **Whitened** removes that component and then amplifies the residual in directions that
  are numerically hopeless -- the covariance's eigenvalue spread is 4.1e6, so the 1e-4
  clip means a 100x gain. A query's own track lands at rank **1948-12821**.
* **Mean-centred** exposes the residual without amplifying it, and won on both tests:
  the same-artist share of the top 10 went **6/10 (raw) -> 8/10 (centred)**, with a
  score spread of 0.051 instead of 0.008 -- scores that actually separate.

For a query whose own track is in the corpus, raw puts it at rank 3/1/1 and centred at
64/1/1: raw is not useless, it is just compressed. Centred is used because it
discriminates better and its numbers are readable. This is not a claim about the corpus
views, which default to whitened and are exact by construction -- the eval suite
measures that space and prefers it.

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
    catalog = get_catalog()
    # Mean-centred: the query is centred with the same corpus mean the matrix was
    # centred with, then compared by cosine. See the module docstring for the numbers
    # behind that choice over raw or whitened.
    query = query - catalog.raw_mean
    norm = float(np.linalg.norm(query)) or 1.0
    query = (query / norm).astype(np.float32)
    sims = catalog.space("centred") @ query
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
        "space": "centred",
        "note": (
            "Embedded from a provider preview — a different recording of the song, so "
            "these neighbours are noisier than the ones a corpus track gets."
        ),
        "items": [catalog.serialize(int(i), score=float(sims[i])) for i in top],
    }
