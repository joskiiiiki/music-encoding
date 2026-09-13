"""Search beyond the corpus, and find an outside song's neighbours in it."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from .. import external

router = APIRouter(tags=["external"])


@router.get("/external/search")
def search(
    q: Annotated[
        str, Query(description="free text; the providers decide what matches")
    ],
    provider: Annotated[str, Query(pattern="^(auto|deezer|itunes)$")] = "auto",
    limit: Annotated[int, Query(ge=1, le=25)] = 8,
) -> dict:
    """Deezer (then iTunes) results for a query.

    Deliberately unfiltered: when the user types a query, the providers' answers *are*
    the request. (Resolving audio for a *corpus* track is the opposite problem and does
    require an artist match -- see previews.search_deezer.)
    """
    return external.search(q, provider=provider, limit=limit)


@router.get("/external/similar")
def similar(
    preview_url: Annotated[str, Query(description="the provider's preview mp3 URL")],
    provider: Annotated[str, Query(pattern="^(deezer|itunes)$")] = "deezer",
    id: Annotated[str | None, Query(description="the provider's track id")] = None,
    title: str = "",
    artist: str = "",
    k: Annotated[int, Query(ge=1, le=100)] = 20,
) -> dict:
    """Embed the preview and return the corpus tracks nearest to it.

    Ranked in the **raw** space, not the whitened one the corpus views default to:
    whitening amplifies a query embedding's error into near-orthogonality (measured), so
    query must not be whitened. See app/external.py.
    """
    try:
        return external.similar(
            preview_url,
            provider=provider,
            external_id=id,
            title=title,
            artist=artist,
            k=k,
        )
    except external.EmbedderUnavailable as exc:
        # 503, not 500: the app is fine, a dependency is not running.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
