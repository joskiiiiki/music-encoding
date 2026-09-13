"""Browsing, searching and metadata endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from ..index import SORTS, get_catalog
from ..schemas import Facets as FacetsModel
from ..schemas import SearchResponse, Track, TrackDetail

router = APIRouter(tags=["catalog"])

# `Annotated[...] = None` rather than `Query(default_factory=...)`: a call in a default
# argument is what flake8-bugbear's B008 flags, and FastAPI's documented style is the
# annotation form. Multi-value filters are optional lists, so `None` means "no filter"
# and each is normalised to a list before use.

QueryText = Annotated[
    str | None, Query(description="free text over title, artist, album")
]
QueryList = Annotated[
    list[str] | None, Query(description="repeatable; OR within a facet")
]
QueryLimit = Annotated[int, Query(ge=1, le=500)]
QueryOffset = Annotated[int, Query(ge=0)]


@router.get("/search", response_model=SearchResponse)
def search(
    q: QueryText = None,
    genre: QueryList = None,
    instrument: QueryList = None,
    mood: QueryList = None,
    year_min: int | None = None,
    year_max: int | None = None,
    playable_only: Annotated[
        bool, Query(description="only tracks with local audio or a resolved preview")
    ] = False,
    sort: Annotated[str, Query(description=f"one of {sorted(SORTS)}")] = "relevance",
    limit: QueryLimit = 50,
    offset: QueryOffset = 0,
) -> SearchResponse:
    result = get_catalog().search(
        q=q,
        genres=genre or [],
        instruments=instrument or [],
        moods=mood or [],
        year_min=year_min,
        year_max=year_max,
        playable_only=playable_only,
        sort=sort,
        limit=limit,
        offset=offset,
    )
    return SearchResponse(**result)


@router.get("/facets", response_model=FacetsModel)
def facets() -> FacetsModel:
    """Corpus-wide facet vocabularies, for building the filter UI."""
    return FacetsModel(**get_catalog().global_facets)


@router.get("/tracks/{idx}", response_model=TrackDetail)
def track(idx: int) -> TrackDetail:
    catalog = get_catalog()
    result = catalog.track(idx)
    if result is None:
        raise HTTPException(status_code=404, detail=f"no track with idx {idx}")
    return TrackDetail(**result)


@router.get("/tracks/{idx}/similar", response_model=list[Track])
def similar(
    idx: int,
    k: Annotated[int, Query(ge=1, le=200)] = 20,
    space: Annotated[str, Query(pattern="^(whitened|raw)$")] = "whitened",
) -> list[Track]:
    catalog = get_catalog()
    if idx not in catalog.tracks:
        raise HTTPException(status_code=404, detail=f"no track with idx {idx}")
    return [
        Track(**catalog.serialize(i, score=score))
        for i, score in catalog.similar(idx, k=k, space=space)
    ]


@router.get("/stats")
def stats() -> dict:
    """Index provenance and audio coverage, for the UI footer."""
    catalog = get_catalog()
    catalog.refresh_audio()
    return catalog.stats()
