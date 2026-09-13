"""Response models.

They serve two purposes beyond documentation: they pin down which fields are
**nullable**, and they generate the OpenAPI schema the frontend's types are written
against.

Nullability is not decoration here. ``build_chroma`` omits a metadata key entirely when
its value is empty rather than writing None, so ``instrument`` is absent for 54% of
tracks and ``mood`` for 66%, and ``released`` is absent on rows the TSVs do not cover.
A schema that declared those as required strings would fail on half the corpus.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Track(BaseModel):
    idx: int = Field(description="position in the corpus the DB was built from")
    track_id: str | None = Field(
        default=None, description="stable MTG id, e.g. track_0000214"
    )
    track_num: int | None = None
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str | None = Field(
        default=None,
        description="alphabetically first genre tag -- not a primary genre",
    )
    tags: list[str] = Field(default_factory=list, description="the full genre tag set")
    instrument: str | None = None
    mood: str | None = None
    released: int | None = Field(default=None, description="year, 2004-2017")
    duration: float | None = Field(default=None, description="full-track seconds")
    audio: str = Field(
        default="unknown",
        description="local | preview | none | unknown (not yet looked up)",
    )
    score: float | None = Field(
        default=None, description="cosine to the query, when ranked"
    )
    degree: int | None = Field(default=None, description="edges in the rendered graph")
    seed: bool | None = Field(
        default=None, description="true for the graph's seed node"
    )


class FacetValue(BaseModel):
    value: str
    count: int


class Facets(BaseModel):
    genre: list[FacetValue] = Field(default_factory=list)
    instrument: list[FacetValue] = Field(default_factory=list)
    mood: list[FacetValue] = Field(default_factory=list)


class SearchResponse(BaseModel):
    total: int
    limit: int
    offset: int
    sort: str
    items: list[Track]
    facets: Facets


class TrackDetail(Track):
    similar: list[Track] = Field(default_factory=list)


class Edge(BaseModel):
    source: int
    target: int
    weight: float = Field(description="cosine similarity between the two nodes")


class NeighbourAgreement(BaseModel):
    """How often a neighbour matches the seed on one label, against the corpus.

    ``chance`` is what a randomly chosen track would score, so ``lift`` is the honest
    multiplier. ``compared``/``skipped`` expose coverage: a dimension the seed itself
    lacks, or that most neighbours lack, cannot support a claim.
    """

    observed: float = 0.0
    chance: float = 0.0
    lift: float | None = None
    compared: int = 0
    skipped: int = 0
    agreeing: int = 0
    seed_value: str | None = None
    reason: str | None = Field(
        default=None, description="set when no meaningful number could be computed"
    )


class TagOverlap(BaseModel):
    mean_overlap: float = 0.0
    chance: float = 0.0
    lift: float | None = None
    compared: int = 0
    seed_tags: list[str] = Field(default_factory=list)
    reason: str | None = None


class Enrichment(BaseModel):
    artist: NeighbourAgreement
    genre: NeighbourAgreement
    tags: TagOverlap


class GraphResponse(BaseModel):
    seed: int
    space: str
    nodes: list[Track]
    edges: list[Edge]
    enrichment: Enrichment


class AudioStats(BaseModel):
    total_tracks: int
    local: int
    deezer: int
    itunes: int
    playable: int
    attempted_misses: int
    unresolved: int
    attempted: int
