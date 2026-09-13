"""The similarity graph: a seed, its neighbourhood, and the edges among them."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from ..index import get_catalog
from ..schemas import GraphResponse

router = APIRouter(tags=["graph"])


@router.get("/graph", response_model=GraphResponse)
def graph(
    seed: Annotated[int, Query(description="idx of the track to centre the graph on")],
    k: Annotated[int, Query(ge=1, le=100, description="neighbours to pull in")] = 20,
    space: Annotated[str, Query(pattern="^(whitened|raw)$")] = "whitened",
    min_sim: Annotated[
        float,
        Query(
            ge=-1.0,
            le=1.0,
            description="drop edges below this cosine, to thin a dense graph",
        ),
    ] = 0.0,
) -> GraphResponse:
    """Nodes are the seed plus its top-k; edges are every cosine among those nodes.

    Deterministic for fixed parameters -- the neighbour ranking is exact (no approximate
    index) and nothing is randomised -- so reloading a view reproduces it exactly.
    """
    catalog = get_catalog()
    if seed not in catalog.tracks:
        raise HTTPException(status_code=404, detail=f"no track with idx {seed}")
    return GraphResponse(**catalog.graph(seed, k=k, space=space, min_sim=min_sim))
