"""FastAPI app for the MTG-Jamendo similarity explorer.

Run it with::

    nix develop .#web --command uvicorn app.main:app --reload --port 8000

(from ``webapp/api``). The frontend dev server proxies ``/api`` here, so nothing needs
CORS in the normal setup; the middleware is for direct calls from elsewhere.

This process reads only the exported artifacts -- numpy + sqlite. It does not import
torch, chromadb, or the ``music_encoding`` package at all, and it needs no GPU and no
checkpoint: similarity is an exact matrix product over the exported embeddings.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .index import get_catalog
from .routers import audio, catalog, graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load eagerly rather than on first request: the vectors are ~34 MB, and failing
    # here with "run export_index.py" is far clearer than a 500 on the first search.
    loaded = get_catalog()
    print(
        f"catalog ready: {loaded.n} tracks x {loaded.dim}-d "
        f"({', '.join(sorted(loaded._spaces))}) from {loaded.catalog_path}"
    )
    yield


app = FastAPI(
    title="MTG-Jamendo similarity explorer",
    description=(
        "Browse, play and explore an MTG-Jamendo corpus by Barlow Twins audio "
        "embeddings. Similarity is exact cosine over the exported index."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *(filter(None, os.environ.get("WEBAPP_EXTRA_ORIGINS", "").split(","))),
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    # Without this the browser cannot read Content-Range off a 206, so the audio
    # element cannot seek even though the server supports ranges.
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)

api = APIRouter(prefix="/api")
api.include_router(catalog.router)
api.include_router(graph.router)
api.include_router(audio.router)
app.include_router(api)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "tracks": get_catalog().n}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
