"""Audio endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Response

from .. import audio, audio_store
from ..index import get_catalog
from ..schemas import AudioStats

router = APIRouter(tags=["audio"])


@router.get(
    "/tracks/{idx}/audio",
    responses={
        200: {"content": {"audio/mpeg": {}}},
        206: {"description": "partial content (Range request)"},
        404: {"description": "nothing to play for this track"},
    },
)
def track_audio(idx: int, range: str | None = Header(default=None)):
    """Stream the local track, or a resolved preview.

    Playback is the one thing that cannot be promised for this corpus: 586 of 32,783
    tracks (~1.8%) exist locally and about 28% of the remainder resolve to a preview.
    A 404 here is an ordinary outcome, not an error, and mirrors the ``none`` badge the
    UI shows for the same track.
    """
    catalog = get_catalog()
    record = catalog.tracks.get(idx)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no track with idx {idx}")
    response = audio.serve(
        idx, record["artist"] or "", record["title"] or "", range_header=range
    )
    if response is None:
        raise HTTPException(
            status_code=404,
            detail="no audio available: not in the local slice and no preview found",
        )
    return response


@router.get("/audio/stats", response_model=AudioStats)
def audio_stats() -> AudioStats:
    catalog = get_catalog()
    catalog.refresh_audio(force=True)
    return AudioStats(**catalog.stats()["audio"])


@router.delete("/tracks/{idx}/audio", status_code=204)
def clear_audio_match(idx: int) -> Response:
    """Forget a cached match, so the next play resolves from scratch.

    Exists because preview matching is fuzzy: when the player lands on a different
    recording the user needs a way to say so, and "not it?" has to actually do
    something rather than just hide the label.
    """
    con = audio_store.connect()
    try:
        audio_store.ensure_schema(con)
        con.execute("DELETE FROM audio WHERE idx = ? AND source != 'local'", (idx,))
        con.commit()
    finally:
        con.close()
    return Response(status_code=204)
