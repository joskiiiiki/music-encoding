"""Shared helpers for reading the Chroma vec DB used by the eval scripts.

Collections (built by build_chroma.py, cosine space):
  tracks_whitened  — one pooled whitened 128-d embedding per track (+ metadata)
  tracks_raw       — one pooled raw 128-d embedding per track (+ metadata)
  windows_raw      — the 5 raw per-window embeddings of each track (+ metadata,
                     `window` key gives the window index; ids track_<idx>_w<k>)
"""

import chromadb
import numpy as np

DEFAULT_DB_DIR = "chroma_db"
COLLECTION_WHITENED = "tracks_whitened"
COLLECTION_RAW = "tracks_raw"
COLLECTION_WINDOWS = "windows_raw"


def get_client(db_dir: str = DEFAULT_DB_DIR):
    return chromadb.PersistentClient(path=db_dir)


def get_collection(db_dir: str, name: str):
    return get_client(db_dir).get_collection(name)


_PAGE = 2000


def load_all(
    db_dir: str = DEFAULT_DB_DIR, name: str = COLLECTION_WHITENED
) -> tuple[list[str], np.ndarray, list[dict]]:
    """Fetch every vector + metadata from a collection.

    Returns (ids, embs, metas) with embs a (N, D) float32 array, all ordered by
    (track idx, window) so pooled and window data come back deterministically.
    Pages through the collection: a single get() hits SQLite's variable limit
    on large collections (windows_raw has ~40k vectors).
    """
    col = get_collection(db_dir, name)
    ids: list[str] = []
    embs: list = []
    metas: list = []
    offset = 0
    while True:
        data = col.get(include=["embeddings", "metadatas"], limit=_PAGE, offset=offset)
        batch = data["ids"]
        if not batch:
            break
        ids.extend(batch)
        embs.extend(data["embeddings"])
        metas.extend(data["metadatas"])
        offset += _PAGE

    embs = np.asarray(embs, dtype=np.float32)

    # ids look like "track_<idx>" or "track_<idx>_w<k>"; sort by (idx, window).
    parts = [id_.split("_") for id_ in ids]
    sort_key = np.array(
        [(int(p[1]), int(p[3]) if len(p) > 3 else 0) for p in parts],
        dtype=[("idx", np.int64), ("win", np.int64)],
    )
    order = np.argsort(sort_key, kind="stable")
    return [ids[i] for i in order], embs[order], [metas[i] for i in order]
