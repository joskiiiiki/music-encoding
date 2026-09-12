"""Shared helpers for reading the Chroma vec DB used by the eval scripts.

Collections (built by build_chroma.py, cosine space; D is the encoder's
`embed_dims`, 128 by default — the projector's z is never stored):
  tracks_whitened  — one pooled whitened embedding per track (+ metadata)
  tracks_raw       — one pooled raw embedding per track (+ metadata)
  windows_raw      — the raw per-window embeddings of each track (+ metadata,
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
    on large collections (windows_raw holds ~164k vectors over the full corpus).

    Each page is converted to a float32 array as it arrives rather than being
    accumulated as Python floats — chroma hands back list[list[float]], and at
    164k x 128 that is ~670 MB of boxed floats on a machine with ~3 GB free.
    """
    col = get_collection(db_dir, name)
    ids: list[str] = []
    pages: list[np.ndarray] = []
    metas: list = []
    offset = 0
    while True:
        data = col.get(include=["embeddings", "metadatas"], limit=_PAGE, offset=offset)
        batch = data["ids"]
        if not batch:
            break
        ids.extend(batch)
        pages.append(np.asarray(data["embeddings"], dtype=np.float32))
        metas.extend(data["metadatas"])
        offset += _PAGE

    embs = (
        np.concatenate(pages, axis=0)
        if pages
        else np.zeros((0, 0), dtype=np.float32)  # empty collection
    )

    # ids look like "track_<idx>" (pooled) or "track_<idx>_w<k>" (per-window);
    # sort by (idx, window) so the documented ordering is actually guaranteed.
    def _key(id_: str) -> tuple[int, int]:
        p = id_.split("_")
        win = int(p[2][1:]) if len(p) > 2 and p[2].startswith("w") else 0
        return int(p[1]), win

    sort_key = np.array(
        [_key(i) for i in ids], dtype=[("idx", np.int64), ("win", np.int64)]
    )
    order = np.argsort(sort_key, kind="stable")
    return [ids[i] for i in order], embs[order], [metas[i] for i in order]
