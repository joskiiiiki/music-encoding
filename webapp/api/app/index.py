"""The catalogue: metadata, facets, exact k-NN, and the similarity graph.

Everything the API serves comes from the artifacts ``export_index.py`` wrote, so this
module needs numpy and sqlite and nothing else -- no chromadb, no torch, no GPU.

Similarity is **exact brute-force cosine**, not HNSW. The whole matrix is 32,783 x 128
float32 (17 MB per space), so ``X @ X[i]`` answers a query in ~2 ms and returns the
true top-k rather than an approximate one. ``verify_export.py`` checks that this path
reproduces chroma's answers on the same data.
"""

from __future__ import annotations

import collections
import re
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from . import audio_store, enrichment
from .paths import audio_db_path, catalog_path, vectors_path

# Facet dimensions and the child table each is stored in.
FACETS = {
    "genre": "track_tags",
    "instrument": "track_instruments",
    "mood": "track_moods",
}

SORTS = {
    "relevance": None,
    "idx": "t.idx",
    "title": "t.title COLLATE NOCASE, t.artist COLLATE NOCASE",
    "artist": "t.artist COLLATE NOCASE, t.title COLLATE NOCASE",
    "released_desc": "t.released IS NULL, t.released DESC, t.artist COLLATE NOCASE",
    "released_asc": "t.released IS NULL, t.released ASC, t.artist COLLATE NOCASE",
    "duration_asc": "t.duration IS NULL, t.duration ASC",
    "duration_desc": "t.duration IS NULL, t.duration DESC",
}

AUDIO_REFRESH_SECONDS = 60.0


def fts_query(text: str) -> str | None:
    """Turn user input into a safe FTS5 expression.

    Every token is quoted so that punctuation a user types (``AC/DC``, ``!!!``) cannot
    be parsed as FTS5 syntax and raise; the final token gets ``*`` so that searching
    "poding" finds "Podington" while typing. Returns ``None`` when there is nothing
    searchable, which the caller treats as "no text filter".
    """
    tokens = re.findall(r"[^\W_]+", text, re.UNICODE)
    if not tokens:
        return None
    quoted = [f'"{t}"' for t in tokens]
    quoted[-1] += "*"
    return " ".join(quoted)


class Catalog:
    def __init__(self, catalog: Path | None = None, vectors: Path | None = None):
        self.catalog_path = catalog or catalog_path()
        self.vectors_path = vectors or vectors_path()
        if not self.catalog_path.exists() or not self.vectors_path.exists():
            raise FileNotFoundError(
                f"missing export artifacts in {self.catalog_path.parent} -- run "
                "webapp/api/scripts/export_index.py (see webapp/README.md)"
            )
        self.con = sqlite3.connect(
            f"file:{self.catalog_path}?mode=ro", uri=True, check_same_thread=False
        )
        self.con.row_factory = sqlite3.Row
        self._lock = (
            threading.Lock()
        )  # sqlite connection + numpy are shared across threads

        with np.load(self.vectors_path) as z:
            self._spaces = {k: z[k] for k in ("whitened", "raw")}
        self.n, self.dim = self._spaces["whitened"].shape
        # Mean-centred, derived from raw at startup. Exists because the raw space is
        # almost entirely a shared component (||mean|| = 0.971, mean pairwise
        # cosine 0.943), so raw cosines are compressed into 0.94-1.0 and barely
        # discriminate; subtracting the mean exposes the ~0.23-norm residual where the
        # content actually lives. It is used for *external queries*, and is deliberately
        # not offered on the corpus routes -- see app/external.py for the measurements.
        self.raw_mean = self._spaces["raw"].mean(axis=0)
        self._spaces["centred"] = self._unit_rows(self._spaces["raw"] - self.raw_mean)

        self._load_metadata()
        self._load_audio()

    # ------------------------------------------------------------------ loading

    def _load_metadata(self) -> None:
        self.tracks: dict[int, dict] = {}
        self.title = {}
        self.artist: dict[int, str] = {}
        self.genre: dict[int, str] = {}
        self.tags: dict[int, set[str]] = collections.defaultdict(set)
        self.instrument: dict[int, str] = {}
        self.mood: dict[int, str] = {}

        for row in self.con.execute("SELECT * FROM tracks"):
            idx = row["idx"]
            self.tracks[idx] = dict(row)
            self.title[idx] = row["title"] or ""
            self.artist[idx] = row["artist"] or ""
            self.genre[idx] = row["genre"] or ""
        for table, target in (
            ("track_tags", self.tags),
            ("track_instruments", self.instrument),
            ("track_moods", self.mood),
        ):
            for idx, tag in self.con.execute(f"SELECT idx, tag FROM {table}"):
                if target is self.tags:
                    target[idx].add(tag)
                else:
                    target[idx] = tag  # single-valued by construction

        # Corpus-wide frequencies drive the enrichment chance rates: the probability a
        # random track agrees with the seed about a label. Tags are interned so 80k rows
        # share a handful of string objects.
        self.tag_prevalence: collections.Counter[str] = collections.Counter(
            tag for tags in self.tags.values() for tag in tags
        )
        self.artist_counts: collections.Counter[str] = collections.Counter(
            a for a in self.artist.values() if a
        )
        self.genre_counts: collections.Counter[str] = collections.Counter(
            g for g in self.genre.values() if g
        )
        self.global_facets = {
            name: [
                {"value": value, "count": count}
                for value, count in self.con.execute(
                    f"SELECT tag, COUNT(*) c FROM {table} "
                    "GROUP BY tag ORDER BY c DESC, tag"
                )
            ]
            for name, table in FACETS.items()
        }
        self.meta = {
            row["key"]: row["value"] for row in self.con.execute("SELECT * FROM meta")
        }

    def _load_audio(self) -> None:
        """Availability, as a plain ``idx -> kind`` map.

        Read from its own file, so the long preview warm can be running while the API
        serves, and refreshed on a timer rather than per request.
        """
        self._audio_at = 0.0
        self._avail: dict[int, str] = {}
        self._playable: set[int] = set()
        self.refresh_audio(force=True)

    def refresh_audio(self, force: bool = False) -> None:
        with self._lock:
            if not force and time.time() - self._audio_at < AUDIO_REFRESH_SECONDS:
                return
            self._audio_at = time.time()
            if not audio_db_path().exists():
                self._avail, self._playable = {}, set()
                return
            con = audio_store.connect()
            try:
                audio_store.ensure_schema(con)
                avail = {
                    row["idx"]: audio_store.kind(dict(row))
                    for row in con.execute("SELECT * FROM audio")
                }
            finally:
                con.close()
            self._avail = avail
            self._playable = {i for i, k in avail.items() if k in ("local", "preview")}

    # ------------------------------------------------------------------- lookups

    @staticmethod
    def _unit_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
        return (matrix / norms).astype(np.float32)

    def space(self, name: str) -> np.ndarray:
        if name not in self._spaces:
            raise KeyError(
                f"unknown space {name!r}; expected one of {sorted(self._spaces)}"
            )
        return self._spaces[name]

    def serialize(self, idx: int, *, score: float | None = None) -> dict:
        t = self.tracks[idx]
        return {
            "idx": idx,
            "track_id": t["track_id"],
            "track_num": t["track_num"],
            "title": t["title"] or "",
            "artist": t["artist"] or "",
            "album": t["album"] or "",
            "genre": t["genre"] or None,
            "tags": sorted(self.tags.get(idx, ())),
            "instrument": self.instrument.get(idx),
            "mood": self.mood.get(idx),
            "released": t["released"],
            "duration": t["duration"],
            "audio": self._avail.get(idx, "unknown"),
            "score": score,
        }

    def track(self, idx: int) -> dict | None:
        if idx not in self.tracks:
            return None
        out = self.serialize(idx)
        out["similar"] = [
            self.serialize(i, score=s) for i, s in self.similar(idx, k=20)
        ]
        return out

    # ---------------------------------------------------------------- similarity

    def similar(
        self, idx: int, k: int = 20, space: str = "whitened"
    ) -> list[tuple[int, float]]:
        """Top-k by exact cosine, self excluded."""
        X = self.space(space)
        k = max(1, min(k, self.n - 1))
        sims = X @ X[idx]
        sims[idx] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def submatrix(self, idxs: list[int], space: str) -> np.ndarray:
        return (
            self.space(space)[np.asarray(idxs)] @ self.space(space)[np.asarray(idxs)].T
        )

    def graph(
        self,
        seed: int,
        k: int = 20,
        space: str = "whitened",
        min_sim: float = 0.0,
    ) -> dict:
        """The seed plus its top-k, with every edge among those nodes.

        Computing the (k+1)x(k+1) submatrix directly -- instead of issuing k+1 separate
        neighbour queries -- is both cheaper and better: it yields the similarity
        *between* displayed nodes, which is what makes this a network rather than a
        star.
        """
        neighbours = self.similar(seed, k=k, space=space)
        nodes = [seed] + [i for i, _ in neighbours]
        sub = self.submatrix(nodes, space)
        edges = [
            {"source": nodes[a], "target": nodes[b], "weight": float(sub[a, b])}
            for a in range(len(nodes))
            for b in range(a + 1, len(nodes))
            if sub[a, b] >= min_sim
        ]
        degree = collections.Counter()
        for e in edges:
            degree[e["source"]] += 1
            degree[e["target"]] += 1

        payload_nodes = []
        for pos, idx in enumerate(nodes):
            node = self.serialize(idx, score=None if pos == 0 else float(sub[0, pos]))
            node["degree"] = degree.get(idx, 0)
            node["seed"] = pos == 0
            payload_nodes.append(node)

        return {
            "seed": seed,
            "space": space,
            "nodes": payload_nodes,
            "edges": edges,
            # Seed-centred, against the corpus. See enrichment.py for why a within-graph
            # permutation null is the wrong baseline for an ego-graph.
            "enrichment": {
                "artist": enrichment.neighbour_agreement(
                    nodes, self.artist, self.artist_counts, self.n
                ),
                "genre": enrichment.neighbour_agreement(
                    nodes, self.genre, self.genre_counts, self.n
                ),
                "tags": enrichment.tag_overlap_enrichment(
                    nodes, self.tags, self.tag_prevalence, self.n
                ),
            },
        }

    # -------------------------------------------------------------------- search

    def _filter_clauses(
        self,
        q: str | None,
        facets: dict[str, list[str]],
        year_min: int | None,
        year_max: int | None,
        playable_only: bool,
        skip: str | None = None,
    ) -> tuple[list[str], list]:
        clauses: list[str] = []
        params: list = []

        if q:
            expr = fts_query(q)
            if expr:
                clauses.append(
                    "t.idx IN (SELECT rowid FROM tracks_fts WHERE tracks_fts MATCH ?)"
                )
                params.append(expr)
        for name, table in FACETS.items():
            values = facets.get(name) or []
            if values and name != skip:
                marks = ",".join("?" * len(values))
                clauses.append(
                    f"t.idx IN (SELECT idx FROM {table} WHERE tag IN ({marks}))"
                )
                params += list(values)
        if year_min is not None:
            clauses.append("t.released >= ?")
            params.append(year_min)
        if year_max is not None:
            clauses.append("t.released <= ?")
            params.append(year_max)
        if playable_only:
            if not self._playable:
                clauses.append("0")
            else:
                marks = ",".join("?" * len(self._playable))
                clauses.append(f"t.idx IN ({marks})")
                params += sorted(self._playable)
        return clauses, params

    def search(
        self,
        q: str | None = None,
        genres: list[str] | None = None,
        instruments: list[str] | None = None,
        moods: list[str] | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        playable_only: bool = False,
        sort: str = "relevance",
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        facets = {
            "genre": genres or [],
            "instrument": instruments or [],
            "mood": moods or [],
        }
        order = SORTS.get(sort) or SORTS["idx"]
        expr = fts_query(q) if q else None

        if expr and sort == "relevance":
            # `rank` is a column of the FTS table, so relevance ordering needs the join
            # form; every other sort uses the cheaper IN-subquery.
            clauses, params = self._filter_clauses(
                None, facets, year_min, year_max, playable_only
            )
            where = "WHERE " + " AND ".join(clauses + ["tracks_fts MATCH ?"])
            params = params + [expr]
            sql = f"FROM tracks t JOIN tracks_fts ON tracks_fts.rowid = t.idx {where}"
            total = self.con.execute(f"SELECT COUNT(*) {sql}", params).fetchone()[0]
            rows = self.con.execute(
                f"SELECT t.* {sql} ORDER BY rank LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            applied_sort = "relevance"
        else:
            clauses, params = self._filter_clauses(
                q, facets, year_min, year_max, playable_only
            )
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            total = self.con.execute(
                f"SELECT COUNT(*) FROM tracks t {where}", params
            ).fetchone()[0]
            rows = self.con.execute(
                f"SELECT t.* FROM tracks t {where} ORDER BY {order} LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            applied_sort = sort if sort in SORTS else "idx"

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "sort": applied_sort,
            "items": [self.serialize(r["idx"]) for r in rows],
            "facets": self.facet_counts(q, facets, year_min, year_max, playable_only),
        }

    def facet_counts(
        self,
        q: str | None,
        facets: dict[str, list[str]],
        year_min: int | None,
        year_max: int | None,
        playable_only: bool,
    ) -> dict:
        """Per-facet value counts for the current filters.

        Each dimension is counted with its **own** filter removed, which is what makes
        the filter UI usable: you can see how many tracks adding a second genre would
        bring in, rather than every unselected genre reading zero.
        """
        out = {}
        for name, table in FACETS.items():
            clauses, params = self._filter_clauses(
                q, facets, year_min, year_max, playable_only, skip=name
            )
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            out[name] = [
                {"value": row["tag"], "count": row["c"]}
                for row in self.con.execute(
                    f"SELECT tag, COUNT(*) c FROM {table} "
                    f"WHERE idx IN (SELECT idx FROM tracks t {where}) "
                    f"GROUP BY tag ORDER BY c DESC, tag",
                    params,
                )
            ]
        return out

    def stats(self) -> dict:
        con = audio_store.connect()
        try:
            audio_store.ensure_schema(con)
            audio = audio_store.stats(con, self.n)
        finally:
            con.close()
        return {
            "tracks": self.n,
            "dim": self.dim,
            "spaces": sorted(self._spaces),
            "meta": self.meta,
            "audio": audio,
            "facets": {
                name: len(values) for name, values in self.global_facets.items()
            },
        }


_catalog: Catalog | None = None
_catalog_lock = threading.Lock()


def get_catalog() -> Catalog:
    """Process-wide singleton; the matrices are 17 MB each, so load them once."""
    global _catalog
    with _catalog_lock:
        if _catalog is None:
            _catalog = Catalog()
        return _catalog
