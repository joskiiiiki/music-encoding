"""Direct retrieval eval: is a window's nearest neighbour its own track?

Genre k-NN and the intra/inter histogram are both *proxies* for embedding quality.
The stated purpose of this model is a Shazam-style fingerprint — "any 5-second
window of a track maps near its other windows" — so the metric that actually
matches that objective is retrieval: for a query window, what fraction of its top-k
nearest neighbours belong to the SAME track?

Why this is worth measuring separately: Barlow Twins pushes two views of one track
together and decorrelates *dimensions*, but never pushes *different tracks apart*.
Redundancy reduction is not instance discrimination. So intra-track similarity can
look excellent (≈0.94) while a window's nearest neighbour is still a different track
more than half the time — which is exactly what this reports, and what the other
evals cannot show.

Whitening is reported alongside because the raw space is known to be poorly
conditioned; if whitening fixes retrieval, the model was fine and the space was not.
(It barely helps — unlike the intra/inter gap — which is how you can tell the
weakness is the objective rather than the geometry.)

Read from the chroma DB's `windows_raw` collection only: no checkpoint, no model
forwards, no GPU. Chance is (W-1)/(N-1) ≈ 2.4e-05 for 5 windows over 163,915.

Usage:
    python -m music_encoding.test_retrieval [--db-dir chroma_db] [--queries 1500] [--k 10]
"""

import argparse

import numpy as np

from music_encoding.db import COLLECTION_WINDOWS, load_all


def hit_rates(
    P: np.ndarray, track: np.ndarray, queries: np.ndarray, ks: tuple[int, ...]
) -> dict[int, float]:
    """Fraction of queries whose top-k neighbours include the same track."""
    hits = {k: 0 for k in ks}
    for qi in queries:
        sim = P @ P[qi]
        sim[qi] = -2.0  # never match the query window itself
        same = track[np.argsort(-sim)[: max(ks)]] == track[qi]
        for k in ks:
            if same[:k].any():
                hits[k] += 1
    return {k: c / len(queries) for k, c in hits.items()}


def whiten_by_track(embs: np.ndarray, track: np.ndarray) -> np.ndarray:
    """ZCA-whiten using per-track pooled embeddings (as in the collapse diagnostic)."""
    n_tracks = len(np.unique(track))
    pooled = np.stack([embs[track == t].mean(axis=0) for t in np.unique(track)])
    mean = pooled.mean(axis=0)
    cov = ((pooled - mean).T @ (pooled - mean)) / (n_tracks - 1)
    evals, evecs = np.linalg.eigh(cov.astype(np.float64))
    evals = np.clip(evals, 1e-4, None)
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    Z = (embs - mean) @ W
    return (Z / np.linalg.norm(Z, axis=1, keepdims=True)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="same-track retrieval from the chroma DB's per-window vectors"
    )
    parser.add_argument(
        "--db-dir", default="chroma_db",
        help="Chroma persistent dir (default: chroma_db)",
    )
    parser.add_argument(
        "--queries", type=int, default=1500,
        help="windows to sample as queries (default: 1500)",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="largest k to report (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="query-sampling seed (default: 42)",
    )
    args = parser.parse_args()

    print(f"[main] loading '{args.db_dir}/{COLLECTION_WINDOWS}'")
    ids, embs, _ = load_all(args.db_dir, COLLECTION_WINDOWS)
    # ids are "track_<idx>_w<k>"
    track = np.array([int(i.split("_")[1]) for i in ids])
    n = len(ids)
    n_tracks = len(np.unique(track))
    n_per = n // n_tracks
    print(f"[main] {n} windows over {n_tracks} tracks ({n_per} windows/track)")

    P = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    ks = tuple(k for k in (1, 3, 5, 10) if k <= args.k)
    rng = np.random.default_rng(args.seed)
    q = rng.choice(n, size=min(args.queries, n), replace=False)

    for label, Pmat in (("raw", P), ("whitened", whiten_by_track(embs, track))):
        rates = hit_rates(Pmat, track, q, ks)
        body = "  ".join(f"top-{k}:{v:.4f}" for k, v in rates.items())
        print(f"[main] {label:<9} {body}")

    print(f"[main] chance (same-track among all windows): {(n_per - 1) / (n - 1):.2e}")
    print(
        "[main] a working fingerprint model should approach 1.0 at top-1; "
        "whitening barely moving it means the weakness is the objective, "
        "not the conditioning"
    )


if __name__ == "__main__":
    main()
