"""k-NN genre accuracy from the Chroma vec DB.

Loads the pooled embeddings + genres from the chroma DB (built by
build_chroma.py) and runs the same seeded 80/20 track split and top-k majority
genre vote as before — no audio decoding or model forwards needed. Defaults to
the *whitened* space, which scores higher for genre (≈0.36 vs ≈0.31 raw).

Usage:
    python -m music_encoding.test_knn [--db-dir chroma_db] [--k 50] [--space whitened|raw]
"""

import argparse
from collections import Counter

import numpy as np

from music_encoding.db import COLLECTION_RAW, COLLECTION_WHITENED, load_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="k-NN genre accuracy from the chroma vec DB"
    )
    parser.add_argument(
        "--db-dir", default="chroma_db",
        help="Chroma persistent dir (default: chroma_db)",
    )
    parser.add_argument(
        "--k", type=int, default=50, help="number of neighbors (default: 50)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="split RNG seed (default: 42)"
    )
    parser.add_argument(
        "--space", choices=["whitened", "raw"], default="whitened",
        help="similarity space for the kNN (default: whitened)",
    )
    args = parser.parse_args()

    collection = COLLECTION_WHITENED if args.space == "whitened" else COLLECTION_RAW
    print(f"[main] loading embeddings from '{args.db_dir}/{collection}' ({args.space} space)")
    ids, embs, metas = load_all(args.db_dir, collection)
    n = len(ids)
    print(f"[main] loaded {n} tracks, emb shape={embs.shape}")

    genres = np.array([m.get("genre") or "" for m in metas], dtype=object)
    track_idx = np.arange(n)

    rng = np.random.default_rng(args.seed)
    rng.shuffle(track_idx)
    split = int(0.8 * n)
    train_idx, test_idx = track_idx[:split], track_idx[split:]
    print(f"[main] split: {len(train_idx)} train / {len(test_idx)} test tracks")

    # normalize rows for cosine similarity
    train = embs[train_idx]
    train = train / np.linalg.norm(train, axis=1, keepdims=True)
    test = embs[test_idx]
    test = test / np.linalg.norm(test, axis=1, keepdims=True)

    print("[main] computing cosine similarity matrix")
    sim = test @ train.T  # (N_test, N_train)

    k = args.k
    print(f"[main] computing top-{k} neighbors")
    topk_idx = np.argsort(-sim, axis=1)[:, :k]

    train_genres = genres[train_idx]
    test_true = genres[test_idx]
    print("[main] voting over neighbors")
    predicted = [
        Counter(train_genres[row].tolist()).most_common(1)[0][0] for row in topk_idx
    ]

    correct = sum(p == t for p, t in zip(predicted, test_true, strict=True))
    accuracy = correct / len(test_true)
    print(f"[main] k-NN genre accuracy (k={k}): {accuracy:.3f} on {len(test_true)} test tracks")

    counts = Counter(g for g in genres if g)
    print("[main]", counts.most_common(10))
    print(f"random baseline: {1 / len(counts):.3f}")


if __name__ == "__main__":
    main()
