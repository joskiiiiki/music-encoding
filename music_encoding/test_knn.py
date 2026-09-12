"""k-NN genre accuracy from the Chroma vec DB.

Loads the pooled embeddings + genres from the chroma DB (built by
build_chroma.py) and runs the same seeded 80/20 track split and top-k majority
genre vote as before — no audio decoding or model forwards needed. Defaults to
the *whitened* space, which scores higher for genre.

THREE SCORINGS, because an MTG-Jamendo track is not single-genre. 82% of the
corpus's tracks carry 2-9 genre tags, and the TSV lists them in ALPHABETICAL
order — so `genre` (= tags[0], the original metric) is "the alphabetically first
genre", not a primary one. A neighbor that shares a genuine genre with the query
but not its first letter is scored wrong. So:

  primary   prediction == the track's tags[0]       — the historical number
  any-tag   prediction ∈ the track's full tag set   — tolerant of label ambiguity
  single    primary, on test tracks with exactly ONE tag — clean labels

Each is printed with the chance level it should be read against (a 125-label
problem and a 16-label problem are not comparable by raw accuracy). `single` is
the closest thing to a like-for-like number across corpora: it removes the
ambiguity that differs between label sets.

`genres` (the '|'-joined full tag set) is written by build_chroma; a DB built
before that key existed only supports `primary`, and this says so.

Usage:
    python -m music_encoding.test_knn [--db-dir chroma_db] [--k 50] [--space whitened|raw]
"""

import argparse
from collections import Counter

import numpy as np

from music_encoding.db import COLLECTION_RAW, COLLECTION_WHITENED, load_all


def tag_sets(metas: list[dict]) -> list[frozenset]:
    """Each row's full genre tag set, falling back to {tags[0]} on old DBs."""
    out: list[frozenset] = []
    for m in metas:
        joined = m.get("genres")
        if joined:
            out.append(frozenset(joined.split("|")))
        else:
            g = m.get("genre") or ""
            out.append(frozenset([g]) if g else frozenset())
    return out


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
    tagsets = tag_sets(metas)
    have_tagsets = any(m.get("genres") for m in metas)
    if not have_tagsets:
        print(
            "[main] NOTE: this DB predates the `genres` key, so any-tag scoring "
            "collapses to primary. Rebuild with build_chroma for the real numbers."
        )
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
    test_primary = genres[test_idx]
    print("[main] voting over neighbors")
    predicted = [
        Counter(train_genres[row].tolist()).most_common(1)[0][0] for row in topk_idx
    ]

    # --- three scorings, each against its own chance level -------------------
    primary_hit = np.array([p == t for p, t in zip(predicted, test_primary, strict=True)])
    any_hit = np.array(
        [p in tagsets[i] for p, i in zip(predicted, test_idx, strict=True)]
    )
    single = np.array([len(tagsets[i]) == 1 for i in test_idx])

    n_labels = len({g for g in genres if g})
    # chance for any-tag: if predictions were drawn from the train label
    # distribution, this is the probability of landing in the query's tag set
    freq = Counter(train_genres)
    chance_any = float(
        np.mean([sum(freq.get(g, 0) for g in tagsets[i]) / len(train_idx) for i in test_idx])
    )

    print()
    print(f"[main] === genre k-NN accuracy (k={k}, {args.space}, n_test={len(test_idx)}) ===")
    print(
        f"[main] primary (tags[0], the historical metric): {primary_hit.mean():.3f}   "
        f"chance {1 / n_labels:.3f}  ({n_labels} labels)"
    )
    print(
        f"[main] any-tag (pred in the track's tag set)  : {any_hit.mean():.3f}   "
        f"chance {chance_any:.3f}"
    )
    if single.any():
        print(
            f"[main] single (clean labels, only 1 tag)     : "
            f"{primary_hit[single].mean():.3f}   chance {1 / n_labels:.3f}   "
            f"on {int(single.sum())} of {len(test_idx)} test tracks"
        )
    else:
        print("[main] single: no single-genre test tracks")
    n_multi = sum(1 for i in test_idx if len(tagsets[i]) > 1)
    print(
        f"[main] label ambiguity in the test split: {n_multi}/{len(test_idx)} tracks "
        f"carry >1 genre tag ({n_multi / len(test_idx):.1%})"
    )

    counts = Counter(g for g in genres if g)
    print("[main]", counts.most_common(10))
    print(f"[main] random baseline (1/n_labels): {1 / n_labels:.3f}")


if __name__ == "__main__":
    main()
