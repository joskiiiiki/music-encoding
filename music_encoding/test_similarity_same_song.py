"""Intra/inter-track similarity (embedding collapse) from the Chroma vec DB.

Loads the raw per-window embeddings from `windows_raw` (built by build_chroma.py),
groups them into (N, W, D), and computes within-track pairwise cosine sim (intra)
vs between-random-tracks sim (inter), plotting histograms + printing means/stds
and the gap. Also applies a post-hoc ZCA whitening and replots — a big separation
gain means the raw space was just poorly conditioned, not that the model failed.

Usage:
    python -m music_encoding.test_similarity_same_song [--db-dir chroma_db]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch as tc

from music_encoding.db import COLLECTION_WINDOWS, load_all


def compute_sims(
    embs: tc.Tensor, n_windows: int
) -> tuple[tc.Tensor, tc.Tensor]:
    """Intra- and inter-track cosine-sim distributions from (N, W, D) embeddings
    (already L2-normalized along D)."""
    n_tracks = embs.shape[0]

    # intra: pairwise cosine sim among the windows of the SAME track
    # (N, W, D) @ (N, D, W) -> (N, W, W), then drop the diagonal
    intra_sim = tc.bmm(embs, embs.transpose(1, 2))
    eye = tc.eye(n_windows, dtype=tc.bool)
    intra_vals = intra_sim[:, ~eye].reshape(n_tracks, -1)  # off-diagonal entries per track

    # inter: first window of each track vs first window of a DIFFERENT track
    rng = np.random.default_rng(42)
    other_track = rng.permutation(n_tracks)
    same = other_track == np.arange(n_tracks)
    other_track[same] = (other_track[same] + 1) % n_tracks

    query_window = embs[:, 0, :]                                  # (N, D)
    other_window = embs[tc.from_numpy(other_track), 0, :]         # (N, D)
    inter_vals = (query_window * other_window).sum(dim=-1)        # cosine sim (already normalized)
    return intra_vals, inter_vals


def whiten_embs(embs_raw: tc.Tensor, eps: float = 1e-4) -> tc.Tensor:
    """ZCA-whiten the (N, W, D) embeddings: center on the track-level mean, then
    decorrelate + unit-scale each dimension via C^{-1/2} estimated from per-track
    pooled embeddings. Returns whitened embeddings L2-normalized along D.
    (Rotation by an orthogonal matrix leaves cosine sim unchanged, so this yields
    the same intra/inter metrics as PCA whitening.)"""
    N, _, D = embs_raw.shape
    pooled = embs_raw.mean(dim=1)                          # one embedding per track
    mean = pooled.mean(dim=0)                              # (D,)
    cov = ((pooled - mean).T @ (pooled - mean)) / (N - 1)  # (D, D)
    eigvals, eigvecs = tc.linalg.eigh(cov)                 # ascending eigenvalues
    eigvals = eigvals.clamp(min=eps)
    W = eigvecs @ tc.diag(1.0 / eigvals.sqrt()) @ eigvecs.T  # C^{-1/2}
    whitened = (embs_raw - mean) @ W
    return tc.nn.functional.normalize(whitened, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="intra/inter-track similarity from the chroma vec DB"
    )
    parser.add_argument(
        "--db-dir", default="chroma_db",
        help="Chroma persistent dir (default: chroma_db)",
    )
    args = parser.parse_args()

    print(f"[main] loading windows from '{args.db_dir}/{COLLECTION_WINDOWS}'")
    ids, embs, _ = load_all(args.db_dir, COLLECTION_WINDOWS)

    idxs = np.array([int(i.split("_")[1]) for i in ids])
    n_tracks = len(np.unique(idxs))
    n_windows = len(ids) // n_tracks
    embs = tc.from_numpy(embs).reshape(n_tracks, n_windows, -1)
    print(f"[main] loaded {n_tracks} tracks x {n_windows} windows, emb shape={tuple(embs.shape)}")

    # keep the raw (pre-L2-norm) embeddings for the whitening diagnostic
    embs_raw = embs
    embs_norm = tc.nn.functional.normalize(embs_raw, dim=-1)
    embs_white = whiten_embs(embs_raw)

    intra_vals, inter_vals = compute_sims(embs_norm, n_windows)
    intra_white, inter_white = compute_sims(embs_white, n_windows)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    for ax, (iv, jv), title in [
        (ax1, (intra_vals, inter_vals), "raw embeddings"),
        (ax2, (intra_white, inter_white), "post-hoc whitened"),
    ]:
        # mean intra-track sim per track (5 windows -> 20 pairwise values each),
        # so one value per track — otherwise hist groups the 20 columns as series
        ax.hist(iv.mean(dim=1).numpy(), bins=50, alpha=0.6, label="intra-track (same song)")
        ax.hist(jv.numpy(), bins=50, alpha=0.6, label="inter-track (different songs)")
        ax.set_title(title)
        ax.set_xlabel("cosine similarity")
        ax.set_ylabel("count")
        ax.legend()
    plt.tight_layout()
    plt.show()

    for name, iv, jv in [
        ("raw", intra_vals, inter_vals),
        ("whitened", intra_white, inter_white),
    ]:
        im = iv.mean().item()
        istd = iv.std().item()
        jm = jv.mean().item()
        jstd = jv.std().item()
        print(f"[main] [{name}] intra-track similarity: mean={im:.4f}, std={istd:.4f}")
        print(f"[main] [{name}] inter-track similarity: mean={jm:.4f}, std={jstd:.4f}")
        print(f"[main] [{name}] gap (intra - inter): {im - jm:.4f}")


if __name__ == "__main__":
    main()
