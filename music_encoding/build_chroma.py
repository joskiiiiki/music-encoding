"""Populate a Chroma vector DB with per-track embeddings + metadata.

Embeds every track's 5 windows (raw), stores their mean-pooled 128-d vector in
raw and corpus-wide-whitened form, and keeps the per-window raw vectors too. All
land in persistent Chroma collections under <db-dir>/ with per-track metadata
(idx, title, artist, primary genre, album, release year, listens). The eval
scripts can then sample / filter / query this DB instead of recomputing
embeddings.

Chroma uses cosine distance for nearest-neighbour queries (whitened space is the
informative one per the similarity diagnostics; raw is kept for the k-NN genre
eval, which operates on raw cosine).

Usage:
    python -m music_encoding.build_chroma checkpoints/checkpoint_e30.pt \
        [--cache-dir cached] [--db-dir chroma_db] [--limit N]
"""

import argparse
import pathlib
import random
from collections import defaultdict

import chromadb
import numpy as np
import torch as tc

from music_encoding.model import SiameseEncoderBT
from music_encoding.mtg import MTGJamendoBase
from music_encoding.twin_dataset import (
    DEFAULT_MIN_OFF_S,
    DEFAULT_TGT_SR,
    DEFAULT_WIN_LEN_S,
    LogMelSpectrogram,
    load_resampled_cached,
    random_window,
)

mel = LogMelSpectrogram()  # plain log-mel (no RRC) for eval

RAW_COLLECTION = "tracks_raw"
WHITENED_COLLECTION = "tracks_whitened"
WINDOWS_COLLECTION = "windows_raw"  # the 5 raw per-window embeddings per track
UPSERT_CHUNK = 2000


def extract_embs(
    model: SiameseEncoderBT,
    ds,
    track_idxs,
    batch_size: int = 256,
    device: str = "cuda",
    tgt_sr: int = DEFAULT_TGT_SR,
    win_s: float = DEFAULT_WIN_LEN_S,
    min_off_s: float = DEFAULT_MIN_OFF_S,
    win_per_track: int = 5,
    cache_dir: str = "cached",
):
    """Per-window embeddings (N, W, D) from cached resampled waveforms, window
    forwards batched across tracks. Returns (unique_track_ids, per_track_embs)."""
    model.eval()
    win_length = int(tgt_sr * win_s)
    min_offset = int(tgt_sr * min_off_s)
    cache = pathlib.Path(cache_dir)

    all_embs: list[tc.Tensor] = []
    all_track_ids: list = []
    buf_wavs: list[tc.Tensor] = []
    buf_ids: list = []

    def flush():
        if not buf_wavs:
            return
        with tc.no_grad():
            wavs = tc.stack(buf_wavs).to(device)
            emb, _ = model.forward(mel(wavs))
            all_embs.append(emb.cpu())
        all_track_ids.extend(buf_ids)
        buf_wavs.clear()
        buf_ids.clear()

    for n, t_idx in enumerate(track_idxs):
        wav_full = load_resampled_cached(ds, int(t_idx), tgt_sr, cache)
        for _ in range(win_per_track):
            wav, _ = random_window(wav_full, win_length, min_offset)
            buf_wavs.append(wav)
            buf_ids.append(int(t_idx))
            if len(buf_wavs) >= batch_size:
                flush()
        del wav_full
        if (n + 1) % 1000 == 0:
            print(f"[extract_embs] {n + 1}/{len(track_idxs)} tracks processed", flush=True)

    flush()
    all_embs_t = tc.cat(all_embs, dim=0)
    track_to_embs: dict = defaultdict(list)
    for tid, emb in zip(all_track_ids, all_embs_t, strict=True):
        track_to_embs[tid].append(emb)
    unique_ids = list(track_to_embs.keys())
    per_track = tc.stack([tc.stack(track_to_embs[tid]) for tid in unique_ids])
    print(f"[extract_embs] done: {len(unique_ids)} unique tracks, shape={tuple(per_track.shape)}", flush=True)
    return unique_ids, per_track


def whiten(X: tc.Tensor, eps: float = 1e-4) -> tc.Tensor:
    """ZCA-whiten a (N, D) embedding matrix: center, decorrelate, unit-scale."""
    mean = X.mean(dim=0)
    cov = ((X - mean).T @ (X - mean)) / (X.shape[0] - 1)
    eigvals, eigvecs = tc.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=eps)
    W = eigvecs @ tc.diag(1.0 / eigvals.sqrt()) @ eigvecs.T  # C^{-1/2}
    return (X - mean) @ W


def build_metadata(ds, n: int) -> list[dict]:
    """Metadata for tracks 0..n-1 from the MTG loader rows (no audio decode)."""
    metas: list[dict] = []
    for i in range(n):
        r = ds.row(i)
        meta = {
            "idx": i,
            "title": r["title"],
            "artist": r["artist"],
            "album": r["album"],
            "track_id": r["track_id"],
            "genre": r["genre"] or "",
        }
        if r["instrument"]:
            meta["instrument"] = r["instrument"]
        if r["mood_theme"]:
            meta["mood_theme"] = r["mood_theme"]
        if r["released"]:
            meta["released"] = int(r["released"][:4])  # 'YYYY-MM-DD' -> year
        metas.append(meta)
    return metas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="populate a Chroma vec DB with track embeddings + metadata"
    )
    parser.add_argument("checkpoint", help="path to a model checkpoint .pt")
    parser.add_argument(
        "--cache-dir", default="cached_mtg",
        help="dir of resampled full-track waveforms (default: cached_mtg)",
    )
    parser.add_argument(
        "--mtg-data", required=True, type=pathlib.Path,
        help="MTG-Jamendo data dir (contains autotagging.tsv, raw.meta.tsv)",
    )
    parser.add_argument(
        "--audio-root", default=None, type=pathlib.Path,
        help="dir where MTG audio unpacked (defaults to --mtg-data)",
    )
    parser.add_argument(
        "--db-dir", default="chroma_db",
        help="Chroma persistent directory (default: chroma_db)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap on how many tracks to embed (handy for testing)",
    )
    args = parser.parse_args()

    device = "cuda"
    print(f"[main] loading checkpoint from {args.checkpoint}")
    checkpoint = tc.load(args.checkpoint, map_location=device)
    proj_dims = checkpoint["model_state_dict"]["projector.6.weight"].shape[0]
    model = SiameseEncoderBT(proj_dims=proj_dims).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("[main] loading MTG-Jamendo dataset")
    ds = MTGJamendoBase(args.mtg_data, audio_root=args.audio_root)

    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    track_idx = np.arange(n)
    print(f"[main] embedding {n} tracks")

    random.seed(42)  # deterministic windows -> reproducible embeddings
    ids, embs = extract_embs(model, ds, track_idx, device=device, cache_dir=args.cache_dir)
    pooled = embs.mean(dim=1)  # (N, D) raw mean-pooled embeddings
    print(f"[main] embeddings shape={tuple(pooled.shape)}")

    whitened = whiten(pooled)  # corpus-wide ZCA transform
    print("[main] whitening computed over the full corpus")

    ids_str = [f"track_{int(i)}" for i in ids]
    metadatas = build_metadata(ds, n)

    # per-window raw collection: 5 windows per track, ids track_<idx>_w<k>
    n_windows = embs.shape[1]
    win_ids: list[str] = []
    win_embs: list[list[float]] = []
    win_metas: list[dict] = []
    for i in range(n):
        for k in range(n_windows):
            win_ids.append(f"track_{int(ids[i])}_w{k}")
            win_embs.append(embs[i, k].tolist())
            m = dict(metadatas[i])
            m["window"] = k
            win_metas.append(m)

    client = chromadb.PersistentClient(path=args.db_dir)
    for name, emb in [
        (WHITENED_COLLECTION, whitened.tolist()),
        (RAW_COLLECTION, pooled.tolist()),
        (WINDOWS_COLLECTION, win_embs),
    ]:
        col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        ids_use = win_ids if name == WINDOWS_COLLECTION else ids_str
        metas_use = win_metas if name == WINDOWS_COLLECTION else metadatas
        for s in range(0, len(ids_use), UPSERT_CHUNK):
            col.upsert(
                ids=ids_use[s : s + UPSERT_CHUNK],
                embeddings=emb[s : s + UPSERT_CHUNK],
                metadatas=metas_use[s : s + UPSERT_CHUNK],
            )
        print(f"[main] '{name}': {col.count()} vectors in {args.db_dir}/")


if __name__ == "__main__":
    main()
