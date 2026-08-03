import sys
import time
from collections import Counter, defaultdict

import numpy as np
import numpy.typing as npt
import torch as tc
from datasets import Dataset, load_dataset

from music_encoding.model import SiameseEncoderBT
from music_encoding.twin_dataset import (
    DEFAULT_MIN_OFF_S,
    DEFAULT_TGT_SR,
    DEFAULT_WIN_LEN_S,
    load,
    random_window,
    resample,
)


def extract_embs(
    model: SiameseEncoderBT,
    ds: Dataset,
    track_idxs: npt.NDArray[np.int64],
    batch_size: int = 256,
    device: str = "cuda",
    tgt_sr: int = DEFAULT_TGT_SR,
    win_s: float = DEFAULT_WIN_LEN_S,
    min_off_s: float = DEFAULT_MIN_OFF_S,
    win_per_track: int = 5,
) -> tuple[list[np.int64], tc.Tensor]:
    model.eval()
    win_length = int(tgt_sr * win_s)
    min_offset = int(tgt_sr * min_off_s)

    total_tracks = len(track_idxs)
    print(f"[extract_embs] start: {total_tracks} tracks, win_per_track={win_per_track}, batch_size={batch_size}")
    t0 = time.monotonic()

    all_embs: list[tc.Tensor] = []
    all_track_ids: list[np.int64] = []

    buf_wavs: list[tc.Tensor] = []
    buf_ids: list[np.int64] = []

    def flush():
        if not buf_wavs:
            return
        with tc.no_grad():
            wavs = tc.stack(buf_wavs).to(device)
            emb, _ = model.forward(wavs)
            all_embs.append(emb.cpu())
        all_track_ids.extend(buf_ids)
        buf_wavs.clear()
        buf_ids.clear()

    for n, t_idx in enumerate(track_idxs):
        audio = load(ds, t_idx).get_all_samples()
        wav_full = resample(audio.data, audio.sample_rate, tgt_sr)
        for _ in range(win_per_track):
            wav, _ = random_window(wav_full, win_length, min_offset)
            buf_wavs.append(wav)
            buf_ids.append(t_idx)
            if len(buf_wavs) >= batch_size:
                flush()
        del audio, wav_full  # let the full-track tensor go before next iteration
        if (n + 1) % 100 == 0:
            print(f"[extract_embs] {n + 1}/{total_tracks} tracks processed", flush=True)

    flush()  # remaining partial batch

    all_embs_t = tc.cat(all_embs, dim=0)
    track_to_embs: defaultdict[np.int64, list[tc.Tensor]] = defaultdict(list)
    for tid, emb in zip(all_track_ids, all_embs_t, strict=True):
        track_to_embs[tid].append(emb)

    unique_track_ids = list(track_to_embs.keys())
    per_track_embs = tc.stack(
        [tc.stack(track_to_embs[tid]).mean(dim=0) for tid in unique_track_ids]
    )
    print(
        f"[extract_embs] done: {len(unique_track_ids)} unique tracks, "
        f"emb shape={tuple(per_track_embs.shape)}, took {time.monotonic() - t0:.1f}s",
        flush=True,
    )
    return unique_track_ids, per_track_embs

def primary_genre(genre_list: list) -> str | None:
    return genre_list[0] if len(genre_list) > 0 else None


def main() -> None:
    device = "cuda"
    checkpoint_path = sys.argv[1]
    print(f"[main] loading checkpoint from {checkpoint_path}")

    model = SiameseEncoderBT(proj_dims=2048).to(device)
    checkpoint = tc.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[main] model loaded and set to eval mode")

    print("[main] loading dataset 'benjamin-paine/free-music-archive-small'")
    ds: Dataset = load_dataset("benjamin-paine/free-music-archive-small")["train"]
    print(f"[main] dataset loaded: {len(ds)} tracks")

    track_idx = np.arange(len(ds))
    rng = np.random.default_rng(42)
    rng.shuffle(track_idx)

    split = int(0.8 * len(track_idx))
    train_track_idx = track_idx[:split]
    test_track_idx = track_idx[split:]
    print(f"[main] split: {len(train_track_idx)} train / {len(test_track_idx)} test tracks")

    print("[main] extracting train embeddings")
    train_ids, train_embs = extract_embs(model, ds, train_track_idx, device=device)
    print("[main] extracting test embeddings")
    test_ids, test_embs = extract_embs(model, ds, test_track_idx, device=device)

    # normalize for cosine similarity — keepdim=True is required here
    print("[main] normalizing embeddings")
    train_norm = train_embs / train_embs.norm(dim=1, keepdim=True)
    test_norm = test_embs / test_embs.norm(dim=1, keepdim=True)

    print("[main] computing cosine similarity matrix")
    sim = test_norm @ train_norm.T  # (N_test, N_train)

    k = 10
    print(f"[main] computing top-{k} neighbors")
    topk_sim, topk_idx = sim.topk(k, dim=1)  # topk_idx: positions WITHIN train_embs

    # translate positions-in-train_embs back to real ds track indices
    train_ids_arr = np.array(train_ids)
    neighbor_track_ids = train_ids_arr[topk_idx.cpu().numpy()]  # (N_test, k)
    print("[main] translated neighbor positions to ds track indices")

    # batch-fetch all genres once instead of one ds[...] call per lookup

    print("[main] loading genre labels")
    all_genres = ds["genres"]   

    print("[main] fetching genres for neighbors")
    flat_neighbor_ids = neighbor_track_ids.flatten().tolist()
    flat_neighbor_genres = [all_genres[i] for i in flat_neighbor_ids]
    neighbor_genres_grid = np.array(
        [primary_genre(g) for g in flat_neighbor_genres], dtype=object
    ).reshape(neighbor_track_ids.shape)    

    print("[main] fetching true genres for test tracks")
    test_true_genres = [all_genres[i] for i in test_ids]
    true_primary_genres = [primary_genre(g) for g in test_true_genres]

    print("[main] voting over neighbors")
    predicted = []
    for row in neighbor_genres_grid:
        vote = Counter(row.tolist()).most_common(1)[0][0]
        predicted.append(vote)

    correct = sum(p == t for p, t in zip(predicted, true_primary_genres, strict=True))
    accuracy = correct / len(true_primary_genres)
    print(f"[main] k-NN genre accuracy (k={k}): {accuracy:.3f} on {len(true_primary_genres)} test tracks")

    genre_counts = Counter(primary_genre(g) for g in all_genres)
    print(genre_counts.most_common())
    n_genres = len(genre_counts)
    print(f"random baseline: {1/n_genres:.3f}")

if __name__ == "__main__":
    main()
