import sys
import time
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
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
        [tc.stack(track_to_embs[tid]) for tid in unique_track_ids]
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
    print("[main] extracting embeddings")
    ids, embs = extract_embs(model, ds, track_idx, device=device)
    # embs: (N_tracks, win_per_track, embed_dim)

    embs = tc.nn.functional.normalize(embs, dim=-1)
    n_tracks, n_windows, embed_dim = embs.shape

    # --- intra-track similarity: pairwise cosine sim among the windows of the SAME track ---
    # (N, W, D) @ (N, D, W) -> (N, W, W)
    intra_sim = tc.bmm(embs, embs.transpose(1, 2))
    # mask out the diagonal (self-similarity with itself is trivially 1.0)
    eye = tc.eye(n_windows, dtype=tc.bool)
    intra_vals = intra_sim[:, ~eye].reshape(n_tracks, -1)  # off-diagonal entries per track
    intra_mean = intra_vals.mean().item()
    intra_std = intra_vals.std().item()

    # --- inter-track similarity: compare one window from each track against a window from a DIFFERENT random track ---
    rng = np.random.default_rng(42)
    other_track = rng.permutation(n_tracks)
    # ensure no track is "paired with itself" by this shuffle
    same = other_track == np.arange(n_tracks)
    other_track[same] = (other_track[same] + 1) % n_tracks

    query_window = embs[:, 0, :]                      # (N, D) — first window of each track
    other_window = embs[tc.from_numpy(other_track), 0, :]  # (N, D) — first window of a different track
    inter_vals = (query_window * other_window).sum(dim=-1)  # cosine sim, since already normalized
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    
    ax1.hist(intra_vals)
    ax2.hist(inter_vals)

    plt.show()
    
    inter_mean = inter_vals.mean().item()
    inter_std = inter_vals.std().item()

    print(f"[main] intra-track similarity: mean={intra_mean:.4f}, std={intra_std:.4f}")
    print(f"[main] inter-track similarity: mean={inter_mean:.4f}, std={inter_std:.4f}")
    print(f"[main] gap (intra - inter): {intra_mean - inter_mean:.4f}")    
 
if __name__ == "__main__":
    main()
