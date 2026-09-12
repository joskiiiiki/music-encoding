"""Populate a Chroma vector DB with per-track embeddings + metadata.

Embeds every track's N windows, stores their mean-pooled vector in raw and
corpus-wide-whitened form, and keeps the per-window raw vectors too. All land in
persistent Chroma collections under <db-dir>/ with per-track metadata (idx,
track_num, track_id, title, artist, primary genre, album, release year). The eval
scripts then sample / filter / query this DB instead of recomputing embeddings, so
re-running an eval is seconds rather than a GPU job.

Two input modes, matching train.py's:

  --mel-root    precomputed MTG-Jamendo log-mel .npy spectrograms: the windows are
                cut straight out of the spectrograms, with no audio decode and no
                mel conversion. This is the mode training runs in, and the only one
                that works on this machine (the MTG audio is still tarballs).
  (default)     decoded MTG-Jamendo audio, resampled + cached, mel computed here.

--mtg-data is required in BOTH modes: in mel mode it supplies metadata only, joined
by MTG track number (the `.npy` stem) rather than by position.

Chroma uses cosine distance for nearest-neighbour queries (whitened space is the
informative one per the similarity diagnostics; raw is kept for the k-NN genre
eval, which operates on raw cosine).

Usage:
    # mel mode — the training path (pass --n-mels to match the run)
    python -m music_encoding.build_chroma checkpoints/checkpoint_e30.pt \\
        --mel-root ~/mtg_jamendo --mtg-data ~/mtg-jamendo-dataset/data \\
        --n-mels 64
    # audio mode
    python -m music_encoding.build_chroma checkpoints/checkpoint_e30.pt \\
        --mtg-data <data> --audio-root <audio> [--cache-dir cached_mtg]

Note on writes: collections are UPSERTed, not reset, so a run covering fewer tracks
than the previous one (e.g. --limit) leaves the surplus vectors in place and the
evals will mix generations. A full run over the whole corpus overwrites cleanly.
"""

import argparse
import pathlib
import random
from functools import partial

import chromadb
import torch as tc

from music_encoding.model import SiameseEncoderBT
from music_encoding.mtg import MTGJamendoBase
from music_encoding.twin_dataset import (
    DEFAULT_MIN_OFF_S,
    DEFAULT_MIN_WINDOW_DB,
    DEFAULT_TGT_SR,
    DEFAULT_WIN_LEN_S,
    LogMelSpectrogram,
    MelSpecWindowSource,
    load_resampled_cached,
    random_window,
)

RAW_COLLECTION = "tracks_raw"
WHITENED_COLLECTION = "tracks_whitened"
WINDOWS_COLLECTION = "windows_raw"  # the per-window raw embeddings per track
UPSERT_CHUNK = 2000


def _iter_waveform_windows(
    ds,
    n_tracks: int,
    win_length: int,
    min_offset: int,
    win_per_track: int,
    cache_dir: str,
    tgt_sr: int = DEFAULT_TGT_SR,
):
    """Yield (track_position, waveform_window) from decoded, cached audio."""
    cache = pathlib.Path(cache_dir)
    for t in range(n_tracks):
        wav_full = load_resampled_cached(ds, t, tgt_sr, cache)
        for _ in range(win_per_track):
            wav, _ = random_window(wav_full, win_length, min_offset)
            yield t, wav
        del wav_full
        if (t + 1) % 1000 == 0:
            print(f"[extract_embs] {t + 1}/{n_tracks} tracks processed", flush=True)


def _iter_mel_windows(
    source: MelSpecWindowSource, n_tracks: int, win_per_track: int
):
    """Yield (track_position, (1,1,n_mels,T)) straight from the .npy spectrograms.

    Draws each window independently (no `exclude`), matching the audio path's
    protocol; the near-silence gate inside `random_window` mirrors training's.
    """
    for t in range(n_tracks):
        spec = source.load(source.ids[t])  # (96, T) float32, memmapped
        for _ in range(win_per_track):
            win, _ = source.random_window(spec)
            yield t, win
        del spec
        if (t + 1) % 1000 == 0:
            print(f"[extract_embs] {t + 1}/{n_tracks} tracks processed", flush=True)


def _prepare_waveform(items: list, mel: LogMelSpectrogram, device: str) -> tc.Tensor:
    """(B, N) waveform windows -> (B, 1, n_mels, T): mel stacked on CPU, then moved,
    exactly as the training path does it."""
    return mel(tc.stack(items).to(device))


def _prepare_mel(items: list, device: str) -> tc.Tensor:
    """Mel-mode windows are already (1, 1, n_mels, T); just collate and move."""
    return tc.cat(items).to(device)


def extract_embs(
    model: SiameseEncoderBT,
    windows,
    prepare,
    n_tracks: int,
    win_per_track: int,
    dim: int,
    batch_size: int = 256,
    device: str = "cuda",
) -> tc.Tensor:
    """Per-window embeddings (N, W, D) float32, in track order.

    `windows` yields (track_position, item) and `prepare(items)` maps a list of
    those items to the model's (B, 1, n_mels, T) input on `device` — so the audio
    and mel modes share this whole loop and differ only in where a window comes
    from. Windows are written into a preallocated tensor as they arrive: the 5
    per-window embeddings per track plus the pooled copy would otherwise mean
    several GB of duplicate tensors on a 15 GB box.
    """
    model.eval()
    embs = tc.empty((n_tracks, win_per_track, dim), dtype=tc.float32)
    seen = tc.zeros(n_tracks, dtype=tc.long)
    buf: list = []
    buf_keys: list[int] = []

    def flush() -> None:
        if not buf_keys:
            return
        with tc.no_grad():
            out, _ = model.forward(prepare(buf))
        out = out.float().cpu()
        for row, t in enumerate(buf_keys):
            embs[t, int(seen[t])] = out[row]
            seen[t] += 1
        buf.clear()
        buf_keys.clear()

    for t, item in windows:
        buf.append(item)
        buf_keys.append(t)
        if len(buf) >= batch_size:
            flush()
    flush()

    # a short track or an off-by-one in the iterators would otherwise show up only
    # as a subtly wrong eval number, so fail loudly here instead
    if int(seen.min()) != win_per_track:
        bad = tc.nonzero(seen != win_per_track).flatten()[:5].tolist()
        raise RuntimeError(
            f"tracks {bad} yielded != {win_per_track} windows "
            f"(counts {seen[bad].tolist()})"
        )
    print(
        f"[extract_embs] done: {n_tracks} tracks x {win_per_track} windows, "
        f"shape={tuple(embs.shape)}",
        flush=True,
    )
    return embs


def whiten(X: tc.Tensor, eps: float = 1e-4) -> tc.Tensor:
    """ZCA-whiten a (N, D) embedding matrix: center, decorrelate, unit-scale."""
    mean = X.mean(dim=0)
    cov = ((X - mean).T @ (X - mean)) / (X.shape[0] - 1)
    eigvals, eigvecs = tc.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=eps)
    W = eigvecs @ tc.diag(1.0 / eigvals.sqrt()) @ eigvecs.T  # C^{-1/2}
    return (X - mean) @ W


def build_metadata(ds: MTGJamendoBase, track_nums: list[int]) -> list[dict]:
    """Metadata rows for MTG track numbers, in the order given.

    `idx` is the position in the DB — what the `track_<idx>` ids refer to — while
    `track_num`/`track_id` carry the MTG identity, so a DB row stays traceable to
    the corpus without the TSVs.

    Chroma metadata accepts only str/int/float/bool, so absent strings become "" and
    absent optionals are omitted outright (never written as None).
    """
    metas: list[dict] = []
    for i, num in enumerate(track_nums):
        r = ds.row_by_num(num)
        if r is None:  # not in the TSVs at all; keep the position aligned anyway
            metas.append(
                {
                    "idx": i,
                    "track_num": int(num),
                    "track_id": "",
                    "title": "",
                    "artist": "",
                    "album": "",
                    "genre": "",
                }
            )
            continue
        # the FULL genre tag set, not just tags[0]: MTG's TSV lists tags in
        # alphabetical order, so `genre` alone is an arbitrary label and the genre
        # k-NN eval would score a neighbor wrong for sharing a real genre under a
        # different first letter. '|'-joined because chroma metadata must be scalar.
        genres = r["genres"]
        meta = {
            "idx": i,
            "track_num": int(num),
            "track_id": r["track_id"],
            "title": r["title"] or "",
            "artist": r["artist"] or "",
            "album": r["album"] or "",
            "genre": r["genre"] or "",
            "genres": "|".join(genres),
            "n_genres": len(genres),
        }
        if r["instrument"]:
            meta["instrument"] = r["instrument"]
        if r["mood_theme"]:
            meta["mood_theme"] = r["mood_theme"]
        if r["released"]:
            meta["released"] = int(r["released"][:4])  # 'YYYY-MM-DD' -> year
        metas.append(meta)
    return metas


def _upsert_chunked(col, ids: list, embeddings, metadatas: list | None) -> None:
    """Chunked upsert taking numpy rows, so no chunk is ever materialised as
    Python floats (163k x 1024 of those would be several GB)."""
    for s in range(0, len(ids), UPSERT_CHUNK):
        e = s + UPSERT_CHUNK
        col.upsert(
            ids=ids[s:e],
            embeddings=embeddings[s:e],
            metadatas=None if metadatas is None else metadatas[s:e],
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="populate a Chroma vec DB with track embeddings + metadata"
    )
    parser.add_argument("checkpoint", help="path to a model checkpoint .pt")
    parser.add_argument(
        "--mel-root", default=None, type=pathlib.Path,
        help="dir of precomputed MTG-Jamendo log-mel .npy files, laid out as "
             "<id mod 100>/<id>.npy. When set, EMBED FROM SPECTROGRAMS: no audio "
             "decode and no mel conversion, matching how training runs. Requires "
             "--mtg-data for metadata; incompatible with --audio-root.",
    )
    parser.add_argument(
        "--mtg-data", required=True, type=pathlib.Path,
        help="MTG-Jamendo data dir (contains autotagging.tsv, raw.meta.tsv). Required "
             "in both modes: in mel mode it supplies metadata only.",
    )
    parser.add_argument(
        "--audio-root", default=None, type=pathlib.Path,
        help="dir where MTG audio unpacked (defaults to --mtg-data); audio mode only",
    )
    parser.add_argument(
        "--cache-dir", default="cached_mtg",
        help="dir of resampled full-track waveforms (default: cached_mtg); audio "
             "mode only, ignored with --mel-root",
    )
    parser.add_argument(
        "--n-mels", default=64, type=int,
        help="mel mode: model-input mel bands. MUST match the training run — the "
             "conv stack is resolution-agnostic, so a mismatched value still runs "
             "and silently returns wrong embeddings. Default 64.",
    )
    parser.add_argument(
        "--min-window-db", default=DEFAULT_MIN_WINDOW_DB, type=float,
        help="mel mode: re-draw windows whose mean is below this dB, mirroring "
             f"training's gate (default {DEFAULT_MIN_WINDOW_DB}). Pass < -90 (the "
             "power floor) to disable.",
    )
    parser.add_argument(
        "--win-per-track", default=5, type=int,
        help="windows to embed per track (default: 5; the window evaluations assume "
             "the per-track count is uniform)",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int,
        help="windows per forward pass (default: 256)",
    )
    parser.add_argument(
        "--db-dir", default="chroma_db",
        help="Chroma persistent directory (default: chroma_db)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="cap on how many tracks to embed (handy for testing). Beware: the DB "
             "is upserted, so a limited run leaves the previous run's surplus "
             "vectors in place and the evals will mix generations.",
    )
    args = parser.parse_args()

    if args.mel_root is not None and args.audio_root is not None:
        parser.error(
            "--audio-root is audio-mode only; with --mel-root no audio is decoded"
        )

    device = "cuda"
    print(f"[main] loading checkpoint from {args.checkpoint}")
    checkpoint = tc.load(args.checkpoint, map_location=device)
    proj_dims = checkpoint["model_state_dict"]["projector.6.weight"].shape[0]
    model = SiameseEncoderBT(proj_dims=proj_dims).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    # The DB stores `embed`, not the projector's z: the projection only exists to be
    # decorrelated by the BT loss and is discarded at eval time (same as before).
    embed_dims = model.embed[-1].out_features
    print(f"[main] model ready (embed_dims={embed_dims}, proj_dims={proj_dims})")

    print(f"[main] loading MTG-Jamendo metadata from {args.mtg_data}")
    ds = MTGJamendoBase(args.mtg_data, audio_root=args.audio_root)

    if args.mel_root is not None:
        print(
            f"[main] MEL mode: reading precomputed log-mel .npy under "
            f"{args.mel_root} (no audio decode). n_mels={args.n_mels} — this MUST "
            f"match the training run; gate={args.min_window_db} dB",
            flush=True,
        )
        source = MelSpecWindowSource(
            args.mel_root, n_mels=args.n_mels, min_window_db=args.min_window_db
        )
        n = len(source) if args.limit is None else min(args.limit, len(source))
        track_nums = source.ids[:n]
        windows = _iter_mel_windows(source, n, args.win_per_track)
        prepare = partial(_prepare_mel, device=device)
    else:
        n = len(ds) if args.limit is None else min(args.limit, len(ds))
        track_nums = ds.track_nums[:n]
        mel = LogMelSpectrogram()
        windows = _iter_waveform_windows(
            ds,
            n,
            win_length=int(DEFAULT_TGT_SR * DEFAULT_WIN_LEN_S),
            min_offset=int(DEFAULT_TGT_SR * DEFAULT_MIN_OFF_S),
            win_per_track=args.win_per_track,
            cache_dir=args.cache_dir,
        )
        prepare = partial(_prepare_waveform, mel=mel, device=device)

    if args.limit is not None:
        print(
            "[main] WARNING: --limit set. The DB is upserted, not reset, so any "
            "tracks beyond this limit from an earlier run stay in the collections "
            "and the evals will mix generations. Use a scratch --db-dir for tests.",
            flush=True,
        )
    print(f"[main] embedding {n} tracks")

    ids_str = [f"track_{i}" for i in range(n)]
    metadatas = build_metadata(ds, track_nums)
    if len(metadatas) != n:
        raise RuntimeError(f"metadata/track mismatch: {len(metadatas)} vs {n}")

    random.seed(42)  # deterministic windows -> reproducible embeddings
    embs = extract_embs(
        model,
        windows,
        prepare,
        n_tracks=n,
        win_per_track=args.win_per_track,
        dim=embed_dims,
        batch_size=args.batch_size,
        device=device,
    )

    pooled = embs.mean(dim=1)  # (N, D) raw mean-pooled embeddings
    whitened = whiten(pooled)  # corpus-wide ZCA transform
    print("[main] whitening computed over the full corpus")

    # NOTE on peak memory: the embeddings go to Chroma as float32 numpy ROWS
    # (views into `embs`), never as Python float lists. At 32.7k tracks x 5 windows
    # x 1024 dims, `.tolist()` alone would be ~5 GB on a box with ~3 GB free.
    client = chromadb.PersistentClient(path=args.db_dir)

    for name, emb in (
        (WHITENED_COLLECTION, whitened.numpy()),
        (RAW_COLLECTION, pooled.numpy()),
    ):
        col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        _upsert_chunked(col, ids_str, emb, metadatas)
        print(f"[main] '{name}': {col.count()} vectors in {args.db_dir}/")

    # per-window raw collection: rows are (track, window), ids track_<idx>_w<k>
    W = args.win_per_track
    win_flat = embs.reshape(n * W, embed_dims).numpy()
    col = client.get_or_create_collection(
        WINDOWS_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    for s in range(0, n * W, UPSERT_CHUNK):
        e = s + UPSERT_CHUNK
        rows = range(s, min(e, n * W))
        col.upsert(
            ids=[f"track_{r // W}_w{r % W}" for r in rows],
            embeddings=win_flat[s:e],
            metadatas=[{**metadatas[r // W], "window": r % W} for r in rows],
        )
    print(f"[main] '{WINDOWS_COLLECTION}': {col.count()} vectors in {args.db_dir}/")


if __name__ == "__main__":
    main()
