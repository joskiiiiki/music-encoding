"""Sample pairs of highly similar tracks and save them as wavs for manual QA.

Ranks pairs of DIFFERENT tracks by their similarity in the whitened embedding
space — read straight from the chroma DB `tracks_whitened` — then writes one
.wav per pair: track A, a 1 s silence, then track B, so the pair is easy to
A/B by ear. The audio is decoded at its native quality (44.1 kHz stereo, not
the 22.05 kHz mono the model's mel uses), so what you hear is the real song.
A manifest of titles/artists/genres accompanies it.

Usage:
    python -m music_encoding.test_similar_pairs [--db-dir chroma_db] [--n-tracks 1000] \
        [--pairs 10] [--out-dir similar_pairs] [--space whitened|raw]
"""

import argparse
import pathlib

import numpy as np
import torch as tc
import torchaudio as ta

from music_encoding.db import COLLECTION_RAW, COLLECTION_WHITENED, load_all
from music_encoding.mtg import MTGJamendoBase
from music_encoding.twin_dataset import load


def load_native(ds, idx: int) -> tuple[tc.Tensor, int]:
    """Full-quality source audio: stereo at the track's native sample rate."""
    samples = load(ds, idx).get_all_samples()
    return samples.data, samples.sample_rate  # (channels, N) float32


def main() -> None:
    parser = argparse.ArgumentParser(
        description="save the most-similar track pairs as wavs for manual QA (from chroma DB)"
    )
    parser.add_argument(
        "--db-dir", default="chroma_db",
        help="Chroma persistent dir (default: chroma_db)",
    )
    parser.add_argument(
        "--n-tracks", type=int, default=1000,
        help="tracks to sample for pair-finding (default: 1000)",
    )
    parser.add_argument(
        "--pairs", type=int, default=10,
        help="number of top pairs to save (default: 10)",
    )
    parser.add_argument(
        "--out-dir", default="similar_pairs",
        help="output dir for wavs + manifest (default: similar_pairs)",
    )
    parser.add_argument(
        "--space", choices=["whitened", "raw"], default="whitened",
        help="similarity space used for ranking (default: whitened)",
    )
    parser.add_argument(
        "--max-raw-sim", type=float, default=0.98,
        help="skip pairs whose RAW cosine sim is >= this (near-duplicates and "
             "same-artist tracks crowd the extreme top; the QA set then shows "
             "varied, genuinely-different tracks). 1.0 keeps everything.",
    )
    parser.add_argument(
        "--mtg-data", required=True, type=pathlib.Path,
        help="MTG-Jamendo data dir (contains autotagging.tsv, raw.meta.tsv)",
    )
    parser.add_argument(
        "--audio-root", default=None, type=pathlib.Path,
        help="dir where MTG audio unpacked (defaults to --mtg-data)",
    )
    args = parser.parse_args()

    print(f"[main] loading embeddings from '{args.db_dir}'")
    _, embs_w, metas = load_all(args.db_dir, COLLECTION_WHITENED)
    _, embs_r, _ = load_all(args.db_dir, COLLECTION_RAW)
    n = len(metas)

    n_samp = min(args.n_tracks, n)
    pick = np.sort(np.random.default_rng(42).choice(n, size=n_samp, replace=False))
    embs_w, embs_r = embs_w[pick], embs_r[pick]
    metas = [metas[i] for i in pick]
    print(f"[main] sampled {n_samp} tracks, ranking in {args.space} space")

    Pw = embs_w / np.linalg.norm(embs_w, axis=1, keepdims=True)
    Pr = embs_r / np.linalg.norm(embs_r, axis=1, keepdims=True)
    S_w = Pw @ Pw.T
    S_r = Pr @ Pr.T
    rank_S = S_w if args.space == "whitened" else S_r

    iu, ju = np.triu_indices(n_samp, 1)
    vals = rank_S[iu, ju]
    raw_vals = S_r[iu, ju]
    # tracks whose raw embedding is near-identical (cos > 0.99) to another
    # sampled track are near-silent/collapsed content — different audio that the
    # raw space maps to one point — so exclude them from pair selection
    np.fill_diagonal(S_r, -1.0)
    collapsed = S_r.max(axis=1) > 0.99
    print(f"[main] {int(collapsed.sum())} collapsed/near-silent tracks excluded from pairs")
    # walk the ranking, skipping collapsed tracks and pairs at/above the raw-sim
    # cutoff, so the QA set shows varied, genuinely-different tracks
    chosen: list[int] = []
    for k in np.argsort(-vals):
        i, j = int(iu[k]), int(ju[k])
        if collapsed[i] or collapsed[j]:
            continue
        if raw_vals[k] < args.max_raw_sim:
            chosen.append(int(k))
            if len(chosen) >= args.pairs:
                break
    topk = np.asarray(chosen)
    print(f"[main] selected {len(topk)} pairs with raw sim < {args.max_raw_sim}")

    print("[main] loading dataset for audio")
    ds = MTGJamendoBase(args.mtg_data, audio_root=args.audio_root)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest: list[str] = []
    same_genre = 0
    for rank, k in enumerate(topk):
        i, j = int(iu[k]), int(ju[k])
        idx_a, idx_b = int(pick[i]), int(pick[j])
        ma, mb = metas[i], metas[j]
        ga, gb = ma.get("genre") or "?", mb.get("genre") or "?"
        art_a, art_b = (ma.get("artist") or ""), (mb.get("artist") or "")
        same_artist = bool(art_a) and art_a == art_b
        same_genre_flag = ga != "?" and ga == gb
        same_track = idx_a == idx_b  # always False here: pairs are distinct tracks
        # tag: 't'/'a'/'g' = same track / artist / genre, 'x' = different
        tag = ("t" if same_track else "x") + ("a" if same_artist else "x") + ("g" if same_genre_flag else "x")

        # full-quality source for listening (native SR, stereo) — the 22.05 kHz
        # mono representation is only what the model's mel sees
        wav_a, sr_a = load_native(ds, idx_a)
        wav_b, sr_b = load_native(ds, idx_b)
        tgt = max(sr_a, sr_b)
        if sr_a < tgt:
            wav_a = ta.functional.resample(wav_a, sr_a, tgt)
        if sr_b < tgt:
            wav_b = ta.functional.resample(wav_b, sr_b, tgt)
        if wav_a.shape[0] != wav_b.shape[0]:  # mismatched channels -> mono both
            wav_a = wav_a.mean(dim=0, keepdim=True)
            wav_b = wav_b.mean(dim=0, keepdim=True)
        gap = tc.zeros((wav_a.shape[0], tgt), dtype=wav_a.dtype)  # 1 s silence
        combined = tc.cat([wav_a, gap, wav_b], dim=1)  # (channels, samples)

        fpath = out / f"pair_{rank:02d}_{S_w[i, j]:.3f}_{tag}.wav"
        ta.save(fpath, combined, tgt)

        same_genre += same_genre_flag
        line = (
            f"{rank:02d} whit={S_w[i, j]:.3f} raw={S_r[i, j]:.3f} [{tag}] | "
            f"{ga:<22} {ma.get('title', '')!r} — {ma.get('artist', '')!r}  vs  "
            f"{gb:<22} {mb.get('title', '')!r} — {mb.get('artist', '')!r} | {fpath.name}"
        )
        manifest.append(line)
        print(f"[main] {line}")

    (out / "pairs.txt").write_text("\n".join(manifest) + "\n")
    n_saved = len(topk)
    print(f"[main] saved {n_saved} wavs (A + 1s gap + B each) + pairs.txt to {out}/")
    print(f"[main] top pairs same-genre: {same_genre}/{n_saved}")


if __name__ == "__main__":
    main()
