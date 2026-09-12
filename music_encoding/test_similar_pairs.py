"""QA the top-ranked track pairs — as spectrograms, or as listenable wavs.

Ranks pairs of DIFFERENT tracks by their similarity in the whitened embedding space
(read straight from the chroma DB `tracks_whitened`), then renders the most similar
pairs so a human can judge whether the model is actually finding related music.

Ranks the WHOLE corpus by default (in query blocks, so the n^2 similarity matrix is
never materialised), because a sample would not surface the corpus's actual top pairs.

Three things are excluded from the QA set, each reported so its composition is
visible rather than implied:
  * collapsed tracks — raw cosine >= `--collapse-sim` to some other track, i.e.
    near-silent/degenerate content that the raw space maps to one point
  * near-duplicate audio — raw cosine >= `--max-raw-sim`, i.e. the same recording
    re-uploaded, which tells you nothing about the model
  * same-artist pairs — unless `--allow-same-artist`; the same act twice dominates
    the extreme top of the ranking, so excluding it is what makes the set show whether
    the model finds ORDINARY-similarity related music

Two render modes, because this machine's MTG audio is still tarballs:

  --mel-root    (works here) one PNG per pair: the full log-mel of track A above
                track B, on a shared dB scale. This is the same representation the
                model sees, so it is the honest check of what the embedding is
                grouping together.
  --mtg-data    one .wav per pair at native quality (44.1 kHz stereo), track A,
                a 1 s gap, then track B, so the pair can be A/B'd by ear.

Either way a `pairs.txt` manifest of titles/artists/genres accompanies the output.

Usage:
    python -m music_encoding.test_similar_pairs --mel-root ~/mtg_jamendo
    python -m music_encoding.test_similar_pairs --mtg-data <data> --audio-root <audio>
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torchaudio as ta

from music_encoding.db import COLLECTION_RAW, COLLECTION_WHITENED, load_all
from music_encoding.mtg import MTGJamendoBase, track_num
from music_encoding.twin_dataset import MEL_FLOOR_DB, load, mel_spec_path


def load_native(ds, idx: int) -> tuple[tc.Tensor, int]:
    """Full-quality source audio: stereo at the track's native sample rate."""
    samples = load(ds, idx).get_all_samples()
    return samples.data, samples.sample_rate  # (channels, N) float32


def _track_number(meta: dict) -> int:
    """A DB row's MTG track number.

    `track_num` is written by build_chroma (both modes); fall back to parsing
    `track_id` so DBs built before that key existed still resolve.
    """
    if "track_num" in meta:
        return int(meta["track_num"])
    return track_num(meta["track_id"])


def _label(meta: dict) -> str:
    return (
        f"{meta.get('genre') or '?'} | {meta.get('title') or ''} "
        f"— {meta.get('artist') or ''}"
    )


def rank_pairs(
    Pw: np.ndarray, Pr: np.ndarray, per_block: int, block: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Candidate similar pairs over the WHOLE corpus, scored a query block at a time.

    Ranking the corpus against itself needs n^2 similarities — 32,783^2 float32 is
    4.3 GB, too much to hold — so score `block` queries at a time against everything
    and keep the best `per_block` pairs from each block. That is exact for the global
    top-`per_block`: every global top pair lies inside its own query block's
    top-`per_block`, so merging the per-block sets cannot miss one.

    Also returns each track's maximum raw similarity to ANY other track, computed in
    the same pass. Every track is a query in exactly one block, so that maximum is
    exact and needs no second pass — it is what identifies collapsed tracks, and
    doing it here avoids the old code's subset-only estimate (whose count moved with
    `--n-tracks` and so was not comparable between runs).

    Returns (i, j, whitened_sim, raw_sim, max_raw) with i < j, candidates unsorted.
    """
    n = Pw.shape[0]
    ci, cj, cw, cr = [], [], [], []
    max_raw = np.full(n, -1.0, dtype=np.float32)
    for s in range(0, n, block):
        e = min(s + block, n)
        Sw = Pw[s:e] @ Pw.T  # (b, n)
        Sr = Pr[s:e] @ Pr.T
        # exclude self and the already-scored lower triangle, so only i < j survive
        for r in range(e - s):
            cut = s + r + 1
            Sw[r, :cut] = -1.0
            Sr[r, :cut] = -1.0
        max_raw[s:e] = Sr.max(axis=1)

        take = min(per_block, Sw.size)
        flat = Sw.ravel()
        idx = np.argpartition(-flat, take - 1)[:take]
        ci.append(s + idx // n)
        cj.append(idx % n)
        cw.append(flat[idx])
        cr.append(Sr.ravel()[idx])
    return (
        np.concatenate(ci),
        np.concatenate(cj),
        np.concatenate(cw),
        np.concatenate(cr),
        max_raw,
    )


def write_spec_pair(
    out: pathlib.Path, rank: int, root, ma: dict, mb: dict, sim: float, tag: str
) -> pathlib.Path:
    """Full log-mel of A above B, on a shared dB scale so they are comparable."""
    specs = [
        np.load(mel_spec_path(root, _track_number(m)), mmap_mode="r") for m in (ma, mb)
    ]
    # one shared dB scale, otherwise the two rows can't be compared visually
    vmax = max(float(s.max()) for s in specs)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    for ax, meta, spec in zip(axes, (ma, mb), specs, strict=True):
        ax.imshow(
            spec,
            aspect="auto",
            origin="lower",  # mel band 0 at the bottom
            vmin=MEL_FLOOR_DB,
            vmax=vmax,
            cmap="magma",
        )
        ax.set_title(_label(meta), fontsize=9)
        ax.set_ylabel("mel band")
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fpath = out / f"pair_{rank:02d}_{sim:.3f}_{tag}.png"
    fig.savefig(fpath, dpi=110)
    plt.close(fig)
    return fpath


def write_wav_pair(
    out: pathlib.Path, rank: int, ds, idx_a: int, idx_b: int, sim: float, tag: str
) -> pathlib.Path:
    """A + 1 s of silence + B at native quality, so the pair can be A/B'd by ear."""
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
    fpath = out / f"pair_{rank:02d}_{sim:.3f}_{tag}.wav"
    ta.save(fpath, combined, tgt)
    return fpath


def main() -> None:
    parser = argparse.ArgumentParser(
        description="render the most-similar track pairs for manual QA (from chroma DB)"
    )
    parser.add_argument(
        "--db-dir", default="chroma_db",
        help="Chroma persistent dir (default: chroma_db)",
    )
    parser.add_argument(
        "--n-tracks", type=int, default=0,
        help="limit to a random sample of this many tracks (default: 0 = the whole "
             "corpus). The corpus is ranked in query blocks, so scanning all 32,783 "
             "costs ~1 GB of scratch and a few seconds — and a sample would not give "
             "the corpus's actual top pairs (a 1,000-track sample is 3%% of it).",
    )
    parser.add_argument(
        "--pairs", type=int, default=10,
        help="number of top pairs to save (default: 10)",
    )
    parser.add_argument(
        "--out-dir", default="similar_pairs",
        help="output dir for PNGs/wavs + manifest (default: similar_pairs)",
    )
    parser.add_argument(
        "--space", choices=["whitened", "raw"], default="whitened",
        help="similarity space used for ranking (default: whitened)",
    )
    parser.add_argument(
        "--max-raw-sim", type=float, default=0.995,
        help="skip pairs whose RAW cosine sim is >= this — near-duplicate audio "
             "(the same recording re-uploaded) is uninformative QA. Default 0.995 "
             "and not the old 0.98: the e1000 model's GENUINE cross-track pairs sit "
             "at 0.97+, so an absolute 0.98 cutoff no longer separates duplicates "
             "from real similarity (it let same-artist near-duplicates through). "
             "1.0 keeps everything.",
    )
    parser.add_argument(
        "--collapse-sim", type=float, default=0.999,
        help="a track whose raw cosine to ANY other track is >= this is treated as "
             "collapsed (near-silent/degenerate content) and excluded from pairs. "
             "Default 0.999, measured against the full 32,783-track corpus: "
             "0.99 flags 374 per 1000 (a THIRD of the corpus — the raw space is "
             "crowded, mean inter-similarity 0.844, and every track is compared "
             "against 32,782 others, so the median track's best match is already "
             "0.9888), 0.995 flags 48, and 0.999 flags 1.0. Note the count scales "
             "with corpus size, so only compare it at a fixed --n-tracks.",
    )
    parser.add_argument(
        "--allow-same-artist", action="store_true",
        help="keep pairs from the same artist. Off by default: same-artist tracks "
             "dominate the extreme top of the ranking, so excluding them is what "
             "makes the QA set show whether the model finds ORDINARY-similarity "
             "related music rather than the same act twice. Skipped counts are "
             "reported either way.",
    )
    parser.add_argument(
        "--mel-root", default=None, type=pathlib.Path,
        help="dir of precomputed MTG-Jamendo log-mel .npy: render each pair as a "
             "spectrogram PNG (no audio decode). Metadata comes from the DB.",
    )
    parser.add_argument(
        "--mtg-data", default=None, type=pathlib.Path,
        help="audio mode: MTG-Jamendo data dir, for decoding the pair to a .wav. "
             "Needs a DB built from the same source (a row's `idx` is a position in "
             "whichever corpus built the DB, so a mel-built DB would decode the "
             "wrong tracks)",
    )
    parser.add_argument(
        "--audio-root", default=None, type=pathlib.Path,
        help="dir where MTG audio unpacked (defaults to --mtg-data); audio mode only",
    )
    args = parser.parse_args()

    if (args.mel_root is None) == (args.mtg_data is None):
        parser.error(
            "set exactly one of --mel-root (spectrogram PNGs, no audio needed) or "
            "--mtg-data (listenable wavs from decoded audio)"
        )

    print(f"[main] loading embeddings from '{args.db_dir}'")
    _, embs_w, metas = load_all(args.db_dir, COLLECTION_WHITENED)
    _, embs_r, _ = load_all(args.db_dir, COLLECTION_RAW)
    n = len(metas)

    if args.n_tracks and args.n_tracks < n:
        pick = np.sort(
            np.random.default_rng(42).choice(n, size=args.n_tracks, replace=False)
        )
        embs_w, embs_r = embs_w[pick], embs_r[pick]
        metas = [metas[i] for i in pick]
        print(f"[main] sampled {len(pick)}/{n} tracks")
    else:
        pick = np.arange(n)
        print(f"[main] ranking all {n} tracks")
    print(f"[main] ranking in {args.space} space")

    Pw = embs_w / np.linalg.norm(embs_w, axis=1, keepdims=True)
    Pr = embs_r / np.linalg.norm(embs_r, axis=1, keepdims=True)

    # generous per-block candidate budget so the filters below still have enough
    # to walk down; the merged set is exact for the global top-`per_block`
    per_block = max(200, args.pairs * 40)
    ci, cj, cw, cr, max_raw = rank_pairs(Pw, Pr, per_block=per_block)
    print(
        f"[main] scored {len(ci)} candidate pairs across {Pw.shape[0]} tracks "
        f"({per_block} kept per query block)"
    )

    collapsed = max_raw > args.collapse_sim
    n_coll = int(collapsed.sum())
    print(
        f"[main] {n_coll} collapsed tracks excluded "
        f"({1000 * n_coll / len(pick):.1f} per 1000; raw cosine >= "
        f"{args.collapse_sim} to some other track)"
    )

    # walk the ranking by the chosen space, skipping collapsed tracks, near-duplicate
    # audio, and (by default) same-artist pairs — reporting each skip count so the
    # QA set's composition is visible rather than implied
    order = np.argsort(-(cw if args.space == "whitened" else cr))
    chosen: list[int] = []
    skip_collapsed = skip_dup = skip_artist = 0
    for k in order:
        i, j = int(ci[k]), int(cj[k])
        if collapsed[i] or collapsed[j]:
            skip_collapsed += 1
            continue
        if cr[k] >= args.max_raw_sim:
            skip_dup += 1
            continue
        if not args.allow_same_artist:
            art_i = metas[i].get("artist") or ""
            if art_i and art_i == (metas[j].get("artist") or ""):
                skip_artist += 1
                continue
        chosen.append(int(k))
        if len(chosen) >= args.pairs:
            break
    topk = np.asarray(chosen)
    print(
        f"[main] skipped {skip_collapsed} collapsed, {skip_dup} near-duplicate "
        f"(raw >= {args.max_raw_sim}), {skip_artist} same-artist"
        + ("" if args.allow_same_artist else " (use --allow-same-artist to keep them)")
    )
    print(f"[main] selected {len(topk)} pairs")

    ds = None
    if args.mel_root is not None:
        print(f"[main] mel mode: rendering spectrogram PNGs from {args.mel_root}")
    else:
        print("[main] audio mode: decoding pairs to wavs")
        ds = MTGJamendoBase(args.mtg_data, audio_root=args.audio_root)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest: list[str] = []
    same_genre = 0
    for rank, k in enumerate(topk):
        i, j = int(ci[k]), int(cj[k])
        idx_a, idx_b = int(pick[i]), int(pick[j])
        ma, mb = metas[i], metas[j]
        whit, raw = float(cw[k]), float(cr[k])
        ga, gb = ma.get("genre") or "?", mb.get("genre") or "?"
        art_a, art_b = (ma.get("artist") or ""), (mb.get("artist") or "")
        same_artist = bool(art_a) and art_a == art_b
        same_genre_flag = ga != "?" and ga == gb
        # tag: 'a'/'g' = same artist / same genre, 'x' = different. There is no
        # same-track case to mark — pairs are distinct tracks by construction.
        tag = "x" + ("a" if same_artist else "x") + ("g" if same_genre_flag else "x")

        if ds is None:
            fpath = write_spec_pair(out, rank, args.mel_root, ma, mb, whit, tag)
        else:
            fpath = write_wav_pair(out, rank, ds, idx_a, idx_b, whit, tag)

        same_genre += same_genre_flag
        line = (
            f"{rank:02d} whit={whit:.3f} raw={raw:.3f} [{tag}] | "
            f"{ga:<22} {ma.get('title', '')!r} — {ma.get('artist', '')!r}  vs  "
            f"{gb:<22} {mb.get('title', '')!r} — {mb.get('artist', '')!r} | {fpath.name}"
        )
        manifest.append(line)
        print(f"[main] {line}")

    (out / "pairs.txt").write_text("\n".join(manifest) + "\n")
    n_saved = len(topk)
    kind = "PNGs (A above B)" if ds is None else "wavs (A + 1s gap + B)"
    print(f"[main] saved {n_saved} {kind} + pairs.txt to {out}/")
    print(f"[main] top pairs same-genre: {same_genre}/{n_saved}")


if __name__ == "__main__":
    main()
