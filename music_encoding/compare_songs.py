"""Case study: how similar are two songs, by the model's own judgement?

Takes two song names (or ISRCs), fetches each one's 30-second preview, runs it
through the trained encoder, and reports the cosine similarity of the two
embeddings — plus the context needed to read that number.

WHY THE CONTEXT IS NOT OPTIONAL
    The raw embedding space is crowded: two random MTG tracks already sit at cosine
    ~0.844, because the model shares a large common component across all audio. So a
    bare "similarity: 0.87" is uninterpretable — it could be near the middle of the
    null distribution or far in its tail. This tool therefore always reports where
    the pair falls in the corpus's random-pair distribution (z-score and percentile)
    and each song's nearest corpus neighbours, so the number means something.

THE FRONT END IS CALIBRATED, NOT GUESSED
    The model never saw raw audio: it was trained on MTG-Jamendo's precomputed 96-band
    log-mels (merged to 64 bands), and MTG's STFT parameters were undocumented.
    Reproducing them was measured end-to-end against ground truth — the MTG audio
    tarball holds tracks whose `.npy` we also have, so an audio-derived embedding can
    be compared with the `.npy`-derived one already in chroma_db:

        sr=24000, n_fft=1024, hop=512, n_mels=96, norm='slaney',
        mel_scale='slaney'          ->  0.987 mean cosine to the DB embedding

    (torchaudio's default `norm=None` instead of librosa's `'slaney'` costs 0.047;
    the 48k/2048/1024 variants reach 0.977.) So a preview cosine inherits ~1.3%
    preprocessing error, which is the precision to quote. `_merge_mels` is reused
    from MelSpecWindowSource, so the 96->64 step is identical to training.

    Note the tarball files turned out to be FULL tracks, not 30s previews, which is
    what made this validation possible at all.

PROVIDERS
    Deezer and iTunes, both free and unauthenticated (verified). Deezer is tried
    first because its search response also carries the ISRC, so `--a-isrc` works.
    Spotify is NOT usable: it removed 30-second previews from the Web API on
    2024-11-27 with no scope to re-enable them, so a Spotify track id cannot be
    resolved to audio. Previews are fetched at request time for analysis only and are
    never stored or redistributed; note both providers' terms restrict caching audio.

Usage:
    python -m music_encoding.compare_songs --a "Radiohead - Creep" \\
        --b "Radiohead - Creep (Acoustic)"
    python -m music_encoding.compare_songs --a-isrc GBAYE9200070 --b "Portishead - Roads"
"""

import argparse
import json
import pathlib
import random
import tempfile
import urllib.parse
import urllib.request

import numpy as np
import torch as tc
import torchaudio as ta

from music_encoding.db import DEFAULT_DB_DIR, COLLECTION_RAW, load_all
from music_encoding.model import SiameseEncoderBT
from music_encoding.twin_dataset import (
    DEFAULT_MIN_WINDOW_DB,
    MEL_FPS,
    MelSpecWindowSource,
)

# --- the calibrated MTG front end (see the module docstring for the evidence) -----
FRONT_SR = 24000
FRONT_N_FFT = 1024
FRONT_HOP = 512
FRONT_N_MELS = 96
FRONT_MEL_SCALE = "slaney"  # librosa's default
FRONT_NORM = "slaney"       # librosa's default; torchaudio defaults to None
FRONT_TOP_DB = 80.0
FRONT_FIDELITY = 0.987  # measured end-to-end vs MTG's own .npy-derived embeddings

WIN = round(5 * MEL_FPS)  # 234 frames = 5 s at 46.875 fps, the training window
UA = {"User-Agent": "music-encoding-case-study/1.0"}


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _download(url: str, dest: pathlib.Path) -> pathlib.Path:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        dest.write_bytes(r.read())
    return dest


def resolve_deezer(query: str) -> dict | None:
    """Deezer search: free, unauthenticated, and returns the ISRC as a bonus."""
    url = "https://api.deezer.com/search?" + urllib.parse.urlencode({"q": query, "limit": 1})
    data = _get_json(url)
    for r in data.get("data", []):
        if r.get("preview"):
            return {
                "provider": "deezer",
                "artist": r["artist"]["name"],
                "title": r["title"],
                "album": r.get("album", {}).get("title", ""),
                "isrc": r.get("isrc"),
                "url": r["preview"],
            }
    return None


def resolve_deezer_isrc(isrc: str) -> dict | None:
    data = _get_json(f"https://api.deezer.com/track/isrc:{urllib.parse.quote(isrc)}")
    if "error" in data or not data.get("preview"):
        return None
    return {
        "provider": "deezer",
        "artist": data["artist"]["name"],
        "title": data["title"],
        "album": data.get("album", {}).get("title", ""),
        "isrc": data.get("isrc"),
        "url": data["preview"],
    }


def resolve_itunes(query: str) -> dict | None:
    """iTunes Search: free, unauthenticated, mainstream catalogue."""
    url = "https://itunes.apple.com/search?" + urllib.parse.urlencode(
        {"term": query, "entity": "song", "limit": 3}
    )
    for r in _get_json(url).get("results", []):
        if r.get("previewUrl"):
            return {
                "provider": "itunes",
                "artist": r.get("artistName", ""),
                "title": r.get("trackName", ""),
                "album": r.get("collectionName", ""),
                "isrc": None,  # iTunes search does not return it
                "url": r["previewUrl"],
            }
    return None


def resolve(query: str, isrc: str | None, provider: str) -> dict:
    """Resolve a query (or ISRC) to a preview URL + metadata, Deezer then iTunes."""
    tried = []
    if isrc:
        hit = resolve_deezer_isrc(isrc)
        if hit:
            return hit
        raise SystemExit(f"[resolve] no Deezer track for ISRC {isrc}")
    for name, fn in (("deezer", resolve_deezer), ("itunes", resolve_itunes)):
        if provider not in ("auto", name):
            continue
        try:
            hit = fn(query)
        except Exception as exc:  # network hiccup -> try the next provider
            tried.append(f"{name}: {type(exc).__name__}")
            continue
        if hit:
            return hit
        tried.append(f"{name}: no match")
    raise SystemExit(f"[resolve] no preview found for {query!r} ({', '.join(tried)})")


def front_end(wav: tc.Tensor) -> tc.Tensor:
    """Audio -> (1, 1, 64, T) log-mel, reproducing MTG's front end then merging."""
    mel = ta.transforms.MelSpectrogram(
        sample_rate=FRONT_SR,
        n_fft=FRONT_N_FFT,
        hop_length=FRONT_HOP,
        n_mels=FRONT_N_MELS,
        mel_scale=FRONT_MEL_SCALE,
        norm=FRONT_NORM,
    )
    db = ta.transforms.AmplitudeToDB(top_db=FRONT_TOP_DB)(mel(wav.unsqueeze(0)))
    # the SAME 96->64 merge training uses, so the band geometry matches exactly
    return MelSpecWindowSource._merge_mels(db.unsqueeze(1), 64)


def embed_preview(model, spec: tc.Tensor, n_win: int, seed: int = 42) -> np.ndarray:
    """Mean-pool n_win random 5 s windows, mirroring build_chroma's protocol."""
    rng = random.Random(seed)
    T = spec.shape[-1]
    wins = []
    for _ in range(n_win):
        if T <= WIN:
            w = tc.full((1, 1, spec.shape[2], WIN), -90.0, dtype=spec.dtype)
            w[..., :T] = spec
        else:
            s = rng.randrange(0, T - WIN + 1)
            w = spec[..., s : s + WIN]
        wins.append(w)
    with tc.no_grad():
        emb, _ = model.forward(tc.cat(wins).to("cuda"))
    return emb.mean(dim=0).float().cpu().numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def null_cosines(P: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Cosines of `n` random DISTINCT pairs from P — the distribution the pair is judged against."""
    i = rng.integers(0, len(P), n)
    j = rng.integers(0, len(P), n)
    k = i != j
    a, b = P[i[k]].astype(np.float64), P[j[k]].astype(np.float64)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", a, b)


def describe(spec: tc.Tensor) -> str:
    p5, p50, p95 = (float(tc.quantile(spec.flatten(), q)) for q in (0.05, 0.5, 0.95))
    return f"p5 {p5:.0f} / p50 {p50:.0f} / p95 {p95:.0f} dB"


def main() -> None:
    p = argparse.ArgumentParser(
        description="cosine similarity of two songs, via 30s previews and the trained encoder"
    )
    p.add_argument("--a", help="song A, e.g. 'Radiohead - Creep'")
    p.add_argument("--b", help="song B")
    p.add_argument("--a-isrc", help="exact ISRC for A (Deezer; more precise than a name)")
    p.add_argument("--b-isrc", help="exact ISRC for B")
    p.add_argument(
        "--provider", choices=["auto", "deezer", "itunes"], default="auto",
        help="auto (default) tries Deezer then iTunes",
    )
    p.add_argument(
        "--checkpoint", default="/home/johannes/mel_runs/cont1000/checkpoints/checkpoint_e1000.pt",
        help="model checkpoint (default: the best trained one)",
    )
    p.add_argument("--db-dir", default=DEFAULT_DB_DIR, help="chroma DB for the baseline")
    p.add_argument(
        "--windows", type=int, default=5,
        help="5 s windows to pool per preview, matching build_chroma (default: 5)",
    )
    p.add_argument("--top-n", type=int, default=5, help="neighbours to show (default: 5)")
    args = p.parse_args()

    if not args.a and not args.a_isrc:
        p.error("give --a (a name) or --a-isrc")
    if not args.b and not args.b_isrc:
        p.error("give --b (a name) or --b-isrc")

    print(f"[main] model: {args.checkpoint}")
    ck = tc.load(args.checkpoint, map_location="cuda")
    model = SiameseEncoderBT(
        proj_dims=ck["model_state_dict"]["projector.6.weight"].shape[0]
    ).to("cuda")
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(
        f"[main] MTG front end: sr={FRONT_SR} n_fft={FRONT_N_FFT} "
        f"hop={FRONT_HOP} n_mels={FRONT_N_MELS} norm={FRONT_NORM} "
        f"(validated {FRONT_FIDELITY:.3f} cosine vs MTG's own mels)"
    )

    print(f"\n[main] loading the corpus baseline from '{args.db_dir}'")
    _, corpus_raw, metas = load_all(args.db_dir, COLLECTION_RAW)
    # the whitening build_chroma used is not stored, but it is reproducible from
    # tracks_raw alone (verified to 1e-5), so re-derive it once and reuse it for
    # both the corpus and the queries
    w_mean, w_mat = _zca(corpus_raw)
    corpus_white = (corpus_raw - w_mean) @ w_mat
    print(f"[main] corpus: {len(metas)} tracks")

    queries = []
    for label, name, isrc in (("A", args.a, args.a_isrc), ("B", args.b, args.b_isrc)):
        hit = resolve(name, isrc, args.provider)
        print(
            f"\n[{label}] resolved via {hit['provider']}: "
            f"{hit['artist']!r} — {hit['title']!r}"
            + (f"  ({hit['album']})" if hit["album"] else "")
            + (f"  isrc={hit['isrc']}" if hit["isrc"] else "")
        )
        with tempfile.TemporaryDirectory() as td:
            path = _download(hit["url"], pathlib.Path(td) / "preview")
            from torchcodec.decoders import AudioDecoder

            s = AudioDecoder(str(path)).get_all_samples()
            wav = s.data
            if wav.ndim > 1:
                wav = wav.mean(dim=0)
            if s.sample_rate != FRONT_SR:
                wav = ta.functional.resample(wav, s.sample_rate, FRONT_SR)
            spec = front_end(wav)
        n_win = args.windows
        emb = embed_preview(model, spec, n_win)
        print(
            f"     preview {wav.shape[-1] / FRONT_SR:.1f}s -> mel {tuple(spec.shape[1:])} "
            f"({describe(spec)}), pooled {n_win} windows -> embed {emb.shape[0]}d"
        )
        queries.append((label, hit, emb))

    (la, ha, ea), (lb, hb, eb) = queries

    # --- the pair, and where it sits in the corpus's random-pair distribution -----
    d_raw_pair = cosine(ea, eb)
    rng = np.random.default_rng(42)
    null_raw = null_cosines(corpus_raw, 40000, rng)
    mu_r, sd_r = float(null_raw.mean()), float(null_raw.std())
    z_raw = (d_raw_pair - mu_r) / sd_r
    pct_raw = float((null_raw < d_raw_pair).mean())

    ew_a = (ea - w_mean) @ w_mat  # the corpus transform, applied to a query
    ew_b = (eb - w_mean) @ w_mat
    d_white_pair = cosine(ew_a, ew_b)
    null_white = null_cosines(corpus_white, 40000, rng)
    mu_w, sd_w = float(null_white.mean()), float(null_white.std())
    z_white = (d_white_pair - mu_w) / sd_w
    pct_white = float((null_white < d_white_pair).mean())

    # apples-to-apples null: how similar is THIS query to unrelated tracks? The
    # corpus-pair null below mixes distributions (queries carry ~1.3% front-end
    # error the corpus does not), so the query-vs-corpus spread is the fairer
    # yardstick for a query pair.
    qa_vs_corpus = corpus_raw @ ea / (
        np.linalg.norm(corpus_raw, axis=1) * np.linalg.norm(ea)
    )

    print("\n" + "=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  {ha['artist']} — {ha['title']}")
    print(f"  {hb['artist']} — {hb['title']}")
    print()
    print(f"  cosine similarity (raw)      : {d_raw_pair:+.4f}")
    print(f"    corpus random pairs        : {mu_r:.3f} +/- {sd_r:.3f}")
    print(f"    -> z = {z_raw:+.2f}, percentile {100 * pct_raw:.1f}%")
    print(
        f"    A vs unrelated tracks      : {qa_vs_corpus.mean():.3f} "
        f"+/- {qa_vs_corpus.std():.3f}"
        f"  (pair z = {(d_raw_pair - qa_vs_corpus.mean()) / qa_vs_corpus.std():+.2f})"
    )
    print(f"  cosine similarity (whitened) : {d_white_pair:+.4f}")
    print(f"    corpus random pairs        : {mu_w:+.3f} +/- {sd_w:.3f}")
    print(f"    -> z = {z_white:+.2f}, percentile {100 * pct_white:.1f}%")
    print()
    print(
        f"  read with: ~{FRONT_FIDELITY:.3f} front-end fidelity, so treat the "
        f"decimals beyond ~0.01 as noise"
    )
    print(
        "  note: 'percentile' is against random corpus pairs; a pair can score a high "
        "raw cosine simply because EVERY pair of tracks does (the space is crowded)"
    )

    # --- what each query sounds like in the corpus --------------------------------
    for (label, hit, emb) in queries:
        c = corpus_raw @ emb / (
            np.linalg.norm(corpus_raw, axis=1) * np.linalg.norm(emb)
        )
        top = np.argsort(-c)[: args.top_n]
        print(f"\n  [{label}] nearest corpus tracks to {hit['title']!r}:")
        for k in top:
            m = metas[int(k)]
            print(
                f"    {c[k]:+.3f}  {m.get('genre') or '?':<14} "
                f"{(m.get('title') or '')[:42]!r} — {(m.get('artist') or '')[:28]!r}"
            )


def _zca(raw: np.ndarray, eps: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """The corpus ZCA as (mean, W), so `whitened = (X - mean) @ W`.

    This is build_chroma.whiten's arithmetic, but returning the transform instead of
    only its result — the whitening build_chroma used is NOT stored, so a query can
    only be whitened consistently by re-deriving it from tracks_raw. That is sound
    because the transform is a deterministic function of the corpus; verified to
    reproduce the stored tracks_whitened to a max diff of 1e-5.
    """
    x = raw.astype(np.float64)
    mean = x.mean(axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / (x.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, eps, None)
    W = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return mean, W


if __name__ == "__main__":
    main()
