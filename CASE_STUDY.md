# Does it know it's the same song?

**A case study of the trained encoder on real, mainstream music.**
2026-09-13 · model `checkpoint_e1000` (1000 epochs, 128k steps) · 32,783-track corpus

---

## The question

The model exists to be a Shazam-style fingerprint: any 5-second window of a track
should map near its other windows, robustly to noise, EQ and compression. So the
obvious test is to hand it two songs by name and ask how similar it thinks they are.

```bash
python -m music_encoding.compare_songs \
    --a "Radiohead - Creep" --b "Radiohead - Creep (Acoustic)"
```

That resolves each name to a 30-second preview (Deezer, iTunes as fallback),
downloads it, runs it through the encoder, and reports the cosine similarity of the
two embeddings — **and where that number sits in the corpus's own distribution**,
because a bare cosine turns out to be close to meaningless.

## The headline: raw cosine cannot tell these pairs apart

| pair | raw | raw z | raw % | **whitened** | whitened z | whitened % |
|---|---|---|---|---|---|---|
| Creep vs Karma Police — *same artist* | 0.9696 | +1.31 | 93.4 | **+0.3495** | **+2.62** | 98.5 |
| Creep vs Creep (Acoustic) — **same song** | 0.9575 | +0.71 | 75.4 | +0.2524 | +1.89 | 95.8 |
| Creep vs Goldberg Variations — *unrelated* | 0.9195 | −1.17 | 11.2 | +0.0246 | +0.18 | 61.1 |
| Wonderwall vs Song 2 — *same era, diff. artists* | 0.9244 | −0.93 | 15.4 | −0.0280 | −0.21 | 44.2 |

Corpus null — random pairs of corpus tracks: **raw 0.943 ± 0.020**, whitened
0.000 ± 0.133. Read the raw column again: an *unrelated* pair scores **0.92**, and two
versions of the *same song* score **0.958**. Every pair sits inside ~1.3 standard
deviations of a random pairing. Quoting a raw cosine of "0.96" as evidence of
similarity would be reporting noise.

The whitened space does separate them — 0.35 / 0.25 / 0.02 / −0.03 against a
near-zero null — and the ordering is sensible.

## The uncomfortable part

**Two *different* Radiohead songs score higher than two versions of the *same*
song** (0.3495 vs 0.2524, z +2.62 vs +1.89) — in both spaces, and by a margin larger
than the difference between "same song" and "unrelated" in the raw space.

That is the finding that matters. The embedding is grouping by **artist and
production style**, not by song identity: *Karma Police* shares Radiohead's
recording signature, while *Creep (Acoustic)* is the same composition re-arranged
with different instrumentation and tempo. For a fingerprinting product this is the
wrong signal — it would prefer a stylistic match by the same act over the actual
recording you played.

It is also consistent with the quantitative evaluation rather than contradicting it.
Over all 163,915 windows, retrieval top-1 is **0.673** (top-10 0.793, chance
2.4e-05): the right track is usually *findable*, but a third of queries still return
the wrong track first, and the ranking is evidently driven more by timbre than by
identity.

And even the strongest pair here reaches only **z = +2.6**. A fingerprint worth
shipping would want z ≫ 3.

## Why these numbers can be trusted — the front end was measured, not guessed

The model never saw raw audio. It was trained on MTG-Jamendo's precomputed 96-band
log-mels (merged to 64 bands), and **MTG's STFT parameters are undocumented**. So
embedding a downloaded preview requires reproducing a front end we did not have, and
getting it wrong would make every number above a measurement of *our preprocessing*
rather than of the model.

Ground truth made this checkable: the MTG audio tarball holds tracks whose `.npy` we
*also* have, so the same track can be embedded from audio and from MTG's own mel, and
the two compared.

| front end | mean cosine to MTG's own embedding |
|---|---|
| **`sr=24000, n_fft=1024, hop=512, n_mels=96, norm='slaney'`** | **0.9872** |
| same, `norm=None` (torchaudio's default) | 0.9475 |
| `sr=48000, n_fft=2048, hop=1024, norm='slaney'` | 0.9768 |
| `sr=48000, n_fft=2048, hop=1024, norm=None` | 0.9044 |

The answer is librosa with essentially all defaults. **`norm='slaney'` was the
missing piece** — torchaudio's `MelSpectrogram` defaults to `norm=None`, which costs
0.047. Two incidental findings: the tarball files are **full tracks, not 30-second
previews** despite the `raw_30s` name, and the frame rate is 46.875 fps (24000/512),
*not* the 43.07 fps of the repo's own waveform path — the 43.07 candidates correlate
only 0.52–0.78 on the energy envelope where 46.875 reaches 0.93–0.98.

So every preview cosine in this study carries a **known ~1.3% preprocessing error**.
Decimals beyond ~0.01 should be treated as noise, and that is why the percentile
columns carry the argument rather than the raw cosines.

One more prerequisite was verified rather than assumed: `build_chroma` whitens with a
ZCA transform it does **not** store, so a query embedding cannot be whitened
consistently unless the transform is reproducible. It is — re-deriving it from
`tracks_raw` reproduces the stored `tracks_whitened` vectors to a max diff of 1e-5 —
so the same transform is applied to the query, making the whitened cosines comparable
to the DB's space.

## Reproducing

```bash
# by name (Deezer search, iTunes fallback)
python -m music_encoding.compare_songs --a "Radiohead - Creep" \
    --b "Radiohead - Creep (Acoustic)"

# exactly, by ISRC — Deezer's search returns the ISRC, so this is unambiguous
python -m music_encoding.compare_songs --a-isrc GBAYE9200070 --b "Portishead - Roads"

# alternatives
--provider deezer|itunes|auto     --top-n 5     --windows 5
```

The tool also reports each song's nearest corpus tracks, which is where it is most
useful in practice — it shows what the model thinks the song *sounds like*:

```
[A] nearest corpus tracks to 'Creep':
  +0.987  alternative    'Tramonti' — 'Introversia'
  +0.987  alternative    'Warmachine' — 'Shearer'
```

## Caveats — what this is and is not

- **n = 4.** This is an illustration, not an evaluation. The quantitative evidence is
  the retrieval metric over 163,915 windows (`test_retrieval.py`); this case study
  explains *why* that metric matters, and should not be quoted as a benchmark.
- **30-second previews are usually the hook**, not a representative sample of the
  track, and they are mastered/encoded differently from the corpus audio (the two
  Creep previews differ by 18 dB in median level). The embedding is not
  loudness-invariant by construction.
- **The preview population is mainstream commercial music**; the corpus is largely
  Creative Commons and indie. The corpus null is the right yardstick for *ranking*,
  but the two populations are not the same distribution — the tool reports a second,
  query-vs-corpus null for this reason (it agreed closely here: 0.946 ± 0.016).
- **Spotify cannot be used.** It removed 30-second previews from the Web API on
  2024-11-27 with no scope to re-enable them for a new app, so a Spotify track id
  cannot be resolved to audio at all. Deezer and iTunes are free and unauthenticated.
- **Previews are fetched at request time for analysis only** and never stored or
  redistributed; both providers' terms restrict caching audio.
- The front-end reproduction is 0.987-faithful, so any *absolute* cosine carries
  ~1.3% error — comparable to a fifth of the null's standard deviation. Rankings and
  percentiles are robust to it; third-decimal differences are not.

## What would move the needle

The failure mode is that the representation encodes **style** more than **identity**.
Three directions, in rough order of promise:

1. **More steps.** The step lever was still paying at 128k (retrieval 0.345 → 0.673
   across the day), so this is the proven one.
2. **Instance discrimination.** Barlow Twins pulls two views of one track together and
   decorrelates dimensions, but never pushes *different tracks apart* — which is
   exactly the signal that would separate "same song" from "same artist". A
   contrastive term was implemented and tested (`InfoNCELoss`); at weight 30 it proved
   to be a reparameterization equivalent to free whitening, so it needs a larger
   weight or hard-negative mining to bite.
3. **A bigger embedding** (128 → 256), which the case study's style-vs-identity
   confusion gives a reason to try.
