"""Is this neighbourhood organised, or is it noise?

The question the similarity graph raises, and the one that needs a real number next to
it: ``CLAUDE.md`` documents that a force-directed k-NN plot renders an embedding with
genre encoded *perfectly* as the same hairball as isotropic noise, so a picture alone
asserts nothing.

Why there is no permutation null here
-------------------------------------
``music_encoding/test_network.py`` compares a same-label edge fraction against a
permutation null that shuffles labels *within the graph*. That is the correct null
there, because the graph is a **sample of the corpus** (400 tracks are drawn, then edges
are derived from them) -- so shuffling labels over those 400 leaves the selection
untouched.

It is the **wrong** null for a seed-centred ego-graph, and measurably so. Here the node
set is *chosen by similarity to the seed*, so it already contains whatever structure is
being tested; shuffling labels inside it leaves the enrichment in place. Measured: the
eight tracks 0-7 are all by David TMX (96 tracks in the corpus) and **61% of their
top-20 neighbours share the artist**, yet the within-graph lift came out at **0.8x** --
below chance -- because 20 of 21 nodes were one artist, so the *null* was ~0.9 too.

So the null used here is the **corpus**, via prevalence: "what would a randomly chosen
track agree with the seed about?" A neighbour set that is 61% same-artist against a
corpus chance of ~0.3% is a 200x result, and that is the honest way to state it.

Chance must be prevalence-weighted, never uniform. Genre tags are heavily skewed
(``electronic`` alone is ~29% of the corpus, and the top tags are not rare), so dividing
by the number of tags would understate chance by roughly an order of magnitude and
inflate every lift.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def neighbour_agreement(
    neighbours: Sequence[int],
    labels: Mapping[int, str | None],
    counts: Mapping[str, int],
    n_tracks: int,
) -> dict:
    """How often a neighbour matches the seed on a single-valued label.

    ``labels`` must include the seed; it is compared against every neighbour. ``counts``
    is the corpus-wide frequency of each label value, used for the chance rate.
    Neighbours with no label are skipped and reported separately, so a dimension with
    poor coverage (54% of tracks have no instrument tag) cannot silently look like a
    strong result.
    """
    seed = neighbours[0]
    rest = list(neighbours[1:])
    seed_label = labels.get(seed)
    if not seed_label:
        return {
            "observed": 0.0,
            "chance": 0.0,
            "lift": None,
            "compared": 0,
            "skipped": len(rest),
            "reason": "the seed itself has no value for this dimension",
        }
    compared = [n for n in rest if labels.get(n)]
    agreeing = sum(1 for n in compared if labels[n] == seed_label)
    observed = agreeing / len(compared) if compared else 0.0
    # A random track shares the seed's label with probability (count-1)/(N-1).
    chance = (counts.get(seed_label, 1) - 1) / (n_tracks - 1) if n_tracks > 1 else 0.0
    return {
        "observed": observed,
        "chance": chance,
        "lift": (observed / chance) if chance else None,
        "compared": len(compared),
        "skipped": len(rest) - len(compared),
        "seed_value": seed_label,
        "agreeing": agreeing,
        "reason": None,
    }


def tag_overlap_enrichment(
    neighbours: Sequence[int],
    tags: Mapping[int, set[str]],
    prevalence: Mapping[str, int],
    n_tracks: int,
) -> dict:
    """Mean shared genre-tag count per neighbour, against a prevalence-weighted chance.

    The multi-valued counterpart of :func:`neighbour_agreement`: a track carries 2-9
    genre tags, so "agreement" is the size of the intersection rather than an equality
    test.
    """
    seed = neighbours[0]
    rest = list(neighbours[1:])
    seed_tags = tags.get(seed, set())
    if not rest or not seed_tags:
        return {
            "mean_overlap": 0.0,
            "chance": 0.0,
            "lift": None,
            "compared": 0,
            "reason": "the seed has no genre tags" if not seed_tags else None,
        }
    observed = sum(len(seed_tags & tags.get(n, set())) for n in rest)
    expected = sum(
        sum((prevalence[t] - 1) / (n_tracks - 1) for t in seed_tags) for _ in rest
    )
    mean_overlap = observed / len(rest)
    chance = expected / len(rest)
    return {
        "mean_overlap": mean_overlap,
        "chance": chance,
        "lift": (mean_overlap / chance) if chance else None,
        "compared": len(rest),
        "seed_tags": sorted(seed_tags),
        "reason": None,
    }
