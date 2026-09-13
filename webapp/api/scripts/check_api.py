#!/usr/bin/env python3
"""Exercise the API's data layer and assert on the results.

    nix develop .#web --command python scripts/check_api.py

Checks the things that would otherwise fail silently -- a search that returns rows in
the wrong order, facets that do not respect the filters, a graph whose edges disagree
with its node similarities, an enrichment panel that reports a lift of 1 for everything.
Runs against ``webapp/data/`` directly, so it needs no server and no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import audio  # noqa: E402
from app.index import Catalog, fts_query  # noqa: E402

problems: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "ok  " if condition else "FAIL"
    suffix = f" -- {detail}" if detail else ""
    print(f"  {status} {label}{suffix}")
    if not condition:
        problems.append(label)


def main() -> int:
    print("loading catalog ...")
    cat = Catalog()
    print(f"  {cat.n} tracks x {cat.dim}-d\n")

    print("search")
    res = cat.search(q="Podington Bear", limit=10)
    artists = {item["artist"] for item in res["items"]}
    check(
        "text search finds the artist",
        res["total"] > 0 and artists == {"Podington Bear"},
        f"{res['total']} hits, artists={artists}",
    )
    check("prefix search works while typing", cat.search(q="poding")["total"] > 0)

    res2 = cat.search(genres=["ambient"], limit=5)
    check(
        "genre filter returns only that genre",
        all("ambient" in i["tags"] for i in res2["items"]),
        f"{res2['total']} ambient tracks",
    )

    both = cat.search(genres=["ambient", "classical"], limit=5)
    check(
        "multi-select genre is an OR within the facet",
        both["total"] >= res2["total"],
        f"ambient={res2['total']} ambient+classical={both['total']}",
    )

    narrowed = cat.search(genres=["ambient"], instruments=["piano"], limit=5)
    check(
        "facets combine as AND across dimensions",
        narrowed["total"] <= res2["total"],
        f"ambient+piano={narrowed['total']}",
    )

    # Facet counts must be computed with the facet's own filter removed, otherwise every
    # unselected option would read zero and the filter UI would be unusable.
    counts = {f["value"]: f["count"] for f in res2["facets"]["genre"]}
    check(
        "facet counts ignore their own dimension's filter",
        counts.get("classical", 0) > 0,
        f"classical={counts.get('classical')} while the genre filter is ambient only",
    )

    check(
        "pagination is stable",
        cat.search(limit=5, offset=0)["items"][0]["idx"]
        != cat.search(limit=5, offset=5)["items"][0]["idx"],
    )

    check(
        "year filter applies",
        all(
            2004 <= (i["released"] or 0) <= 2010
            for i in cat.search(year_max=2010, limit=20)["items"]
        ),
    )

    print("\nsimilarity")
    seed = 0
    sims = cat.similar(seed, k=10)
    check(
        "k neighbours, self excluded",
        len(sims) == 10 and all(i != seed for i, _ in sims),
    )
    check(
        "sorted by descending cosine",
        all(sims[i][1] >= sims[i + 1][1] for i in range(9)),
        f"{sims[0][1]:.3f} -> {sims[-1][1]:.3f}",
    )

    # The neighbours must be the true top-k. Recomputing independently from the raw
    # matrix is a check on the argpartition logic, which is easy to get subtly wrong.
    import numpy as np

    X = cat.space("whitened")
    expected = np.argsort(-(X @ X[seed]))[:2]
    check(
        "matches an independent argsort",
        int(expected[0]) == seed and int(expected[1]) == sims[0][0],
        f"argsort top={expected.tolist()}",
    )

    print("\ngraph")
    g = cat.graph(seed, k=20)
    check("node count is seed + k", len(g["nodes"]) == 21)
    check("seed is flagged", g["nodes"][0]["seed"] is True)
    node_ids = {n["idx"] for n in g["nodes"]}
    check(
        "edges stay inside the node set",
        all(e["source"] in node_ids and e["target"] in node_ids for e in g["edges"]),
    )
    check("no self edges", all(e["source"] != e["target"] for e in g["edges"]))

    # Edge weights must equal the cosine of the two endpoints -- the failure mode being
    # an edge list built from a differently-ordered matrix.
    sub = cat.submatrix([n["idx"] for n in g["nodes"]], "whitened")
    pos = {n["idx"]: i for i, n in enumerate(g["nodes"])}
    worst = max(
        (
            abs(sub[pos[e["source"]], pos[e["target"]]] - e["weight"])
            for e in g["edges"]
        ),
        default=0.0,
    )
    check(
        "edge weights match the node similarities",
        worst < 1e-5,
        f"max diff {worst:.2e}",
    )

    g2 = cat.graph(seed, k=20)
    check(
        "graph is deterministic",
        [n["idx"] for n in g["nodes"]] == [n["idx"] for n in g2["nodes"]]
        and g["edges"] == g2["edges"],
    )
    check(
        "every neighbour is reachable from the seed",
        all(n["degree"] > 0 for n in g["nodes"]),
    )

    e = g["enrichment"]
    rng_seeds = [
        int(x) for x in np.random.default_rng(7).choice(cat.n, size=12, replace=False)
    ]
    for name in ("artist", "genre"):
        m = e[name]
        lift = f"{m['lift']:.1f}x" if m["lift"] else "n/a"
        print(
            f"  {name:6s}: {m['agreeing']}/{m['compared']} neighbours agree "
            f"({m['observed']:.3f}) vs corpus chance {m['chance']:.4f} -> {lift}"
        )
    print(
        f"  tags  : {e['tags']['mean_overlap']:.3f} shared tags vs chance "
        f"{e['tags']['chance']:.4f} = {e['tags']['lift']:.2f}x"
    )

    # Track 0 is a deliberate probe: an atypical "Intro" track whose neighbours are
    # other artists and other genres, so it must NOT look enriched, and must not look
    # like an error either. It is the case that exposed the ego-graph null being wrong.
    check(
        "an atypical seed reports low/no enrichment, not an error",
        e["artist"]["lift"] is None or e["artist"]["lift"] < 1.0,
        f"artist lift {e['artist']['lift']}",
    )

    # The corpus-wide measurement is ~3.3-4.0x for tags and tens-to-hundreds for artist.
    # Averaged over seeds, a lift at or below 1 would mean the panel reports noise.
    lifts = {"artist": [], "genre": [], "tags": []}
    for s in rng_seeds:
        ee = cat.graph(s, k=20)["enrichment"]
        for name in ("artist", "genre"):
            if ee[name]["lift"] is not None:
                lifts[name].append(ee[name]["lift"])
        if ee["tags"]["lift"] is not None:
            lifts["tags"].append(ee["tags"]["lift"])
    means = {k: (sum(v) / len(v) if v else 0.0) for k, v in lifts.items()}
    print(
        f"  mean lift over {len(rng_seeds)} random seeds: "
        f"artist {means['artist']:.0f}x  genre {means['genre']:.1f}x  "
        f"tags {means['tags']:.2f}x"
    )
    for name in ("artist", "genre", "tags"):
        check(
            f"{name} enrichment above chance in aggregate",
            means[name] > 1.0,
            f"{means[name]:.2f}x",
        )

    print("\nrange parsing")
    cases = [
        (None, 100, None),
        ("bytes=0-49", 100, (0, 49)),
        ("bytes=50-", 100, (50, 99)),
        ("bytes=-20", 100, (80, 99)),
        ("bytes=0-999", 100, (0, 99)),
        ("bytes=200-300", 100, None),
        ("rows=1-2", 100, None),
        ("bytes=abc", 100, None),
        ("bytes=10-5", 100, None),
    ]
    for header, size, want in cases:
        got = audio.parse_range(header, size)
        check(f"parse_range({header!r}, {size})", got == want, f"got {got}")
    check(
        "range end is clamped, not rejected",
        audio.parse_range("bytes=90-999", 100) == (90, 99),
    )

    print("\nfts input handling")
    check(
        "punctuation cannot break the query",
        fts_query("AC/DC") == '"AC" "DC"*',
        fts_query("AC/DC"),
    )
    check("empty text yields no filter", fts_query("!!!") is None)
    check(
        "last token gets a prefix star", fts_query("poding bear") == '"poding" "bear"*'
    )
    check("search with punctuation does not raise", cat.search(q="AC/DC")["total"] >= 0)

    print("\nstats")
    s = cat.stats()
    check("stats report the corpus size", s["tracks"] == cat.n)
    print(f"  audio: {s['audio']}")

    if problems:
        print("\nFAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK: all API checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
