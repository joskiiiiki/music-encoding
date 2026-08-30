"""k-NN network of tracks colored by metadata — from the Chroma vec DB.

Nodes = tracks; edges = top-k inter-track cosine sims in the *whitened* space,
loaded straight from `tracks_whitened` (built by build_chroma.py) — no audio
decoding or model forwards. Force-directed layout so clusters of similar tracks
appear spatially; node color is a metadata column so you can eyeball whether
clusters correspond to it.

Note: the DB metadata has no `bpm`; numeric columns are `released` (year) and
`listens`.

Usage:
    python -m music_encoding.test_network [--db-dir chroma_db] [--n-tracks 400] [--color genres] [--interactive]

--color in {genres, artist, album, instrument, mood_theme, released}
--interactive writes an interactive HTML (pyvis/vis.js) instead of a PNG.
"""

import argparse
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from music_encoding.db import COLLECTION_WHITENED, load_all

K = 6             # neighbors per node in the k-NN graph
SIM_FLOOR = 0.05  # drop edges below this cosine sim
NODE_SIZE = 20    # ~1/3 of the original dot size
MAX_CATEGORIES = 20  # legend size for categorical coloring; the rest go gray


def node_colors(metas: list[dict], field: str):
    """Return (colors, cmap, norm, legend) from the DB metadata dicts.

    Categorical -> tab20 for the top MAX_CATEGORIES values (rest gray) + legend.
    Numeric (`released`, `listens`) -> viridis + colorbar. Missing -> gray."""
    # CLI uses the dataset column name (genres); the DB metadata key is the
    # singular primary genre
    field = {"genres": "genre"}.get(field, field)
    raw = [m.get(field) for m in metas]

    if field == "released":
        nums = [None if v is None else float(v) for v in raw]
        valid = [v for v in nums if v is not None]
        if not valid:
            return ["#cccccc"] * len(raw), None, None, None
        vmin, vmax = min(valid), max(valid)
        if vmin == vmax:
            vmax = vmin + 1
        cmap = plt.cm.viridis
        norm = matplotlib.colors.Normalize(vmin, vmax)
        colors = [cmap(norm(v)) if v is not None else "#cccccc" for v in nums]
        return colors, cmap, norm, None

    cats = [("other" if v is None else str(v)) for v in raw]
    top = [c for c, _ in Counter(cats).most_common(MAX_CATEGORIES)]
    tab = plt.cm.tab20.colors
    cmap_dict = {c: tab[i % len(tab)] for i, c in enumerate(top)}
    colors = [cmap_dict.get(c, "#dddddd") for c in cats]
    return colors, None, None, [(c, cmap_dict[c]) for c in top]


def save_interactive_html(
    G: nx.Graph, metas: list[dict], color_by: str, out: str = "test_network_interactive.html"
) -> None:
    """Render G as an interactive vis.js HTML (pyvis): hover a node to see its
    metadata, drag nodes (force physics), zoom in/out. Nodes keep the --color
    coloring; edge thickness follows similarity."""
    from pyvis.network import Network

    colors, _, _, _ = node_colors(metas, color_by)
    hex_colors = []
    for c in colors:
        if isinstance(c, str):
            hex_colors.append(c)
        else:
            r, g, b = (int(round(v * 255)) for v in c[:3])
            hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")

    net = Network(
        height="800px", width="100%", directed=False, font_color="#222222",
        cdn_resources="in_line",  # embed vis.js so the HTML works offline
    )
    for i, m in enumerate(metas):
        tooltip = (
            f"<b>{m.get('title', '')}</b> — {m.get('artist', '')}<br>"
            f"genre: {m.get('genre', '?')} | released: {m.get('released', '?')}<br>"
            f"listens: {m.get('listens', '?')}"
        )
        net.add_node(i, label="", title=tooltip, color=hex_colors[i])
    for i, j, d in G.edges(data=True):
        net.add_edge(i, j, value=float(d["weight"]))
    net.write_html(out, open_browser=False)
    print(f"[main] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="k-NN track-similarity network colored by metadata (from chroma DB)"
    )
    parser.add_argument(
        "--db-dir", default="chroma_db",
        help="Chroma persistent dir (default: chroma_db)",
    )
    parser.add_argument(
        "--n-tracks", type=int, default=400,
        help="tracks to sample for the plot (default: 400)",
    )
    parser.add_argument(
        "--color", default="genres",
        choices=["genres", "artist", "album", "instrument", "mood_theme", "released"],
        help="metadata column to color nodes by (default: genres)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="write an interactive HTML (pyvis) instead of a static PNG",
    )
    args = parser.parse_args()

    print(f"[main] loading '{args.db_dir}/{COLLECTION_WHITENED}'")
    ids, embs, metas = load_all(args.db_dir, COLLECTION_WHITENED)
    n = len(ids)

    n_samp = min(args.n_tracks, n)
    pick = np.sort(np.random.default_rng(42).choice(n, size=n_samp, replace=False))
    embs = embs[pick]
    metas = [metas[i] for i in pick]
    print(f"[main] sampled {n_samp} tracks, colored by '{args.color}'")

    P = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    S = P @ P.T
    np.fill_diagonal(S, -1.0)

    print("[main] building k-NN graph (whitened space)")
    G = nx.Graph()
    G.add_nodes_from(range(n_samp))
    for i in range(n_samp):
        for j in np.argsort(-S[i])[:K]:
            if S[i, j] > SIM_FLOOR:
                G.add_edge(int(i), int(j), weight=float(S[i, j]))  # int, not np.int64 (JSON)
    print(f"[main] graph: {n_samp} nodes, {G.number_of_edges()} edges (k={K}, sim>{SIM_FLOOR})")

    artists = [m.get("artist") or "" for m in metas]
    genres = [m.get("genre") or "" for m in metas]
    n_e = G.number_of_edges()
    if n_e:
        same_artist = sum(1 for i, j in G.edges() if artists[i] and artists[i] == artists[j])
        same_genre = sum(1 for i, j in G.edges() if genres[i] and genres[i] == genres[j])
        n_s = len(artists)
        counts_g = Counter(g for g in genres if g)
        counts_a = Counter(a for a in artists if a)
        base_genre = sum((c / n_s) ** 2 for c in counts_g.values())
        base_artist = sum((c / n_s) ** 2 for c in counts_a.values())
        print(f"[main] edges connecting same artist: {same_artist / n_e:.3f} (random {base_artist:.3f})")
        print(f"[main] edges connecting same genre: {same_genre / n_e:.3f} (random {base_genre:.3f})")

    if args.interactive:
        save_interactive_html(G, metas, args.color)
        return

    print("[main] laying out graph (spring)")
    pos = nx.spring_layout(G, seed=42, weight="weight")

    colors, cmap, norm, legend = node_colors(metas, args.color)

    fig, ax = plt.subplots(figsize=(12, 12))
    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    wmax = max(weights) if weights else 1.0
    widths = [0.2 + 2.0 * (w / wmax) for w in weights]
    alphas = [0.06 + 0.44 * (w / wmax) for w in weights]  # tuned-down, still by strength
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=alphas, edge_color="#8a8a8a")
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=NODE_SIZE, node_color=colors,
        linewidths=0.3, edgecolors="#444444",
    )
    if legend is not None:
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                       markersize=8, label=lab)
            for lab, c in legend
        ]
        ax.legend(handles=handles, loc="best", fontsize=7, frameon=True, title=args.color)
    elif cmap is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=args.color)
    ax.set_title(f"k-NN track-similarity network (whitened) — colored by {args.color}")
    ax.axis("off")
    fig.tight_layout()
    fname = f"test_network_{args.color}.png"
    fig.savefig(fname, dpi=150)
    print(f"[main] saved {fname}")
    plt.show()


if __name__ == "__main__":
    main()
