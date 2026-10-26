#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Draw one legible, full EGoD layered-circular network without changing originals.

Run from any directory:
    python code_3_2_viz_EGoD_layered_circular_optimized.py

All corrected nodes and unique undirected links are retained. Highlighted nodes
are selected by graph degree within each layer; this is a topology summary, not
a biological importance or causal-effect estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "ebio_summary_data" / "buildModel"
DEFAULT_OUT = ROOT / "results_viz_ebio" / "viz_EGoD" / "set_node_size_vs_degree_optimized"
GROUPS = ("Exposure", "GO", "Disease")
RADII = {"Exposure": 3.0, "GO": 2.05, "Disease": 1.10}
COLORS = {"Exposure": "#3D6DAA", "GO": "#3E947C", "Disease": "#C6605A"}
TOP_N = {"Exposure": 5, "GO": 3, "Disease": 5}
PHASE = {"Exposure": 0.08, "GO": 0.44, "Disease": 0.78}


def load_network() -> tuple[nx.Graph, dict[str, str], dict[str, str]]:
    nodes = pd.read_csv(DATA / "EGoD_nodes_corrected.csv", dtype={"EXID": str})
    edges = pd.read_csv(DATA / "EGoD_edges_corrected.csv", usecols=["Source", "Target"], dtype=str)
    needed = {"EXID", "Group name", "Preferred name"}
    if not needed.issubset(nodes) or not {"Source", "Target"}.issubset(edges):
        raise ValueError("Corrected EGoD files lack required node or edge fields")
    nodes = nodes.drop_duplicates("EXID", keep="first")
    groups = dict(zip(nodes["EXID"], nodes["Group name"]))
    names = dict(zip(nodes["EXID"], nodes["Preferred name"].fillna(nodes["EXID"])))
    g = nx.Graph()
    g.add_nodes_from(nodes["EXID"].astype(str))
    g.add_edges_from(edges[["Source", "Target"]].itertuples(index=False, name=None))
    unknown = set(g) - set(groups)
    if unknown:
        raise ValueError(f"{len(unknown)} edge endpoints absent from corrected node table")
    if set(groups.values()) - set(GROUPS):
        raise ValueError("Unexpected EGoD node group")
    return g, groups, names


def layout(g: nx.Graph, groups: dict[str, str]) -> dict[str, tuple[float, float]]:
    degree = dict(g.degree())
    pos = {}
    for group in GROUPS:
        members = sorted((n for n in g if groups[n] == group), key=lambda n: (-degree[n], n))
        # Place highlighted nodes at separated angular slots. All remaining
        # nodes fill the unused slots; no observations or links are removed.
        count = len(members)
        selected = members[:TOP_N[group]]
        slots = [round((i + PHASE[group]) * count / len(selected)) % count
                 for i in range(len(selected))]
        used = set(slots)
        free = [i for i in range(count) if i not in used]
        order = list(zip(selected, slots)) + list(zip(members[len(selected):], free))
        for node, slot in order:
            angle = np.pi / 2 + 2 * np.pi * slot / count
            pos[node] = (RADII[group] * np.cos(angle), RADII[group] * np.sin(angle))
    return pos


def draw(g: nx.Graph, groups: dict[str, str], names: dict[str, str], out: Path, dpi: int) -> None:
    degree = dict(g.degree())
    pos = layout(g, groups)
    top = {
        group: sorted((n for n in g if groups[n] == group), key=lambda n: (-degree[n], n))[:TOP_N[group]]
        for group in GROUPS
    }
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"],
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(12.6, 11.2), facecolor="white")
    ax = fig.add_axes([0.035, 0.075, 0.80, 0.83])
    ax.set_aspect("equal")
    ax.set_xlim(-4.75, 4.75)
    ax.set_ylim(-4.35, 4.35)
    ax.axis("off")

    # Draw every unique edge. Rasterizing only edges keeps the PDF compact and
    # leaves circles and labels editable. Lines are deliberately very thin.
    edge_pairs = {("Exposure", "GO"): [], ("GO", "Disease"): []}
    for u, v in g.edges:
        pair = tuple(sorted((groups[u], groups[v])))
        if pair == ("Exposure", "GO"):
            edge_pairs[("Exposure", "GO")].append([pos[u], pos[v]])
        elif pair == ("Disease", "GO"):
            edge_pairs[("GO", "Disease")].append([pos[u], pos[v]])
    for pair, color, alpha in [
        (("GO", "Disease"), "#7A8290", 0.012),
        (("Exposure", "GO"), "#6B8099", 0.045),
    ]:
        if edge_pairs[pair]:
            coll = LineCollection(edge_pairs[pair], colors=color, linewidths=0.20,
                                  alpha=alpha, rasterized=True, zorder=1)
            ax.add_collection(coll)

    for group in GROUPS:
        ax.add_patch(Circle((0, 0), RADII[group], fill=False, lw=0.55,
                            ls=(0, (3, 4)), ec="#BDC4CA", zorder=2))

    # Node area follows unique-link degree on a restrained square-root scale.
    # This avoids hiding neighboring nodes on the dense GO and disease rings.
    for group in GROUPS:
        members = [n for n in g if groups[n] == group]
        x = [pos[n][0] for n in members]
        y = [pos[n][1] for n in members]
        d = np.asarray([degree[n] for n in members], float)
        areas = 9 + 4.8 * np.sqrt(d)
        if group == "Disease":
            areas *= 0.70
        ax.scatter(x, y, s=areas, c=COLORS[group], alpha=0.88,
                   linewidths=0.18, edgecolors="white", zorder=4, rasterized=True)
        selected = top[group]
        ax.scatter([pos[n][0] for n in selected], [pos[n][1] for n in selected],
                   s=[(9 + 4.8 * np.sqrt(degree[n])) * (0.70 if group == "Disease" else 1)
                      for n in selected], facecolors="none", edgecolors="#172938",
                   linewidths=0.75, zorder=5)

    fig.text(0.055, 0.944, "EGoD network", fontsize=17, fontweight="bold", color="#193047")
    fig.text(0.055, 0.914, "Exposure  ·  Gene Ontology  ·  Disease", fontsize=10.5, color="#5B6770")

    # Direct labels follow their nodes around each ring. Long GO terms have a
    # short display label here; their full names are in highlighted_nodes.csv.
    go_short = {
        "positive regulation of reactive oxygen species biosynthetic process": "ROS synthesis (+)",
        "negative regulation of cell population proliferation": "Proliferation (−)",
        "cell population proliferation": "Proliferation",
    }
    label_radius = {"Exposure": 3.58, "GO": 2.43, "Disease": 1.43}
    for group in GROUPS:
        for node in top[group]:
            x, y = pos[node]
            angle = np.arctan2(y, x)
            lx = label_radius[group] * np.cos(angle)
            ly = label_radius[group] * np.sin(angle)
            label = go_short.get(str(names[node]).strip().lower(), names[node]) if group == "GO" else names[node]
            if str(label) == "ROS synthesis (+)":
                # This left-side GO label otherwise crosses the outer guide ring.
                lx = -3.25
            ha = "left" if lx > 0.25 else "right" if lx < -0.25 else "center"
            ax.annotate(str(label), xy=(x, y), xytext=(lx, ly), textcoords="data",
                        fontsize=8.4 if group != "GO" else 7.9,
                        color="#243642", ha=ha, va="center", zorder=8,
                        arrowprops={"arrowstyle": "-", "color": COLORS[group],
                                    "lw": 0.55, "alpha": 0.75, "shrinkA": 12, "shrinkB": 5})

    legend = [Line2D([], [], marker="o", ls="", color=COLORS[k], markersize=6, label=k)
              for k in GROUPS]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(-0.02, -0.045),
              ncol=3, frameon=False, fontsize=9, columnspacing=1.8, handletextpad=0.4)
    fig.text(0.855, 0.767, "DEGREE", fontsize=10, fontweight="bold", color="#193047")
    for group, y0, values in [
        ("Exposure", 0.698, [1, 100, 350]),
        ("GO", 0.515, [1, 20, 40]),
        ("Disease", 0.333, [1, 400, 800]),
    ]:
        fig.text(0.855, y0 + 0.036, group, fontsize=9, fontweight="bold", color=COLORS[group])
        for i, value in enumerate(values):
            y = y0 - i * 0.039
            area = (9 + 4.8 * np.sqrt(value)) * (0.70 if group == "Disease" else 1)
            fig.add_artist(Line2D([0.872], [y], marker="o", ls="", markersize=np.sqrt(area),
                                  color=COLORS[group], transform=fig.transFigure))
            fig.text(0.917, y, f"{value:,}", fontsize=8.3, va="center", color="#52606A")
    fig.text(0.855, 0.168, f"{g.number_of_nodes():,} nodes\n{g.number_of_edges():,} unique links",
             fontsize=8.3, color="#52606A", linespacing=1.5)
    fig.text(0.855, 0.102, "Node area increases\nwith graph degree.",
             fontsize=8.3, color="#52606A", linespacing=1.35)
    fig.text(0.055, 0.026, "Topology reflects database connectivity; it does not establish toxicity or causality.",
             fontsize=8.4, color="#69757D")

    out.mkdir(parents=True, exist_ok=True)
    stem = out / "EGoD_network_layered_circular_optimized"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), dpi=dpi, facecolor="white")
    plt.close(fig)
    rows = [{"Group": group, "Rank": i, "EXID": node, "Preferred_name": names[node],
             "Unique_graph_degree": degree[node]} for group in GROUPS
            for i, node in enumerate(top[group], 1)]
    pd.DataFrame(rows).to_csv(out / "highlighted_nodes.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {stem}.png/.pdf/.svg and highlighted_nodes.csv")
    print(f"Network: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} unique links")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 100:
        parser.error("--dpi must be at least 100")
    graph, groups, names = load_network()
    draw(graph, groups, names, args.output_dir, args.dpi)


if __name__ == "__main__":
    main()
