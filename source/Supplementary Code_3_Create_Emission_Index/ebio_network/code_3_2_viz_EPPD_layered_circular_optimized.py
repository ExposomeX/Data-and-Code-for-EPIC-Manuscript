#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Draw the full corrected EPPD network as one layered circular figure.

Run from any directory:
    python code_3_2_viz_EPPD_layered_circular_optimized.py

All corrected nodes and unique undirected links are shown. Labels mark the
highest-degree nodes in each actual data layer, not biological/causal priority.
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
DEFAULT_OUT = ROOT / "results_viz_ebio" / "viz_EPPD" / "set_node_size_vs_degree_optimized"
GROUPS = ("Exposure", "Protein_1", "Protein_2", "Disease")
RADII = {"Exposure": 4.0, "Protein_1": 3.0, "Protein_2": 2.0, "Disease": 1.0}
COLORS = {"Exposure": "#3D6DAA", "Protein_1": "#3E947C",
          "Protein_2": "#B7934D", "Disease": "#C6605A"}
TOP_N = {"Exposure": 5, "Protein_1": 3, "Protein_2": 3, "Disease": 5}
PHASE = {"Exposure": 0.08, "Protein_1": 0.45,
         "Protein_2": 0.20, "Disease": 0.75}
AREA_FACTOR = {"Exposure": 1.0, "Protein_1": 0.44,
               "Protein_2": 0.65, "Disease": 0.8}


def load_network() -> tuple[nx.Graph, dict[str, str], dict[str, str]]:
    nodes = pd.read_csv(DATA / "EPPD_nodes_corrected.csv", dtype={"EXID": str})
    edges = pd.read_csv(DATA / "EPPD_edges_corrected.csv",
                        usecols=["Source", "Target"], dtype=str)
    if not {"EXID", "Group name", "Preferred name"}.issubset(nodes):
        raise ValueError("Corrected EPPD node file is missing required fields")
    nodes = nodes.drop_duplicates("EXID", keep="first")
    groups = dict(zip(nodes["EXID"], nodes["Group name"]))
    names = dict(zip(nodes["EXID"], nodes["Preferred name"].fillna(nodes["EXID"])))
    graph = nx.Graph()
    graph.add_nodes_from(nodes["EXID"].astype(str))
    graph.add_edges_from(edges[["Source", "Target"]].itertuples(index=False, name=None))
    unknown = set(graph) - set(groups)
    if unknown:
        raise ValueError(f"{len(unknown)} edge endpoints are absent from the node table")
    if set(groups.values()) != set(GROUPS):
        raise ValueError(f"Unexpected EPPD groups: {set(groups.values())}")
    return graph, groups, names


def layout(graph: nx.Graph, groups: dict[str, str]) -> dict[str, tuple[float, float]]:
    degree = dict(graph.degree())
    positions = {}
    for group in GROUPS:
        members = sorted((node for node in graph if groups[node] == group),
                         key=lambda node: (-degree[node], node))
        count = len(members)
        top = members[:TOP_N[group]]
        slots = [round((i + PHASE[group]) * count / len(top)) % count
                 for i in range(len(top))]
        used = set(slots)
        free = [i for i in range(count) if i not in used]
        order = list(zip(top, slots)) + list(zip(members[len(top):], free))
        for node, slot in order:
            angle = np.pi / 2 + 2 * np.pi * slot / count
            positions[node] = (RADII[group] * np.cos(angle),
                               RADII[group] * np.sin(angle))
    return positions


def node_area(group: str, degree: int) -> float:
    return (9 + 4.8 * np.sqrt(degree)) * AREA_FACTOR[group]


def draw(graph: nx.Graph, groups: dict[str, str], names: dict[str, str],
         output_dir: Path, dpi: int) -> None:
    degree = dict(graph.degree())
    positions = layout(graph, groups)
    top = {group: sorted((n for n in graph if groups[n] == group),
                         key=lambda n: (-degree[n], n))[:TOP_N[group]]
           for group in GROUPS}
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.unicode_minus": False})
    fig = plt.figure(figsize=(13.2, 12.2), facecolor="white")
    ax = fig.add_axes([0.025, 0.073, 0.81, 0.83])
    ax.set_aspect("equal")
    ax.set_xlim(-5.35, 5.35)
    ax.set_ylim(-4.95, 4.95)
    ax.axis("off")

    # Draw all unique links, separated by actual layer transition. Rasterizing
    # the links makes the vector PDF smaller while preserving editable labels.
    segments: dict[tuple[str, str], list] = {}
    for source, target in graph.edges:
        pair = tuple(sorted((groups[source], groups[target])))
        segments.setdefault(pair, []).append([positions[source], positions[target]])
    for pair, lines in segments.items():
        if not lines:
            continue
        alpha = 0.060 if "Exposure" in pair else 0.035 if "Disease" in pair else 0.022
        ax.add_collection(LineCollection(lines, colors="#788797", linewidths=0.18,
                                         alpha=alpha, rasterized=True, zorder=1))

    for group in GROUPS:
        ax.add_patch(Circle((0, 0), RADII[group], fill=False, lw=0.5,
                            ls=(0, (3, 4)), ec="#BDC4CA", zorder=2))
        members = [n for n in graph if groups[n] == group]
        ax.scatter([positions[n][0] for n in members],
                   [positions[n][1] for n in members],
                   s=[node_area(group, degree[n]) for n in members],
                   c=COLORS[group], alpha=0.86, edgecolors="white",
                   linewidths=0.13, rasterized=True, zorder=4)
        ax.scatter([positions[n][0] for n in top[group]],
                   [positions[n][1] for n in top[group]],
                   s=[node_area(group, degree[n]) for n in top[group]],
                   facecolors="none", edgecolors="#172938", linewidths=0.7, zorder=5)

    fig.text(0.055, 0.946, "EPPD network", fontsize=17, fontweight="bold", color="#193047")
    fig.text(0.055, 0.918, "Exposure  ·  Protein 1  ·  Protein 2  ·  Disease",
             fontsize=10.5, color="#5B6770")

    label_radius = {"Exposure": 4.52, "Protein_1": 3.40,
                    "Protein_2": 2.36, "Disease": 1.40}
    for group in GROUPS:
        for node in top[group]:
            x, y = positions[node]
            angle = np.arctan2(y, x)
            lx = label_radius[group] * np.cos(angle)
            ly = label_radius[group] * np.sin(angle)
            horizontal = "left" if lx > 0.25 else "right" if lx < -0.25 else "center"
            ax.annotate(" ".join(str(names[node]).split()), xy=(x, y),
                        xytext=(lx, ly), textcoords="data", ha=horizontal, va="center",
                        fontsize=8.2, color="#243642", zorder=8,
                        arrowprops={"arrowstyle": "-", "color": COLORS[group],
                                    "lw": 0.5, "alpha": 0.78,
                                    "shrinkA": 11, "shrinkB": 4})

    category_legend = [Line2D([], [], marker="o", ls="", color=COLORS[group],
                              markersize=6, label=group.replace("_", " "))
                       for group in GROUPS]
    ax.legend(handles=category_legend, loc="lower left", bbox_to_anchor=(-0.02, -0.04),
              ncol=4, frameon=False, fontsize=8.6, columnspacing=1.3, handletextpad=0.4)

    fig.text(0.848, 0.797, "DEGREE", fontsize=10, fontweight="bold", color="#193047")
    examples = [
        ("Exposure", 0.735, [1, 250, 900]),
        ("Protein_1", 0.565, [1, 100, 300]),
        ("Protein_2", 0.395, [1, 30, 70]),
        ("Disease", 0.225, [1, 100, 250]),
    ]
    for group, y0, values in examples:
        fig.text(0.848, y0 + 0.033, group.replace("_", " "), fontsize=9,
                 fontweight="bold", color=COLORS[group])
        for i, value in enumerate(values):
            y = y0 - 0.037 * i
            fig.add_artist(Line2D([0.866], [y], marker="o", ls="",
                                  markersize=np.sqrt(node_area(group, value)),
                                  color=COLORS[group], transform=fig.transFigure))
            fig.text(0.915, y, f"{value:,}", fontsize=8.2,
                     va="center", color="#52606A")

    fig.text(0.848, 0.080,
             f"{graph.number_of_nodes():,} nodes  ·  {graph.number_of_edges():,} unique links\n"
             "Node area increases with graph degree.",
             fontsize=8.2, color="#52606A", linespacing=1.45)
    fig.text(0.055, 0.028,
             "Topology reflects database connectivity; it does not establish toxicity or causality.",
             fontsize=8.4, color="#69757D")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "EPPD_network_layered_circular_optimized"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=dpi, facecolor="white")
    plt.close(fig)
    rows = [{"Group": group, "Rank": rank, "EXID": node,
             "Preferred_name": names[node], "Unique_graph_degree": degree[node]}
            for group in GROUPS for rank, node in enumerate(top[group], 1)]
    pd.DataFrame(rows).to_csv(output_dir / "highlighted_nodes.csv",
                              index=False, encoding="utf-8-sig")
    print(f"Saved {stem}.png/.pdf/.svg and highlighted_nodes.csv")
    print(f"Network: {graph.number_of_nodes():,} nodes, "
          f"{graph.number_of_edges():,} unique links")


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
