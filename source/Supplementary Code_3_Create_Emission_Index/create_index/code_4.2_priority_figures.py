# =============================================================================
# PART 2: PRIORITY-RANKING FIGURE SUITE
# =============================================================================

#%%
'\ncode_4.2_priority_figs.py\n------------------------------\nGenerate a suite of publication-quality priority-ranking figures from the\nPriority Index and Top-50% target list. The original code_4 outputs are not\noverwritten; results are written to code_4_results_update/.\n'

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE_DIR = Path(__file__).resolve().parent
INPUT = BASE_DIR / "code_3_results" / "3_Final_Top50pct_Categorized_Targets.csv"
OUT = BASE_DIR / "code_4_results"
FIG = OUT / "code_4.2_priority_figures"

CAT_SHORT = {
    "Heavy metal": "Heavy metal",
    "Halogenated carbazole": "HCZ",
    "Metalloid": "Metalloid",
    "Mineral Dust": "Mineral dust",
    "Halogen": "Halogen",
    "Halogenated aliphatic hydrocarbon": "HAH",
    "Light metal": "Light metal",
    "Parent polycyclic aromatic hydrocarbon": "pPAH",
    "Methylated polycyclic aromatic hydrocarbon": "MePAH",
}

SECTOR_COLOR = {
    "CK": "#2171b5",
    "ico": "#238b45",
    "MSWI": "#6a51a3",
    "pow": "#cb181d",
    "ipr": "#b35806",
    "Other": "#636363",
}
SECTOR_LABEL = {
    "CK": "CK (coke oven)",
    "ico": "ico (iron / industrial)",
    "MSWI": "MSWI (waste incineration)",
    "pow": "pow (power plants)",
    "ipr": "ipr (non-ferrous)",
    "Other": "Other",
}

CMAP_BLUE = mcolors.LinearSegmentedColormap.from_list(
    "nat_blue", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Times New Roman", "Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["sector"] = d["Exposure"].astype(str).apply(
        lambda s: s.split("_")[-1] if "_" in s else "Other"
    )
    d["pol"] = d["name_en_abbr"].fillna(
        d["Exposure"].astype(str).str.replace(r"^X_", "", regex=True).str.replace(r"_[^_]+$", "", regex=True)
    )
    d["pol"] = d["pol"].replace({"Hexachlorobutadiene": "HCBD"})
    d["cat_short"] = d["sub category_en"].map(CAT_SHORT).fillna(d["sub category_en"])
    # Implementation note.
    tot = d.groupby("Disease_Name")["Priority_Index"].transform("sum")
    d["share_pct"] = 100.0 * d["Priority_Index"] / tot
    return d


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png")
    fig.savefig(FIG / f"{stem}.pdf")
    plt.close(fig)
    print(f"  saved: {stem}")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig02_stacked_class(d: pd.DataFrame) -> None:
    mat = (
        d.groupby(["Disease_Name", "cat_short"], as_index=False)["Priority_Index"]
        .sum()
    )
    # Implementation note.
    mat["pct"] = mat.groupby("Disease_Name")["Priority_Index"].transform(
        lambda x: 100.0 * x / x.sum()
    )
    disease_order = (
        mat.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index.tolist()
    )
    cat_order = (
        mat.groupby("cat_short")["Priority_Index"].sum().sort_values(ascending=False).index.tolist()
    )
    pivot = mat.pivot(index="Disease_Name", columns="cat_short", values="pct").fillna(0)
    pivot = pivot.reindex(index=disease_order, columns=cat_order)

    colors = sns.color_palette("tab10", n_colors=len(cat_order))
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for i, cat in enumerate(cat_order):
        vals = pivot[cat].to_numpy()
        ax.barh(y, vals, left=left, color=colors[i], height=0.72, edgecolor="white", linewidth=0.4, label=cat)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Within-disease contribution (%)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_title("Chemical-class composition of priority risk", fontsize=12, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(5, len(cat_order)),
        frameon=False,
        fontsize=7.5,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "02_stacked_contribution_by_class")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig03_bio_epi(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    diseases = sorted(d["Disease_Name"].unique())
    palette = dict(zip(diseases, sns.color_palette("tab10", n_colors=len(diseases))))
    # Implementation note.
    pmin, pmax = d["Priority_Index"].min(), d["Priority_Index"].max()
    sizes = 40 + 220 * (d["Priority_Index"] - pmin) / (pmax - pmin + 1e-9)

    for dis in diseases:
        sub = d[d["Disease_Name"] == dis]
        s = 40 + 220 * (sub["Priority_Index"] - pmin) / (pmax - pmin + 1e-9)
        ax.scatter(
            sub["Unified_Bio_Score"],
            sub["Epi_Score"],
            s=s,
            c=[palette[dis]],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            label=dis,
            zorder=3,
        )
 
    # Implementation note.
    top = d.nlargest(8, "Priority_Index")
    for _, r in top.iterrows():
        ax.annotate(
            f"{r['pol']}|{r['Disease_Name'][:3]}",
            (r["Unified_Bio_Score"], r["Epi_Score"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6.5,
            color="#222222",
        )

    ax.set_xlabel("Biological evidence score", fontsize=10)
    ax.set_ylabel("Epidemiological score", fontsize=10)
    ax.set_title("Priority landscape: biology × epidemiology", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, title="Disease", title_fontsize=8, loc="best")
    # Implementation note.
    for pi, lab in [(pmin, "low PI"), (pmax, "high PI")]:
        s = 40 + 220 * (pi - pmin) / (pmax - pmin + 1e-9)
        ax.scatter([], [], s=s, c="#888888", alpha=0.5, label=lab)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, ls=":", lw=0.5, color="#cccccc", zorder=0)
    save(fig, "03_bio_vs_epi_priority_scatter")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig04_pareto(d: pd.DataFrame) -> None:
    diseases = (
        d.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index.tolist()
    )
    n = len(diseases)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.0 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, dis in zip(axes, diseases):
        sub = d[d["Disease_Name"] == dis].sort_values("Priority_Index", ascending=False).copy()
        sub["cum"] = 100.0 * sub["Priority_Index"].cumsum() / sub["Priority_Index"].sum()
        x = np.arange(1, len(sub) + 1)
        ax.fill_between(x, sub["cum"], color="#6baed6", alpha=0.35)
        ax.plot(x, sub["cum"], color="#08519c", lw=1.8, marker="o", ms=3.5)
        ax.axhline(50, color="#cb181d", ls="--", lw=0.9, alpha=0.8)
        ax.set_title(dis, fontsize=10, fontweight="bold")
        ax.set_xlabel("Ranked targets", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_xlim(0.5, max(len(sub), 1) + 0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        # Implementation note.
        n50 = int((sub["cum"] <= 50).sum()) + 1
        n50 = min(n50, len(sub))
        ax.text(0.97, 0.08, f"n50~{n50}/{len(sub)}", transform=ax.transAxes, ha="right", fontsize=7.5, color="#444")

    for j in range(len(diseases), len(axes)):
        axes[j].set_axis_off()
    axes[0].set_ylabel("Cumulative contribution (%)", fontsize=9)
    fig.suptitle("Pareto curves of priority targets (50% line dashed)", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "04_pareto_cumulative_by_disease")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig05_top_lollipop(d: pd.DataFrame, top_n: int = 25) -> None:
    top = d.nlargest(top_n, "Priority_Index").iloc[::-1].copy()
    top["label"] = top["pol"] + " · " + top["Disease_Name"]
    colors = [SECTOR_COLOR.get(s, "#666") for s in top["sector"]]

    fig, ax = plt.subplots(figsize=(8.2, 0.34 * top_n + 1.4))
    y = np.arange(len(top))
    ax.hlines(y, 0, top["Priority_Index"], color="#bdbdbd", lw=1.2, zorder=1)
    ax.scatter(top["Priority_Index"], y, c=colors, s=55, zorder=3, edgecolors="white", linewidths=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(top["label"], fontsize=8)
    ax.set_xlabel("Priority Index", fontsize=10)
    ax.set_title(f"Top {top_n} prioritized exposure–disease pairs", fontsize=12, fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SECTOR_COLOR[s], markersize=7, label=SECTOR_LABEL[s])
        for s in ["CK", "ico", "MSWI", "pow"]
        if s in set(top["sector"])
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlim(0, top["Priority_Index"].max() * 1.08)
    save(fig, "05_top_priority_lollipop")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig06_heatmap(d: pd.DataFrame) -> None:
    # Implementation note.
    pol_rank = d.groupby("Exposure")["Priority_Index"].sum().sort_values(ascending=False)
    keep = pol_rank.head(18).index.tolist()
    sub = d[d["Exposure"].isin(keep)]
    mat = sub.pivot_table(index="Disease_Name", columns="Exposure", values="Priority_Index", aggfunc="max")
    # Implementation note.
    dis_order = mat.sum(axis=1).sort_values(ascending=False).index
    pol_order = mat.sum(axis=0).sort_values(ascending=False).index
    mat = mat.reindex(index=dis_order, columns=pol_order)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.heatmap(
        mat,
        ax=ax,
        cmap=CMAP_BLUE,
        linewidths=0.5,
        linecolor="white",
        mask=mat.isna(),
        cbar_kws={"label": "Priority Index", "shrink": 0.75},
        square=False,
    )
    ax.set_xlabel("Pollutant", fontsize=10)
    ax.set_ylabel("Disease", fontsize=10)
    ax.set_title("Priority Index heatmap (selected pollutants)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    plt.setp(ax.get_xticklabels(), ha="right")
    save(fig, "06_priority_heatmap_disease_pollutant")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig07_sector_stack(d: pd.DataFrame) -> None:
    mat = d.groupby(["Disease_Name", "sector"], as_index=False)["Priority_Index"].sum()
    mat["pct"] = mat.groupby("Disease_Name")["Priority_Index"].transform(lambda x: 100 * x / x.sum())
    dis_order = mat.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index
    sectors = [s for s in ["CK", "ico", "MSWI", "pow", "ipr", "Other"] if s in set(mat["sector"])]
    pivot = mat.pivot(index="Disease_Name", columns="sector", values="pct").fillna(0)
    pivot = pivot.reindex(index=dis_order, columns=sectors)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for sec in sectors:
        vals = pivot[sec].to_numpy()
        ax.barh(y, vals, left=left, color=SECTOR_COLOR.get(sec, "#666"), height=0.7, edgecolor="white", lw=0.4, label=SECTOR_LABEL.get(sec, sec))
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Within-disease contribution (%)", fontsize=10)
    ax.set_title("Industrial sector drivers of prioritized targets", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "07_stacked_contribution_by_sector")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig08_top5_panels(d: pd.DataFrame) -> None:
    diseases = (
        d.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index.tolist()
    )
    n = len(diseases)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, dis in zip(axes, diseases):
        sub = d[d["Disease_Name"] == dis].nlargest(5, "Priority_Index").iloc[::-1]
        colors = [SECTOR_COLOR.get(s, "#666") for s in sub["sector"]]
        ax.barh(sub["pol"], sub["Priority_Index"], color=colors, height=0.65, edgecolor="white", lw=0.4)
        ax.set_title(dis, fontsize=10, fontweight="bold")
        ax.set_xlabel("Priority Index", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for j in range(len(diseases), len(axes)):
        axes[j].set_axis_off()

    handles = [
        Patch(facecolor=SECTOR_COLOR[s], edgecolor="white", label=SECTOR_LABEL[s])
        for s in ["CK", "ico", "MSWI", "pow"]
        if s in set(d["sector"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Top-5 priority targets by disease", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "08_top5_by_disease")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig09_rr_priority(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    diseases = sorted(d["Disease_Name"].unique())
    palette = dict(zip(diseases, sns.color_palette("tab10", n_colors=len(diseases))))
    for dis in diseases:
        sub = d[d["Disease_Name"] == dis]
        ax.scatter(
            sub["RR"],
            sub["Priority_Index"],
            c=[palette[dis]],
            s=42,
            alpha=0.75,
            edgecolors="white",
            lw=0.4,
            label=dis,
        )
    ax.axvline(1.0, color="#999999", ls="--", lw=0.8)
    ax.set_xlabel("Relative risk (RR)", fontsize=10)
    ax.set_ylabel("Priority Index", fontsize=10)
    ax.set_title("Epidemiological effect size vs priority rank score", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="best")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, ls=":", lw=0.45, color="#cccccc")
    save(fig, "09_RR_vs_priority")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig10_count_tile(d: pd.DataFrame) -> None:
    ct = d.groupby(["Disease_Name", "cat_short"]).size().reset_index(name="n")
    dis_order = d.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index
    cat_order = d.groupby("cat_short")["Priority_Index"].sum().sort_values(ascending=False).index
    mat = ct.pivot(index="Disease_Name", columns="cat_short", values="n").fillna(0)
    mat = mat.reindex(index=dis_order, columns=cat_order)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    sns.heatmap(
        mat,
        ax=ax,
        cmap="Greys",
        annot=True,
        fmt=".0f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "No. of selected targets", "shrink": 0.75},
        annot_kws={"fontsize": 8},
    )
    ax.set_title("Number of top-50% priority targets", fontsize=12, fontweight="bold")
    ax.set_xlabel("Chemical class", fontsize=10)
    ax.set_ylabel("Disease", fontsize=10)
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    plt.setp(ax.get_xticklabels(), ha="right")
    save(fig, "10_count_matrix_disease_class")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig11_score_decomposition(d: pd.DataFrame) -> None:
    top = d.nlargest(15, "Priority_Index").iloc[::-1].copy()
    top["label"] = top["pol"] + " · " + top["Disease_Name"]
    y = np.arange(len(top))
    h = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(y + h, top["Unified_Bio_Score"], height=h, color="#9ecae1", label="Bio score", edgecolor="white", lw=0.3)
    ax.barh(y, top["Epi_Score"], height=h, color="#fc9272", label="Epi score", edgecolor="white", lw=0.3)
    ax.barh(y - h, top["Priority_Index"], height=h, color="#08519c", label="Priority Index", edgecolor="white", lw=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(top["label"], fontsize=8)
    ax.set_xlabel("Score", fontsize=10)
    ax.set_title("Score decomposition for top-15 priority pairs", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "11_score_decomposition_top15")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig12_integrated_atlas(d: pd.DataFrame) -> None:
    """
    Integrated overview of Bio, Epi and Priority scores:
    A: Bio versus Epi; point size represents Priority and colour represents disease (all targets).
    B: Three-score heatmap for all exposure–disease pairs (rows=pairs; columns=Bio/Epi/Priority).
    """
    diseases = (
        d.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index.tolist()
    )
    palette = dict(zip(diseases, sns.color_palette("tab10", n_colors=len(diseases))))

    # Implementation note.
    rows = []
    for dis in diseases:
        sub = d[d["Disease_Name"] == dis].sort_values("Priority_Index", ascending=False)
        for _, r in sub.iterrows():
            rows.append(r)
    ordered = pd.DataFrame(rows)
    ordered["row_label"] = ordered["Exposure"] + " · " + ordered["Disease_Name"]
    score_mat = ordered[["Unified_Bio_Score", "Epi_Score", "Priority_Index"]].to_numpy(dtype=float)
    score_df = pd.DataFrame(
        score_mat,
        index=ordered["row_label"],
        columns=["Bio", "Epi", "Priority"],
    )

    n = len(score_df)
    fig = plt.figure(figsize=(12.5, max(8.5, 0.175 * n + 2.5)))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.05, 1.35],
        wspace=0.28,
        left=0.07,
        right=0.92,
        top=0.93,
        bottom=0.06,
    )
    ax_sc = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])

    # ---- A: scatter ----
    pmin, pmax = d["Priority_Index"].min(), d["Priority_Index"].max()
    for dis in diseases:
        sub = d[d["Disease_Name"] == dis]
        s = 35 + 260 * (sub["Priority_Index"] - pmin) / (pmax - pmin + 1e-9)
        ax_sc.scatter(
            sub["Unified_Bio_Score"],
            sub["Epi_Score"],
            s=s,
            c=[palette[dis]],
            alpha=0.78,
            edgecolors="white",
            linewidths=0.45,
            label=dis,
            zorder=3,
        )
    # Implementation note.
    for pi, lab in [(pmin, "Low priority"), ((pmin + pmax) / 2, "Mid"), (pmax, "High priority")]:
        s = 35 + 260 * (pi - pmin) / (pmax - pmin + 1e-9)
        ax_sc.scatter([], [], s=s, c="#888888", alpha=0.55, label=lab)

    ax_sc.set_xlabel("Biological evidence score (Bio)", fontsize=9)
    ax_sc.set_ylabel("Epidemiological score (Epi)", fontsize=9)
    ax_sc.set_title("A  Priority landscape", fontsize=11, fontweight="bold", loc="left")
    ax_sc.grid(True, ls=":", lw=0.45, color="#cccccc", zorder=0)
    for spine in ("top", "right"):
        ax_sc.spines[spine].set_visible(False)
    leg = ax_sc.legend(
        frameon=False,
        fontsize=6.5,
        loc="upper left",
        title="Disease / size",
        title_fontsize=7,
        markerscale=0.9,
    )

    # ---- B: heatmap of three scores for ALL pairs ----
    sns.heatmap(
        score_df,
        ax=ax_hm,
        cmap=CMAP_BLUE,
        vmin=0,
        vmax=max(1.0, float(np.nanmax(score_mat))),
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Score", "shrink": 0.45, "pad": 0.02},
        yticklabels=True,
    )
    ax_hm.set_title("B  Score decomposition (all pairs)", fontsize=11, fontweight="bold", loc="left")
    ax_hm.set_xlabel("")
    ax_hm.set_ylabel("")
    ax_hm.tick_params(axis="y", labelsize=5.5)
    ax_hm.tick_params(axis="x", labelsize=9, labelrotation=0)
    # Implementation note.
    counts = ordered.groupby("Disease_Name", sort=False).size()
    # ordered diseases follow `diseases` order
    ypos = 0
    for dis in diseases:
        ypos += int(counts.get(dis, 0))
        if ypos < n:
            ax_hm.axhline(ypos, color="#333333", lw=0.6)

    fig.suptitle(
        "Integrated Bio / Epi / Priority atlas (all prioritized pairs)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    save(fig, "12_integrated_bio_epi_priority_atlas")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig13_decomposition_all_by_disease(d: pd.DataFrame) -> None:
    diseases = (
        d.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index.tolist()
    )
    n = len(diseases)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    # Implementation note.
    max_per = int(d.groupby("Disease_Name").size().max())
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12.0, max(3.2, 0.28 * max_per) * nrows),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()
    h = 0.22

    for ax, dis in zip(axes, diseases):
        sub = d[d["Disease_Name"] == dis].sort_values("Priority_Index", ascending=True)
        y = np.arange(len(sub))
        ax.barh(y + h, sub["Unified_Bio_Score"], height=h, color="#9ecae1", edgecolor="white", lw=0.2)
        ax.barh(y, sub["Epi_Score"], height=h, color="#fc9272", edgecolor="white", lw=0.2)
        ax.barh(y - h, sub["Priority_Index"], height=h, color="#08519c", edgecolor="white", lw=0.2)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["pol"], fontsize=15)
        ax.set_title(f"{dis} (n={len(sub)})", fontsize=15, fontweight="bold")
        ax.set_xlim(0, max(1.05, d[["Unified_Bio_Score", "Epi_Score", "Priority_Index"]].max().max() * 1.02))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, axis="x", ls=":", lw=0.4, color="#dddddd")

    for j in range(len(diseases), len(axes)):
        axes[j].set_axis_off()

    handles = [
        Patch(facecolor="#9ecae1", edgecolor="white", label="Bio score"),
        Patch(facecolor="#fc9272", edgecolor="white", label="Epi score"),
        Patch(facecolor="#08519c", edgecolor="white", label="Priority Index"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Bio / Epi / Priority decomposition — all pollutants by disease",
        fontsize=20,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save(fig, "13_decomposition_all_by_disease")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
from adjustText import adjust_text

# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig14_rr_priority_bio_size(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    diseases = sorted(d["Disease_Name"].unique())
    
    # Implementation note.
    palette = dict(zip(diseases, sns.color_palette("tab10", n_colors=len(diseases))))
    bmin, bmax = d["Unified_Bio_Score"].min(), d["Unified_Bio_Score"].max()

    # Implementation note.
    for dis in diseases:
        sub = d[d["Disease_Name"] == dis]
        # Implementation note.
        s = 50 + 400 * (sub["Unified_Bio_Score"] - bmin) / (bmax - bmin + 1e-9)
        ax.scatter(
            sub["RR"],
            sub["Priority_Index"],
            s=s,
            c=[palette[dis]],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

    # Implementation note.
    # texts = []
    # for _, r in d.nsmallest(20, "Priority_Index").iterrows():
    # Implementation note.
    #     label_text = r["Exposure"][2:] if str(r["Exposure"]).startswith("X_") else r["Exposure"]
    #     texts.append(
    #         ax.text(
    #             r["RR"], 
    #             r["Priority_Index"], 
    #             label_text, 
    #             fontsize=7.5, 
    #             color="#333333",
    #             fontweight="medium",
    #             zorder=4
    #         )
    #     )
    
    # Implementation note.
    # adjust_text(
    #     texts, 
    #     ax=ax, 
    #     expand_points=(1.5, 1.5),
    #     arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, alpha=0.8)
    # )

    # Implementation note.
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=dis,
               markerfacecolor=palette[dis], markersize=9, alpha=0.9)
        for dis in diseases
    ]
    ax.legend(
        handles=legend_elements, 
        frameon=False, 
        fontsize=8.5, 
        ncol=2, 
        loc="center left", 
        bbox_to_anchor=(1.02, 0.5), # Implementation note.
        title="Disease Category", 
        title_fontproperties={'weight':'bold', 'size':9}
    )

    # Implementation note.
    ax.axvline(1.0, color="#bbbbbb", ls="--", lw=1.2, zorder=1)
    ax.set_xlabel("Relative risk (RR)", fontsize=11, fontweight='medium', labelpad=8)
    ax.set_ylabel("Priority Index", fontsize=11, fontweight='medium', labelpad=8)
    ax.set_title("Epidemiological effect size vs Priority Index", fontsize=13, fontweight="bold", pad=15)
    
    # Implementation note.
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color("#333333")

    # Implementation note.
    # ax.text(
    #     0.98, 0.02,
    #     "Circle size encodes Biological Evidence Score",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="bottom",
    #     fontsize=8,
    #     color="#777777",
    #     style="italic"
    # )
    
    fig.tight_layout()
    save(fig, "14_RR_vs_priority_sizeBio")


def write_readme() -> None:
    text = """# Priority ranking figures (code_4_update)

Data: code_3_results/3_Final_Top50pct_Categorized_Targets.csv

| File | Content |
|------|---------|
| 01 (existing) | Equal-size exposome matrix, fill = contribution |
| 02_stacked_contribution_by_class | Disease × chemical-class stacked % |
| 03_bio_vs_epi_priority_scatter | Biology vs epidemiology landscape |
| 04_pareto_cumulative_by_disease | Cumulative contribution / Pareto |
| 05_top_priority_lollipop | Global top priority pairs |
| 06_priority_heatmap_disease_pollutant | Disease × pollutant Priority heatmap |
| 07_stacked_contribution_by_sector | Disease × sector stacked % |
| 08_top5_by_disease | Top-5 targets per disease |
| 09_RR_vs_priority | RR versus Priority Index |
| 10_count_matrix_disease_class | Counts of selected targets |
| 11_score_decomposition_top15 | Bio / Epi / Priority bars (top 15) |
| 12_integrated_bio_epi_priority_atlas | COMPREHENSIVE: scatter + all-pair score heatmap |
| 13_decomposition_all_by_disease | COMPREHENSIVE: fig11-style bars for ALL pairs by disease |
| 14_RR_vs_priority_sizeBio | RR vs Priority, size = Bio (three metrics) |
"""
    (OUT / "README_priority_figures.txt").write_text(text, encoding="utf-8")

#%%
def main() -> None:
    set_style()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    d = prep(pd.read_csv(INPUT))
    d.to_csv(OUT / "2_priority_targets_plotready.csv", index=False)

    print(">>> plotting priority figure suite ...")
    fig02_stacked_class(d)
    fig03_bio_epi(d)
    fig04_pareto(d)
    fig05_top_lollipop(d)
    fig06_heatmap(d)
    fig07_sector_stack(d)
    fig08_top5_panels(d)
    fig09_rr_priority(d)
    fig10_count_tile(d)
    fig11_score_decomposition(d)
    fig12_integrated_atlas(d)
    fig13_decomposition_all_by_disease(d)
    fig14_rr_priority_bio_size(d)
    write_readme()
    print(f"done → {FIG}")

if __name__ == "__main__":
    main()

# %%
