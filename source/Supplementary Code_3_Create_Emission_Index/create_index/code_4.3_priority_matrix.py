# =============================================================================
# PART 3: EXPOSOME PRIORITY MATRIX
# =============================================================================

#%%
'\ncode_4.3_priority_matrix.py\n----------------\nGenerate a publication-quality exposome priority matrix without overwriting\nthe original code_4 outputs.\n- All circles have equal size.\n- Fill intensity represents within-disease contribution (%).\n- Pollutant names are displayed without percentage labels.\n- Circle-edge colour represents the primary emission sector.\nResults are written to code_4_results_update/.\n'

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "code_3_results" / "3_Final_Top50pct_Categorized_Targets.csv"
OUT_DIR = BASE_DIR / "code_4_results" / "code_4.3_priority_matrix"

# Implementation note.
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

# Implementation note.
SECTOR_STYLE = {
    "pow": {"edge": "#2171b5", "label": "CK (coke oven)"},
    "ico": {"edge": "#238b45", "label": "ico (iron / industrial)"},
    "MSWI": {"edge": "#6a51a3", "label": "MSWI (waste incineration)"},
    "CK": {"edge": "#cb181d", "label": "pow (power plants)"},
    "ipr": {"edge": "#b35806", "label": "ipr (non-ferrous)"},
    "Other": {"edge": "#636363", "label": "Other"},
}

# Implementation note.
NATURE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "nature_contrib",
    ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Times new roman"],
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


ABBR_FIX = {
    "Hexachlorobutadiene": "HCBD",
    "1,3,6,8-CCZ": "1368-CCZ",
    "1,3,6-BCZ": "136-BCZ",
}


def format_pollutant_label(names: list[str], max_n: int = 3) -> str:
    'Generate the corresponding publication-quality visualization.'
    cleaned = []
    for n in names:
        n = ABBR_FIX.get(n, n)
        if n not in cleaned:
            cleaned.append(n)
    # Implementation note.
    if any(len(n) > 7 for n in cleaned):
        cleaned = cleaned[:2]
    else:
        cleaned = cleaned[:max_n]
    return "\n".join(cleaned)


def get_primary_sector(x: pd.Series) -> str:
    sectors = x.astype(str).apply(lambda s: s.split("_")[-1] if "_" in s else "Other")
    mode = sectors.mode()
    return mode.iloc[0] if not mode.empty else "Other"

def aggregate(df_raw: pd.DataFrame):
    disease_col = "Disease_Name"
    category_col = "sub category_en"

    df_raw = df_raw.sort_values(
        by=[disease_col, category_col, "Priority_Index"],
        ascending=[True, True, False],
    )

    def _top_names(g: pd.DataFrame) -> str:
        # Implementation note.
        if "name_en_abbr" in g.columns:
            names = g["name_en_abbr"].astype(str).tolist()
        else:
            names = (
                g["Exposure"]
                .astype(str)
                .str.replace(r"^X_", "", regex=True)
                .str.replace(r"_[^_]+$", "", regex=True)
                .tolist()
            )
        return format_pollutant_label(names, 3)

    rows = []
    for (dis, cat), g in df_raw.groupby([disease_col, category_col], sort=False):
        rows.append(
            {
                disease_col: dis,
                category_col: cat,
                "Priority_Index_Sum": g["Priority_Index"].sum(),
                "Contribution_Pct": g["Contribution_Pct"].sum(),
                "Top3_Unique_Pollutants": _top_names(g),
                "Primary_Sector": get_primary_sector(g["Exposure"]),
            }
        )
    df_summary = pd.DataFrame(rows)

    disease_totals = (
        df_raw.groupby(disease_col)["Priority_Index"]
        .sum()
        .reset_index(name="Group_Total_Sum")
    )
    df_summary = df_summary.merge(disease_totals, on=disease_col, how="left")
    df_summary["Contribution_Pct"] = (
        df_summary["Priority_Index_Sum"] / df_summary["Group_Total_Sum"] * 100.0
    )

    disease_order = (
        df_summary.groupby(disease_col)["Priority_Index_Sum"]
        .sum()
        .sort_values(ascending=False)
        .index
    )
    category_order = (
        df_summary.groupby(category_col)["Priority_Index_Sum"]
        .sum()
        .sort_values(ascending=False)
        .index
    )

    df_summary[disease_col] = pd.Categorical(
        df_summary[disease_col], categories=disease_order, ordered=True
    )
    df_summary[category_col] = pd.Categorical(
        df_summary[category_col], categories=category_order, ordered=True
    )
    df_summary["x_coord"] = df_summary[category_col].map(
        {c: i for i, c in enumerate(category_order)}
    )
    df_summary["y_coord"] = df_summary[disease_col].map(
        {d: i for i, d in enumerate(disease_order)}
    )
    return df_summary, list(disease_order), list(category_order)


def _text_color(rgba) -> str:
    """Automatically select black or white text according to fill luminance."""
    r, g, b = rgba[:3]
    # Implementation note.
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if lum > 0.55 else "#ffffff"


def plot_matrix(df: pd.DataFrame, disease_order: list, category_order: list) -> None:
    n_x, n_y = len(category_order), len(disease_order)
    fig = plt.figure(figsize=(11.2, 8.6), dpi=300)
    # Implementation note.
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[4.6, 1.15],
        wspace=0.08,
        left=0.10,
        right=0.98,
        top=0.92,
        bottom=0.16,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    ax.set_xlim(-0.72, n_x - 0.28)
    ax.set_ylim(n_y - 0.28, -0.72)
    ax.set_xticks(range(n_x))
    ax.set_yticks(range(n_y))
    ax.set_xticklabels(
        [CAT_SHORT.get(c, c) for c in category_order],
        rotation=35,
        ha="right",
        fontsize=9,
        color="#222222",
    )
    ax.set_yticklabels(disease_order, fontsize=10, color="#222222")
    ax.tick_params(length=0)
    ax.set_xlabel("Chemical class", fontsize=10, color="#333333", labelpad=8)
    ax.set_ylabel("Disease outcome", fontsize=10, color="#333333", labelpad=8)

    # Implementation note.
    ax.set_xticks(np.arange(-0.5, n_x, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_y, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.35, color="#efefef", zorder=0)
    ax.grid(False, which="major")
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.7)

    vmin, vmax = 0.0, 100.0
    cmap = NATURE_CMAP
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    bubble_size = 1850
    for _, row in df.iterrows():
        pct = float(row["Contribution_Pct"])
        face = cmap(norm(pct))
        sec = str(row["Primary_Sector"])
        edge = SECTOR_STYLE.get(sec, SECTOR_STYLE["Other"])["edge"]
        ax.scatter(
            float(row["x_coord"]),
            float(row["y_coord"]),
            s=bubble_size,
            c=[face],
            edgecolors=edge,
            linewidths=1.35,
            alpha=1.0,
            zorder=3,
        )
        labels = str(row["Top3_Unique_Pollutants"]).strip()
        if labels and labels.lower() != "nan":
            n_lines = labels.count("\n") + 1
            fs = 9.0 if n_lines >= 3 else (9 if n_lines == 2 else 9)
            ax.text(
                float(row["x_coord"]),
                float(row["y_coord"]),
                labels,
                ha="center",
                va="center",
                fontsize=fs,
                fontweight="semibold",
                color=_text_color(face),
                linespacing=1.12,
                zorder=4,
                clip_on=True,
            )

    # Implementation note.
    y0 = 0.98
    ax_leg.text(
        0.0,
        y0,
        "Contribution",
        fontsize=9,
        fontweight="bold",
        color="#222222",
        transform=ax_leg.transAxes,
        va="top",
    )
    # Implementation note.
    cax = ax_leg.inset_axes([0.08, 0.52, 0.18, 0.38])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.ax.tick_params(labelsize=7.5, length=2.5, width=0.6, colors="#333333")
    cbar.outline.set_linewidth(0.6)
    ax_leg.text(
        0.42,
        0.71,
        "(%)",
        fontsize=8,
        color="#444444",
        transform=ax_leg.transAxes,
        va="center",
    )

    ax_leg.text(
        0.0,
        0.44,
        "Primary sector\n(circle edge)",
        fontsize=9,
        fontweight="bold",
        color="#222222",
        transform=ax_leg.transAxes,
        va="top",
        linespacing=1.25,
    )

    # Implementation note.
    sector_order = ["CK", "ico", "MSWI", "pow", "ipr", "Other"]
    present = set(df["Primary_Sector"].astype(str))
    sectors_show = [s for s in sector_order if s in present]
    for s in sorted(present):
        if s not in sectors_show:
            sectors_show.append(s)

    handles = []
    labels = []
    for sec in sectors_show:
        sty = SECTOR_STYLE.get(sec, SECTOR_STYLE["Other"])
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=9,
                markerfacecolor="#f7fbff",
                markeredgecolor=sty["edge"],
                markeredgewidth=2.0,
            )
        )
        labels.append(sty["label"])

    leg = ax_leg.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.38),
        frameon=False,
        fontsize=7.5,
        handletextpad=0.5,
        borderaxespad=0.0,
        labelspacing=0.85,
    )
    for t in leg.get_texts():
        t.set_color("#222222")

    ax_leg.text(
        0.0,
        0.02,
        "Circle size equal.\nFill depth = contribution.\nLabels = top pollutants.",
        fontsize=6.5,
        color="#777777",
        transform=ax_leg.transAxes,
        va="bottom",
        linespacing=1.35,
    )

    fig.suptitle(
        "Exposome priority matrix (top 50% cumulative risk)",
        fontsize=12,
        fontweight="bold",
        color="#111111",
        y=0.97,
    )

    out_png = OUT_DIR / "1_Matrix_EqualSize_ColorByPct.png"
    out_pdf = OUT_DIR / "1_Matrix_EqualSize_ColorByPct.pdf"
    fig.savefig(out_png, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_png.name}")


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    print(">>> aggregate ...")
    df_raw = pd.read_csv(INPUT_FILE)
    df_summary, disease_order, category_order = aggregate(df_raw)
    df_summary.to_csv(OUT_DIR / "1_Aggregated_Matrix_Data.csv", index=False)

    print(">>> plot matrix ...")
    plot_matrix(df_summary, disease_order, category_order)

    print(f"done → {OUT_DIR}")


if __name__ == "__main__":
    main()

#%%
