"""Create the Fig. 13 horizontal score bars for one pollutant–sector target.

Example
-------
python priority_fig13_single_target.py --pollutant As --sector CK

The chart contains only disease rows present in the original selected-target
table. Missing pollutant–disease combinations are not treated as zero.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "code_3_results" / "3_Final_Top50pct_Categorized_Targets.csv"
DEFAULT_OUTPUT = ROOT / "code_4_results" / "code_4.4_priority_single_target_figures"
SCORES = ["Unified_Bio_Score", "Epi_Score", "Priority_Index"]
DISPLAY = ["Bio score", "Epi score", "Priority Index"]
COLORS = ["#9ecae1", "#f28b72", "#0d4f91"]
INK = "#172b4d"
MUTED = "#66778c"
GRID = "#dce5ee"


def load_target(input_csv: Path, pollutant: str, sector: str) -> pd.DataFrame:
    if not input_csv.is_file():
        raise FileNotFoundError(f"not find：{input_csv}")

    df = pd.read_csv(input_csv)
    required = {"Disease_Name", "Exposure", "name_en_abbr", *SCORES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"not find：{sorted(missing)}")

    df["pollutant"] = df["name_en_abbr"].astype(str).str.strip()
    df["sector"] = df["Exposure"].astype(str).str.rsplit("_", n=1).str[-1]
    match = (
        df["pollutant"].str.casefold().eq(pollutant.strip().casefold())
        & df["sector"].str.casefold().eq(sector.strip().casefold())
    )
    selected = df.loc[match, ["Disease_Name", "Exposure", "pollutant", "sector", *SCORES]].copy()
    if selected.empty:
        choices = sorted(
            f"{p} · {s}" for p, s in zip(df["pollutant"].unique(), df["sector"].unique())
        )
        raise ValueError(
            f"not find：{pollutant}_{sector}。please check the pollutant abbreviation and sector code"
            f"available sectors：{sorted(df['sector'].dropna().unique())}"
        )
    if selected["Disease_Name"].duplicated().any():
        dup = selected.loc[selected["Disease_Name"].duplicated(False), "Disease_Name"].tolist()
        raise ValueError(f"repeated records for the same target：{dup}")

    for col in SCORES:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")
    values = selected[SCORES].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("contain invalid values")
    if not np.allclose(
        selected["Unified_Bio_Score"] * selected["Epi_Score"],
        selected["Priority_Index"], atol=1e-7, rtol=1e-5,
    ):
        raise ValueError("please check the data.")

    return selected.sort_values(
        ["Priority_Index", "Disease_Name"], ascending=[True, True]
    ).reset_index(drop=True)


def plot_target(df: pd.DataFrame, pollutant: str, sector: str,
                output_dir: Path, dpi: int = 600) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_pol = re.sub(r"[^A-Za-z0-9.-]+", "_", pollutant).strip("_")
    safe_sector = re.sub(r"[^A-Za-z0-9.-]+", "_", sector).strip("_")
    stem = f"Fig13_{safe_pol}_{safe_sector}"

    n = len(df)
    fig_height = max(3.4, 1.35 + 0.58 * n)
    fig, ax = plt.subplots(figsize=(9.2, fig_height), facecolor="white")
    y = np.arange(n)
    offsets = [0.24, 0.0, -0.24]
    max_score = float(df[SCORES].to_numpy(float).max())
    xmax = max_score * 1.32 if max_score > 0 else 1.0

    for score, label, color, offset in zip(SCORES, DISPLAY, COLORS, offsets):
        vals = df[score].to_numpy(float)
        bars = ax.barh(
            y + offset, vals, height=0.19, color=color,
            edgecolor="white", linewidth=0.45, label=label, zorder=3,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                val + xmax * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", ha="left", fontsize=8,
                color=INK, clip_on=False,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(df["Disease_Name"], fontsize=10, color=INK)
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Score", fontsize=10, color=INK, labelpad=8)
    ax.grid(axis="x", color=GRID, linestyle=":", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=3)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(0.8)

    fig.suptitle(
        f"{pollutant} · {sector}", x=0.24, y=0.98, ha="left",
        fontsize=15, fontweight="bold", color=INK,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper right", bbox_to_anchor=(0.91, 0.94),
        frameon=False, ncol=3, fontsize=8.5, handlelength=1.2,
        columnspacing=1.0, handletextpad=0.45,
    )
   
    fig.subplots_adjust(left=0.24, right=0.91, top=0.86, bottom=0.15)

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw the Fig. 13 grouped horizontal bars for one pollutant and sector."
    )
    parser.add_argument("--pollutant", default=None, help="pollutant abbreviation, e.g., As, Cd, BaP")
    parser.add_argument("--sector", default=None, help="sector code, e.g., CK, CFPP, ICO, MSWI")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="priority results CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--all", action="store_true", help="run all pollutant × sector combinations")
    args = parser.parse_args()

    if args.all:
        df_all = pd.read_csv(args.input)

        required = {"name_en_abbr", "Exposure"}
        missing = required - set(df_all.columns)
        if missing:
            raise ValueError(f"Priority results CSV missing columns: {sorted(missing)}")

        df_all["_pollutant"] = df_all["name_en_abbr"].astype(str).str.strip()
        df_all["_sector"] = df_all["Exposure"].astype(str).str.rsplit("_", n=1).str[-1]

        pairs = sorted(
            df_all[["_pollutant", "_sector"]]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

        for pollutant, sector in pairs:
            try:
                data = load_target(args.input, pollutant, sector)
            except Exception as e:
                print(f"skip {pollutant}_{sector}：{e}")
                continue

            png, pdf = plot_target(data, pollutant, sector, args.output_dir, args.dpi)
            print(f"Target: {pollutant}_{sector}")
            print(f"Disease rows plotted: {len(data)}")
            print(f"PNG: {png}")
            print(f"PDF: {pdf}")
            print("-" * 60)

        return

    pollutant = args.pollutant or input("please enter pollutant abbreviation (e.g., As, Cd, BaP): ").strip()
    sector = args.sector or input("please enter sector code (e.g., CK, CFPP, ICO, MSWI): ").strip()
    if not pollutant or not sector:
        parser.error("must provide both pollutant and sector; can be specified with --pollutant and --sector.")

    data = load_target(args.input, pollutant, sector)
    png, pdf = plot_target(data, pollutant, sector, args.output_dir, args.dpi)
    print(f"Target: {pollutant}_{sector}")
    print(f"Disease rows plotted: {len(data)}")
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")

if __name__ == "__main__":
    main()
