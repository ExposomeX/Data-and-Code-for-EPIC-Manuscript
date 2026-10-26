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
DEFAULT_SOURCE = ROOT / "code_1_results" / "EWAS_Sensitivity_Results_All.csv"
DEFAULT_OUTPUT_DIR = ROOT / "M4_Forest_Plots"

MODEL = "M4_Full"
EXPECTED_OUTCOMES = 15
NAVY = "#23466B"
BLUE = "#3E78A8"
ORANGE = "#D47B45"
GREY = "#6D7782"
GRID = "#E8EDF1"


def exposure_name(pollutant: str, sector: str) -> str:
    """Build the Exposure key used in the EWAS result table."""
    pollutant = pollutant.strip()
    sector = sector.strip()
    pollutant = re.sub(r"^X_", "", pollutant, flags=re.IGNORECASE)
    return f"X_{pollutant}_{sector}"


def load_m4_data(source: Path, exposure: str) -> pd.DataFrame:
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    df = pd.read_csv(source)
    required = {
        "Model_Type", "Outcome", "Exposure", "RR", "RR_low", "RR_high",
        "P_value", "Samples", "Significance_raw",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Result table is missing required columns: {sorted(missing)}")

    all_exposures = sorted(df["Exposure"].dropna().astype(str).unique().tolist())
    df = df.loc[
        (df["Model_Type"].astype(str) == MODEL)
        & (df["Exposure"].astype(str) == exposure)
    ].copy()

    if df.empty:
        examples = [e for e in all_exposures if e.lower().endswith("_" + exposure.split("_")[-1].lower())][:12]
        raise ValueError(
            f"No {MODEL} results found for Exposure={exposure!r}. "
            f"Please check the pollutant abbreviation and sector code. "
            f"Example exposures with similar sectors: {examples}"
        )

    for col in ("RR", "RR_low", "RR_high", "P_value", "Samples"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    invalid = (
        ~np.isfinite(df[["RR", "RR_low", "RR_high", "P_value"]]).all(axis=1)

        | (df["RR"] <= 0)
        | (df["RR_low"] <= 0)
        | (df["RR_high"] <= 0)
        | (df["RR_low"] > df["RR"])
        | (df["RR"] > df["RR_high"])
    )
    if invalid.any():
        bad = df.loc[invalid, "Outcome"].tolist()
        raise ValueError(f"The following outcomes have invalid RR or confidence intervals and were not plotted: {bad}")

    if df["Outcome"].duplicated().any():
        duplicates = df.loc[df["Outcome"].duplicated(False), "Outcome"].tolist()
        raise ValueError(f"Duplicated outcome records found in {MODEL}: {duplicates}")

    if len(df) != EXPECTED_OUTCOMES:
        raise ValueError(
            f"The {MODEL} results for {exposure} contain {len(df)} outcomes, "
            f"but {EXPECTED_OUTCOMES} are expected. Please check the input table before plotting."
        )

    df["Outcome_label"] = (
        df["Outcome"].astype(str)
        .str.replace(r"^Y_", "", regex=True)
        .str.replace("_", " ", regex=False)
    )
    # Keep all 15 outcomes and order them by the M4 point estimate.
    return df.sort_values(["RR", "Outcome_label"], ascending=[False, True]).reset_index(drop=True)


def format_p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def make_forest(df: pd.DataFrame, pollutant: str, sector: str,
                output_dir: Path, dpi: int = 600) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_pollutant = re.sub(r"[^A-Za-z0-9.-]+", "_", pollutant).strip("_")
    safe_sector = re.sub(r"[^A-Za-z0-9.-]+", "_", sector).strip("_")
    stem = f"Forest_M4_{safe_pollutant}_{safe_sector}"

    n = len(df)
    height = max(4.8, 0.29 * n + 0.95)
    fig = plt.figure(figsize=(9.0, height), facecolor="white")
    grid = fig.add_gridspec(
        nrows=1, ncols=2, width_ratios=[4.5, 2.0],
        left=0.26, right=0.98, bottom=0.16, top=0.87, wspace=0.035,
    )
    ax = fig.add_subplot(grid[0, 0])
    ax_text = fig.add_subplot(grid[0, 1], sharey=ax)

    y = np.arange(n)[::-1]
    low = df["RR_low"].to_numpy(float)
    rr = df["RR"].to_numpy(float)
    high = df["RR_high"].to_numpy(float)
    colors = np.where(
        (df["Significance_raw"].astype(str).to_numpy() == "Significant Harm")
        & (rr > 1), ORANGE, BLUE
    )

    xmin = max(0.08, min(0.8, float(low.min()) * 0.82))
    xmax = max(2.5, float(high.max()) * 1.16)
    #ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)
    ax.axvline(1.0, color=GREY, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)

    for yi, lo, point, hi, color in zip(y, low, rr, high, colors):
        ax.hlines(yi, lo, hi, color=color, linewidth=1.7, zorder=2)
        ax.vlines([lo, hi], yi - 0.09, yi + 0.09, color=color, linewidth=1.0, zorder=2)
        ax.scatter(point, yi, s=34, color=color, edgecolor="white", linewidth=0.55, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(df["Outcome_label"], fontsize=9)
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_xlabel("Rate ratio per emission IQR (95% CI; log scale)", labelpad=7, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.75)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", labelsize=8, length=3, color=GREY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#AEB8C2")
    ax.spines["bottom"].set_linewidth(0.7)

    ax_text.set_xlim(0, 1)
    ax.set_ylim(-0.65, n - 0.05)
    ax_text.set_ylim(ax.get_ylim())
    ax_text.axis("off")
    ax_text.text(0.02, n - 0.24, "RR (95% CI)", fontsize=8.5,
                 color=GREY, weight="bold", va="bottom")
    ax_text.text(0.98, n - 0.24, "P", fontsize=8.5,
                 color=GREY, weight="bold", va="bottom", ha="right")
    for yi, row in zip(y, df.itertuples(index=False)):
        ax_text.text(
            0.02, yi, f"{row.RR:.2f} ({row.RR_low:.2f}–{row.RR_high:.2f})",
            fontsize=8.3, color="#354657", va="center",
        )
        ax_text.text(0.98, yi, format_p(float(row.P_value)), fontsize=8.3,
                     color="#354657", va="center", ha="right")

    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a 15-outcome M4_Full forest plot for one pollutant and sector."
    )
    parser.add_argument("--pollutant", default=None,
                        help="Pollutant abbreviation, e.g., As, BaP, or 1-MePHE")
    parser.add_argument("--sector", default=None,
                        help="Sector code, e.g., CK, CFPP, ico, ipr, or pow")
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE,
                        help="EWAS_Sensitivity_Results_All.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--all", action="store_true",
                        help="Automatically run all pollutant × sector combinations under M4_Full")
    args = parser.parse_args()

    if args.all:
        df_all = pd.read_csv(args.input)

        required = {
            "Model_Type", "Outcome", "Exposure", "RR", "RR_low", "RR_high",
            "P_value", "Samples", "Significance_raw",
        }
        missing = required - set(df_all.columns)
        if missing:
            raise ValueError(f"Result table is missing required columns: {sorted(missing)}")

        m4 = df_all.loc[df_all["Model_Type"].astype(str) == MODEL].copy()
        if m4.empty:
            raise ValueError(f"No records found for Model_Type={MODEL!r}")

        exposures = sorted(m4["Exposure"].dropna().astype(str).unique())

        for exposure in exposures:
            try:
                data = load_m4_data(args.input, exposure)
            except ValueError as e:
                print(f"Skipping Exposure={exposure!r}: {e}")
                continue

            parts = exposure.split("_", 2)
            if len(parts) == 3:
                pollutant = parts[1]
                sector = parts[2]
            else:
                pollutant = exposure
                sector = ""

            png_path, pdf_path = make_forest(
                data, pollutant, sector, args.output_dir, args.dpi
            )
            print(f"Exposure: {exposure}")
            print(f"Model: {MODEL}; outcomes: {len(data)}")
            print(f"PNG: {png_path}")
            print(f"PDF: {pdf_path}")
            print("-" * 60)

        return

    if not args.pollutant or not args.sector:
        parser.error("When --all is not used, both --pollutant and --sector must be provided")

    exposure = exposure_name(args.pollutant, args.sector)
    data = load_m4_data(args.input, exposure)
    png_path, pdf_path = make_forest(
        data, args.pollutant, args.sector, args.output_dir, args.dpi
    )
    print(f"Exposure: {exposure}")
    print(f"Model: {MODEL}; outcomes: {len(data)}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()