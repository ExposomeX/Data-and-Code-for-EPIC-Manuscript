"""
Temporal validation for POW, ICO, IPR and WST emission targets.

Validation schemes
------------------
forward : train 2016-2018, predict 2019-2021

The script uses the same feature definition as the supplied workflow: all
available proxy columns ending in ``_emission`` except the target itself,
plus time, latitude and longitude. Standardisation is fitted separately on
each training period to prevent temporal leakage.

Input files are expected to have one row per time-lat-lon cell and a ``time``
column containing years. The default paths match the supplied workflow. Use
--industry-train and --power-train when the files are stored elsewhere.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = BASE_DIR / "results" / "code_4.1_predict_PAH"
DEFAULT_OUTPUT = BASE_DIR / 'results' / "code_4.2_temporal_validation_PAH"

DEPARTMENT_ORDER = ["pow", "ico", "ipr"]
DEPARTMENT_LABELS = {
    "pow": "Power generation",
    "ico": "Industrial combustion",
    "ipr": "Industrial processes",
}
DEPARTMENT_COLORS = {
    "pow": "#3C5488",
    "ico": "#B23A48",
    "ipr": "#00A087",
}

SCHEME_ORDER = ["forward"]
SCHEME_LABELS = {
    "forward": "2016–2018 → 2019–2021",
}
SCHEME_COLORS = {"forward": "#3C5488"}

RANDOM_STATE = 42
USE_LOG1P = False  # Keep False to reproduce the original target scale.
MIN_TRAIN_ROWS = 20


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["times new roman", "arial", "helvetica", "verdana"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_alignment_helper():
    candidates = []
    env_path = os.environ.get("NATURE_FIGURE_SKILL_SCRIPTS")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path.home() / ".codex" / "skills" / "nature-figure" / "scripts",
            Path.home() / ".agents" / "skills" / "nature-figure" / "scripts",
        ]
    )
    for candidate in candidates:
        if (candidate / "audit_panel_alignment.py").exists():
            sys.path.insert(0, str(candidate))
            from audit_panel_alignment import require_matplotlib_panel_alignment

            return require_matplotlib_panel_alignment
    raise ImportError(
        "Panel-alignment helper not found. Set NATURE_FIGURE_SKILL_SCRIPTS "
        "to the nature-figure scripts directory."
    )


def save_figure(
    fig: plt.Figure,
    base: Path,
    *,
    multipanel: bool = False,
    axes=None,
    panel_ids=None,
    raster_dpi: int = 600,
) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    if multipanel:
        require_matplotlib_panel_alignment = load_alignment_helper()
        require_matplotlib_panel_alignment(
            fig,
            axes=axes,
            panel_ids=panel_ids,
            json_out=str(base) + ".alignment.json",
            overlay_svg=str(base) + ".alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=True,
            strict=True,
        )
    fig.savefig(base.with_suffix(".png"), dpi=raster_dpi, bbox_inches="tight")
    plt.close(fig)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError(f"Missing 'time' column in {path}")
    df = df.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df["time"] = df["time"].astype(int)
    for col in ["lat", "lon"]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' column in {path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def merge_sources(industry_train: Path, power_train: Path) -> pd.DataFrame:
    """Read industry and power panels separately, then stack them with a source flag.

    The two panels commonly use the same proxy names for different sectors.
    They must therefore not be merged horizontally or their sector-specific
    proxy values would be silently mixed. Each target is later evaluated only
    on its own source group.
    """
    def read_if_exists(path: Path, source_group: str, year_low: int, year_high: int) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        frame = read_table(path).copy()
        frame["_source_group"] = source_group
        frame = frame[frame["time"].between(year_low, year_high)]
        keys = ["time", "lat", "lon"]
        return frame.sort_values(keys).drop_duplicates(keys, keep="first").reset_index(drop=True)

    train_parts = [
        read_if_exists(industry_train, "industry", 2016, 2021),
        read_if_exists(power_train, "power", 2016, 2021),
    ]
    train = (
        pd.concat(
            [x for x in train_parts if not x.empty],
            ignore_index=True,
            sort=False,
        )
        if any(not x.empty for x in train_parts)
        else pd.DataFrame()
    )
    if train.empty:
        raise ValueError("No training file was found")
    return train


def department_from_target(target: str) -> str | None:
    lower = str(target).lower()
    for dep in DEPARTMENT_ORDER:
        if lower == dep or lower.startswith(dep + "_") or lower.startswith(dep):
            return dep
    return None


def infer_targets(df: pd.DataFrame) -> list[str]:
    targets = []
    for col in df.columns:
        if department_from_target(col) and pd.api.types.is_numeric_dtype(df[col]):
            if not str(col).endswith("_pred"):
                targets.append(col)
    return sorted(set(targets))


def infer_features(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    emission_features = [
        c for c in df.columns
        if str(c).endswith("emission") and c != target and pd.api.types.is_numeric_dtype(df[c])
    ]
    meta = [c for c in ["time", "lat", "lon", "mask"] if c in df.columns]
    if not emission_features:
        raise ValueError("No proxy columns ending in '_emission' were found")
    return emission_features, meta


def metrics(y_true, y_pred) -> dict:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[keep], pred[keep]
    if len(true) == 0:
        return {
            "n": 0,
            "observed_sum": np.nan,
            "predicted_sum": np.nan,
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "spearman": np.nan,
            "pearson": np.nan,
            "relative_sum_bias": np.nan,
        }
    out = {
        "n": int(len(true)),
        "observed_sum": float(true.sum()),
        "predicted_sum": float(pred.sum()),
        "r2": float(r2_score(true, pred)) if len(true) > 1 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(true, pred))),
        "mae": float(mean_absolute_error(true, pred)),
        "bias": float(np.mean(pred - true)),
        "spearman": float(spearmanr(true, pred).statistic) if len(true) > 2 else np.nan,
        "pearson": float(pearsonr(true, pred).statistic) if len(true) > 2 else np.nan,
    }
    out["relative_sum_bias"] = (
        (out["predicted_sum"] - out["observed_sum"]) / out["observed_sum"]
        if out["observed_sum"] != 0 else np.nan
    )
    return out


def build_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def fit_predict(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target: str,
    features: list[str],
    meta: list[str],
):
    n_before_train = len(train_df)
    train_use = train_df.dropna(subset=features + meta + [target]).copy()
    n_after_train = len(train_use)
    eval_required = features + meta + ([target] if target in eval_df.columns else [])
    n_before_eval = len(eval_df)
    eval_use = eval_df.dropna(subset=eval_required).copy()
    n_after_eval = len(eval_use)
    if len(train_use) < MIN_TRAIN_ROWS or eval_use.empty:
        return pd.DataFrame(), None, {
            "n_train": len(train_use),
            "n_eval": len(eval_use),
            "n_before_train": n_before_train,
            "n_after_train": n_after_train,
            "n_before_eval": n_before_eval,
            "n_after_eval": n_after_eval,
        }

    scaler = StandardScaler()
    x_train_features = scaler.fit_transform(train_use[features])
    x_eval_features = scaler.transform(eval_use[features])
    x_train = pd.DataFrame(x_train_features, columns=features, index=train_use.index)
    x_eval = pd.DataFrame(x_eval_features, columns=features, index=eval_use.index)
    for col in meta:
        x_train[col] = train_use[col].to_numpy()
        x_eval[col] = eval_use[col].to_numpy()

    y_train = train_use[target].astype(float).to_numpy()
    has_observed = target in eval_use.columns
    y_eval = eval_use[target].astype(float).to_numpy() if has_observed else np.full(len(eval_use), np.nan)
    if USE_LOG1P:
        y_model = np.log1p(np.clip(y_train, 0, None))
    else:
        y_model = y_train

    model = build_model()
    model.fit(x_train, y_model, verbose=False)
    pred_model = model.predict(x_eval)
    pred = np.expm1(pred_model) if USE_LOG1P else pred_model
    pred = np.clip(pred, 0, None)
    pred_table = eval_use[["time", "lat", "lon"]].copy()
    pred_table["observed"] = y_eval
    pred_table["predicted"] = pred
    return pred_table, model, {
        "n_train": len(train_use),
        "n_eval": len(eval_use),
        "n_before_train": n_before_train,
        "n_after_train": n_after_train,
        "n_before_eval": n_before_eval,
        "n_after_eval": n_after_eval,
    }


def temporal_splits(train_df: pd.DataFrame):
    return {
        "forward": (
            train_df[train_df["time"].between(2016, 2018)],
            train_df[train_df["time"].between(2019, 2021)],
        ),
    }


def run_validation(train_df: pd.DataFrame, out: Path):
    targets = infer_targets(train_df)
    if not targets:
        raise ValueError("No POW/ICO/IPR/WST target columns were found")
    metric_rows = []
    prediction_rows = []
    for number, target in enumerate(targets, start=1):
        department = department_from_target(target)
        source_group = "power" if department == "pow" else "industry"
        source_train = train_df[train_df["_source_group"] == source_group].copy()
        features, meta = infer_features(source_train, target)
        print(f"[{number}/{len(targets)}] {target} ({department})")
        splits = temporal_splits(source_train)
        for scheme in SCHEME_ORDER:
            fit_df, eval_df = splits[scheme]
            pred_table, model, sample_counts = fit_predict(
                fit_df, eval_df, target, features, meta
            )
            if pred_table.empty:
                continue
            pred_table.insert(0, "target", target)
            pred_table.insert(1, "department", department)
            pred_table.insert(2, "scheme", scheme)
            prediction_rows.append(pred_table)
            row = {
                "target": target,
                "department": department,
                "department_label": DEPARTMENT_LABELS[department],
                "scheme": scheme,
                "scheme_label": SCHEME_LABELS[scheme],
                "train_years": "2016-2018",
                "eval_years": "2019-2021",
                **sample_counts,
                **metrics(pred_table["observed"], pred_table["predicted"]),
            }
            metric_rows.append(row)

            target_dir = out / "target_predictions" / target
            target_dir.mkdir(parents=True, exist_ok=True)
            pred_table.to_csv(target_dir / f"{scheme}_predictions.csv", index=False)
            if model is not None:
                model.save_model(target_dir / f"xgb_model_{target}_{scheme}.json")

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    metrics_df.to_csv(out / "Table_temporal_validation_target_metrics.csv", index=False)
    predictions_df.to_csv(out / "SourceData_temporal_validation_predictions.csv", index=False)
    return metrics_df, predictions_df


def summarize(metrics_df: pd.DataFrame, out: Path):
    rows = []
    for (department, scheme), group in metrics_df.groupby(["department", "scheme"], observed=True):
        values = group["r2"].dropna().to_numpy(float)
        mean = float(np.mean(values)) if len(values) else np.nan
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        half = 2.776 * sd / np.sqrt(len(values)) if len(values) > 1 else np.nan
        rows.append(
            {
                "department": department,
                "department_label": DEPARTMENT_LABELS[department],
                "scheme": scheme,
                "scheme_label": SCHEME_LABELS[scheme],
                "n_targets": len(values),
                "r2_mean": mean,
                "r2_sd": sd,
                "r2_ci95_low": mean - half if np.isfinite(half) else mean,
                "r2_ci95_high": mean + half if np.isfinite(half) else mean,
                "r2_median": float(np.median(values)) if len(values) else np.nan,
                "r2_min": float(np.min(values)) if len(values) else np.nan,
                "r2_max": float(np.max(values)) if len(values) else np.nan,
                "mean_relative_sum_bias": float(group["relative_sum_bias"].mean()),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out / "Table_temporal_validation_department_summary.csv", index=False)
    with pd.ExcelWriter(out / "Tables_temporal_validation.xlsx") as writer:
        summary_df.to_excel(writer, sheet_name="Department_summary", index=False)
        metrics_df.to_excel(writer, sheet_name="Target_metrics", index=False)
    return summary_df


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def plot_summary(summary_df: pd.DataFrame, metrics_df: pd.DataFrame, out: Path) -> None:
    """Plot forward-validation results only."""
    summary_df = summary_df[summary_df["scheme"].eq("forward")].copy()
    metrics_df = metrics_df[metrics_df["scheme"].eq("forward")].copy()
    if summary_df.empty or metrics_df.empty:
        raise ValueError("No forward-validation results were found.")

    department_order = [
        dep for dep in DEPARTMENT_ORDER
        if dep in set(metrics_df["department"])
    ]
    department_names = {
        "pow": "POW",
        "ico": "ICO",
        "ipr": "IPR",
        "wst": "WST",
    }
    colors = [DEPARTMENT_COLORS[dep] for dep in department_order]
    x = np.arange(len(department_order))
    rng = np.random.default_rng(RANDOM_STATE)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), constrained_layout=False)
    ax_a, ax_b, ax_c = axes
    for ax, label in zip(axes, ["a", "b", "c"]):
        add_panel_label(ax, label)

    # a: department-level mean R² and 95% CI
    department_summary = (
        summary_df.set_index("department").reindex(department_order)
    )
    r2_mean = department_summary["r2_mean"].to_numpy(float)
    r2_low = department_summary["r2_ci95_low"].to_numpy(float)
    r2_high = department_summary["r2_ci95_high"].to_numpy(float)
    ax_a.bar(x, r2_mean, width=0.62, color=colors, alpha=0.84,
             edgecolor="#41464D", linewidth=0.6, zorder=2)
    ax_a.errorbar(
        x,
        r2_mean,
        yerr=[np.maximum(r2_mean - r2_low, 0),
              np.maximum(r2_high - r2_mean, 0)],
        fmt="none", ecolor="#20242B", elinewidth=0.8,
        capsize=2.2, capthick=0.8, zorder=3,
    )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([department_names[d] for d in department_order])
    ax_a.set_ylabel("Mean validation R²")
    ax_a.set_title("Performance by department", loc="left")
    ax_a.set_ylim(min(-0.05, np.nanmin(r2_low) - 0.04), 1.03)
    ax_a.axhline(0, color="#777C83", linestyle="--", linewidth=0.65)
    ax_a.grid(axis="y", linestyle=":", linewidth=0.55,
              color="#D3D7DC", zorder=0)

    # b: target-level forward R² distribution
    target_r2 = metrics_df.groupby(["department", "target"], as_index=False)["r2"].mean()
    r2_values = [
        target_r2.loc[target_r2["department"].eq(dep), "r2"]
        .dropna().to_numpy(float)
        for dep in department_order
    ]
    box_values = [values if len(values) else np.array([np.nan]) for values in r2_values]
    bp = ax_b.boxplot(
        box_values, positions=x, widths=0.58, patch_artist=True,
        showfliers=False,
        medianprops={"color": "#20242B", "linewidth": 1.15},
        whiskerprops={"color": "#636A73", "linewidth": 0.75},
        capprops={"color": "#636A73", "linewidth": 0.75},
        boxprops={"edgecolor": "#4A5058", "linewidth": 0.65},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    for position, values, color in zip(x, r2_values, colors):
        if len(values):
            ax_b.scatter(
                position + rng.uniform(-0.20, 0.20, len(values)),
                values + rng.normal(0, 0.0018, len(values)),
                s=6, color=color, alpha=0.22, edgecolor="none",
                rasterized=True, zorder=2,
            )
    all_r2 = target_r2["r2"].dropna().to_numpy(float)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([department_names[d] for d in department_order])
    ax_b.set_ylabel("Target-level R²")
    ax_b.set_title("Distribution across targets", loc="left")
    ax_b.set_ylim(min(-0.05, np.nanmin(all_r2) - 0.04), 1.03)
    ax_b.axhline(0, color="#777C83", linestyle="--", linewidth=0.65)
    ax_b.grid(axis="y", linestyle=":", linewidth=0.55,
              color="#D3D7DC", zorder=0)

    # c: relative total-emission bias
    target_bias = metrics_df.groupby(["department", "target"], as_index=False)["relative_sum_bias"].mean()
    bias_values = [
        target_bias.loc[target_bias["department"].eq(dep), "relative_sum_bias"]
        .dropna().to_numpy(float)
        for dep in department_order
    ]
    bias_means = [np.mean(values) if len(values) else np.nan for values in bias_values]
    bias_ci95 = [
        1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
        if len(values) > 1 else np.nan
        for values in bias_values
    ]
    for position, values, color in zip(x, bias_values, colors):
        if len(values):
            ax_c.scatter(
                position + rng.uniform(-0.18, 0.18, len(values)),
                values, s=6, color=color, alpha=0.20,
                edgecolor="none", rasterized=True, zorder=1,
            )
    ax_c.errorbar(
        x, bias_means, yerr=bias_ci95, fmt="none", ecolor="#30343B",
        elinewidth=0.85, capsize=2.2, capthick=0.8, zorder=3,
    )
    ax_c.scatter(x, bias_means, s=24, c=colors, marker="D",
                 edgecolors="#20242B", linewidths=0.45, zorder=4)
    ax_c.axhline(0, color="#777C83", linestyle="--", linewidth=0.7)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([department_names[d] for d in department_order])
    ax_c.set_ylabel("Relative total-emission bias")
    ax_c.set_title("Conservation of total emissions", loc="left")
    ax_c.grid(axis="y", linestyle=":", linewidth=0.55,
              color="#D3D7DC", zorder=0)

    fig.suptitle(
        "Forward temporal validation across four emission departments",
        fontsize=10.5, fontweight="bold", x=0.02, ha="left", y=0.99,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19,
                        top=0.82, wspace=0.38)
    save_figure(
        fig,
        out / "Figure_forward_validation_summary_4departments",
        multipanel=True,
        axes=list(axes),
        panel_ids=["a", "b", "c"],
        raster_dpi=600,
    )


def plot_target_figures(predictions_df: pd.DataFrame, metrics_df: pd.DataFrame, out: Path) -> None:
    """Write compact per-target scatter and annual total figures."""
    if predictions_df.empty:
        return
    target_root = out / "target_figures"
    for target, group in predictions_df.groupby("target"):
        dep = group["department"].iloc[0]
        d = group[group["scheme"] == "forward"].copy()
        fig, ax = plt.subplots(figsize=(3.0, 2.8))
        if d.empty or d["observed"].notna().sum() == 0:
            ax.text(0.5, 0.5, "No forward-validation data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7)
            ax.set_axis_off()
        else:
            ax.scatter(
                d["observed"], d["predicted"], s=4, alpha=0.25,
                color=DEPARTMENT_COLORS[dep], linewidth=0, rasterized=True,
            )
            lim = np.nanmax(np.abs(np.r_[d["observed"], d["predicted"]]))
            lim = max(lim, 1e-12)
            ax.plot([0, lim], [0, lim], "--", color="#6B7280", lw=0.7)
            row = metrics_df[
                (metrics_df["target"] == target)
                & (metrics_df["scheme"] == "forward")
            ]
            r2 = row["r2"].iloc[0] if not row.empty else np.nan
            ax.text(
                0.04, 0.94,
                f"R² = {r2:.2f}" if np.isfinite(r2) else "R² = NA",
                transform=ax.transAxes, va="top", fontsize=6.5,
            )
            ax.set_xlabel("Observed")
            ax.set_ylabel("Predicted")
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.grid(ls=":", lw=0.5, color="#D5D9DE")
        fig.suptitle(
            f"{target} ({DEPARTMENT_LABELS[dep]})",
            fontsize=9, fontweight="bold", x=0.02, ha="left",
        )
        fig.tight_layout(pad=1.0)
        save_figure(fig, target_root / f"{target}_forward_scatter", raster_dpi=300)

        annual = (
            d.groupby("time", as_index=False)
            .agg(
                observed=("observed", lambda values: values.sum(min_count=1)),
                predicted=("predicted", "sum"),
            )
        )
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        if not annual.empty and annual["observed"].notna().any():
            ax.plot(annual["time"], annual["observed"], "o-",
                    color="#20242B", ms=2.6, lw=0.8, label="Observed")
        if not annual.empty:
            ax.plot(annual["time"], annual["predicted"], "--",
                    color=SCHEME_COLORS["forward"], lw=1.0,
                    label="Predicted")
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual total")
        ax.set_title(f"Annual total: {target}", loc="left")
        ax.grid(ls=":", lw=0.6, color="#D5D9DE")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout(pad=1.0)
        save_figure(fig, target_root / f"{target}_forward_annual_totals", raster_dpi=300)


def write_notes(out: Path, train: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    out.joinpath("Temporal_validation_QA_notes.txt").write_text(
        "Temporal validation design\n"
        "- Forward: train 2016-2018; evaluate 2019-2021.\n"
        "- Feature scaling is fitted within each training period only.\n"
        "- Predictions are clipped at zero after prediction; observed test values are never used to decide whether a prediction is zero.\n"
        f"- Training rows loaded: {len(train):,}.\n"
        f"- Target-scheme metric records: {len(metrics_df):,}.\n"
        "- R² confidence intervals in the department summary are t-based intervals across target substances.\n"
        "- This is temporal generalisation validation; it is not an independent external emissions inventory validation.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--industry-train", type=Path, default=None)
    parser.add_argument("--power-train", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    set_publication_style()
    industry_train = args.industry_train or args.root / "train_for_gems_emi_industry.csv"
    power_train = args.power_train or args.root / "train_for_gems_emi_power.csv"

    train = merge_sources(industry_train, power_train)
    args.out.mkdir(parents=True, exist_ok=True)
    metrics_df, predictions_df = run_validation(train, args.out)
    summary_df = summarize(metrics_df, args.out)
    plot_summary(summary_df, metrics_df, args.out)
    plot_target_figures(predictions_df, metrics_df, args.out)
    write_notes(args.out, train, metrics_df)
    print(f"Training rows: {len(train):,}")
    print(f"Targets: {metrics_df['target'].nunique() if not metrics_df.empty else 0:,}")
    print(f"Results written to: {args.out}")


if __name__ == "__main__":
    main()

# %%
