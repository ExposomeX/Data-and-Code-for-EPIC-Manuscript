#%%
"""
Future emission prediction: model training and publication-ready figures
=========================================================================

This script follows the supplied modelling workflow:
    1. merge pollutant, proxy-emission, GDP and population data;
    2. split data into 70% training, 15% validation and 15% testing;
    3. tune XGBoost with 15 random parameter combinations and 3-fold CV;
    4. fit the selected model and evaluate it on the held-out test set;
    5. save the model, predictions, metrics and publication-ready figures.

The training logic is retained. The script removes the duplicated broken
TabPFN block, corrects target-column selection and maps the four source groups
explicitly:
    pow -> proxy sector 1
    ico/ipr -> proxy sector 2
    wst -> proxy sector 6

Run:
    python code_future_prediction_paper.py

Jupyter:
    from code_future_prediction_paper import run_pipeline
    metrics, importance = run_pipeline()
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split


# ============================================================================
# Configuration: paths retained from the supplied script
# ============================================================================

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
POL_PATH = os.path.join(BASE_PATH, "code_1_results")
PROXY_PATH = os.path.join(BASE_PATH, "code_2_results")
OUTPUT = os.path.join(BASE_PATH, "code_4_results")

SSPS = ["ssp126", "ssp245", "ssp370"]
RANDOM_STATE = 42
N_SEARCH_ITER = 15
SEARCH_CV = 3
MAKE_SHAP = False

DEPARTMENT_ORDER = ["pow", "ipr", "ico", "wst"]
DEPARTMENT_LABELS = {
    "pow": "Power generation",
    "ipr": "Industrial processes",
    "ico": "Industrial combustion",
    "wst": "Waste treatment",
}
DEPARTMENT_COLORS = {
    "pow": "#3C5488",
    "ipr": "#00A087",
    "ico": "#B23A48",
    "wst": "#E69F00",
}
SCENARIO_COLORS = {
    "ssp126": "#3C5488",
    "ssp245": "#7A5195",
    "ssp370": "#B23A48",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )
    sns.set_theme(style="whitegrid", context="paper")


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"{name}.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "index_y" in data.columns:
        data = data.drop(columns="index_y")
    if "index_x" in data.columns:
        data = data.rename(columns={"index_x": "name"})
    return data


def load_panel(ssp: str) -> pd.DataFrame:
    df_pol = clean_data(pd.read_csv(os.path.join(POL_PATH, "grid_level_pollutants.csv")))
    df_proxy = clean_data(
        pd.read_csv(os.path.join(PROXY_PATH, f"grid_level_proxy_6_pols_{ssp}.csv"))
    )
    df_gdp = clean_data(pd.read_csv(os.path.join(PROXY_PATH, "grid_level_gdp.csv")))
    df_pop = clean_data(pd.read_csv(os.path.join(PROXY_PATH, "grid_level_population.csv")))

    merge_cols = ["time", "lon", "lat"]
    for name, df in {
        "pollutant": df_pol,
        "proxy": df_proxy,
        "GDP": df_gdp,
        "population": df_pop,
    }.items():
        missing = [c for c in merge_cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name} data are missing merge columns: {missing}")

    pol_cols = [c for c in df_pol.columns if c not in merge_cols]
    proxy_cols = [c for c in df_proxy.columns if "emission" in str(c)]
    if "GDP" not in df_gdp.columns or "population" not in df_pop.columns:
        raise KeyError("GDP or population column not found")

    panel = df_pol[merge_cols + pol_cols].merge(
        df_proxy[merge_cols + proxy_cols], on=merge_cols, how="inner"
    )
    panel = panel.merge(df_gdp[merge_cols + ["GDP"]], on=merge_cols, how="inner")
    panel = panel.merge(df_pop[merge_cols + ["population"]], on=merge_cols, how="inner")
    panel.columns = [str(c).replace(".0", "") for c in panel.columns]
    return panel


def infer_targets_and_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    merge_cols = ["time", "lon", "lat"]
    emission_cols = [c for c in df.columns if "emission" in str(c)]
    features = emission_cols + ["lat", "lon", "population"]
    targets = [c for c in df.columns if c not in set(features + merge_cols + ["GDP"])]
    targets = [c for c in targets if pd.api.types.is_numeric_dtype(df[c])]
    if not targets:
        raise ValueError("No numeric target columns were found")
    return targets, features


def target_department(target: str) -> str:
    suffix = str(target).split("_")[-1].lower()
    if suffix in {"pow", "cfpp"}:
        return "pow"
    if suffix in {"ipr", "pcu", "scu", "sal", "spb", "szn"}:
        return "ipr"
    if suffix in {"ico", "bfi", "ios", "eaf", "ck", "cop"}:
        return "ico"
    if suffix in {"wst", "mswi", "hwi"}:
        return "wst"
    raise ValueError(f"Cannot assign target to a department: {target}")


def target_ratio(target: str) -> float:
    if target in {"Gd_MSWI", "Hexachlorobutadiene_MSWI"}:
        return 1e5
    return 1.0


def select_features(target: str, features: list[str]) -> list[str]:
    department = target_department(target)
    proxy_suffix = {"pow": "1", "ipr": "2", "ico": "2", "wst": "6"}[department]
    selected_emission = [
        feature for feature in features
        if str(feature).endswith(proxy_suffix)
    ]
    selected = selected_emission + ["lat", "lon", "population"]
    missing = [feature for feature in selected if feature not in features]
    if missing:
        raise KeyError(f"Features missing for {target}: {missing}")
    return selected


def train_one_target(
    df: pd.DataFrame,
    target: str,
    all_features: list[str],
    ssp: str,
    output_dir: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    features = select_features(target, all_features)
    ratio = target_ratio(target)
    temp_df = df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(temp_df) < 20 or np.isclose(temp_df[target].sum(), 0):
        return {}, pd.DataFrame(), pd.DataFrame()

    X = temp_df[features]
    y = temp_df[target] * ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    param_dist = {
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [200, 500, 800],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    base_model = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1
    )
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=N_SEARCH_ITER,
        scoring="r2",
        cv=SEARCH_CV,
        verbose=0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    random_search.fit(X_train, y_train)

    best_model = xgb.XGBRegressor(
        **random_search.best_params_,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    # Keep the original training logic. The validation set is retained as an
    # evaluation set, without silently introducing early stopping.
    best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred_scaled = best_model.predict(X_test)
    pred = np.clip(pred_scaled / ratio, 0, None)
    observed = y_test.reset_index(drop=True).to_numpy(dtype=float) / ratio
    department = target_department(target)
    r2 = float(r2_score(observed, pred))
    rmse = float(np.sqrt(mean_squared_error(observed, pred)))

    target_dir = output_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)
    best_model.save_model(target_dir / f"xgb_model_{target}_{ssp}.json")
    pd.DataFrame({
        "observed": observed,
        "predicted": pred,
        "residual": pred - observed,
        "target": target,
        "ssp": ssp,
    }).to_csv(target_dir / f"test_predictions_{target}_{ssp}.csv", index=False)
    with open(target_dir / f"best_params_{target}_{ssp}.json", "w", encoding="utf-8") as handle:
        json.dump(random_search.best_params_, handle, indent=2)

    metrics_row = {
        "ssp": ssp,
        "target": target,
        "department": department,
        "department_label": DEPARTMENT_LABELS[department],
        "n_total": len(temp_df),
        "n_train": len(X_train),
        "n_validation": len(X_val),
        "n_test": len(X_test),
        "r2": r2,
        "rmse": rmse,
        "best_params": json.dumps(random_search.best_params_),
    }
    importance = pd.DataFrame({
        "ssp": ssp,
        "target": target,
        "department": department,
        "feature": features,
        "importance": best_model.feature_importances_,
    })
    return metrics_row, importance, pd.DataFrame({
        "ssp": ssp,
        "target": target,
        "department": department,
        "observed": observed,
        "predicted": pred,
    })


def plot_r2_forest(metrics_df: pd.DataFrame, fig_dir: Path) -> None:
    d = metrics_df.sort_values(["department", "r2"]).reset_index(drop=True)
    d["label"] = d["target"] + "  (" + d["ssp"] + ")"
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.2, max(5.0, 0.18 * len(d) + 1.5)))
    for i, row in d.iterrows():
        color = SCENARIO_COLORS.get(row["ssp"], "#555555")
        ax.plot([0, row["r2"]], [i, i], color="#D3D7DE", lw=0.8, zorder=1)
        ax.plot(row["r2"], i, "o", color=color, ms=4, mec="white", mew=0.5, zorder=2)
    ax.axvline(0, color="#777777", ls="--", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(d["label"], fontsize=6.5)
    ax.set_xlabel("Held-out test-set R²")
    ax.set_title("Predictive performance for all modeled emission targets", loc="left")
    ax.grid(axis="x", ls=":", lw=0.6, color="#CDD2D9")
    handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", color=color, label=ssp)
        for ssp, color in SCENARIO_COLORS.items()
    ]
    ax.legend(handles=handles, title="Scenario", fontsize=7)
    save_figure(fig, fig_dir, "Fig_future_prediction_R2_forest_all_targets")


def plot_department_boxplot(metrics_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    data = [metrics_df.loc[metrics_df["department"] == d, "r2"].dropna().to_numpy() for d in DEPARTMENT_ORDER]
    bp = ax.boxplot(data, positions=np.arange(1, 5), widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={"color": "#20242B", "linewidth": 1.2},
                    whiskerprops={"color": "#5B6573", "linewidth": 0.9},
                    capprops={"color": "#5B6573", "linewidth": 0.9},
                    boxprops={"edgecolor": "#4A5563", "linewidth": 0.8})
    for patch, dep in zip(bp["boxes"], DEPARTMENT_ORDER):
        patch.set_facecolor(DEPARTMENT_COLORS[dep])
        patch.set_alpha(0.82)
    rng = np.random.default_rng(42)
    for i, values in enumerate(data, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values, s=8, color="#222222", alpha=0.22, linewidth=0)
    ax.set_xticks(np.arange(1, 5))
    ax.set_xticklabels(["pow", "ipr", "ico", "wst"])
    ax.set_xlabel("Industrial department")
    ax.set_ylabel("Held-out test-set R²")
    ax.set_title("Prediction performance by industrial department")
    ax.axhline(0, color="#777777", ls="--", lw=0.7)
    ax.grid(axis="y", ls=":", lw=0.7, color="#CDD2D9")
    save_figure(fig, fig_dir, "Fig_future_prediction_R2_boxplot_4departments")


def plot_representative_scatter(predictions: pd.DataFrame, metrics_df: pd.DataFrame, fig_dir: Path) -> None:
    selected = (
        metrics_df.sort_values("r2", ascending=False)
        .groupby(["department", "ssp"], as_index=False)
        .head(1)
    )
    selected_keys = set(zip(selected["department"], selected["ssp"], selected["target"]))
    d = predictions[predictions.apply(lambda x: (x["department"], x["ssp"], x["target"]) in selected_keys, axis=1)]
    if d.empty:
        return
    keys = list(zip(selected["department"], selected["ssp"], selected["target"]))
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.2))
    axes = axes.ravel()
    for ax, dep in zip(axes, DEPARTMENT_ORDER):
        candidates = selected[selected["department"] == dep]
        if candidates.empty:
            ax.axis("off")
            continue
        row = candidates.iloc[0]
        s = d[(d["department"] == dep) & (d["ssp"] == row["ssp"]) & (d["target"] == row["target"])]
        x, y = s["observed"].to_numpy(float), s["predicted"].to_numpy(float)
        positive = (x > 0) & (y > 0)
        ax.scatter(x[positive], y[positive], s=8, alpha=0.28, color=DEPARTMENT_COLORS[dep], linewidth=0)
        if (~positive).any():
            ax.scatter(x[~positive], y[~positive], s=8, alpha=0.2, color="#777777", linewidth=0)
        values = np.r_[x, y]
        values = values[np.isfinite(values)]
        if len(values) and np.nanmax(values) > 0:
            lo = max(np.nanmin(values[values > 0]) * 0.7, 1e-12)
            hi = np.nanmax(values) * 1.4
            ax.plot([lo, hi], [lo, hi], "--", color="#555555", lw=0.8)
            ax.set_xscale("symlog", linthresh=max(lo, 1e-12))
            ax.set_yscale("symlog", linthresh=max(lo, 1e-12))
            ax.set_xlim(0, hi)
            ax.set_ylim(0, hi)
        ax.text(0.05, 0.95, f"R² = {row['r2']:.2f}\n{row['ssp']}\n{row['target']}", transform=ax.transAxes, va="top", fontsize=7)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.grid(ls=":", lw=0.5, color="#CDD2D9")
    fig.suptitle("Representative held-out predictions", x=0.08, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, fig_dir, "Fig_future_prediction_observed_vs_predicted")


def plot_feature_importance(importance_df: pd.DataFrame, fig_dir: Path) -> None:
    d = importance_df.groupby(["department", "feature"], as_index=False)["importance"].mean()
    d["importance"] = d.groupby("department")["importance"].transform(lambda x: x / x.sum() if x.sum() else x)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    pivot = d.pivot(index="department", columns="feature", values="importance").fillna(0)
    pivot = pivot.reindex([x for x in DEPARTMENT_ORDER if x in pivot.index])
    pivot.plot.barh(stacked=True, ax=ax, colormap="Blues", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Mean relative feature importance")
    ax.set_ylabel("Industrial department")
    ax.set_title("Proxy-variable importance across emission targets")
    ax.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(axis="x", ls=":", lw=0.6, color="#CDD2D9")
    save_figure(fig, fig_dir, "Fig_future_prediction_feature_importance")



started = time.perf_counter()
set_style()
output_dir = Path(OUTPUT)
figure_dir = output_dir / "paper_figures"
output_dir.mkdir(parents=True, exist_ok=True)
'''
metric_rows = []
importance_tables = []
prediction_tables = []

for ssp in SSPS:
    print(f"\nProcessing {ssp}...")
    df = load_panel(ssp)
    targets, all_features = infer_targets_and_features(df)
    df.to_csv(output_dir / f"training_panel_{ssp}.csv", index=False)
    print(f"Rows: {len(df):,}; targets: {len(targets)}")
    for target in targets:
        try:
            row, importance, predictions = train_one_target(
                df, target, all_features, ssp, output_dir
            )
            if row:
                metric_rows.append(row)
                importance_tables.append(importance)
                prediction_tables.append(predictions)
                print(f"  {target}: test R²={row['r2']:.3f}")
        except Exception as exc:
            print(f"  [warning] skipped {target}: {exc}")

metrics_df = pd.DataFrame(metric_rows)
importance_df = pd.concat(importance_tables, ignore_index=True) if importance_tables else pd.DataFrame()
predictions_df = pd.concat(prediction_tables, ignore_index=True) if prediction_tables else pd.DataFrame()
metrics_df.to_csv(output_dir / "Table_future_prediction_test_metrics.csv", index=False)
importance_df.to_csv(output_dir / "Table_future_prediction_feature_importance.csv", index=False)
predictions_df.to_csv(output_dir / "Table_future_prediction_test_predictions.csv", index=False)
'''
metrics_df = pd.read_csv(output_dir / "Table_future_prediction_test_metrics.csv")
importance_df = pd.read_csv(output_dir / "Table_future_prediction_feature_importance.csv")
predictions_df = pd.read_csv(output_dir / "Table_future_prediction_test_predictions.csv")

if not metrics_df.empty:
    plot_r2_forest(metrics_df, figure_dir)
    plot_department_boxplot(metrics_df, figure_dir)
if not predictions_df.empty and not metrics_df.empty:
    plot_representative_scatter(predictions_df, metrics_df, figure_dir)
if not importance_df.empty:
    plot_feature_importance(importance_df, figure_dir)

elapsed = time.perf_counter() - started
print(f"\nFinished in {elapsed / 60:.2f} min ({elapsed:.1f} s)")
print(f"Results written to: {output_dir}")




# %%
# %%
"""
Department-level visualization of future emission model performance.

Reads existing held-out test metrics; does not retrain models.

Statistical unit:
    one prediction target within one SSP scenario.

Outputs:
    1. Department × scenario boxplots
    2. Department summary forest plot: median and interquartile range
    3. Department × scenario heatmap
    4. Department summary table and plotting source data

R² values retain the scale used in the original validation.
"""

from pathlib import Path
import time
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ============================================================================
# Configuration
# ============================================================================

OUTPUT = Path(BASE_PATH) / "code_4.2_results"

METRICS_FILE = OUTPUT / "Table_future_prediction_test_metrics.csv"
FIG_DIR = OUTPUT / "01_paper_figures_department"

DEPARTMENT_ORDER = ["pow", "ipr", "ico", "wst"]
SSPS = ["ssp126", "ssp245", "ssp370"]

DEPARTMENT_LABELS = {
    "pow": "Power generation",
    "ipr": "Industrial processes",
    "ico": "Industrial combustion",
    "wst": "Waste treatment",
}

SCENARIO_LABELS = {
    "ssp126": "SSP1–2.6",
    "ssp245": "SSP2–4.5",
    "ssp370": "SSP3–7.0",
}

SCENARIO_COLORS = {
    "ssp126": "#55789B",
    "ssp245": "#6A9A8B",
    "ssp370": "#C47770",
}

SHOW_IN_JUPYTER = True
RANDOM_STATE = 42
DPI = 600


# ============================================================================
# Style and export
# ============================================================================

def set_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "axes.grid": False
    })


def save_figure(fig, name):
    """Fixed-size exports avoid image heights increasing with target count."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig.canvas.draw()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            FIG_DIR / f"{name}.{extension}",
            dpi=DPI,
            bbox_inches="tight",
            pad_inches=0.08,
        )

    if SHOW_IN_JUPYTER:
        plt.show()
    plt.close(fig)


# ============================================================================
# Data preparation
# ============================================================================

def load_metrics():
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_FILE}")

    data = pd.read_csv(METRICS_FILE)

    required = {"department", "ssp", "target", "r2"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data["department"] = (
        data["department"].astype(str).str.strip().str.lower()
    )
    data["ssp"] = data["ssp"].astype(str).str.strip().str.lower()
    data["r2"] = pd.to_numeric(data["r2"], errors="coerce")

    valid_group = (
        data["department"].isin(DEPARTMENT_ORDER)
        & data["ssp"].isin(SSPS)
    )
    finite_r2 = np.isfinite(data["r2"])

    audit = {
        "input_rows": int(len(data)),
        "unsupported_group_rows": int((~valid_group).sum()),
        "nonfinite_r2_rows_in_supported_groups": int(
            (valid_group & ~finite_r2).sum()
        ),
        "statistical_unit": "one target within one scenario",
        "forest_interval": "25th–75th percentile across target-specific R2",
        "r2_scale": "unchanged from input metrics",
        "jitter": "horizontal only; R2 values unchanged",
    }

    clean = data.loc[valid_group & finite_r2].copy()

    if clean.empty:
        raise ValueError("No valid department–scenario R² records found.")

    # Multiple records for the same target would change its statistical weight.
    keys = ["department", "ssp", "target"]
    duplicated = clean.duplicated(keys, keep=False)
    if duplicated.any():
        example = clean.loc[duplicated, keys].head(10)
        raise ValueError(
            "Duplicate target–scenario records found. Resolve duplicates "
            "before plotting:\n" + example.to_string(index=False)
        )

    audit["plotted_rows"] = int(len(clean))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(FIG_DIR / "plotting_data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    clean.to_csv(
        FIG_DIR / "Source_data_target_test_R2.csv", index=False
    )

    print(
        f"Loaded {len(data):,} rows; "
        f"{len(clean):,} valid target–scenario records retained."
    )
    if audit["unsupported_group_rows"]:
        print(
            "Excluded unsupported department/scenario rows:",
            audit["unsupported_group_rows"],
        )
    if audit["nonfinite_r2_rows_in_supported_groups"]:
        print(
            "Excluded non-finite R² records:",
            audit["nonfinite_r2_rows_in_supported_groups"],
        )

    return clean


def summarize_metrics(data):
    rows = []

    for department in DEPARTMENT_ORDER:
        for ssp in SSPS:
            values = data.loc[
                (data["department"] == department)
                & (data["ssp"] == ssp), "r2"
            ].to_numpy(float)

            n = len(values)
            rows.append({
                "department": department,
                "department_label": DEPARTMENT_LABELS[department],
                "ssp": ssp,
                "n_targets": n,
                "r2_mean": np.mean(values) if n else np.nan,
                "r2_median": np.median(values) if n else np.nan,
                "r2_sd": np.std(values, ddof=1) if n > 1 else np.nan,
                "r2_q25": np.quantile(values, 0.25) if n else np.nan,
                "r2_q75": np.quantile(values, 0.75) if n else np.nan,
                "r2_min": np.min(values) if n else np.nan,
                "r2_max": np.max(values) if n else np.nan,
                "n_negative_r2": int((values < 0).sum()),
                "negative_r2_percent": (
                    100 * np.mean(values < 0) if n else np.nan
                ),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(
        FIG_DIR / "Table_department_scenario_performance.csv",
        index=False,
    )
    return summary


def format_r2_axis(ax, orientation="y"):
    if orientation == "y":
        ax.set_ylabel(r"Held-out test $R^2$")
        #ax.grid(axis="y", color="#E3E6E8", linewidth=0.5)
    else:
        ax.set_xlabel(r"Held-out test $R^2$")
        #ax.grid(axis="x", color="#E3E6E8", linewidth=0.5)
    #ax.set_axisbelow(True)


# ============================================================================
# Figure 1: grouped boxplots
# ============================================================================

def plot_department_boxplots(data):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.18, top=0.82)

    rng = np.random.default_rng(RANDOM_STATE)
    offsets = [-0.23, 0.0, 0.23]

    for ssp, offset in zip(SSPS, offsets):
        for i, department in enumerate(DEPARTMENT_ORDER):
            values = data.loc[
                (data["department"] == department)
                & (data["ssp"] == ssp), "r2"
            ].to_numpy(float)

            if not len(values):
                continue

            position = i + offset
            bp = ax.boxplot(
                [values],
                positions=[position],
                widths=0.19,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                medianprops={
                    "color": "#25313A", "linewidth": 1.2,
                },
                whiskerprops={
                    "color": "#65717A", "linewidth": 0.75,
                },
                capprops={
                    "color": "#65717A", "linewidth": 0.75,
                },
                boxprops={
                    "edgecolor": "#65717A", "linewidth": 0.75,
                },
            )
            bp["boxes"][0].set_facecolor(SCENARIO_COLORS[ssp])
            bp["boxes"][0].set_alpha(0.72)

            # Horizontal jitter only; every finite R² value is shown.
            jitter = rng.uniform(-0.055, 0.055, len(values))
            ax.scatter(
                position + jitter,
                values,
                s=7,
                color=SCENARIO_COLORS[ssp],
                alpha=0.32,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )

    ax.set_xticks(range(4))
    ax.set_xticklabels([d.upper() for d in DEPARTMENT_ORDER])
    ax.set_xlim(-0.55, 3.55)
    ax.set_xlabel("Emission department")

    # Keep negative R² values visible; no clipping or vertical perturbation.
    lower = min(float(data["r2"].min()), 0.0)
    upper = max(float(data["r2"].max()), 1.0)
    padding = max(0.035, 0.04 * (upper - lower))
    ax.set_ylim(lower - padding, upper + padding)

    ax.axhline(0, color="#8A9196", linestyle="--", linewidth=0.7)
    format_r2_axis(ax)

    handles = [
        Patch(
            facecolor=SCENARIO_COLORS[s],
            edgecolor="none",
            label=SCENARIO_LABELS[s],
        )
        for s in SSPS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.98),
        ncol=3,
        handlelength=1.3,
        columnspacing=2.0,
    )

    save_figure(fig, "Fig_department_R2_boxplots_by_scenario")


# ============================================================================
# Figure 2: compact forest plot
# ============================================================================

def plot_department_forest(summary):
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.24, top=0.82)

    offsets = [-0.20, 0.0, 0.20]
    markers = ["o", "s", "^"]

    for ssp, offset, marker in zip(SSPS, offsets, markers):
        for i, department in enumerate(DEPARTMENT_ORDER):
            row = summary.loc[
                (summary["department"] == department)
                & (summary["ssp"] == ssp)
            ].iloc[0]

            if row["n_targets"] == 0:
                continue

            median = float(row["r2_median"])
            low = float(row["r2_q25"])
            high = float(row["r2_q75"])

            ax.errorbar(
                median,
                i + offset,
                xerr=np.array([[median - low], [high - median]]),
                fmt=marker,
                color=SCENARIO_COLORS[ssp],
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.5,
                elinewidth=1.6,
                capsize=2.5,
                capthick=0.9,
            )

    ax.set_yticks(range(4))
    ax.set_yticklabels([
        DEPARTMENT_LABELS[d] for d in DEPARTMENT_ORDER
    ])
    ax.set_ylim(3.5, -0.5)
    #ax.axvline(0, color="#8A9196", linestyle="--", linewidth=0.7)
    format_r2_axis(ax, orientation="x")

    handles = [
        Line2D(
            [], [],
            marker=marker,
            color=SCENARIO_COLORS[ssp],
            linewidth=1.5,
            markersize=5,
            label=SCENARIO_LABELS[ssp],
        )
        for ssp, marker in zip(SSPS, markers)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.61, 0.98),
        ncol=3,
        columnspacing=1.7,
    )
    fig.text(
        0.61, 0.025,
        "Points: median; intervals: interquartile range across targets",
        ha="center",
        fontsize=7,
        color="#65717A",
    )

    save_figure(fig, "Fig_department_R2_summary_forest")


# ============================================================================
# Figure 3: department × scenario heatmap
# ============================================================================

def plot_department_heatmap(summary):
    medians = summary.pivot(
        index="department", columns="ssp", values="r2_median"
    ).reindex(index=DEPARTMENT_ORDER, columns=SSPS)

    counts = summary.pivot(
        index="department", columns="ssp", values="n_targets"
    ).reindex(index=DEPARTMENT_ORDER, columns=SSPS)

    matrix = medians.to_numpy(float)
    finite_values = matrix[np.isfinite(matrix)]
    if not len(finite_values):
        return

    fig, ax = plt.subplots(figsize=(5.7, 3.7))
    fig.subplots_adjust(left=0.32, right=0.82, bottom=0.20, top=0.92)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "muted_red",
        ["#FFF7F4", "#F2D0C5", "#D99484", "#B55B51", "#843B38"],
    )
    cmap.set_bad("#F1F2F3")

    # Use the full range of displayed medians.
    vmin = float(finite_values.min())
    vmax = float(finite_values.max())
    if np.isclose(vmin, vmax):
        vmin -= 0.01
        vmax += 0.01

    norm = mpl.colors.Normalize(vmin=0.9, vmax=1.0)
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="nearest",
    )

    for i in range(4):
        for j in range(3):
            value = matrix[i, j]
            n = int(counts.iloc[i, j])

            if np.isfinite(value):
                colour = "white" if norm(value) > 0.64 else "#31383E"
                text = f"{value:.2f}\n$n$ = {n}"
            else:
                colour = "#858C92"
                text = "NA"

            ax.text(
                j, i, text,
                ha="center", va="center",
                fontsize=8, color=colour,
            )

    ax.set_xticks(range(3))
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SSPS])
    ax.set_yticks(range(4))
    ax.set_yticklabels([
        DEPARTMENT_LABELS[d] for d in DEPARTMENT_ORDER
    ])
    #ax.tick_params(axis="both", which="both", length=0, pad=7)

    #ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    #ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    #ax.grid(which="minor", color="white", linewidth=1.8)
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    colour_ax = fig.add_axes([0.86, 0.20, 0.025, 0.72])
    colourbar = fig.colorbar(image, cax=colour_ax)
    colourbar.set_label(r"Median held-out test $R^2$", fontsize=8)
    colourbar.ax.tick_params(labelsize=7)
    colourbar.outline.set_visible(False)

    save_figure(fig, "Fig_department_scenario_R2_heatmap")


# ============================================================================
# Run: suitable for Jupyter or a Python script
# ============================================================================

def run_visualization():
    started = time.perf_counter()
    set_style()

    data = load_metrics()
    summary = summarize_metrics(data)

    plot_department_boxplots(data)
    plot_department_forest(summary)
    plot_department_heatmap(summary)

    print("\nDepartment-level performance:")
    print(
        summary[
            ["department", "ssp", "n_targets",
             "r2_median", "r2_q25", "r2_q75"]
        ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    elapsed = time.perf_counter() - started
    print(f"\nFinished in {elapsed:.1f} s ({elapsed / 60:.2f} min)")
    print(f"Figures and tables: {FIG_DIR}")

    return data, summary


metrics_df, department_summary_df = run_visualization()
# %%
