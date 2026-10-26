"""
Model validation and publication figures for four industrial departments
==========================================================================

Departments
------------
pow : Electric generation (CFPP)
ico : Industrial combustion (BFI, IOS, EAF, CK, COP)
ipr : Industrial processes (PCu, SCu, SAl, SPb, SZn)
wst : Waste treatment (MSWI, HWI)

Important modelling rule
------------------------
The original provincial model has three proxy sectors: electric generation,
industrial combustion and waste treatment. Industrial-combustion and
industrial-process targets both use the original sector-2 industrial proxy.
This script preserves that training logic while reporting the two target
families separately as ico and ipr.

Preserved from the original training code
------------------------------------------
* proxy variables: BC, OC, NOx, CO2, SO2 and CO densities
* only proxy densities are standardized; Year is appended unchanged
* target-specific order-of-magnitude scaling
* XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05)
* shuffled five-fold cross-validation, random_state=42

Run on the Linux environment containing the data:
    python code_model_validation_4departments.py

Jupyter:
    from code_model_validation_4departments import run_pipeline
    fold_metrics, oof_predictions, target_summary = run_pipeline()
"""
#%%
from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Configuration: paths are retained from the supplied script
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, 'data_raw')
PROV_EXCEL = os.path.join(SCRIPT_DIR, 'results','code_2.1_cal_emission', 'df_emi.xlsx') 
PROV_PROXY_DIR = BASE_PATH
PROV_SHP =  os.path.join(SCRIPT_DIR, 'data_raw','china_shp','geoBoundaries-CHN-ADM1.shp')
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, 'results','code_3_pred_cross_validation')

CHEM_PROXY_COLS = ["BC", "OC", "NOx", "CO2", "SO2", "CO"]
RANDOM_STATE = 42
N_SPLITS = 5

DEPARTMENT_ORDER = ["pow", "ipr", "ico", "wst"]
DEPARTMENT_LABELS = {
    "pow": "Electric generation",
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

TARGET_SUFFIXES = {
    "pow": {"CFPP"},
    "ico": {"BFI", "IOS", "EAF", "CK", "COP"},
    "ipr": {"PCu", "SCu", "SAl", "SPb", "SZn"},
    "wst": {"MSWI", "HWI"},
}
ALL_TARGET_SUFFIXES = set().union(*TARGET_SUFFIXES.values())


# ============================================================================
# General utilities
# ============================================================================

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
            "ytick.labelsize": 8,
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


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[keep], y_pred[keep]

    if len(y_true) == 0:
        return {
            "n": 0, "r2": np.nan, "rmse": np.nan, "mae": np.nan,
            "bias": np.nan, "spearman": np.nan,
        }

    rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 2 else np.nan
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "spearman": float(rho),
    }


def safe_target_scale(y: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce").dropna()
    positive = y[y > 0]
    if positive.empty:
        return 1.0
    return float(1 / (10 ** round(np.log10(float(positive.mean())))))


def extract_year(value):
    """Handle cftime.DatetimeNoLeap and ordinary date-like values."""
    if hasattr(value, "year"):
        return int(value.year)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return int(value)
    parsed = pd.to_datetime(value, errors="coerce")
    return int(parsed.year) if not pd.isna(parsed) else np.nan


def target_department(target: str) -> str | None:
    suffix = str(target).split("_")[-1]
    for department, suffixes in TARGET_SUFFIXES.items():
        if suffix in suffixes:
            return department
    return None


def target_proxy_sector(department: str) -> str:
    """Map four target families onto the original three proxy sectors."""
    if department == "pow":
        return "Electric generation"
    if department in {"ico", "ipr"}:
        return "Industrial combustion"
    if department == "wst":
        return "Waste treatment"
    raise ValueError(f"Unknown department: {department}")


def resolve_prov_shp() -> str:
    if os.path.exists(PROV_SHP):
        return PROV_SHP

# ============================================================================
# Provincial training data preparation
# ============================================================================

def grid_area_km2(lon: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    radius = 6371.0
    lon_rad = np.deg2rad(lon.values)
    lat_rad = np.deg2rad(lat.values)
    dlon = abs(lon_rad[1] - lon_rad[0]) if len(lon_rad) > 1 else np.deg2rad(1)
    dlat = abs(lat_rad[1] - lat_rad[0]) if len(lat_rad) > 1 else np.deg2rad(1)
    bounds = np.zeros(len(lat_rad) + 1)
    bounds[1:-1] = (lat_rad[:-1] + lat_rad[1:]) / 2
    bounds[0] = max(-np.pi / 2, lat_rad[0] - dlat / 2)
    bounds[-1] = min(np.pi / 2, lat_rad[-1] + dlat / 2)
    area_lat = radius**2 * dlon * abs(np.sin(bounds[1:]) - np.sin(bounds[:-1]))
    return xr.DataArray(
        np.tile(area_lat[:, None], (1, len(lon))),
        coords=[lat, lon],
        dims=["lat", "lon"],
    )


def prepare_training_data() -> pd.DataFrame:
    excel = pd.read_excel(PROV_EXCEL)
    provinces = gpd.read_file(resolve_prov_shp()).to_crs("EPSG:4326")
    # EPSG:6933 is a global equal-area projection and is safer for China-wide areas.
    provinces["Area_km2"] = provinces.geometry.to_crs("EPSG:6933").area / 1e6
    provinces = provinces.reset_index(drop=True)

    nc_files = glob.glob(os.path.join(PROV_PROXY_DIR, "*1950_2100.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No proxy files found in {PROV_PROXY_DIR}")
    sample = xr.open_dataset(nc_files[0])
    lon, lat = sample.lon, sample.lat

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon_grid.ravel(), lat_grid.ravel()),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        points,
        provinces[["geometry", "shapeName"]],
        how="left",
        predicate="within",
    )
    labels = (
    joined["shapeName"]
    .astype(object)
    .to_numpy()
    .reshape(len(lat), len(lon))
    )

    mask = xr.DataArray(
        np.stack([(labels == name) for name in provinces.shapeName], axis=0),
        dims=["region", "lat", "lon"],
        coords={"region": provinces.shapeName.values, "lat": lat, "lon": lon},
    )

    cleaning = {
        "Beijing Municipality": "Beijing",
        "Tianjin Municipality": "Tianjin",
        "Shanghai Municipality": "Shanghai",
        "Chongqing Municipality": "Chongqing",
        "Inner Mongolia Autonomous Region": "Inner Mongolia",
        "Guangxi Zhuang Autonomous Region": "Guangxi",
        "Tibet Autonomous Region": "Xizang",
        "Ningxia Ningxia Hui Autonomous Region": "Ningxia",
        "Xinjiang Uyghur Autonomous Region": "Xinjiang",
        "Guangzhou Province": "Guangdong",
    }
    # Preserve the original three proxy sectors.
    proxy_sectors = {
        1: "Electric generation",
        2: "Industrial combustion",
        6: "Waste treatment",
    }
    train_tables = []

    for sector_id, sector_name in proxy_sectors.items():
        pieces = []
        for nc in nc_files:
            ds = xr.open_dataset(nc).sel(sector=sector_id)
            variable = list(ds.data_vars)[0]
            pollutant = Path(nc).name.split("_")[0]
            values = []
            for province in mask.region.values:
                one = ds[variable].where(mask.sel(region=province)).sum(dim=["lat", "lon"])
                one = one.assign_coords(Province_Name=province)
                values.append(one)
            totals = xr.concat(values, dim="Province_Name")
            tmp = totals.to_dataframe(name=pollutant).reset_index()
            pieces.append(tmp[["time", "Province_Name", pollutant]])

        x = pieces[0]
        for piece in pieces[1:]:
            x = x.merge(piece, on=["time", "Province_Name"], how="inner")
        x["Year"] = x["time"].map(extract_year)
        x = x.dropna(subset=["Year"])
        x["Year"] = x["Year"].astype(int)
        x = x.rename(columns={"Province_Name": "Province"})
        x["Province"] = x["Province"].replace(cleaning)

        p = provinces.copy()
        p["Province"] = p["shapeName"].replace(cleaning)
        x = x.merge(p[["Province", "Area_km2"]], on="Province", how="inner")
        for col in CHEM_PROXY_COLS:
            if col in x.columns:
                x[f"{col}_density"] = x[col] / x["Area_km2"]
        x["model_sector"] = sector_name
        train_tables.append(x)

    data = pd.concat(train_tables, ignore_index=True)
    data = data.merge(excel, on=["Year", "Province"], how="inner")
    return data


def infer_targets(data: pd.DataFrame) -> list[str]:
    excluded = {
        "Year", "Province", "Area_km2", "time", "model_sector",
        *CHEM_PROXY_COLS,
        *[f"{x}_density" for x in CHEM_PROXY_COLS],
    }
    targets = []
    for col in data.columns:
        if col in excluded or str(col).startswith("shape"):
            continue
        if target_department(col) and pd.api.types.is_numeric_dtype(data[col]):
            targets.append(col)
    return targets


# ============================================================================
# Five-fold validation
# ============================================================================

def run_validation(data: pd.DataFrame, output_dir: Path):
    feature_cols = [
        f"{x}_density"
        for x in CHEM_PROXY_COLS
        if f"{x}_density" in data.columns
    ]
    if not feature_cols:
        raise KeyError("No proxy density features found")

    target_cols = infer_targets(data)
    if not target_cols:
        raise KeyError("No target columns found for pow/ipr/ico/wst")

    metric_rows = []
    prediction_rows = []
    importance_rows = []

    for number, target in enumerate(target_cols, start=1):
        department = target_department(target)
        proxy_sector = target_proxy_sector(department)
        sub = data[data["model_sector"] == proxy_sector].dropna(
            subset=feature_cols + [target, "Province", "Year"]
        ).copy()
        if len(sub) < N_SPLITS or sub[target].sum() == 0:
            continue

        # Preserve original target scaling.
        y_scale = safe_target_scale(sub[target])
        scaler = StandardScaler()
        x_scaled = pd.DataFrame(
            scaler.fit_transform(sub[feature_cols]),
            columns=feature_cols,
            index=sub.index,
        )
        # Preserve the original logic: Year is not standardized.
        x_scaled["Year"] = sub["Year"].values
        y = sub[target].astype(float) * y_scale

        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(x_scaled), start=1):
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                objective="reg:squarederror",
            )
            model.fit(x_scaled.iloc[train_idx], y.iloc[train_idx])
            pred = np.clip(model.predict(x_scaled.iloc[valid_idx]) / y_scale, 0, None)
            true = sub[target].iloc[valid_idx].to_numpy(dtype=float)

            metric_rows.append(
                {
                    "workflow": "provincial_downscaling",
                    "department": department,
                    "department_label": DEPARTMENT_LABELS[department],
                    "proxy_sector": proxy_sector,
                    "target": target,
                    "fold": fold,
                    "validation_type": "random_5fold_cv",
                    **calculate_metrics(true, pred),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "workflow": "provincial_downscaling",
                        "department": department,
                        "target": target,
                        "fold": fold,
                        "Province": sub["Province"].iloc[valid_idx].values,
                        "Year": sub["Year"].iloc[valid_idx].values,
                        "observed": true,
                        "predicted": pred,
                    }
                )
            )
            importance_rows.extend(
                {
                    "department": department,
                    "target": target,
                    "fold": fold,
                    "feature": feature,
                    "importance": float(value),
                }
                for feature, value in zip(feature_cols + ["Year"], model.feature_importances_)
            )

        print(f"[{number}/{len(target_cols)}] {target} -> {department}")

    metrics_df = pd.DataFrame(metric_rows)
    preds_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    importance_df = pd.DataFrame(importance_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "provincial_downscaling_fold_metrics_4departments.csv", index=False)
    preds_df.to_csv(output_dir / "provincial_downscaling_oof_predictions_4departments.csv", index=False)
    importance_df.to_csv(output_dir / "provincial_downscaling_feature_importance_4departments.csv", index=False)
    return metrics_df, preds_df, importance_df


# ============================================================================
# Tables and figures
# ============================================================================

def summarize_results(metrics_df: pd.DataFrame, preds_df: pd.DataFrame, out: Path):
    fold_summary = (
        metrics_df.groupby(["department", "department_label", "target"], as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            n_total=("n", "sum"),
            R2_mean=("r2", "mean"),
            R2_sd=("r2", "std"),
            R2_min=("r2", "min"),
            R2_max=("r2", "max"),
            RMSE_mean=("rmse", "mean"),
            MAE_mean=("mae", "mean"),
            Bias_mean=("bias", "mean"),
            Spearman_mean=("spearman", "mean"),
        )
    )
    if preds_df.empty:
        oof_summary = pd.DataFrame()
    else:
        rows = []
        for (department, target), group in preds_df.groupby(["department", "target"]):
            rows.append(
                {
                    "department": department,
                    "target": target,
                    **calculate_metrics(group["observed"], group["predicted"]),
                }
            )
        oof_summary = pd.DataFrame(rows)

    out.mkdir(parents=True, exist_ok=True)
    fold_summary.to_csv(out / "Table_all_substances_R2_summary.csv", index=False)
    oof_summary.to_csv(out / "Table_all_substances_OOF_summary.csv", index=False)

    department_summary = (
        metrics_df.groupby(["department", "department_label"], as_index=False)["r2"]
        .agg(n_records="count", mean="mean", sd="std", median="median", min="min", max="max")
    )
    department_summary.to_csv(out / "Table_four_department_summary.csv", index=False)

    try:
        with pd.ExcelWriter(out / "Tables_four_department_validation.xlsx") as writer:
            department_summary.to_excel(writer, sheet_name="Department_summary", index=False)
            fold_summary.to_excel(writer, sheet_name="All_substances_R2", index=False)
            oof_summary.to_excel(writer, sheet_name="OOF_summary", index=False)
            metrics_df.to_excel(writer, sheet_name="Fold_level", index=False)
    except Exception as exc:
        print(f"[warning] Excel export skipped: {exc}")
    return fold_summary, oof_summary


def plot_forest(fold_summary: pd.DataFrame, out: Path) -> None:
    panel_data = {
        dep: fold_summary[fold_summary["department"] == dep]
        .sort_values("R2_mean")
        .reset_index(drop=True)
        for dep in DEPARTMENT_ORDER
    }
    heights = [max(2.5, 0.19 * len(panel_data[d]) + 0.8) for d in DEPARTMENT_ORDER]
    fig, axes = plt.subplots(
        4, 1,
        figsize=(8.6, sum(heights)),
        sharex=True,
        gridspec_kw={"height_ratios": heights},
    )
    axes = np.atleast_1d(axes)
    for ax, dep in zip(axes, DEPARTMENT_ORDER):
        d = panel_data[dep]
        if d.empty:
            ax.set_visible(False)
            continue
        y = np.arange(len(d))
        sd = d["R2_sd"].fillna(0)
        ci = 2.776 * sd / np.sqrt(d["n_folds"].clip(lower=1))
        color = DEPARTMENT_COLORS[dep]
        ax.hlines(y, d["R2_mean"] - ci, d["R2_mean"] + ci, color=color, lw=1.0)
        ax.plot(d["R2_mean"], y, "o", color=color, ms=4, mec="white", mew=0.5, zorder=3)
        ax.axvline(0, color="#777777", ls="--", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(d["target"], fontsize=6.5)
        ax.set_title(
            f"{dep}  |  {DEPARTMENT_LABELS[dep]}  |  {len(d)} substances",
            loc="left", color=color, pad=4,
        )
        ax.grid(axis="x", ls=":", lw=0.6, color="#CDD2D9")
        ax.tick_params(axis="y", length=0, pad=3)
    axes[-1].set_xlabel("Five-fold validation R² (mean ± 95% t-based CI)")
    axes[-1].set_xlim(min(-0.05, fold_summary["R2_mean"].min() - 0.05), 1.02)
    fig.suptitle("Validation performance for all modeled substances", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.26, right=0.98, top=0.985, bottom=0.035, hspace=0.28)
    save_figure(fig, out / "FigS_all_substances_R2_forest_4departments")


def plot_boxplot(metrics_df: pd.DataFrame, out: Path) -> None:
    data = [
        metrics_df.loc[metrics_df["department"] == dep, "r2"].dropna().to_numpy()
        for dep in DEPARTMENT_ORDER
    ]
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    bp = ax.boxplot(
        data,
        positions=np.arange(1, 5),
        widths=0.56,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#20242B", "linewidth": 1.2},
        whiskerprops={"color": "#5B6573", "linewidth": 0.9},
        capprops={"color": "#5B6573", "linewidth": 0.9},
        boxprops={"edgecolor": "#4A5563", "linewidth": 0.8},
    )
    for patch, dep in zip(bp["boxes"], DEPARTMENT_ORDER):
        patch.set_facecolor(DEPARTMENT_COLORS[dep])
        patch.set_alpha(0.82)

    rng = np.random.default_rng(42)
    for i, values in enumerate(data, start=1):
        ax.scatter(
            i + rng.uniform(-0.12, 0.12, len(values)),
            values,
            s=5,
            color="#222222",
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
    ax.set_xticks(np.arange(1, 5))
    ax.set_xticklabels(DEPARTMENT_ORDER)
    ax.set_ylabel("Validation R²")
    ax.set_title("Five-fold validation performance by industrial department")
    ax.set_ylim(min(-0.05, metrics_df["r2"].min() - 0.03), 1.02)
    ax.axhline(0, color="#777777", ls="--", lw=0.7)
    ax.grid(axis="y", ls=":", lw=0.7, color="#CDD2D9")
    fig.tight_layout()
    save_figure(fig, out / "FigS_validation_R2_boxplot_4departments")


def plot_target_heatmap(fold_summary: pd.DataFrame, out: Path) -> None:
    d = fold_summary.copy()
    d["substance"] = d["target"].str.replace(r"^[^_]+_", "", regex=True)
    d = d.sort_values(["department", "R2_mean"])
    matrix = d.pivot_table(index="substance", columns="department", values="R2_mean", aggfunc="mean")
    matrix = matrix.reindex(columns=DEPARTMENT_ORDER)
    fig, ax = plt.subplots(figsize=(5.4, max(5.0, 0.18 * len(matrix) + 1.5)))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Mean five-fold R²"},
    )
    ax.set_xlabel("Industrial department")
    ax.set_ylabel("Substance")
    ax.set_title("Target-specific validation performance", loc="left")
    ax.set_xticklabels(DEPARTMENT_ORDER, rotation=0)
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    save_figure(fig, out / "FigS_all_substances_R2_heatmap")


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    set_style()
    root = Path(OUTPUT_ROOT)
    fig_dir = root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing provincial training data...")
    data = prepare_training_data()
    print(f"Training rows: {len(data):,}")
    print("Running ordinary random five-fold validation...")
    metrics_df, preds_df, importance_df = run_validation(data, root)
    fold_summary, oof_summary = summarize_results(metrics_df, preds_df, root)

    plot_forest(fold_summary, fig_dir)
    plot_boxplot(metrics_df, fig_dir)
    plot_target_heatmap(fold_summary, fig_dir)

    elapsed = time.perf_counter() - started
    print(f"Validation records: {len(metrics_df):,}")
    print(f"Targets: {fold_summary['target'].nunique():,}")
    print(f"Finished in {elapsed / 60:.2f} min ({elapsed:.1f} s)")
    print(f"Results written to: {root}")
    return metrics_df, preds_df, fold_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_pipeline()


if __name__ == "__main__":
    main()

