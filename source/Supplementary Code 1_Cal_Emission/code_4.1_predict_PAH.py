"""
PAH emission prediction for POW, IPR and ICO
=============================================

1. Extract CMIP6 proxy-emission features for GEMS grid points.
2. Train sector-specific XGBoost models using 2016–2021 observations.
3. Evaluate models using 10-fold random cross-validation.
4. Predict PAH emissions for 2022–2024.
5. Export all intermediate data, predictions, metrics and figures to:
   results/code_4.2_predict_PAH

Sector mapping
--------------
sector = 1: POW
sector = 2: IPR and ICO
"""

from __future__ import annotations

import glob
import os
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# 0. Paths and settings
# =============================================================================

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Compatible with Jupyter Notebook
    SCRIPT_DIR = Path.cwd()

BASE_PATH = SCRIPT_DIR / "data_raw"
OUTPUT_PATH = SCRIPT_DIR / "results" / "code_4.2_predict_PAH"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

GEMS_FILE = BASE_PATH / "0.5deg_gems_emission.csv"

POLLUTANTS = ["BC", "CO2", "NOx", "SO2", "OC"]
TARGET_YEARS = np.arange(2016, 2025)
PREDICTION_YEARS = np.arange(2022, 2025)

# Two formerly duplicated workflows
SECTOR_CONFIGS = {
    "industry": {
        "sector_index": 2,
        "target_prefixes": ("ico_", "ipr_"),
        "use_land_mask": False,
    },
    "power": {
        "sector_index": 1,
        "target_prefixes": ("pow_",),
        "use_land_mask": True,
    },
}

N_SPLITS = 10
USE_LOG_TARGET = False
RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "objective": "reg:squarederror",
}


# =============================================================================
# 1. General helper functions
# =============================================================================

def set_plot_style() -> None:
    """Set a clean publication-compatible plotting style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    """Save each figure as PNG and PDF."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def extract_year(time_values) -> np.ndarray:
    """
    Convert integer, datetime64 or cftime coordinates to integer years.
    """
    values = np.asarray(time_values)

    if np.issubdtype(values.dtype, np.number):
        return values.astype(int)

    years = []
    for value in values:
        if hasattr(value, "year"):
            years.append(int(value.year))
        else:
            years.append(pd.Timestamp(value).year)

    return np.asarray(years, dtype=int)


def find_pollutant_file(pollutant: str) -> Path:
    """Locate one CMIP6 annual-total file for a proxy pollutant."""
    patterns = [
        str(BASE_PATH / f"{pollutant}_*annual_total*.nc"),
        str(BASE_PATH / f"{pollutant.lower()}_*annual_total*.nc"),
        str(BASE_PATH / f"{pollutant.upper()}_*annual_total*.nc"),
    ]

    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))

    matches = sorted(set(matches))

    if not matches:
        raise FileNotFoundError(
            f"No annual-total NetCDF file found for {pollutant} in {BASE_PATH}"
        )

    if len(matches) > 1:
        print(
            f"[warning] Multiple files found for {pollutant}; "
            f"using {matches[0]}"
        )

    return Path(matches[0])


def identify_data_variable(ds: xr.Dataset, pollutant: str) -> str:
    """Identify the principal data variable in a NetCDF dataset."""
    candidates = list(ds.data_vars)

    if not candidates:
        raise ValueError(f"No data variable found for {pollutant}")

    exact_candidates = [
        pollutant,
        pollutant.lower(),
        pollutant.upper(),
        f"{pollutant}_emission",
        f"{pollutant.lower()}_emission",
    ]

    for candidate in exact_candidates:
        if candidate in candidates:
            return candidate

    return candidates[0]


def select_target_years(da: xr.DataArray) -> xr.DataArray:
    """Select 2016–2024 for integer, datetime64 or cftime time axes."""
    if "time" not in da.dims:
        raise ValueError("The emission feature has no time dimension.")

    years = extract_year(da["time"].values)
    keep = np.isin(years, TARGET_YEARS)

    if not keep.any():
        raise ValueError(
            f"No target years {TARGET_YEARS.min()}–{TARGET_YEARS.max()} "
            "were found in the NetCDF time coordinate."
        )

    return da.isel(time=np.where(keep)[0])


# =============================================================================
# 2. Extract proxy-emission features
# =============================================================================

def extract_proxy_feature(
    pollutant: str,
    sector_index: int,
    target_geo: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract one pollutant proxy for all requested GEMS grid points.

    Requested GEMS coordinates are retained after nearest-neighbour extraction,
    preventing small coordinate differences from breaking the later merge.
    """
    nc_file = find_pollutant_file(pollutant)

    print(
        f"  Extracting {pollutant}: "
        f"sector={sector_index}, file={nc_file.name}"
    )

    with xr.open_dataset(nc_file) as ds:
        if "sector" in ds.dims or "sector" in ds.coords:
            available_sectors = np.asarray(ds["sector"].values)

            if sector_index not in available_sectors:
                raise ValueError(
                    f"Sector {sector_index} is unavailable in {nc_file.name}; "
                    f"available sectors: {available_sectors.tolist()}"
                )

            ds = ds.sel(sector=sector_index)

        var_name = identify_data_variable(ds, pollutant)
        da = select_target_years(ds[var_name])

        point_table = target_geo.reset_index(drop=True).copy()
        point_table["points"] = np.arange(len(point_table))

        target_lon = xr.DataArray(
            point_table["lon"].to_numpy(),
            dims="points",
            coords={"points": point_table["points"].to_numpy()},
        )
        target_lat = xr.DataArray(
            point_table["lat"].to_numpy(),
            dims="points",
            coords={"points": point_table["points"].to_numpy()},
        )

        matched = da.sel(
            lon=target_lon,
            lat=target_lat,
            method="nearest",
        )

        feature_name = f"{pollutant}_emission"

        feature = matched.to_dataframe(name=feature_name).reset_index()

        # Remove source-grid coordinates because they can differ slightly from
        # the requested GEMS coordinates after nearest-neighbour extraction.
        feature = feature.drop(
            columns=[
                column
                for column in ["lon", "lat", "sector"]
                if column in feature.columns
            ],
            errors="ignore",
        )

        feature["time"] = extract_year(feature["time"].values)

        feature = feature.merge(
            point_table[["points", "lon", "lat"]],
            on="points",
            how="left",
            validate="many_to_one",
        )

        return feature[["time", "points", "lon", "lat", feature_name]]


def extract_land_mask(target_geo: pd.DataFrame) -> pd.DataFrame | None:
    """
    Extract a land–sea mask at GEMS locations if a mask file is available.
    """
    mask_candidates = [
        BASE_PATH / "landseamask_0.5deg.nc",
        SCRIPT_DIR / "landseamask_0.5deg.nc",
    ]

    mask_file = next((p for p in mask_candidates if p.exists()), None)

    if mask_file is None:
        print(
            "[warning] landseamask_0.5deg.nc was not found; "
            "POW models will be fitted without the mask feature."
        )
        return None

    with xr.open_dataset(mask_file) as ds:
        var_name = list(ds.data_vars)[0]
        mask_da = ds[var_name]

        target_lon = xr.DataArray(
            target_geo["lon"].to_numpy(), dims="points"
        )
        target_lat = xr.DataArray(
            target_geo["lat"].to_numpy(), dims="points"
        )

        matched = mask_da.sel(
            lon=target_lon,
            lat=target_lat,
            method="nearest",
        )

        values = np.asarray(matched.values).squeeze()

    return pd.DataFrame(
        {
            "lon": target_geo["lon"].to_numpy(),
            "lat": target_geo["lat"].to_numpy(),
            "mask": values,
        }
    )


def build_sector_datasets(
    gems_dataset: pd.DataFrame,
    sector_name: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build training and future-prediction tables for one sector group."""
    print(f"\n{'=' * 72}")
    print(f"Preparing features for {sector_name}")
    print(f"{'=' * 72}")

    target_geo = (
        gems_dataset[["lon", "lat"]]
        .drop_duplicates()
        .sort_values(["lat", "lon"])
        .reset_index(drop=True)
    )

    feature_tables = []

    for pollutant in POLLUTANTS:
        table = extract_proxy_feature(
            pollutant=pollutant,
            sector_index=config["sector_index"],
            target_geo=target_geo,
        )
        feature_tables.append(table)

    if not feature_tables:
        raise RuntimeError(f"No proxy features were extracted for {sector_name}")

    final_features = feature_tables[0]

    for table in feature_tables[1:]:
        final_features = final_features.merge(
            table,
            on=["time", "points", "lon", "lat"],
            how="outer",
            validate="one_to_one",
        )

    if config["use_land_mask"]:
        mask_table = extract_land_mask(target_geo)

        if mask_table is not None:
            final_features = final_features.merge(
                mask_table,
                on=["lon", "lat"],
                how="left",
                validate="many_to_one",
            )

    gems = gems_dataset.copy()
    gems["time"] = pd.to_numeric(gems["time"], errors="coerce")
    gems = gems.dropna(subset=["time", "lon", "lat"]).copy()
    gems["time"] = gems["time"].astype(int)

    train_dataset = gems.merge(
        final_features.drop(columns="points"),
        on=["time", "lon", "lat"],
        how="left",
        validate="many_to_one",
    )

    pred_dataset = final_features[
        final_features["time"].isin(PREDICTION_YEARS)
    ].copy()

    pred_dataset = pred_dataset.drop(columns="points")
    pred_dataset = pred_dataset.sort_values(
        ["time", "lat", "lon"]
    ).reset_index(drop=True)

    train_file = OUTPUT_PATH / f"train_for_gems_emi_{sector_name}.csv"
    pred_file = OUTPUT_PATH / f"pred_for_gems_emi_{sector_name}.csv"

    train_dataset.to_csv(train_file, index=False)
    pred_dataset.to_csv(pred_file, index=False)

    print(f"  Training table: {train_file}")
    print(f"  Prediction table: {pred_file}")
    print(f"  Training rows: {len(train_dataset):,}")
    print(f"  Prediction rows: {len(pred_dataset):,}")

    return train_dataset, pred_dataset


# =============================================================================
# 3. Model validation and prediction
# =============================================================================

def plot_cv_scatter(
    observed: np.ndarray,
    predicted: np.ndarray,
    target: str,
    r2_value: float,
    target_dir: Path,
) -> None:
    """Plot observed versus cross-validated predictions."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite]
    predicted = predicted[finite]

    min_value = min(observed.min(), predicted.min())
    max_value = max(observed.max(), predicted.max())

    if np.isclose(min_value, max_value):
        margin = max(abs(max_value) * 0.05, 1.0)
    else:
        margin = (max_value - min_value) * 0.05

    limits = [min_value - margin, max_value + margin]

    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    ax.scatter(
        observed,
        predicted,
        s=12,
        alpha=0.30,
        color="#3B6FB6",
        edgecolors="none",
        rasterized=True,
    )

    ax.plot(
        limits,
        limits,
        linestyle="--",
        linewidth=1.0,
        color="#333333",
        label="1:1 line",
    )

    if len(np.unique(observed)) > 1:
        slope, intercept = np.polyfit(observed, predicted, 1)
        x_line = np.linspace(limits[0], limits[1], 200)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            linewidth=1.3,
            color="#C44E52",
            label="Linear fit",
        )

    ax.text(
        0.04,
        0.96,
        f"10-fold CV $R^2$ = {r2_value:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    ax.set(
        xlabel="Observed emission",
        ylabel="Cross-validated prediction",
        title=target,
        xlim=limits,
        ylim=limits,
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="both", linestyle=":", linewidth=0.6, alpha=0.45)

    save_figure(fig, target_dir / f"{target}_cv_scatter")


def plot_time_series(
    time_series: pd.DataFrame,
    target: str,
    target_dir: Path,
) -> None:
    """Plot annual spatial median, IQR and mean."""
    yearly = (
        time_series.groupby("time")[target]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            mean="mean",
        )
        .reset_index()
        .sort_values("time")
    )

    yearly.to_csv(
        target_dir / f"{target}_annual_summary.csv",
        index=False,
    )

    # Median and IQR
    fig, ax = plt.subplots(figsize=(6.3, 4.0))

    ax.fill_between(
        yearly["time"].to_numpy(),
        yearly["q25"].to_numpy(),
        yearly["q75"].to_numpy(),
        color="#9ECAE1",
        alpha=0.45,
        linewidth=0,
        label="Interquartile range",
    )

    ax.plot(
        yearly["time"],
        yearly["median"],
        marker="o",
        markersize=3.5,
        linewidth=1.6,
        color="#2166AC",
        label="Median",
    )

    ax.axvline(
        2021.5,
        color="#777777",
        linestyle="--",
        linewidth=0.8,
    )

    positive_values = yearly[["q25", "median", "q75"]].to_numpy()
    if np.all(positive_values[np.isfinite(positive_values)] > 0):
        ax.set_yscale("log")
        ylabel = "Emission (log scale)"
    else:
        ylabel = "Emission"

    ax.set(
        xlabel="Year",
        ylabel=ylabel,
        title=target,
        xticks=np.arange(2016, 2025),
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.45)

    save_figure(fig, target_dir / f"{target}_median_iqr_series")

    # Mean
    fig, ax = plt.subplots(figsize=(6.3, 4.0))

    ax.plot(
        yearly["time"],
        yearly["mean"],
        marker="o",
        markersize=3.5,
        linewidth=1.6,
        color="#B2182B",
    )

    ax.axvline(
        2021.5,
        color="#777777",
        linestyle="--",
        linewidth=0.8,
    )

    ax.set(
        xlabel="Year",
        ylabel="Mean emission",
        title=target,
        xticks=np.arange(2016, 2025),
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.45)

    save_figure(fig, target_dir / f"{target}_mean_series")


def run_sector_models(
    gems_dataset: pd.DataFrame,
    train_dataset: pd.DataFrame,
    pred_dataset: pd.DataFrame,
    sector_name: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and fit every target belonging to one sector group."""
    target_columns = [
        column
        for column in gems_dataset.columns
        if column.startswith(config["target_prefixes"])
    ]

    if not target_columns:
        raise ValueError(
            f"No target columns beginning with "
            f"{config['target_prefixes']} were found for {sector_name}."
        )

    feature_columns = [
        column
        for column in train_dataset.columns
        if column.endswith("_emission")
    ]

    meta_columns = ["time", "lat", "lon"]

    if config["use_land_mask"] and "mask" in train_dataset.columns:
        meta_columns.append("mask")

    print(f"\nTargets for {sector_name}: {len(target_columns)}")
    print(f"Features: {feature_columns + meta_columns}")

    prediction_results = pred_dataset[
        ["time", "lat", "lon"]
    ].copy()

    if "mask" in pred_dataset.columns:
        prediction_results["mask"] = pred_dataset["mask"].to_numpy()

    metric_rows = []

    for target_index, target in enumerate(target_columns, start=1):
        model_start = time.perf_counter()

        print(f"\n[{sector_name} {target_index}/{len(target_columns)}] {target}")

        target_dir = OUTPUT_PATH / sector_name / target
        target_dir.mkdir(parents=True, exist_ok=True)

        required_train = [target] + feature_columns + meta_columns

        valid_train = train_dataset.dropna(
            subset=required_train
        ).copy()

        if valid_train.empty:
            print("  [skip] No complete training observations.")
            continue

        y_original = valid_train[target].astype(float).to_numpy()

        if np.allclose(y_original, 0):
            print("  [skip] Target values are all zero.")
            continue

        scaler = StandardScaler()

        scaled_features = scaler.fit_transform(
            valid_train[feature_columns]
        )

        X_train = pd.DataFrame(
            scaled_features,
            columns=feature_columns,
            index=valid_train.index,
        )

        for column in meta_columns:
            X_train[column] = valid_train[column].astype(float)

        X_train = X_train[feature_columns + meta_columns]

        if USE_LOG_TARGET:
            y_model = np.log10(np.clip(y_original, 0, None) + 0.001)
        else:
            y_model = y_original.copy()

        n_splits = min(N_SPLITS, len(valid_train))

        if n_splits < 2:
            print("  [skip] Fewer than two complete observations.")
            continue

        cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        cv_model = xgb.XGBRegressor(**MODEL_PARAMS)

        y_cv_model = cross_val_predict(
            cv_model,
            X_train,
            y_model,
            cv=cv,
            n_jobs=1,
        )

        if USE_LOG_TARGET:
            y_cv = np.power(10.0, y_cv_model) - 0.001
        else:
            y_cv = y_cv_model

        y_cv = np.clip(y_cv, 0, None)

        r2_original = r2_score(y_original, y_cv)
        rmse_original = np.sqrt(
            mean_squared_error(y_original, y_cv)
        )
        spearman_rho, spearman_p = spearmanr(
            y_original,
            y_cv,
            nan_policy="omit",
        )

        r2_log = r2_score(
            np.log10(np.clip(y_original, 0, None) + 0.001),
            np.log10(np.clip(y_cv, 0, None) + 0.001),
        )

        print(f"  CV R², original scale: {r2_original:.4f}")
        print(f"  CV R², log10 scale:    {r2_log:.4f}")
        print(f"  CV RMSE:               {rmse_original:.4e}")
        print(f"  Spearman rho:          {spearman_rho:.4f}")

        cv_output = valid_train[
            ["time", "lat", "lon", target]
        ].copy()

        cv_output["cv_prediction"] = y_cv
        cv_output["residual"] = y_original - y_cv

        cv_output.to_csv(
            target_dir / f"{target}_cv_predictions.csv",
            index=False,
        )

        plot_cv_scatter(
            observed=y_original,
            predicted=y_cv,
            target=target,
            r2_value=r2_original,
            target_dir=target_dir,
        )

        # Fit final model using all historical observations
        final_model = xgb.XGBRegressor(**MODEL_PARAMS)
        final_model.fit(X_train, y_model)

        importance_df = pd.DataFrame(
            {
                "feature": feature_columns + meta_columns,
                "importance": final_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        importance_df.to_csv(
            target_dir / f"{target}_feature_importance.csv",
            index=False,
        )

        valid_pred = pred_dataset.dropna(
            subset=feature_columns + meta_columns
        ).copy()

        X_pred = pd.DataFrame(
            scaler.transform(valid_pred[feature_columns]),
            columns=feature_columns,
            index=valid_pred.index,
        )

        for column in meta_columns:
            X_pred[column] = valid_pred[column].astype(float)

        X_pred = X_pred[feature_columns + meta_columns]

        pred_model_scale = final_model.predict(X_pred)

        if USE_LOG_TARGET:
            target_prediction = (
                np.power(10.0, pred_model_scale) - 0.001
            )
        else:
            target_prediction = pred_model_scale

        target_prediction = np.clip(
            target_prediction,
            0,
            None,
        )

        prediction_results[target] = np.nan
        prediction_results.loc[
            valid_pred.index,
            target,
        ] = target_prediction

        future_target = valid_pred[["time", "lat", "lon"]].copy()
        future_target[target] = target_prediction

        historical_target = train_dataset[
            ["time", "lat", "lon", target]
        ].dropna(subset=[target])

        time_series = pd.concat(
            [historical_target, future_target],
            ignore_index=True,
        ).sort_values(["time", "lat", "lon"])

        time_series.to_csv(
            target_dir / f"{target}_2016_2024.csv",
            index=False,
        )

        plot_time_series(
            time_series=time_series,
            target=target,
            target_dir=target_dir,
        )

        elapsed_seconds = time.perf_counter() - model_start

        metric_rows.append(
            {
                "sector_group": sector_name,
                "target": target,
                "n_train": len(valid_train),
                "n_folds": n_splits,
                "r2_original": r2_original,
                "r2_log10": r2_log,
                "rmse_original": rmse_original,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
                "runtime_seconds": elapsed_seconds,
            }
        )

    metrics = pd.DataFrame(metric_rows)

    predictions_file = (
        OUTPUT_PATH /
        f"Predicted_PAH_2022_2024_{sector_name}.csv"
    )
    metrics_file = (
        OUTPUT_PATH /
        f"PAH_10fold_metrics_{sector_name}.csv"
    )

    prediction_results.to_csv(predictions_file, index=False)
    metrics.to_csv(metrics_file, index=False)

    print(f"\nSaved predictions: {predictions_file}")
    print(f"Saved metrics: {metrics_file}")

    return prediction_results, metrics


# =============================================================================
# 4. Combine POW, IPR and ICO predictions
# =============================================================================

def combine_predictions(
    sector_predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge sector prediction tables using time, latitude and longitude."""
    combined = None

    for sector_name, table in sector_predictions.items():
        target_columns = [
            column
            for column in table.columns
            if column not in ["time", "lat", "lon", "mask"]
        ]

        current = table[
            ["time", "lat", "lon"] + target_columns
        ].copy()

        if combined is None:
            combined = current
        else:
            combined = combined.merge(
                current,
                on=["time", "lat", "lon"],
                how="outer",
                validate="one_to_one",
            )

    if combined is None:
        raise RuntimeError("No sector prediction table was produced.")

    combined = combined.sort_values(
        ["time", "lat", "lon"]
    ).reset_index(drop=True)

    combined_file = (
        OUTPUT_PATH / "Predicted_PAH_2022_2024.csv"
    )
    combined.to_csv(combined_file, index=False)

    print(f"\nCombined prediction file: {combined_file}")

    return combined


# =============================================================================
# 5. Main workflow
# =============================================================================

def main() -> None:
    total_start = time.perf_counter()
    set_plot_style()

    if not GEMS_FILE.exists():
        raise FileNotFoundError(
            f"GEMS emission file not found: {GEMS_FILE}"
        )

    print(f"Reading GEMS data: {GEMS_FILE}")
    gems_dataset = pd.read_csv(GEMS_FILE)

    required_columns = {"time", "lon", "lat"}
    missing_columns = required_columns - set(gems_dataset.columns)

    if missing_columns:
        raise ValueError(
            f"GEMS file is missing columns: {sorted(missing_columns)}"
        )

    sector_predictions = {}
    all_metrics = []

    for sector_name, config in SECTOR_CONFIGS.items():
        train_dataset, pred_dataset = build_sector_datasets(
            gems_dataset=gems_dataset,
            sector_name=sector_name,
            config=config,
        )

        predictions, metrics = run_sector_models(
            gems_dataset=gems_dataset,
            train_dataset=train_dataset,
            pred_dataset=pred_dataset,
            sector_name=sector_name,
            config=config,
        )

        sector_predictions[sector_name] = predictions
        all_metrics.append(metrics)

    combined_predictions = combine_predictions(
        sector_predictions
    )

    if all_metrics:
        combined_metrics = pd.concat(
            all_metrics,
            ignore_index=True,
        )

        combined_metrics.to_csv(
            OUTPUT_PATH / "PAH_10fold_metrics_all_sectors.csv",
            index=False,
        )

    elapsed_minutes = (time.perf_counter() - total_start) / 60

    print("\n" + "=" * 72)
    print("PAH prediction completed")
    print(f"Output directory: {OUTPUT_PATH}")
    print(f"Combined rows: {len(combined_predictions):,}")
    print(f"Total runtime: {elapsed_minutes:.2f} minutes")
    print("=" * 72)


if __name__ == "__main__":
    main()