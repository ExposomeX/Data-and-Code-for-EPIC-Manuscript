"""Predict all future pollutant-sector emissions into one CSV for three SSPs.

Inputs: code_3_results future proxy/GDP/population tables and the trained
models in code_4.2_results/<target>/xgb_model_<target>_<ssp>.json.
Output: code_5_results/pred_results_all.csv only.

Each output row identifies a scenario, year and grid cell. The other columns
are pollutant-sector predictions. No per-target CSVs or figures are produced.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


BASE = Path(__file__).resolve().parent
SCENARIOS = ("ssp126", "ssp245", "ssp370")
KEYS = ["lat", "lon", "time"]
SCALED_TARGETS = {"Gd_MSWI", "Hexachlorobutadiene_MSWI"}


def read_table(path: Path, keep: list[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    needed = KEYS if keep is None else keep
    missing = sorted(set(needed) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} lacks columns: {missing}")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{path.name} has duplicate grid/year keys")
    return frame if keep is None else frame[keep]


def load_future_panel(data_dir: Path, scenario: str) -> pd.DataFrame:
    proxy = read_table(data_dir / f"grid_level_proxy_6_pols_fut_{scenario}.csv")
    proxy_columns = [name for name in proxy.columns if "emission" in str(name)]
    if not proxy_columns:
        raise ValueError(f"{scenario}: no emission proxy columns")
    proxy = proxy[KEYS + proxy_columns]
    gdp = read_table(data_dir / f"grid_level_gdp_{scenario[:4]}.csv", KEYS + ["GDP"])
    population = read_table(
        data_dir / f"grid_level_population_{scenario}.csv", KEYS + ["population"]
    )
    population = population.loc[population["time"] != 2020]
    panel = proxy.merge(gdp, on=KEYS, how="inner", validate="one_to_one")
    panel = panel.merge(population, on=KEYS, how="inner", validate="one_to_one")
    if panel.empty:
        raise ValueError(f"{scenario}: no matched future grid records")
    panel.columns = [str(name).replace(".0", "") for name in panel.columns]
    if panel.columns.duplicated().any():
        raise ValueError(f"{scenario}: feature names collide after normalization")
    if panel[KEYS].isna().any().any():
        raise ValueError(f"{scenario}: missing grid coordinate or year")
    return panel


def find_targets(model_dir: Path) -> list[str]:
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    targets = sorted(
        folder.name for folder in model_dir.iterdir()
        if folder.is_dir() and
        (folder / f"xgb_model_{folder.name}_ssp126.json").is_file()
    )
    if not targets:
        raise FileNotFoundError(f"No trained models in {model_dir}")
    missing = [
        model_dir / target / f"xgb_model_{target}_{scenario}.json"
        for target in targets for scenario in SCENARIOS
        if not (model_dir / target / f"xgb_model_{target}_{scenario}.json").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} target-scenario models missing; first: {missing[0]}"
        )
    return targets


def predict_target(panel: pd.DataFrame, model_path: Path, target: str) -> np.ndarray:
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    features = model.get_booster().feature_names
    if not features:
        raise ValueError(f"Saved model lacks feature names: {model_path}")
    missing = [name for name in features if name not in panel.columns]
    if missing:
        raise ValueError(f"{target}: missing trained predictors {missing}")
    prediction = np.asarray(model.predict(panel[features]), dtype=np.float64)
    prediction = np.maximum(prediction, 0)
    if target in SCALED_TARGETS:
        prediction /= 1e5
    if not np.isfinite(prediction).all():
        raise ValueError(f"{target}: non-finite predictions")
    return prediction


def run(data_dir: Path, model_dir: Path, output_file: Path) -> None:
    targets = find_targets(model_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_name(output_file.name + ".incomplete")
    if temporary.exists():
        temporary.unlink()
    total_rows = 0
    try:
        for index, scenario in enumerate(SCENARIOS):
            panel = load_future_panel(data_dir, scenario)
            predictions = {}
            for number, target in enumerate(targets, start=1):
                path = model_dir / target / f"xgb_model_{target}_{scenario}.json"
                predictions[target] = predict_target(panel, path, target)
                if number % 25 == 0 or number == len(targets):
                    print(f"{scenario}: {number}/{len(targets)} targets", flush=True)
            result = pd.concat(
                [
                    pd.DataFrame({"scenario": scenario}, index=panel.index),
                    panel[KEYS],
                    pd.DataFrame(predictions, index=panel.index),
                ],
                axis=1,
            )
            result.to_csv(
                temporary, mode="w" if index == 0 else "a",
                header=index == 0, index=False,
            )
            total_rows += len(result)
            print(f"{scenario}: {len(result):,} grid/year records", flush=True)
        temporary.replace(output_file)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Saved {output_file}: {total_rows:,} rows, {len(targets)} targets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE / "code_3_results")
    parser.add_argument("--model-dir", type=Path, default=BASE / "code_4_results")
    parser.add_argument(
        "--output", type=Path,
        default=BASE / "code_5_results" / "pred_results_all.csv",
    )
    args = parser.parse_args()
    run(args.data_dir, args.model_dir, args.output)


if __name__ == "__main__":
    main()
