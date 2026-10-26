# -*- coding: utf-8 -*-
"""Merge non-PAH and PAH emissions and export annual pollutant NetCDF files.

Run this script before code_5_plot_selected_emission.py. No maps are made here.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
GRID_INPUT = RESULTS_DIR / "Grid_Emission"
GRID_INPUT_FALLBACK = RESULTS_DIR / "code_3.2_pred_grid_emission" / "Grid_Emission.csv"
PAH_INPUT = RESULTS_DIR / "code_4.1_predict_PAH" / "Predicted_PAH_2022_2024.csv"
PAH_HISTORICAL_INPUT = SCRIPT_DIR / "data_raw" / "0.5deg_gems_emission.csv"
POLLUTANT_LIST = SCRIPT_DIR / "data_raw" / "pollutant list.xlsx"
OUTPUT_DIR = RESULTS_DIR / "code_5_all_emissions" / "grid_level_emission"
CSV_OUTPUT = OUTPUT_DIR / "grid_level_emission.csv"
NC_ROOT = OUTPUT_DIR / "netcdf_by_pollutant_year"
YEAR_MIN, YEAR_MAX = 1990, 2023
PAH_YEAR_MIN, PAH_YEAR_MAX = 2016, 2023
SECTORS = ["BFI", "CFPP", "CK", "COP", "HWI", "IOS", "MSWI", "PCu", "SAl", "SCu", "SZn", "EAF", "SPb"]
GEMS_SECTORS = ["ico", "ipr", "pow"]
KEYS = ["time", "lat", "lon"]
CHUNK_SIZE = 20_000
LOGGER = logging.getLogger("grid_level_emission")


def safe(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_.") or "unnamed"


def resolve_grid_input() -> Path:
    """Resolve a CSV file or directory named results/Grid_Emission."""
    if GRID_INPUT.is_file():
        return GRID_INPUT
    if GRID_INPUT.is_dir():
        candidates = sorted(GRID_INPUT.glob("*.csv"))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(f"No CSV file found in {GRID_INPUT}")
        named = [path for path in candidates if path.stem.lower() == "grid_emission"]
        if len(named) == 1:
            return named[0]
        raise ValueError(f"Multiple CSV files in {GRID_INPUT}; expected one Grid_Emission CSV.")
    if GRID_INPUT.with_suffix(".csv").is_file():
        return GRID_INPUT.with_suffix(".csv")
    if GRID_INPUT_FALLBACK.is_file():
        LOGGER.warning("Using existing Grid_Emission output at %s", GRID_INPUT_FALLBACK)
        return GRID_INPUT_FALLBACK
    raise FileNotFoundError(
        "Could not find the non-PAH Grid_Emission input. Checked "
        f"{GRID_INPUT}, {GRID_INPUT.with_suffix('.csv')} and {GRID_INPUT_FALLBACK}."
    )


def load_pollutant_lists() -> tuple[list[str], list[str], dict[str, str]]:
    if not POLLUTANT_LIST.is_file():
        raise FileNotFoundError(f"Pollutant list workbook not found: {POLLUTANT_LIST}")
    table = pd.read_excel(POLLUTANT_LIST)
    required = {"pollutant abbr.", "source"}
    if not required.issubset(table.columns):
        raise ValueError(f"Pollutant list must contain columns {sorted(required)}")
    source = table["source"].astype(str).str.strip().str.casefold()
    liu = table.loc[source.isin({"l", "liu"}), "pollutant abbr."].dropna().astype(str).str.strip().drop_duplicates().tolist()
    gems = table.loc[source.eq("gems"), "pollutant abbr."].dropna().astype(str).str.strip().drop_duplicates().tolist()
    if len(liu) != 28:
        raise ValueError(f"Expected 28 Liu pollutants in the workbook, found {len(liu)}")
    if not gems:
        raise ValueError("No GEMS pollutants were found in the pollutant list workbook.")
    units = {
        str(row["pollutant abbr."]).strip(): str(row["unit"]).strip()
        for _, row in table.loc[table["pollutant abbr."].notna()].iterrows()
    }
    return liu, gems, units


def discover_nonpah_columns(columns: list[str], allowed_pollutants: list[str]) -> dict[str, list[str]]:
    """Find pollutant-sector values in Grid_Emission; ignore *_Sum columns."""
    result: dict[str, list[str]] = {}
    sector_lookup = {sector.lower(): sector for sector in SECTORS}
    for column in columns:
        if column in {"Year", "time", "lat", "lon"} or column.endswith("_Sum"):
            continue
        if "_" not in column:
            continue
        pollutant, sector = column.rsplit("_", 1)
        normalized_sector = sector_lookup.get(sector.lower())
        if normalized_sector and pollutant in allowed_pollutants:
            result.setdefault(pollutant, []).append(column)
    sector_rank = {sector.lower(): index for index, sector in enumerate(SECTORS)}
    for pollutant in result:
        result[pollutant].sort(key=lambda c: sector_rank[c.rsplit("_", 1)[-1].lower()])
    return result


def read_nonpah_grid(path: Path, allowed_pollutants: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]], list[int]]:
    """Load and standardize the non-PAH wide table."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    key_year = "Year" if "Year" in header else "time" if "time" in header else None
    if key_year is None or not {"lat", "lon"}.issubset(header):
        raise ValueError(f"Non-PAH input must contain Year/time, lat and lon: {path}")
    pollutant_columns = discover_nonpah_columns(header, allowed_pollutants)
    if not pollutant_columns:
        raise ValueError(f"No pollutant-sector columns found in {path}")
    missing_pollutants = sorted(set(allowed_pollutants) - set(pollutant_columns))
    if missing_pollutants:
        raise ValueError(f"These Liu pollutants are absent from Grid_Emission: {missing_pollutants}")
    value_columns = [column for cols in pollutant_columns.values() for column in cols]
    usecols = [key_year, "lat", "lon", *value_columns]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        chunk = chunk.rename(columns={key_year: "time"})
        chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
        chunk = chunk.loc[chunk["time"].between(YEAR_MIN, YEAR_MAX)].copy()
        if chunk.empty:
            continue
        chunk["time"] = chunk["time"].astype("int16")
        for column in value_columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("float32")
        chunks.append(chunk)
        LOGGER.info("Loaded non-PAH rows: %s", f"{sum(len(c) for c in chunks):,}")
    if not chunks:
        raise ValueError("The non-PAH input contains no rows between 1990 and 2023.")
    data = pd.concat(chunks, ignore_index=True)
    data = data.groupby(KEYS, as_index=False, sort=False)[value_columns].sum(min_count=1)
    years = sorted(data["time"].unique().astype(int).tolist())
    return data, pollutant_columns, years


def read_pah_grid(path: Path, allowed_species: list[str], year_min: int, year_max: int) -> tuple[pd.DataFrame, dict[str, list[str]], list[int]]:
    """Convert source-sector-first PAH names (ico_BaP) to pollutant_sector."""
    if not path.is_file():
        raise FileNotFoundError(f"PAH input not found: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    year_col = "time" if "time" in header else "Year" if "Year" in header else None
    if year_col is None or not {"lat", "lon"}.issubset(header):
        raise ValueError(f"PAH input must contain time/Year, lat and lon: {path}")
    source_columns: dict[str, str] = {}
    pollutant_columns: dict[str, list[str]] = {}
    for source_column in header:
        if source_column in {year_col, "lat", "lon"} or "_" not in source_column:
            continue
        sector, species = source_column.split("_", 1)
        if sector not in GEMS_SECTORS or species not in allowed_species:
            continue
        target_column = f"{species}_{sector}"
        source_columns[source_column] = target_column
        pollutant_columns.setdefault(species, []).append(target_column)
    if not source_columns:
        raise ValueError(f"No PAH species-sector columns found in {path}")
    data = pd.read_csv(path, usecols=[year_col, "lat", "lon", *source_columns], low_memory=False)
    data = data.rename(columns={year_col: "time", **source_columns})
    data["time"] = pd.to_numeric(data["time"], errors="coerce")
    data = data.loc[data["time"].between(year_min, year_max)].copy()
    data["time"] = data["time"].astype("int16")
    target_columns = list(source_columns.values())
    data[target_columns] = data[target_columns].apply(pd.to_numeric, errors="coerce").astype("float32")
    data = data.copy()
    data = data.groupby(KEYS, as_index=False, sort=False)[target_columns].sum(min_count=1)
    years = sorted(data["time"].unique().astype(int).tolist())
    for species in pollutant_columns:
        pollutant_columns[species] = [f"{species}_{sector}" for sector in GEMS_SECTORS if f"{species}_{sector}" in target_columns]
    return data, pollutant_columns, years


def merge_grid_emissions() -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, list[int]], dict[str, str]]:
    grid_path = resolve_grid_input()
    LOGGER.info("Reading non-PAH source: %s", grid_path)
    liu_pollutants, gems_pollutants, pollutant_units = load_pollutant_lists()
    grid, grid_map, grid_years = read_nonpah_grid(grid_path, liu_pollutants)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    liu_filtered_path = OUTPUT_DIR / "filtered_liu_grid_emission.csv"
    grid.to_csv(liu_filtered_path, index=False, float_format="%.8g", encoding="utf-8-sig")
    LOGGER.info("Saved Liu-filtered result: %s", liu_filtered_path)

    LOGGER.info("Reading historical GEMS PAHs: %s", PAH_HISTORICAL_INPUT)
    pah_historical, historical_map, historical_years = read_pah_grid(
        PAH_HISTORICAL_INPUT, gems_pollutants, PAH_YEAR_MIN, 2021
    )
    LOGGER.info("Reading predicted GEMS PAHs: %s", PAH_INPUT)
    pah_future, future_map, future_years = read_pah_grid(
        PAH_INPUT, gems_pollutants, 2022, PAH_YEAR_MAX
    )
    pah = pd.concat([pah_historical, pah_future], ignore_index=True, sort=False)
    pah_columns = sorted({column for mapping in (historical_map, future_map) for values in mapping.values() for column in values})
    pah = pah.groupby(KEYS, as_index=False, sort=False)[pah_columns].sum(min_count=1)
    pah_map = {
        pollutant: [f"{pollutant}_{sector}" for sector in GEMS_SECTORS if f"{pollutant}_{sector}" in pah_columns]
        for pollutant in gems_pollutants
        if any(column.startswith(f"{pollutant}_") for column in pah_columns)
    }
    pah.to_csv(OUTPUT_DIR / "filtered_pah_2016_2023.csv", index=False, float_format="%.8g", encoding="utf-8-sig")
    overlap = set(grid_map).intersection(pah_map)
    if overlap:
        raise ValueError(f"Pollutants occur in both inputs; prevent ambiguous replacement: {sorted(overlap)}")
    merged = grid.merge(pah, on=KEYS, how="outer", validate="one_to_one", sort=False)
    merged = merged.sort_values(["time", "lat", "lon"], kind="stable").reset_index(drop=True)
    pollutant_map = {**grid_map, **pah_map}
    merged.to_csv(CSV_OUTPUT, index=False, float_format="%.8g", encoding="utf-8-sig")
    LOGGER.info("Saved merged grid-level emission table: %s (%s rows)", CSV_OUTPUT, f"{len(merged):,}")
    coverage = {
        "Liu_non_PAH": grid_years,
        "GEMS_PAH_historical": historical_years,
        "GEMS_PAH_predicted": future_years,
        "PAH_combined": sorted(pah["time"].unique().astype(int).tolist()),
    }
    return merged, pollutant_map, coverage, pollutant_units


def nc_path(pollutant: str, year: int) -> Path:
    return NC_ROOT / safe(pollutant) / f"{safe(pollutant)}_{year}.nc"


def export_netcdf(merged: pd.DataFrame, pollutant_map: dict[str, list[str]], pollutant_units: dict[str, str]) -> pd.DataFrame:
    NC_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for pollutant, columns in pollutant_map.items():
        sector_codes = [column.rsplit("_", 1)[-1] for column in columns]
        sector_order = GEMS_SECTORS if set(sector_codes).issubset(GEMS_SECTORS) else SECTORS
        pollutant_data = merged.loc[merged[columns].notna().any(axis=1), [*KEYS, *columns]].copy()
        years = sorted(pollutant_data["time"].unique().astype(int).tolist())
        lats = np.sort(pollutant_data["lat"].unique().astype(float))
        lons = np.sort(pollutant_data["lon"].unique().astype(float))
        folder = NC_ROOT / safe(pollutant)
        folder.mkdir(parents=True, exist_ok=True)
        for year in years:
            subset = pollutant_data.loc[pollutant_data["time"].eq(year)].copy()
            cube = np.full((len(sector_order), len(lats), len(lons)), np.nan, dtype=np.float32)
            xi = np.searchsorted(lons, subset["lon"].to_numpy(float))
            yi = np.searchsorted(lats, subset["lat"].to_numpy(float))
            for si, sector in enumerate(sector_order):
                column = f"{pollutant}_{sector}"
                if column in subset:
                    cube[si, yi, xi] = pd.to_numeric(subset[column], errors="coerce").to_numpy(np.float32)
            if np.any(cube[np.isfinite(cube)] < 0):
                raise ValueError(f"Negative emission values found for {pollutant}, {year}.")
            path = nc_path(pollutant, year)
            ds = xr.Dataset(
                {"emission": (("time", "sector", "lat", "lon"), cube[None, ...])},
                coords={"time": [np.datetime64(f"{year}-01-01")], "sector": sector_order, "lat": lats, "lon": lons},
            )
            ds["emission"].attrs.update({"long_name": f"Annual {pollutant} emissions by sector", "units": pollutant_units.get(pollutant, "source CSV units")})
            ds.attrs.update({"pollutant": pollutant, "year": year, "missing_values": "NaN; no gap filling"})
            ds.to_netcdf(path, engine="netcdf4", encoding={"emission": {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": np.float32(-9999)}})
            manifest.append({"pollutant": pollutant, "year": year, "netcdf": str(path.relative_to(OUTPUT_DIR)), "sectors": ";".join(sector_order), "n_grid_cells": int(subset[columns].notna().any(axis=1).sum())})
    result = pd.DataFrame(manifest)
    result.to_csv(OUTPUT_DIR / "netcdf_manifest.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    """Merge source tables and export the grid-level CSV and annual NetCDF files."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged, pollutant_map, coverage, pollutant_units = merge_grid_emissions()
    for source_name, years in coverage.items():
        LOGGER.info("%s years present: %s", source_name, years)
    grid_years = coverage["Liu_non_PAH"]
    pah_years = coverage["PAH_combined"]
    if grid_years and (min(grid_years) > YEAR_MIN or max(grid_years) < YEAR_MAX):
        LOGGER.warning("Non-PAH source does not cover every year from %s to %s; no missing years will be fabricated.", YEAR_MIN, YEAR_MAX)
    if pah_years and (min(pah_years) > PAH_YEAR_MIN or max(pah_years) < PAH_YEAR_MAX):
        LOGGER.warning("PAH input does not span the full requested %s-%s period.", PAH_YEAR_MIN, PAH_YEAR_MAX)
    pd.DataFrame([
        {"source": source_name, "first_year": min(years) if years else np.nan, "last_year": max(years) if years else np.nan, "years_present": ";".join(map(str, years))}
        for source_name, years in coverage.items()
    ]).to_csv(OUTPUT_DIR / "source_year_coverage.csv", index=False, encoding="utf-8-sig")
    nc_manifest = export_netcdf(merged, pollutant_map, pollutant_units)
    LOGGER.info("Exported %s annual NetCDF files for %s pollutants", len(nc_manifest),
                nc_manifest["pollutant"].nunique())
    LOGGER.info("All outputs are under %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
