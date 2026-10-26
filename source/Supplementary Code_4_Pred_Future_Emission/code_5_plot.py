"""Export annual pollutant NetCDF files and sector-specific emission maps.

Input: code_5_results/pred_results_all.csv
Output: code_5_plot/{netcdf_by_pollutant_year, selected_maps,
                    pollutant_year_file_manifest.csv,
                    export_validation_summary.csv,source_coverage.json}

Each NetCDF stores emission(time, scenario, sector, lat, lon). Missing source
combinations and grid cells remain missing rather than being interpreted as zero.
Only pollutants with units in the input workbook are exported. Each map shows
one pollutant, one source sector, one scenario, and one year.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from netCDF4 import Dataset, date2num
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.ndimage import gaussian_filter


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = Path(BASE_PATH) / "code_5_results" / "pred_results.csv"
OUTPUT_PATH = Path(BASE_PATH) / "code_5_plot"
BOUNDARY_DIR = Path(BASE_PATH).parent / "Supplementary Code_3_Create_Emission_Index" / "create_index" / "data_raw" / "shp版"
UNIT_FILE = Path(BASE_PATH).parent / "Supplementary Code 1_Cal_Emission" / "data_raw" / "pollutant list.xlsx"
SECTORS = ["BFI", "CFPP", "CK", "COP", "HWI", "IOS", "MSWI", "PCu", "SAl", "SCu", "SZn", "EAF", "SPb", "ipr", "ico", "pow"]
SCENARIO_LABELS = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5", "ssp370": "SSP3-7.0"}
CMAP = LinearSegmentedColormap.from_list("reference_log", [
    "#5D55A5", "#4E91B5", "#76BFA5", "#DDF08A",
    "#FFF08A", "#F2A15A", "#A52B4F",
], N=256)
CMAP.set_bad("white")
PLATE = ccrs.PlateCarree()


def load_boundaries():
    province_path = BOUNDARY_DIR / "中国_省_面.shp"
    dash_path = BOUNDARY_DIR / "china_nine_dotted_line.shp"
    if not province_path.exists() or not dash_path.exists():
        raise FileNotFoundError("Province polygons or South China Sea boundary are missing")
    provinces = gpd.read_file(province_path)
    if len(provinces) != 34 or provinces.crs is None:
        raise ValueError("Province shapefile must contain 34 georeferenced polygons")
    provinces = provinces.to_crs("EPSG:4326").reset_index(drop=True)
    provinces["_pid"] = np.arange(len(provinces))
    dash = gpd.read_file(dash_path).to_crs("EPSG:4326")
    return provinces, dash


def load_units(path):
    if not path.exists():
        raise FileNotFoundError(f"Pollutant unit workbook not found: {path}")
    table = pd.read_excel(path)
    required = {"pollutant abbr.", "unit"}
    if not required.issubset(table.columns):
        raise ValueError(f"Unit workbook is missing columns: {sorted(required - set(table.columns))}")
    table = table.loc[table["pollutant abbr."].notna() & table["unit"].notna()].copy()
    table["pollutant abbr."] = table["pollutant abbr."].astype(str).str.strip()
    table["unit"] = table["unit"].astype(str).str.strip()
    if table["pollutant abbr."].duplicated().any():
        raise ValueError("Pollutant unit workbook contains duplicate abbreviations")
    return dict(zip(table["pollutant abbr."], table["unit"]))


def grid_edges(values):
    mid = (values[:-1] + values[1:]) / 2
    return np.r_[values[0] - (mid[0] - values[0]), mid, values[-1] + (values[-1] - mid[-1])]


def province_ids(lats, lons, provinces):
    xx, yy = np.meshgrid(lons, lats)
    points = gpd.GeoDataFrame(
        {"cell": np.arange(xx.size)},
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, provinces[["_pid", "geometry"]], how="left", predicate="within")
    joined = joined.sort_values("_pid", na_position="last").drop_duplicates("cell")
    ids = np.full(xx.size, -1, dtype=np.int32)
    matched = joined["_pid"].notna()
    ids[joined.loc[matched, "cell"].to_numpy(int)] = joined.loc[matched, "_pid"].to_numpy(np.int32)
    return ids


def province_sum(grid, ids, n_provinces):
    flat = grid.ravel()
    valid = (ids >= 0) & np.isfinite(flat)
    out = np.full(n_provinces, np.nan, dtype=float)
    if valid.any():
        sums = np.bincount(ids[valid], weights=flat[valid], minlength=n_provinces)
        counts = np.bincount(ids[valid], minlength=n_provinces)
        out[counts > 0] = sums[counts > 0]
    return out


def positive_limits(values):
    positive = np.concatenate([np.asarray(v).ravel() for v in values])
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if not positive.size:
        return 1e-3, 1.0
    low, high = np.quantile(positive, [0.005, 0.995])
    low = max(float(low), float(positive.min()))
    high = max(float(high), 10 * low)
    # Preserve the pooled scale, but label it using complete decades so the
    # first and last ticks are the actual colorbar endpoints.
    return 10.0 ** np.floor(np.log10(low)), 10.0 ** np.ceil(np.log10(high))


def display_values(values, low, high, sigma=0.0):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    shown = np.full(values.shape, np.nan)
    if not valid.any():
        return shown
    shown[valid] = np.clip(values[valid], low, high)
    if sigma > 0 and values.ndim == 2:
        log_values = np.zeros(values.shape)
        log_values[valid] = np.log10(shown[valid])
        weight = gaussian_filter(valid.astype(float), sigma, mode="nearest")
        numerator = gaussian_filter(log_values, sigma, mode="nearest")
        smoothed = np.divide(numerator, weight, out=np.zeros_like(numerator), where=weight > 1e-8)
        shown[valid] = 10 ** smoothed[valid]
    return shown


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    required = ["scenario", "time", "lat", "lon"]
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing = set(required) - set(header)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    lookup = {}
    for col in header:
        if "_" not in col:
            continue
        pollutant, sector = col.rsplit("_", 1)
        if sector in SECTORS:
            lookup.setdefault(pollutant, {})[sector] = col
    if not lookup:
        raise ValueError("No pollutant_sector columns were found.")
    columns = [col for sectors in lookup.values() for col in sectors.values()]
    raw = pd.read_csv(
        path, usecols=required + columns,
        dtype={col: "float32" for col in columns}, low_memory=False,
    )
    incomplete = raw[required].isna().any(axis=1)
    dropped = int(incomplete.sum())
    if dropped:
        warnings.warn(f"Skipping {dropped} row(s) without scenario, time, latitude or longitude.")
    # The source CSV repeats every scenario-year-cell three times: one row has
    # all emission predictions and two rows have no emission values at all.
    # Discard only those completely empty prediction rows; keep true zeroes.
    empty_predictions = ~raw[columns].notna().any(axis=1)
    empty_dropped = int((~incomplete & empty_predictions).sum())
    df = raw.loc[~incomplete & ~empty_predictions].copy()
    for key in ("time", "lat", "lon"):
        df[key] = pd.to_numeric(df[key], errors="raise")
    if not df["time"].mod(1).eq(0).all():
        raise ValueError("Year values must be integers.")
    df["time"] = df["time"].astype(int)
    df["lat"] = df["lat"].round(6)
    df["lon"] = df["lon"].round(6)
    df["scenario"] = df["scenario"].astype(str).str.strip()
    if df.duplicated(required).any():
        raise ValueError("Duplicate scenario-year-lat-lon rows were found.")

    return df, lookup, dropped, empty_dropped, len(raw)


def make_cube(frame, sector_columns, scenarios, sectors, lats, lons, lat_index, lon_index):
    cube = np.full((len(scenarios), len(sectors), len(lats), len(lons)), np.nan, dtype=np.float32)
    for si, scenario in enumerate(scenarios):
        rows = frame.loc[frame["scenario"] == scenario]
        yi = rows["lat"].map(lat_index).to_numpy(dtype=int)
        xi = rows["lon"].map(lon_index).to_numpy(dtype=int)
        for sec_i, sector in enumerate(sectors):
            col = sector_columns.get(sector)
            if col is not None:
                cube[si, sec_i, yi, xi] = rows[col].to_numpy(dtype=np.float32)
    return cube


def save_netcdf(path, pollutant, year, cube, scenarios, sectors, lats, lons, columns, unit):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w", format="NETCDF4") as ds:
        for name, length in (("time", 1), ("scenario", len(scenarios)), ("sector", len(sectors)), ("lat", len(lats)), ("lon", len(lons))):
            ds.createDimension(name, length)
        ds.title = f"Predicted China emissions of {pollutant} by scenario and sector, {year}"
        ds.pollutant = pollutant
        ds.year = year
        ds.crs = "EPSG:4326"
        ds.source_columns = json.dumps(columns, ensure_ascii=False)
        ds.unit_source = str(UNIT_FILE)
        t = ds.createVariable("time", "i4", ("time",))
        t.units = "days since 1970-01-01"
        t.calendar = "standard"
        t.standard_name = "time"
        t[:] = [date2num(datetime(year, 1, 1), t.units, t.calendar)]
        for name, values in (("scenario", scenarios), ("sector", sectors)):
            v = ds.createVariable(name, str, (name,))
            v[:] = np.asarray(values, dtype=object)
        for name, values, standard, units in (("lat", lats, "latitude", "degrees_north"), ("lon", lons, "longitude", "degrees_east")):
            v = ds.createVariable(name, "f4", (name,))
            v.standard_name = standard
            v.units = units
            v[:] = values.astype(np.float32)
        e = ds.createVariable("emission", "f4", ("time", "scenario", "sector", "lat", "lon"), zlib=True, complevel=4, shuffle=True, fill_value=np.float32(-9999))
        e.long_name = f"Annual predicted {pollutant} emissions by sector"
        e.units = f"{unit} yr-1"
        e.comment = "Unavailable combinations and grid cells are missing, not zero."
        e[0, :, :, :, :] = cube


class MapRenderer:
    """Reuse one Cartopy canvas for every scenario and year of a pollutant."""

    def __init__(self, map_type, lats, lons, provinces, dash, low, high, dpi, unit,
                 cell_provinces):
        self.map_type = map_type
        self.low, self.high, self.dpi = low, high, dpi
        self.unit = unit
        self.land_mask = cell_provinces.reshape(len(lats), len(lons)) >= 0
        self.norm = LogNorm(low, high, clip=True)
        self.fig = plt.figure(figsize=(9.0, 6.0), facecolor="white")
        self.ax = self.fig.add_axes([0.075, 0.07, 0.89, 0.82], projection=PLATE)
        self.ax.set_extent([72.5, 136, 17, 55], crs=PLATE)
        self.ax.set_facecolor("white")
        self.ax.set_xticks([75, 90, 105, 120, 135], crs=PLATE)
        self.ax.set_yticks([20, 30, 40, 50], crs=PLATE)
        self.ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol="°"))
        self.ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol="°"))
        self.ax.tick_params(axis="both", labelsize=9, length=4, width=0.8, pad=5)
        for spine in self.ax.spines.values():
            spine.set_linewidth(1.05)
            spine.set_color("#222222")
        if map_type == "province_sum":
            self.ax.add_geometries(list(provinces.geometry), crs=PLATE,
                                   facecolor="none", edgecolor="#8A8A8A", linewidth=0.32, zorder=4)
            vertices, province_indices = [], []
            for i, geometry in enumerate(provinces.geometry):
                parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
                for part in parts:
                    vertices.append(np.asarray(part.exterior.coords, dtype=np.float32))
                    province_indices.append(i)
            self.province_indices = np.asarray(province_indices, dtype=int)
            self.artist = PolyCollection(
                vertices, facecolors="none", edgecolors="#737373", linewidths=0.35,
                closed=True, rasterized=True, transform=PLATE, zorder=2,
            )
            self.ax.add_collection(self.artist)
            self.province_vertices = vertices
        else:
            outline = provinces.geometry.union_all().boundary
            self.ax.add_geometries([outline], crs=PLATE, facecolor="none",
                                   edgecolor="#A8AEB2", linewidth=0.9, zorder=4)
            blank = np.full((len(lats), len(lons)), np.nan)
            self.artist = self.ax.pcolormesh(
                grid_edges(lons), grid_edges(lats), blank, shading="flat",
                transform=PLATE, cmap=CMAP, norm=self.norm, rasterized=True, zorder=2,
            )

        inset = self.ax.inset_axes([0.785, 0.045, 0.175, 0.255], projection=PLATE)
        inset.set_extent([105, 125, 2, 25], crs=PLATE)
        inset.set_zorder(10)
        inset.patch.set_facecolor("white")
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color("#222222")
            spine.set_linewidth(0.8)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.add_geometries(list(dash.geometry), crs=PLATE,
                             facecolor="none", edgecolor="#555555", linewidth=0.65, zorder=5)
        if map_type == "province_sum":
            inset.add_geometries(list(provinces.geometry), crs=PLATE,
                                 facecolor="none", edgecolor="#777777", linewidth=0.32, zorder=4)
        if map_type == "province_sum":
            self.inset_artist = PolyCollection(
                self.province_vertices, facecolors="none", edgecolors="none",
                closed=True, rasterized=True, transform=PLATE, zorder=2,
            )
            inset.add_collection(self.inset_artist)
        else:
            self.inset_artist = inset.pcolormesh(
                grid_edges(lons), grid_edges(lats), np.full((len(lats), len(lons)), np.nan),
                shading="flat", transform=PLATE, cmap=CMAP, norm=self.norm,
                rasterized=True, zorder=2,
            )

    def save(self, path, title, grid, province_values=None, sigma=0.9):
        grid_view = display_values(grid, self.low, self.high, sigma)
        if self.map_type == "grid_cell":
            grid_view = np.where(self.land_mask, grid_view, np.nan)
            self.artist.set_array(np.ma.masked_invalid(grid_view).ravel())
            self.inset_artist.set_array(np.ma.masked_invalid(grid_view).ravel())
        else:
            values = display_values(province_values, self.low, self.high)
            colors = CMAP(self.norm(np.where(np.isfinite(values), values, self.low)[self.province_indices]))
            colors[~np.isfinite(values[self.province_indices])] = [0, 0, 0, 0]
            self.artist.set_facecolors(colors)
            self.inset_artist.set_facecolors(colors)
        title_size = 14 if len(title) <= 60 else 11
        self.ax.set_title(title, loc="left", fontsize=title_size, fontweight="bold", pad=12)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.stem + ".rendering.png")
        last_error = None
        for attempt in range(8):
            try:
                self.fig.savefig(temporary, dpi=self.dpi, facecolor="white")
                temporary.replace(path)
                last_error = None
                break
            except (OSError, PermissionError) as exc:
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        if last_error is not None:
            raise last_error

    def save_colorbar(self, path):
        """Save the map's pooled, rounded log scale as its own image."""
        fig = plt.figure(figsize=(9.0, 1.2), facecolor="white")
        try:
            cax = fig.add_axes([0.07, 0.29, 0.86, 0.24])
            cb = fig.colorbar(ScalarMappable(norm=self.norm, cmap=CMAP), cax=cax,
                              orientation="horizontal")
            first = int(round(np.log10(self.low)))
            last = int(round(np.log10(self.high)))
            powers = range(first, last + 1)
            cb.set_ticks([10.0 ** power for power in powers])
            cb.set_ticklabels([rf"$10^{{{power}}}$" for power in powers])
            cb.ax.xaxis.set_ticks_position("top")
            cb.ax.tick_params(axis="x", labelsize=10, length=4, width=0.8, pad=3)
            cb.ax.minorticks_off()
            cb.ax.set_xlim(self.low, self.high)
            labels = cb.ax.get_xticklabels()
            labels[0].set_ha("left")
            labels[-1].set_ha("right")
            cb.outline.set_linewidth(0.8)
            unit_label = "province sum" if self.map_type == "province_sum" else "grid cell"
            fig.text(0.5, 0.12,
                     f"Annual emission ({self.unit} yr⁻¹; {unit_label}; logarithmic scale)",
                     ha="center", va="center", fontsize=9)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.stem + ".rendering.png")
            fig.savefig(temporary, dpi=self.dpi, facecolor="white")
            temporary.replace(path)
        finally:
            plt.close(fig)

    def close(self):
        plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--smooth-sigma", type=float, default=0.90)
    parser.add_argument("--map-types", nargs="+", choices=["grid_cell", "province_sum"],
                        default=["grid_cell", "province_sum"])
    parser.add_argument("--pollutants", nargs="+", help="Optional subset for a test run")
    parser.add_argument("--years", nargs="+", type=int, help="Optional subset for a test run")
    parser.add_argument("--reuse-netcdf", action="store_true",
                        help="Reuse previously exported NetCDF files after checking their dimensions")
    parser.add_argument("--overwrite-maps", action="store_true",
                        help="Re-render maps and separate colorbars that already exist")
    args = parser.parse_args()
    if args.smooth_sigma < 0 or args.dpi <= 0:
        raise ValueError("DPI must be positive and smoothing sigma cannot be negative")
    start = time.perf_counter()
    df, lookup, dropped, empty_dropped, input_rows = load_data(args.input)
    units = load_units(UNIT_FILE)
    skipped_no_unit = sorted(set(lookup) - set(units))
    lookup = {pollutant: columns for pollutant, columns in lookup.items() if pollutant in units}
    if skipped_no_unit:
        print(f"Skipping pollutants absent from unit workbook: {skipped_no_unit}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    scenarios = sorted(df["scenario"].unique().tolist())
    years = sorted(df["time"].unique().tolist())
    if args.pollutants:
        unknown = set(args.pollutants) - set(lookup)
        if unknown:
            raise ValueError(f"Unknown pollutants: {sorted(unknown)}")
        lookup = {p: lookup[p] for p in args.pollutants}
    if args.years:
        unknown = set(args.years) - set(years)
        if unknown:
            raise ValueError(f"Unknown years: {sorted(unknown)}")
        years = sorted(set(args.years))
    lats = np.sort(df["lat"].unique())
    lons = np.sort(df["lon"].unique())
    lat_index = {float(v): i for i, v in enumerate(lats)}
    lon_index = {float(v): i for i, v in enumerate(lons)}
    sectors = [s for s in SECTORS if any(s in item for item in lookup.values())]
    provinces, dash = load_boundaries()
    cell_provinces = province_ids(lats, lons, provinces)
    coverage = {
        "source_file": str(args.input), "source_file_bytes": args.input.stat().st_size,
        "unit_file": str(UNIT_FILE),
        "pollutants_skipped_no_unit": skipped_no_unit,
        "input_rows": input_rows,
        "incomplete_key_rows_skipped": dropped,
        "empty_prediction_rows_skipped": empty_dropped,
        "usable_rows": len(df),
        "scenarios": scenarios, "years": years, "pollutants": len(lookup),
        "unique_grid_cells": int(df[["lat", "lon"]].drop_duplicates().shape[0]),
        "latitude_range": [float(lats.min()), float(lats.max())],
        "longitude_range": [float(lons.min()), float(lons.max())],
        "province_boundary_loaded": True,
        "nine_dash_line_loaded": True,
        "map_types": args.map_types,
        "map_target": "individual pollutant-sector emissions",
        "smoothing_sigma_grid_cells": args.smooth_sigma,
    }
    (args.out / "source_coverage.json").write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Loaded {len(df)} valid rows; {len(lookup)} pollutants; scenarios={scenarios}; years={years}", flush=True)
    if dropped or len(scenarios) < 3 or coverage["unique_grid_cells"] < 1000:
        warnings.warn("Source coverage is incomplete. The maps show only supplied cells; see source_coverage.json.")

    manifest, validation, scales = [], [], []
    total_pairs = len(lookup) * len(years)
    done = 0
    for pollutant, sector_columns in lookup.items():
        unit = units[pollutant]
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", pollutant).strip("_") or "pollutant"
        relevant = df[["scenario", "time", "lat", "lon"] + list(sector_columns.values())]
        records = []
        for year in years:
            frame = relevant.loc[relevant["time"] == year]
            if not frame[list(sector_columns.values())].notna().any().any():
                continue
            cube = make_cube(frame, sector_columns, scenarios, sectors, lats, lons, lat_index, lon_index)
            nc_path = args.out / "netcdf_by_pollutant_year" / safe / f"{safe}_{year}.nc"
            reuse = False
            if args.reuse_netcdf and nc_path.exists():
                with Dataset(nc_path) as existing:
                    reuse = existing.variables["emission"].shape == (
                        1, len(scenarios), len(sectors), len(lats), len(lons)
                    ) and existing.pollutant == pollutant and int(existing.year) == year \
                      and getattr(existing.variables["emission"], "units", None) == f"{unit} yr-1"
            if not reuse:
                save_netcdf(nc_path, pollutant, year, cube, scenarios, sectors, lats, lons, sector_columns, unit)
            for si, scenario in enumerate(scenarios):
                source = frame.loc[frame["scenario"] == scenario]
                if source.empty:
                    continue
                scenario_cube = cube[si]
                if not np.isfinite(scenario_cube).any():
                    continue
                records.append({"year": year, "scenario": scenario, "cube": scenario_cube,
                                "nc_path": nc_path})
                for sector, column in sector_columns.items():
                    grid = scenario_cube[sectors.index(sector)]
                    source_total = float(source[column].sum())
                    nc_total = float(np.nansum(grid, dtype=np.float64))
                    difference = abs(source_total - nc_total)
                    tolerance = max(1e-5, abs(source_total) * 1e-5)
                    validation.append({"pollutant": pollutant, "sector": sector,
                                       "year": year, "scenario": scenario,
                                       "source_total": source_total, "netcdf_total": nc_total,
                                       "absolute_difference": difference,
                                       "within_tolerance": difference <= tolerance,
                                       "grid_cells_with_data": int(np.isfinite(grid).sum())})
            done += 1
            if done % 20 == 0 or done == total_pairs:
                print(f"Exported {done}/{total_pairs} pollutant-year NetCDF files", flush=True)

        for sector in sector_columns:
            index = sectors.index(sector)
            for record in records:
                grid = record["cube"][index]
                record[(sector, "grid")] = grid
                record[(sector, "province")] = province_sum(grid, cell_provinces, len(provinces))
        for map_type in args.map_types:
            key = "grid" if map_type == "grid_cell" else "province"
            low, high = positive_limits([record[(sector, key)]
                                         for sector in sector_columns for record in records])
            scales.append({"pollutant": pollutant, "map_type": map_type,
                           "unit": unit,
                           "vmin": low, "vmax": high,
                           "basis": "pooled positive 0.5th and 99.5th percentiles across sectors, SSPs and years; rounded outward to complete decades"})
            renderer = MapRenderer(map_type, lats, lons, provinces, dash, low, high,
                                   args.dpi, unit, cell_provinces)
            try:
                for sector in sector_columns:
                    for record in records:
                        year, scenario = record["year"], record["scenario"]
                        suffix = "grid" if map_type == "grid_cell" else "province"
                        stem = f"{safe}_{sector}_{scenario}_{year}_{suffix}"
                        map_path = args.out / "selected_maps" / f"{stem}.png"
                        colorbar_path = args.out / "selected_maps" / f"{stem}_colorbar.png"
                        title = f"{pollutant} emissions, {sector}, {SCENARIO_LABELS.get(scenario, scenario)}, {year}"
                        if args.overwrite_maps or not (map_path.exists() and map_path.stat().st_size > 0):
                            renderer.save(map_path, title, record[(sector, "grid")],
                                          record[(sector, "province")], args.smooth_sigma)
                        if args.overwrite_maps or not (colorbar_path.exists() and colorbar_path.stat().st_size > 0):
                            renderer.save_colorbar(colorbar_path)
                        manifest.append({"pollutant": pollutant, "sector": sector,
                                         "unit": unit,
                                         "year": year, "scenario": scenario,
                                         "map_type": map_type,
                                         "netcdf": str(record["nc_path"].relative_to(args.out)),
                                         "map_png": str(map_path.relative_to(args.out)),
                                         "colorbar_png": str(colorbar_path.relative_to(args.out)),
                                         "available_sector_count": len(sector_columns),
                                         "sector_dimension_count": len(sectors)})
            finally:
                renderer.close()
        print(f"Finished maps for {pollutant}; {done}/{total_pairs} NetCDF files", flush=True)
        pd.DataFrame(manifest).to_csv(args.out / "pollutant_year_file_manifest.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(validation).to_csv(args.out / "export_validation_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(scales).to_csv(args.out / "colorbar_scales.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(manifest).to_csv(args.out / "pollutant_year_file_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scales).to_csv(args.out / "colorbar_scales.csv", index=False, encoding="utf-8-sig")
    checks = pd.DataFrame(validation)
    checks.to_csv(args.out / "export_validation_summary.csv", index=False, encoding="utf-8-sig")
    failures = checks.loc[~checks["within_tolerance"]]
    if not failures.empty:
        raise RuntimeError(f"Mass check failed for {len(failures)} map(s).")
    print(f"Completed: {done} NetCDF files, {len(manifest)} maps; elapsed {(time.perf_counter()-start)/60:.1f} min")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
