# -*- coding: utf-8 -*-
"""Plot a selected pollutant, sector, year and scenario from annual NetCDF.

Each map type is exported as a standalone spatial map and a matching standalone
colorbar. Historical files have no scenario dimension; future files are selected
by their actual ``scenario`` coordinate. This script does not create NetCDF.
"""
from __future__ import annotations
import argparse
import logging
import re
import warnings
from functools import lru_cache
from pathlib import Path
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.ndimage import gaussian_filter
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "results" / "code_5_all_emissions" / "grid_level_emission"
NC_ROOT = OUTPUT_DIR / "netcdf_by_pollutant_year"
FUTURE_NC_ROOT = (SCRIPT_DIR.parent / "Supplementary Code_4_Pred_Future_Emission"
                  / "code_5_plot" / "netcdf_by_pollutant_year")
MAP_ROOT = OUTPUT_DIR / "selected_maps"
LOGGER = logging.getLogger("selected_emission_map")
PLATE = ccrs.PlateCarree()
CMAP = LinearSegmentedColormap.from_list("reference_log", ["#5D55A5", "#4E91B5", "#76BFA5", "#DDF08A", "#FFF08A", "#F2A15A", "#A52B4F"], N=256)
CMAP.set_bad("white")

def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_.") or "unnamed"

def output_stem(name: str, sector: str, scenario: str, year: int,
                map_type: str) -> str:
    """Build the flat pollutant-sector-period-year-map filename."""
    suffix = {"grid_cell": "grid", "province_sum": "province"}[map_type]
    return f"{safe(name)}_{safe(sector)}_{safe(scenario)}_{year}_{suffix}"

@lru_cache(maxsize=1)
def get_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]:
    candidates = [
        SCRIPT_DIR / "data_raw" / "China_shp",
        SCRIPT_DIR.parent.parent / "Supplementary Code_3_Create_Emission_Index" / "create_index" / "data_raw",
        SCRIPT_DIR.parent.parent.parent / "Supplementary Code_3_Create_Emission_Index" / "create_index" / "data_raw",
    ]
    shapefiles = []
    for folder in candidates:
        if folder.exists():
            shapefiles.extend(folder.rglob("*.shp"))
    provinces = None
    for path in sorted(set(shapefiles), key=str):
        try:
            shape = gpd.read_file(path)
            if (len(shape) == 34 and shape.crs is not None and shape.geometry.notna().all()
                    and shape.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all()):
                provinces = shape.to_crs("EPSG:4326").reset_index(drop=True)
                provinces["_pid"] = np.arange(34)
                LOGGER.info("Using province polygons: %s", path)
                break
        except Exception:
            continue
    if provinces is None:
        raise FileNotFoundError("Could not locate a valid 34-province polygon shapefile.")
    dash = None
    for path in shapefiles:
        if path.stem.lower() == "china_nine_dotted_line":
            try:
                dash = gpd.read_file(path).to_crs("EPSG:4326")
                break
            except Exception:
                pass
    return provinces, dash


_PROVINCE_ID_CACHE: dict[tuple[bytes, bytes], np.ndarray] = {}


def province_ids(lat: np.ndarray, lon: np.ndarray, provinces: gpd.GeoDataFrame) -> np.ndarray:
    cache_key = (np.asarray(lat, dtype=np.float64).tobytes(), np.asarray(lon, dtype=np.float64).tobytes())
    cached = _PROVINCE_ID_CACHE.get(cache_key)
    if cached is not None:
        return cached
    xx, yy = np.meshgrid(lon, lat)
    points = gpd.GeoDataFrame(
        {"cell": np.arange(xx.size)}, geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()), crs="EPSG:4326"
    )
    joined = gpd.sjoin(points, provinces[["_pid", "geometry"]], how="left", predicate="within")
    joined = joined.sort_values("_pid", na_position="last").drop_duplicates("cell")
    ids = np.full(xx.size, -1, np.int32)
    valid = joined["_pid"].notna()
    ids[joined.loc[valid, "cell"].to_numpy(int)] = joined.loc[valid, "_pid"].to_numpy(np.int32)
    _PROVINCE_ID_CACHE[cache_key] = ids
    return ids


def aggregate_provinces(arrays: list[np.ndarray], ids: np.ndarray, n_provinces: int) -> list[np.ndarray]:
    output = []
    for array in arrays:
        result = np.full((array.shape[0], n_provinces), np.nan, dtype=np.float64)
        for sector in range(array.shape[0]):
            values = array[sector].ravel()
            valid = (ids >= 0) & np.isfinite(values)
            sums = np.bincount(ids[valid], weights=values[valid], minlength=n_provinces)[:n_provinces]
            counts = np.bincount(ids[valid], minlength=n_provinces)[:n_provinces]
            result[sector, counts > 0] = sums[counts > 0]
        output.append(result)
    return output


def edges(values: np.ndarray) -> np.ndarray:
    mid = (values[:-1] + values[1:]) / 2
    return np.r_[values[0] - (mid[0] - values[0]), mid, values[-1] + (values[-1] - mid[-1])]


def get_bounds(arrays: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([array.ravel() for array in arrays])
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return 1e-3, 1.0
    lo, hi = np.quantile(values, [0.005, 0.995])
    lo = max(float(lo), float(values.min()))
    hi = max(float(hi), 10 * lo)
    # Use enclosing decades for both the map normalization and its separate
    # colorbar: e.g. 10^2.04..10^5.69 becomes 10^2..10^6.
    return 10.0 ** np.floor(np.log10(lo)), 10.0 ** np.ceil(np.log10(hi))


def smooth_display(values: np.ndarray, lo: float, hi: float, sigma: float) -> np.ndarray:
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    result = np.full(values.shape, np.nan)
    if not valid.any():
        return result
    clipped = np.clip(values[valid], lo, hi)
    if sigma > 0 and values.ndim == 2:
        log_values = np.zeros(values.shape)
        log_values[valid] = np.log10(clipped)
        weight = gaussian_filter(valid.astype(float), sigma, mode="nearest")
        numerator = gaussian_filter(log_values, sigma, mode="nearest")
        smooth = np.divide(numerator, weight, out=np.zeros_like(numerator), where=weight > 1e-8)
        result[valid] = 10 ** smooth[valid]
    else:
        result[valid] = clipped
    return result


def make_map_axis(fig, extent, provinces, draw_province_lines: bool):
    ax = fig.add_axes([0.075, 0.07, 0.89, 0.82], projection=PLATE)
    ax.set_extent(extent, crs=PLATE)
    ax.set_facecolor("white")
    if draw_province_lines:
        ax.add_geometries(list(provinces.geometry), crs=PLATE, facecolor="none", edgecolor="#8A8A8A", linewidth=0.32, zorder=4)
    else:
        outline = provinces.geometry.union_all().boundary
        ax.add_geometries([outline], crs=PLATE, facecolor="none", edgecolor="#A8AEB2", linewidth=0.9, zorder=4)
    return ax


def make_map_figure(map_type, lat, lon, provinces, dash, polygon_vertices, polygon_ids, lo, hi, units):
    fig = plt.figure(figsize=(9.0, 6.0), facecolor="white")
    ax = make_map_axis(fig, [72.5, 136, 17, 55], provinces, map_type == "province_sum")
    ax.set_xticks([75, 90, 105, 120, 135], crs=PLATE)
    ax.set_yticks([20, 30, 40, 50], crs=PLATE)
    ax.xaxis.set_major_formatter(LongitudeFormatter(degree_symbol="°"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(degree_symbol="°"))
    ax.tick_params(axis="both", labelsize=9, length=4, width=0.8, pad=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color("#222222")
    norm = LogNorm(lo, hi, clip=True)
    if map_type == "grid_cell":
        artist = ax.pcolormesh(edges(lon), edges(lat), np.full((len(lat), len(lon)), np.nan), shading="flat", transform=PLATE, cmap=CMAP, norm=norm, rasterized=True, zorder=2)
    else:
        artist = PolyCollection(polygon_vertices, facecolors=np.tile([[1, 1, 1, 0]], (len(polygon_vertices), 1)), edgecolors="#737373", linewidths=0.35, closed=True, rasterized=True, transform=PLATE, zorder=2)
        ax.add_collection(artist)
    inset = ax.inset_axes([0.785, 0.045, 0.175, 0.255], projection=PLATE)
    inset.set_extent([105, 125, 2, 25], crs=PLATE)
    inset.set_zorder(10)
    inset.patch.set_facecolor("white")
    inset.patch.set_alpha(1.0)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(0.8)
    inset.set_xticks([])
    inset.set_yticks([])
    if map_type == "grid_cell":
        inset_artist = inset.pcolormesh(
            edges(lon), edges(lat), np.full((len(lat), len(lon)), np.nan),
            shading="flat", transform=PLATE, cmap=CMAP, norm=norm,
            rasterized=True, zorder=2)
    else:
        # A province-total map must use those same province totals in the inset.
        # Plotting raw grid cells here would give the inset a different meaning.
        inset_artist = PolyCollection(
            polygon_vertices,
            facecolors=np.tile([[1, 1, 1, 0]], (len(polygon_vertices), 1)),
            edgecolors="none", closed=True, rasterized=True,
            transform=PLATE, zorder=2)
        inset.add_collection(inset_artist)
    inset.add_geometries(list(provinces.geometry), crs=PLATE,
                         facecolor="none", edgecolor="#929A9D",
                         linewidth=0.35, zorder=4)
    if dash is not None:
        inset.add_geometries(list(dash.geometry), crs=PLATE, facecolor="none",
                             edgecolor="#52616A", linewidth=0.75, zorder=5)
    return fig, ax, artist, inset_artist


def make_colorbar_figure(lo: float, hi: float, units: str):
    """Draw the map color scale with integer-decade labels at both ends."""
    norm = LogNorm(lo, hi, clip=True)
    fig = plt.figure(figsize=(9.0, 1.2), facecolor="white")
    cax = fig.add_axes([0.07, 0.29, 0.86, 0.24])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                      orientation="horizontal")
    decade_min = int(round(np.log10(lo)))
    decade_max = int(round(np.log10(hi)))
    powers = range(decade_min, decade_max + 1)
    ticks = [10.0 ** power for power in powers]
    cb.set_ticks(ticks)
    cb.set_ticklabels([rf"$10^{{{power}}}$" for power in powers])
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.tick_params(axis="x", labelsize=10, length=4, width=0.8, pad=3)
    cb.ax.minorticks_off()
    cb.ax.set_xlim(lo, hi)
    labels = cb.ax.get_xticklabels()
    labels[0].set_ha("left")
    labels[-1].set_ha("right")
    cb.outline.set_linewidth(0.8)
    fig.text(0.5, 0.12, f"Annual emission ({units}; logarithmic scale)",
             ha="center", va="center", fontsize=9)
    return fig


def choose_pollutant(requested: str, nc_root: Path) -> tuple[str, Path]:
    """Find the annual NetCDF by pollutant name, ignoring letter case."""
    if not nc_root.is_dir():
        raise FileNotFoundError(
            f"NetCDF directory does not exist: {nc_root}."
        )
    matches = [folder for folder in nc_root.iterdir()
               if folder.is_dir() and folder.name.casefold() == safe(requested).casefold()]
    if len(matches) != 1:
        available = sorted(folder.name for folder in nc_root.iterdir() if folder.is_dir())
        raise ValueError(f"Pollutant {requested!r} not found. Available pollutants: {available}")
    return matches[0].name, matches[0]


def select_scenario(ds: xr.Dataset, scenario: str, path: Path) -> xr.Dataset:
    """Select an SSP from future NetCDF, or validate a historical NetCDF."""
    if "scenario" in ds.dims:
        choices = [str(value) for value in ds["scenario"].values]
        found = [value for value in choices if value.casefold() == scenario.casefold()]
        if len(found) != 1:
            raise ValueError(f"Scenario {scenario!r} unavailable in {path}; available: {choices}")
        return ds.sel(scenario=found[0])
    if scenario != "historical":
        raise ValueError(f"{path} has no scenario dimension; cannot label it {scenario!r}")
    return ds


def load_selected(pollutant: str, year: int, sector: str,
                  scenario: str, nc_root: Path):
    name, folder = choose_pollutant(pollutant, nc_root)
    path = folder / f"{name}_{year}.nc"
    if not path.is_file():
        years = sorted(int(match.group(1)) for item in folder.glob("*.nc")
                       if (match := re.search(r"_(\d{4})\.nc$", item.name)))
        raise FileNotFoundError(f"No NetCDF for {name} in {year}. Available years: {years}")
    with xr.open_dataset(path, engine="netcdf4") as ds:
        if "emission" not in ds or "sector" not in ds.coords:
            raise ValueError(f"Invalid emission NetCDF: {path}")
        selected = select_scenario(ds, scenario, path)
        sectors = [str(item) for item in ds["sector"].values]
        found = [item for item in sectors if item.casefold() == sector.strip().casefold()]
        if len(found) != 1:
            raise ValueError(f"Sector {sector!r} not found for {name}. Available sectors: {sectors}")
        actual_sector = found[0]
        values = np.asarray(selected["emission"].sel(sector=actual_sector).isel(time=0).values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        units = str(ds["emission"].attrs.get("units", "source CSV units"))
    if values.shape != (len(lat), len(lon)):
        raise ValueError(f"Unexpected grid shape in {path}: {values.shape}")
    if lat[0] > lat[-1]:
        lat, values = lat[::-1], values[::-1, :]
    if lon[0] > lon[-1]:
        lon, values = lon[::-1], values[:, ::-1]
    if not np.isfinite(values).any():
        raise ValueError(f"No emission data for {name}, {actual_sector}, {year}")
    return name, actual_sector, path, lat, lon, values, units


def plot_selected(pollutant: str, year: int, sector: str, map_types: list[str],
                  scenario: str, nc_root: Path, output_dir: Path, sigma: float,
                  dpi: int, export_data: bool) -> list[Path]:
    name, sector, nc_file, lat, lon, grid, units = load_selected(
        pollutant, year, sector, scenario, nc_root)
    provinces, dash = get_boundaries()
    output_files: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    if export_data:
        xx, yy = np.meshgrid(lon, lat)
        data_file = output_dir / f"{safe(name)}_{safe(sector)}_{safe(scenario)}_{year}_data.csv"
        pd.DataFrame({"time": year, "lat": yy.ravel(), "lon": xx.ravel(),
                      "sector": sector, "scenario": scenario,
                      "emission": grid.ravel(), "unit": units}).to_csv(
            data_file, index=False, encoding="utf-8-sig")
        output_files.append(data_file)

    ids = province_ids(lat, lon, provinces)
    province_values = None
    if "province_sum" in map_types:
        province_values = aggregate_provinces([grid[None, ...]], ids, len(provinces))[0][0]

    polygon_vertices, polygon_ids = [], []
    if "province_sum" in map_types:
        for pid, geometry in enumerate(provinces.geometry):
            parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
            for part in parts:
                polygon_vertices.append(np.asarray(part.exterior.coords, np.float32))
                polygon_ids.append(pid)
    polygon_ids = np.asarray(polygon_ids, dtype=int)

    for map_type in map_types:
        stem = output_stem(name, sector, scenario, year, map_type)
        raw = grid if map_type == "grid_cell" else province_values
        lo, hi = get_bounds([raw])
        fig, ax, artist, inset_artist = make_map_figure(
            map_type, lat, lon, provinces, dash, polygon_vertices, polygon_ids, lo, hi, units)
        try:
            display = smooth_display(raw, lo, hi, sigma if map_type == "grid_cell" else 0)
            if map_type == "grid_cell":
                display = np.where(ids.reshape(grid.shape) >= 0, display, np.nan)
                artist.set_array(np.ma.masked_invalid(display).ravel())
            else:
                colors = CMAP(LogNorm(lo, hi, clip=True)(
                    np.where(np.isfinite(display), display, lo)[polygon_ids]))
                colors[~np.isfinite(display[polygon_ids])] = [0, 0, 0, 0]
                artist.set_facecolors(colors)
                inset_artist.set_facecolors(colors)
            if map_type == "grid_cell":
                # Grid inset shares the land mask and display scale with the grid map.
                inset_grid = smooth_display(grid, lo, hi, sigma)
                inset_grid = np.where(ids.reshape(grid.shape) >= 0, inset_grid, np.nan)
                inset_artist.set_array(np.ma.masked_invalid(inset_grid).ravel())
            ax.set_title(f"{name} emissions, {sector}, {year}, {scenario}", loc="left",
                         fontsize=14, fontweight="bold", pad=12)
            image_file = output_dir / f"{stem}.png"
            # Fixed canvas avoids pathological dimensions from tight bounding boxes.
            fig.savefig(image_file, dpi=dpi, facecolor="white")
            output_files.append(image_file)
        finally:
            plt.close(fig)
        colorbar_file = output_dir / f"{stem}_colorbar.png"
        cb_fig = make_colorbar_figure(lo, hi, units)
        try:
            cb_fig.savefig(colorbar_file, dpi=dpi, facecolor="white")
            output_files.append(colorbar_file)
        finally:
            plt.close(cb_fig)
    LOGGER.info("Source NetCDF: %s; selected sector: %s; year: %s; scenario: %s",
                nc_file, sector, year, scenario)
    return output_files


def available_targets(pollutant: str | None, year: int | None,
                      sector: str | None, scenario: str,
                      nc_root: Path) -> list[tuple[str, int, str]]:
    """List only pollutant/year/sector combinations with actual grid values."""
    if not nc_root.is_dir():
        raise FileNotFoundError(f"NetCDF directory does not exist: {nc_root}")
    folders = ([choose_pollutant(pollutant, nc_root)[1]] if pollutant
               else sorted((p for p in nc_root.iterdir() if p.is_dir()), key=lambda p: p.name.casefold()))
    targets = []
    for folder in folders:
        for path in sorted(folder.glob("*.nc")):
            match = re.search(r"_(\d{4})\.nc$", path.name)
            if not match:
                continue
            file_year = int(match.group(1))
            if year is not None and file_year != year:
                continue
            with xr.open_dataset(path, engine="netcdf4") as ds:
                if "emission" not in ds or "sector" not in ds.coords:
                    raise ValueError(f"Invalid emission NetCDF: {path}")
                if "scenario" in ds.dims and scenario not in [str(v) for v in ds["scenario"].values]:
                    continue
                if "scenario" not in ds.dims and scenario != "historical":
                    continue
                selected = select_scenario(ds, scenario, path)
                sectors = [str(value) for value in ds["sector"].values]
                if sector:
                    indices = [i for i, value in enumerate(sectors)
                               if value.casefold() == sector.strip().casefold()]
                else:
                    indices = list(range(len(sectors)))
                for i in indices:
                    values = np.asarray(selected["emission"].isel(time=0, sector=i).values)
                    if np.isfinite(values).any():
                        targets.append((folder.name, file_year, sectors[i]))
    if not targets:
        raise ValueError("No NetCDF data matched the requested pollutant, year and sector filters")
    return targets


def output_paths(output_dir: Path, name: str, year: int, sector: str,
                 scenario: str, map_types: list[str], export_data: bool) -> list[Path]:
    files = [output_dir / f"{output_stem(name, sector, scenario, year, map_type)}{suffix}.png"
             for map_type in map_types for suffix in ("", "_colorbar")]
    if export_data:
        files.append(output_dir / f"{safe(name)}_{safe(sector)}_{safe(scenario)}_{year}_data.csv")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pollutant", help="Pollutant abbreviation, for example BaP")
    parser.add_argument("--year", "--time", dest="year", type=int, help="Year, for example 2023")
    parser.add_argument("--sector", "--department", dest="sector", help="Sector, for example ico or CFPP")
    parser.add_argument("--scenario", choices=["historical", "ssp126", "ssp245", "ssp370"],
                        default="historical", help="Historical (default) or actual future SSP coordinate")
    parser.add_argument("--nc-root", type=Path,
                        help="Override the NetCDF input root; defaults to historical/future root by scenario")
    parser.add_argument("--map-type", choices=["grid", "province", "both"],
                        help="Select grid, province totals, or both (default: both)")
    parser.add_argument("--map-types", nargs="+", choices=["grid_cell", "province_sum"],
                        help="Previous syntax; use --map-type for new commands")
    parser.add_argument("--all", action="store_true",
                        help="Draw every available pollutant/year/sector combination; optional filters apply")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --all, list the number of matching maps without drawing them")
    parser.add_argument("--overwrite", action="store_true",
                        help="With --all, redraw maps even when outputs already exist")
    parser.add_argument("--output-dir", type=Path, default=MAP_ROOT)
    parser.add_argument("--smooth-sigma", type=float, default=0.90)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--export-data", action="store_true", help="Also export the selected grid to CSV")
    args = parser.parse_args()
    if args.map_type and args.map_types:
        parser.error("Use either --map-type or --map-types, not both")
    if args.dry_run and not args.all:
        parser.error("--dry-run requires --all")
    map_types = (args.map_types if args.map_types else
                 {"grid": ["grid_cell"], "province": ["province_sum"],
                  "both": ["grid_cell", "province_sum"]}[args.map_type or "both"])
    if not args.all and (args.pollutant is None or args.year is None or args.sector is None):
        import sys
        if not sys.stdin.isatty():
            parser.error("Specify --pollutant, --year and --sector")
        args.pollutant = args.pollutant or input("Pollutant: ").strip()
        args.year = args.year if args.year is not None else int(input("Year: ").strip())
        args.sector = args.sector or input("Sector: ").strip()
    if args.smooth_sigma < 0 or args.dpi <= 0:
        parser.error("--smooth-sigma must be non-negative and --dpi must be positive")
    nc_root = args.nc_root or (NC_ROOT if args.scenario == "historical" else FUTURE_NC_ROOT)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")
    if not args.all:
        files = plot_selected(args.pollutant, args.year, args.sector, map_types,
                              args.scenario, nc_root, args.output_dir, args.smooth_sigma,
                              args.dpi, args.export_data)
        for file in files:
            print(file)
        return

    targets = available_targets(args.pollutant, args.year, args.sector,
                                args.scenario, nc_root)
    print(f"Found {len(targets)} pollutant/year/sector combinations; "
          f"{len(targets) * len(map_types)} maps and matching colorbars requested", flush=True)
    if args.dry_run:
        for name, file_year, code in targets[:20]:
            print(f"{name}, {file_year}, {code}")
        if len(targets) > 20:
            print(f"... and {len(targets) - 20} more combinations")
        return

    records = []
    manifest_path = args.output_dir / "batch_map_manifest.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, file_year, code) in enumerate(targets, 1):
        expected = output_paths(args.output_dir, name, file_year, code,
                                args.scenario, map_types, args.export_data)
        row = {"pollutant": name, "year": file_year, "sector": code,
               "scenario": args.scenario,
               "map_types": ";".join(map_types), "files": ";".join(str(p) for p in expected)}
        if not args.overwrite and all(p.is_file() and p.stat().st_size > 0 for p in expected):
            row["status"] = "skipped_existing"
        else:
            try:
                plot_selected(name, file_year, code, map_types, args.scenario,
                              nc_root, args.output_dir, args.smooth_sigma,
                              args.dpi, args.export_data)
                row["status"] = "created"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                LOGGER.exception("Failed to draw %s, %s, %s", name, file_year, code)
        records.append(row)
        if index % 20 == 0 or index == len(targets):
            pd.DataFrame(records).to_csv(manifest_path, index=False, encoding="utf-8-sig")
            print(f"Processed {index}/{len(targets)} combinations", flush=True)
    failed = sum(row["status"] == "failed" for row in records)
    print(f"Batch manifest: {manifest_path}; failures: {failed}")
    if failed:
        raise RuntimeError(f"{failed} combination(s) failed; see {manifest_path}")


if __name__ == "__main__":
    main()


