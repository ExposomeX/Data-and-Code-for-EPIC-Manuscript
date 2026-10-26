# -*- coding: utf-8 -*-
from __future__ import annotations

# =============================================================================
# PART 1: CEI CONSTRUCTION AND COUNTY MERGING
# =============================================================================

#%%
import warnings
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
import matplotlib.colors as mcolors
import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =========================================================
# Implementation note.
# =========================================================
base_dir = os.path.dirname(os.path.abspath(__file__))

FUT_DATA_PATH = os.path.join(base_dir,'code_3_results')
PRED_PATH = os.path.join(base_dir,'data_raw')
OUTPUT = os.path.join(base_dir,'code_4_results')
os.makedirs(OUTPUT, exist_ok=True)

HIST_POL_PATH = os.path.join(PRED_PATH,'grid_level_pollutants.csv')
PAIR_PATH = os.path.join(base_dir,'code_3_results/3_Final_Top50pct_Categorized_Targets.csv')
STATISIC_PATH = os.path.abspath(os.path.join(base_dir,'../../1_data_process/code_5_results/df_hospitalization_for_association.csv'))
SHP_PATH = os.path.join(PRED_PATH,'shp版/中国_县_面.shp')

TARGETS = pd.read_csv(os.path.join(base_dir,'code_3_results/4_selected_pols_list.csv')).iloc[:,-1].tolist()


def set_nature_style():
    'Generate the corresponding publication-quality visualization.'
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "sans-serif"],
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.2,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.fontsize": 11
    })

set_nature_style()

# =========================================================
# Implementation note.
# =========================================================
print(">>> [1/6] Loading and parsing the target-weight dictionary for wCEI (composite emission index)...")
df_priority = pd.read_csv(PAIR_PATH)
# Implementation note.
# Implementation note.
if 'Label' in df_priority.columns:
    disease_col = 'Label'
elif 'Disease_Name' in df_priority.columns:
    disease_col = 'Disease_Name'
else:
    raise KeyError("No disease column was found in PAIR_PATH.")

cei_weights = {}
all_targets_to_extract = set()

for disease in df_priority[disease_col].unique():
    disease_df = df_priority[df_priority[disease_col] == disease]
    temp_dict = {}
    
    # Implementation note.
    for idx, row in disease_df.iterrows():
        weight = row.get('Priority_Index', 1.0)
        expo_raw = str(row['Exposure']).strip()
        
        # Implementation note.
        # Implementation note.
        # Implementation note.
        expos = re.split(r',\s*(?=X_)', expo_raw)
        
        # Implementation note.
        expos = [e.strip() for e in expos if e.strip()]
        
        for e in expos:
            # Implementation note.
            formatted_expo = f'X_{e}' if not e.startswith('X_') else e
            
            if formatted_expo in temp_dict:
                temp_dict[formatted_expo] = max(temp_dict[formatted_expo], weight)
            else:
                temp_dict[formatted_expo] = weight
                
            all_targets_to_extract.add(formatted_expo)
            
    cei_weights[disease] = temp_dict

print(f"    - Parsed target networks for {len(cei_weights)} disease outcomes.")
print(f"    - A total of {len(all_targets_to_extract)} unique priority pollutants entered the calculation.")

# =========================================================
# Implementation note.
# =========================================================
print("\n>>> [2/6] Calculating Z-score standardization parameters...")
df_hist = pd.read_csv(os.path.join(base_dir,'data_raw/grid_level_pollutants.csv'))

df_hist = df_hist[(df_hist['time'] >= 2017) & (df_hist['time'] <= 2023)]

scaler_params = {}
valid_targets_count = 0
for target in all_targets_to_extract:
    target = target[2:]
    if target in df_hist.columns:
        # Implementation note.
        # Implementation note.
        valid_series = df_hist[target].dropna()
        
        mean_val = valid_series.mean()
        std_val = valid_series.std()
        
        if std_val > 0:
            scaler_params[target] = {'mean': mean_val, 'std': std_val}
            valid_targets_count += 1
        else:
            scaler_params[target] = None 
    else:
        scaler_params[target] = None

print(f"    - Successfully calculated means and standard deviations for {valid_targets_count} target pollutants.")

# =========================================================
# Implementation note.
# =========================================================
print("\n>>> [3/6] Loading the grid-level emission panel and constructing CEIs...")
# Implementation note.
df_grid_pol = pd.read_csv(HIST_POL_PATH)
df_grid_pol_cols = df_grid_pol.columns[:5].tolist() + ['X_'+ i for i in df_grid_pol.columns[5:]]
df_grid_pol.columns = df_grid_pol_cols
# Implementation note.
df_grid_cei = df_grid_pol[['lon', 'lat', 'time']].copy()

for disease, weights_dict in cei_weights.items():
    cei_col_name = f'CEI_{disease}'
    df_grid_cei[cei_col_name] = 0.0 # Implementation note.
    
    components_used = 0
    for target, w in weights_dict.items():
       
        # Implementation note.
        if target in df_grid_pol.columns and scaler_params.get(target[2:]) is not None:
            mean_val = scaler_params[target[2:]]['mean']
            std_val = scaler_params[target[2:]]['std']
            
            # Implementation note.
            # Implementation note.
            z_score = (df_grid_pol[target] - mean_val) / std_val
            
            # Implementation note.
            z_score = z_score.fillna(0)
            
            # Implementation note.
            df_grid_cei[cei_col_name] += z_score * w
            components_used += 1
            
    print(f"    - [{disease}] Composite index constructed using {components_used} priority pollutants.")

# Implementation note.
out_cei_path = os.path.join(OUTPUT, '1_Historical_Grid_CEI.csv')
df_grid_cei.to_csv(out_cei_path, index=False)
print(f"\nHistorical grid-level wCEI panel completed and saved to: {out_cei_path}")

# =========================================================
# Implementation note.
# =========================================================
print("\n>>> [4/6] Generating disease-specific spatial CEI maps...")

# Implementation note.
target_year = df_grid_cei['time'].max()
df_map = df_grid_cei[df_grid_cei['time'] == target_year].copy()

# Implementation note.
diseases_to_plot = list(cei_weights.keys())
disease_to_plot = diseases_to_plot.remove('Cancer')
diseases_to_plot.remove('Neural')
cols = 3
rows = 3
fig, axes = plt.subplots(rows, cols, figsize=(16, 12), dpi=300)
axes = axes.flatten()

# Implementation note.
has_shp = False
if os.path.exists(SHP_PATH):
    china_map = gpd.read_file(SHP_PATH)
    if china_map.crs != "EPSG:4326":
        china_map = china_map.to_crs("EPSG:4326")
    has_shp = True

for i, disease in enumerate(diseases_to_plot):
    ax = axes[i]
    cei_col = f'CEI_{disease}'
    
    # Implementation note.
    # Implementation note.
    # Implementation note.
    df_sub = df_map[df_map[cei_col] > df_map[cei_col].quantile(0.1)]
    
    if has_shp:
        china_map.plot(ax=ax, color='#f4f5f7', edgecolor='#cccccc', linewidth=0.3)
    
    # Implementation note.
    vmax_val = df_sub[cei_col].quantile(0.99)
    vmin_val = df_sub[cei_col].quantile(0.1)
    
    scatter = ax.scatter(df_sub['lon'], df_sub['lat'], 
                         c=df_sub[cei_col], 
                         cmap='YlOrRd', 
                         marker='s',
                         norm=mcolors.Normalize(vmin=vmin_val, vmax=vmax_val),
                         s=15, alpha=0.85, edgecolors='none')
    
    ax.set_xlim(73, 136)
    ax.set_ylim(17, 54)
    
    # Implementation note.
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.03)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('Composite Exposure Index (CEI)', fontsize=10, fontweight='bold')
    
    ax.set_title(f"{disease} ({target_year})", fontsize=14, fontweight='bold')
    ax.set_axis_off()

plt.suptitle(f"Grid-level Integrated Exposome Risk Map (wCEI)", fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()

map_path = os.path.join(OUTPUT, f'Grid_CEI_Spatial_Map_{target_year}.png')
plt.savefig(map_path, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Spatial CEI map generated and saved to: {map_path}")






#%%
"""
Grid-level CEI spatial hotspot maps (3x3, 7 diseases).

Style aligned with plot_high_res_spatial_maps in plot_emission_figures.py:
- multi-year annual mean per grid cell
- province / country / nine-dash-line basemap
- Spectral_r + PowerNorm, inset South-China-Sea map
- colorbar from 0 to maximum
"""


import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running from project root or this file's directory
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
if BASE_PATH not in sys.path:
    sys.path.insert(0, BASE_PATH)

SHP_DIR = os.path.join(BASE_PATH, 'data_raw', 'shp版')
COUNTY_SHP = os.path.join(SHP_DIR, "中国_县_面.shp")
NINE_DASH_SHP = os.path.join(SHP_DIR, "china_nine_dotted_line.shp")

MAIN_XLIM = (73, 136)
MAIN_YLIM = (17, 54)
INSET_XLIM = (105, 125)
INSET_YLIM = (2, 25)
def load_china_boundary_layers(
    county_shp: str = COUNTY_SHP,
    nine_dash_shp: str = NINE_DASH_SHP,
):
    """
    Load county, province (dissolved from adcode), country outline, and nine-dash line.
    Province code derived from gb: (gb // 10000) * 10000  -> 34 provincial units.
    """
    layers = {
        "county": None,
        "province": None,
        "country": None,
        "nine_dash": None,
        "has_shp": False,
    }

    if os.path.exists(county_shp):
        county = gpd.read_file(county_shp)
        if county.crs is not None and county.crs.to_epsg() != 4326:
            county = county.to_crs(epsg=4326)

        county = county.copy()
        county["prov_code"] = (county["gb"].astype(int) // 10000) * 10000

        layers["county"] = county
        layers["province"] = county.dissolve(by="prov_code", as_index=False)
        layers["country"] = county.dissolve(as_index=False)
        layers["has_shp"] = True

    if os.path.exists(nine_dash_shp):
        nine = gpd.read_file(nine_dash_shp)
        if nine.crs is not None and nine.crs.to_epsg() != 4326:
            nine = nine.to_crs(epsg=4326)
        layers["nine_dash"] = nine

    return layers


def draw_china_basemap(
    ax,
    layers,
    *,
    fill_county: bool = True,
    show_province: bool = True,
    show_country: bool = True,
    show_nine_dash: bool = True,
    county_facecolor: str = "#f7f8fa",
    county_edgecolor: str = "#d0d0d0",
    county_linewidth: float = 0.15,
    province_edgecolor: str = "#888888",
    province_linewidth: float = 0.45,
    country_edgecolor: str = "#1A1A1A",
    country_linewidth: float = 0.9,
    nine_dash_edgecolor: str = "#1A1A1A",
    nine_dash_linewidth: float = 0.8,
):
    """Draw county fill + province boundaries + national boundary + nine-dash line."""
    if not layers.get("has_shp"):
        return

    county = layers.get("county")
    if fill_county and county is not None:
        county.plot(
            ax=ax,
            color=county_facecolor,
            edgecolor=county_edgecolor,
            linewidth=county_linewidth,
            zorder=0,
        )

    province = layers.get("province")
    if show_province and province is not None:
        province.boundary.plot(
            ax=ax,
            color=province_edgecolor,
            linewidth=province_linewidth,
            zorder=4,
        )

    country = layers.get("country")
    if show_country and country is not None:
        country.boundary.plot(
            ax=ax,
            color=country_edgecolor,
            linewidth=country_linewidth,
            zorder=5,
        )

    nine = layers.get("nine_dash")
    if show_nine_dash and nine is not None:
        nine.plot(
            ax=ax,
            facecolor="none",
            edgecolor=nine_dash_edgecolor,
            linewidth=nine_dash_linewidth,
            zorder=5,
        )

EXCLUDE_DISEASES = ("Cancer", "Neural")
ROWS = 3
COLS = 3
CMAP = "Spectral_r"
GAMMA = 0.5

DISEASE_COLORS = [
    "#DC0000",
    "#3C5488",
    "#00A087",
    "#E64B35",
    "#4DBBD5",
    "#8491B4",
    "#F39B7F",
]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _annual_mean_grid(df_grid_cei: pd.DataFrame, cei_cols: list[str]) -> pd.DataFrame:
    return df_grid_cei.groupby(["lat", "lon"])[cei_cols].mean().reset_index()


def plot_grid_cei_spatial_maps(
    df_grid_cei: pd.DataFrame,
    cei_weights: dict[str, float],
    output_dir: str,
    *,
    exclude_diseases: tuple[str, ...] = EXCLUDE_DISEASES,
    rows: int = ROWS,
    cols: int = COLS,
    county_shp: str = COUNTY_SHP,
    nine_dash_shp: str = NINE_DASH_SHP,
    filename: str | None = None,
) -> str:
    print("\n>>> [4/6] Generating disease-specific spatial CEI maps...")

    diseases_to_plot = [d for d in cei_weights if d not in exclude_diseases]
    if len(diseases_to_plot) != 7:
        print(f"  Diseases to plot ({len(diseases_to_plot)}): {diseases_to_plot}")

    cei_cols = [f"CEI_{d}" for d in diseases_to_plot]
    missing = [c for c in cei_cols if c not in df_grid_cei.columns]
    if missing:
        raise KeyError(f"df_grid_cei is missing columns: {missing}")

    year_min = int(df_grid_cei["time"].min())
    year_max = int(df_grid_cei["time"].max())
    year_label = f"{year_min}-{year_max}" if year_min != year_max else str(year_min)

    df_map = _annual_mean_grid(df_grid_cei, cei_cols)

    boundary_layers = load_china_boundary_layers(county_shp, nine_dash_shp)
    if not boundary_layers["has_shp"]:
        print(f"  Warning: county shapefile not found at {county_shp}; map boundaries will be omitted.")

    n_axes = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, 14), dpi=300)
    axes = np.atleast_1d(axes).flatten()

    for i, disease in enumerate(diseases_to_plot):
        ax = axes[i]
        cei_col = f"CEI_{disease}"

        draw_china_basemap(ax, boundary_layers)

        df_sub = df_map[df_map[cei_col] > 0].copy()
        if df_sub.empty:
            ax.set_title(disease, fontsize=15, fontweight="bold", pad=12, color="#1A1A1A")
            ax.set_xlim(*MAIN_XLIM)
            ax.set_ylim(*MAIN_YLIM)
            ax.axis("off")
            continue

        vmax_val = float(df_map[cei_col].max())
        norm = mcolors.PowerNorm(gamma=GAMMA, vmin=0, vmax=vmax_val)

        scatter = ax.scatter(
            df_sub["lon"],
            df_sub["lat"],
            c=df_sub[cei_col],
            cmap=CMAP,
            marker="s",
            s=20,
            alpha=0.9,
            edgecolors="none",
            norm=norm,
            zorder=2,
        )

        ax.set_xlim(*MAIN_XLIM)
        ax.set_ylim(*MAIN_YLIM)

        inset_ax = ax.inset_axes([0.80, 0.03, 0.18, 0.25])
        draw_china_basemap(
            inset_ax,
            boundary_layers,
            county_linewidth=0.1,
            province_linewidth=0.35,
            country_linewidth=0.7,
            nine_dash_linewidth=0.6,
        )
        inset_ax.scatter(
            df_sub["lon"],
            df_sub["lat"],
            c=df_sub[cei_col],
            cmap=CMAP,
            marker="s",
            s=3,
            alpha=0.9,
            edgecolors="none",
            norm=norm,
            zorder=2,
        )
        inset_ax.set_xlim(*INSET_XLIM)
        inset_ax.set_ylim(*INSET_YLIM)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        for spine in inset_ax.spines.values():
            spine.set_edgecolor("#a0a0a0")
            spine.set_linewidth(0.8)

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.75, pad=0.01, aspect=30)
        cbar.set_ticks([0, vmax_val])
        cbar.set_ticklabels(["0", f"{vmax_val:.2e}"])
        cbar.ax.tick_params(labelsize=10, length=4, direction="out")
        cbar.set_label(
            "Composite Exposure Index (CEI)",
            fontsize=11,
            fontweight="bold",
            labelpad=-10,
        )

        title_color = DISEASE_COLORS[i % len(DISEASE_COLORS)]
        ax.set_title(
            disease,
            fontsize=15,
            fontweight="bold",
            pad=12,
            color=title_color,
        )
        ax.axis("off")

    for j in range(len(diseases_to_plot), n_axes):
        axes[j].axis("off")

    plt.suptitle(
        f"Grid-level Integrated Exposome Risk Map (wCEI, Annual Mean, {year_label})",
        fontsize=24,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_name = filename or f"Grid_CEI_Spatial_Map_AnnualMean_{year_label}.png"
    map_path = os.path.join(output_dir, out_name)
    plt.savefig(map_path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  Saved: {map_path}")
    return map_path


def load_df_grid_cei(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if ext in {".feather", ".ft"}:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file type: {path}")


if __name__ == "__main__":
    plot_grid_cei_spatial_maps(df_grid_cei, cei_weights, OUTPUT)
#%%



























print("\n>>> [6/6] Aggregating grid-level data to counties...")
target_path = os.path.join(base_dir,'data_raw/Lat_Lon.xlsx')

# Implementation note.
print("Loading the shapefile and extracting geometries...")
counties = gpd.read_file(SHP_PATH)
if counties.crs != "EPSG:4326":
    counties = counties.to_crs("EPSG:4326")

counties['unique_id'] = counties.index
counties_clean = counties[['unique_id', 'name', 'geometry']].copy()

# Implementation note.
print("Processing target coordinates...")
target_df = pd.read_excel(target_path)
target_gdf = gpd.GeoDataFrame(
    target_df,
    geometry=gpd.points_from_xy(target_df['lon'], target_df['lat']),
    crs="EPSG:4326"
)
target_gdf = gpd.sjoin(target_gdf, counties_clean, how='left', predicate='within')
target_gdf.rename(columns={'name': 'matched_county_name'}, inplace=True)

df_sample = df_grid_cei
unique_grids = df_sample[['lon', 'lat']].drop_duplicates().copy()

mapping_df = pd.read_csv(os.path.join(base_dir,'data_raw/mapping.csv'))
merged_ts = pd.merge(df_sample, mapping_df, on=['lon', 'lat'], how='inner')

TARGET_VAL_COL = [i for i in merged_ts.columns if i.startswith('CEI')]
county_yearly = merged_ts.groupby(['unique_id', 'time'])[TARGET_VAL_COL].mean().reset_index()
            
print("Merging with the target data and exporting results...")
final_result = target_gdf.merge(county_yearly, on='unique_id', how='left')

# Implementation note.
cols_to_drop = ['geometry', 'index_right', 'unique_id']
final_result.drop(columns=[c for c in cols_to_drop if c in final_result.columns], inplace=True, errors='ignore')
final_result.rename(columns={'county':'B_county_id'},inplace=True)
final_result.rename(columns={'time':'B_year'},inplace=True)
df_year = pd.read_csv(STATISIC_PATH)

merged_result = pd.merge(df_year,final_result[['B_county_id','B_year']+TARGET_VAL_COL],on=['B_county_id','B_year'],how='left')
merged_result.to_csv(os.path.join(OUTPUT,'2_merged_yearly_data_new.csv'))
print('Saved')
# %%


# =============================================================================
# PART 2: PRIORITY-RANKING FIGURE SUITE
# =============================================================================

#%%
'\ncode_4_update_priority_figs.py\n------------------------------\nGenerate a suite of publication-quality priority-ranking figures from the\nPriority Index and Top-50% target list. The original code_4 outputs are not\noverwritten; results are written to code_4_results_update/.\n'

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE_DIR = Path(__file__).resolve().parent
INPUT = BASE_DIR / "code_3_results" / "3_Final_Top50pct_Categorized_Targets.csv"
OUT = BASE_DIR / "code_4_results_update"
FIG = OUT / "priority_figures"

CAT_SHORT = {
    "Heavy metal": "Heavy metal",
    "Halogenated carbazole": "HCZ",
    "Metalloid": "Metalloid",
    "Mineral Dust": "Mineral dust",
    "Halogen": "Halogen",
    "Halogenated aliphatic hydrocarbon": "HAH",
    "Light metal": "Light metal",
    "Parent polycyclic aromatic hydrocarbon": "pPAH",
    "Methylated polycyclic aromatic hydrocarbon": "MePAH",
}

SECTOR_COLOR = {
    "CK": "#2171b5",
    "ico": "#238b45",
    "MSWI": "#6a51a3",
    "pow": "#cb181d",
    "ipr": "#b35806",
    "Other": "#636363",
}
SECTOR_LABEL = {
    "CK": "CK (coke oven)",
    "ico": "ico (iron / industrial)",
    "MSWI": "MSWI (waste incineration)",
    "pow": "pow (power plants)",
    "ipr": "ipr (non-ferrous)",
    "Other": "Other",
}

CMAP_BLUE = mcolors.LinearSegmentedColormap.from_list(
    "nat_blue", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Times New Roman", "Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["sector"] = d["Exposure"].astype(str).apply(
        lambda s: s.split("_")[-1] if "_" in s else "Other"
    )
    d["pol"] = d["name_en_abbr"].fillna(
        d["Exposure"].astype(str).str.replace(r"^X_", "", regex=True).str.replace(r"_[^_]+$", "", regex=True)
    )
    d["pol"] = d["pol"].replace({"Hexachlorobutadiene": "HCBD"})
    d["cat_short"] = d["sub category_en"].map(CAT_SHORT).fillna(d["sub category_en"])
    # Implementation note.
    tot = d.groupby("Disease_Name")["Priority_Index"].transform("sum")
    d["share_pct"] = 100.0 * d["Priority_Index"] / tot
    return d


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png")
    fig.savefig(FIG / f"{stem}.pdf")
    plt.close(fig)
    print(f"  saved: {stem}")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig02_stacked_class(d: pd.DataFrame) -> None:
    mat = (
        d.groupby(["Disease_Name", "cat_short"], as_index=False)["Priority_Index"]
        .sum()
    )
    # Implementation note.
    mat["pct"] = mat.groupby("Disease_Name")["Priority_Index"].transform(
        lambda x: 100.0 * x / x.sum()
    )
    disease_order = (
        mat.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index.tolist()
    )
    cat_order = (
        mat.groupby("cat_short")["Priority_Index"].sum().sort_values(ascending=False).index.tolist()
    )
    pivot = mat.pivot(index="Disease_Name", columns="cat_short", values="pct").fillna(0)
    pivot = pivot.reindex(index=disease_order, columns=cat_order)

    colors = sns.color_palette("tab10", n_colors=len(cat_order))
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for i, cat in enumerate(cat_order):
        vals = pivot[cat].to_numpy()
        ax.barh(y, vals, left=left, color=colors[i], height=0.72, edgecolor="white", linewidth=0.4, label=cat)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Within-disease contribution (%)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_title("Chemical-class composition of priority risk", fontsize=12, fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(5, len(cat_order)),
        frameon=False,
        fontsize=7.5,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "02_stacked_contribution_by_class")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig03_bio_epi(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    diseases = sorted(d["Disease_Name"].unique())
    palette = dict(zip(diseases, sns.color_palette("tab10", n_colors=len(diseases))))
    # Implementation note.
    pmin, pmax = d["Priority_Index"].min(), d["Priority_Index"].max()
    sizes = 40 + 220 * (d["Priority_Index"] - pmin) / (pmax - pmin + 1e-9)

    for dis in diseases:
        sub = d[d["Disease_Name"] == dis]
        s = 40 + 220 * (sub["Priority_Index"] - pmin) / (pmax - pmin + 1e-9)
        ax.scatter(
            sub["Unified_Bio_Score"],
            sub["Epi_Score"],
            s=s,
            c=[palette[dis]],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            label=dis,
            zorder=3,
        )
 
    # Implementation note.
    top = d.nlargest(8, "Priority_Index")
    for _, r in top.iterrows():
        ax.annotate(
            f"{r['pol']}|{r['Disease_Name'][:3]}",
            (r["Unified_Bio_Score"], r["Epi_Score"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=6.5,
            color="#222222",
        )

    ax.set_xlabel("Biological evidence score", fontsize=10)
    ax.set_ylabel("Epidemiological score", fontsize=10)
    ax.set_title("Priority landscape: biology × epidemiology", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, title="Disease", title_fontsize=8, loc="best")
    # Implementation note.
    for pi, lab in [(pmin, "low PI"), (pmax, "high PI")]:
        s = 40 + 220 * (pi - pmin) / (pmax - pmin + 1e-9)
        ax.scatter([], [], s=s, c="#888888", alpha=0.5, label=lab)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, ls=":", lw=0.5, color="#cccccc", zorder=0)
    save(fig, "03_bio_vs_epi_priority_scatter")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig04_pareto(d: pd.DataFrame) -> None:
    diseases = (
        d.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index.tolist()
    )
    n = len(diseases)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.0 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, dis in zip(axes, diseases):
        sub = d[d["Disease_Name"] == dis].sort_values("Priority_Index", ascending=False).copy()
        sub["cum"] = 100.0 * sub["Priority_Index"].cumsum() / sub["Priority_Index"].sum()
        x = np.arange(1, len(sub) + 1)
        ax.fill_between(x, sub["cum"], color="#6baed6", alpha=0.35)
        ax.plot(x, sub["cum"], color="#08519c", lw=1.8, marker="o", ms=3.5)
        ax.axhline(50, color="#cb181d", ls="--", lw=0.9, alpha=0.8)
        ax.set_title(dis, fontsize=10, fontweight="bold")
        ax.set_xlabel("Ranked targets", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_xlim(0.5, max(len(sub), 1) + 0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        # Implementation note.
        n50 = int((sub["cum"] <= 50).sum()) + 1
        n50 = min(n50, len(sub))
        ax.text(0.97, 0.08, f"n50~{n50}/{len(sub)}", transform=ax.transAxes, ha="right", fontsize=7.5, color="#444")

    for j in range(len(diseases), len(axes)):
        axes[j].set_axis_off()
    axes[0].set_ylabel("Cumulative contribution (%)", fontsize=9)
    fig.suptitle("Pareto curves of priority targets (50% line dashed)", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "04_pareto_cumulative_by_disease")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig05_top_lollipop(d: pd.DataFrame, top_n: int = 25) -> None:
    top = d.nlargest(top_n, "Priority_Index").iloc[::-1].copy()
    top["label"] = top["pol"] + " · " + top["Disease_Name"]
    colors = [SECTOR_COLOR.get(s, "#666") for s in top["sector"]]

    fig, ax = plt.subplots(figsize=(8.2, 0.34 * top_n + 1.4))
    y = np.arange(len(top))
    ax.hlines(y, 0, top["Priority_Index"], color="#bdbdbd", lw=1.2, zorder=1)
    ax.scatter(top["Priority_Index"], y, c=colors, s=55, zorder=3, edgecolors="white", linewidths=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(top["label"], fontsize=8)
    ax.set_xlabel("Priority Index", fontsize=10)
    ax.set_title(f"Top {top_n} prioritized exposure–disease pairs", fontsize=12, fontweight="bold")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SECTOR_COLOR[s], markersize=7, label=SECTOR_LABEL[s])
        for s in ["CK", "ico", "MSWI", "pow"]
        if s in set(top["sector"])
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlim(0, top["Priority_Index"].max() * 1.08)
    save(fig, "05_top_priority_lollipop")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig06_heatmap(d: pd.DataFrame) -> None:
    # Implementation note.
    pol_rank = d.groupby("Exposure")["Priority_Index"].sum().sort_values(ascending=False)
    keep = pol_rank.head(18).index.tolist()
    sub = d[d["Exposure"].isin(keep)]
    mat = sub.pivot_table(index="Disease_Name", columns="Exposure", values="Priority_Index", aggfunc="max")
    # Implementation note.
    dis_order = mat.sum(axis=1).sort_values(ascending=False).index
    pol_order = mat.sum(axis=0).sort_values(ascending=False).index
    mat = mat.reindex(index=dis_order, columns=pol_order)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    sns.heatmap(
        mat,
        ax=ax,
        cmap=CMAP_BLUE,
        linewidths=0.5,
        linecolor="white",
        mask=mat.isna(),
        cbar_kws={"label": "Priority Index", "shrink": 0.75},
        square=False,
    )
    ax.set_xlabel("Pollutant", fontsize=10)
    ax.set_ylabel("Disease", fontsize=10)
    ax.set_title("Priority Index heatmap (selected pollutants)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    plt.setp(ax.get_xticklabels(), ha="right")
    save(fig, "06_priority_heatmap_disease_pollutant")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig07_sector_stack(d: pd.DataFrame) -> None:
    mat = d.groupby(["Disease_Name", "sector"], as_index=False)["Priority_Index"].sum()
    mat["pct"] = mat.groupby("Disease_Name")["Priority_Index"].transform(lambda x: 100 * x / x.sum())
    dis_order = mat.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index
    sectors = [s for s in ["CK", "ico", "MSWI", "pow", "ipr", "Other"] if s in set(mat["sector"])]
    pivot = mat.pivot(index="Disease_Name", columns="sector", values="pct").fillna(0)
    pivot = pivot.reindex(index=dis_order, columns=sectors)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for sec in sectors:
        vals = pivot[sec].to_numpy()
        ax.barh(y, vals, left=left, color=SECTOR_COLOR.get(sec, "#666"), height=0.7, edgecolor="white", lw=0.4, label=SECTOR_LABEL.get(sec, sec))
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Within-disease contribution (%)", fontsize=10)
    ax.set_title("Industrial sector drivers of prioritized targets", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "07_stacked_contribution_by_sector")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig08_top5_panels(d: pd.DataFrame) -> None:
    diseases = (
        d.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index.tolist()
    )
    n = len(diseases)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, dis in zip(axes, diseases):
        sub = d[d["Disease_Name"] == dis].nlargest(5, "Priority_Index").iloc[::-1]
        colors = [SECTOR_COLOR.get(s, "#666") for s in sub["sector"]]
        ax.barh(sub["pol"], sub["Priority_Index"], color=colors, height=0.65, edgecolor="white", lw=0.4)
        ax.set_title(dis, fontsize=10, fontweight="bold")
        ax.set_xlabel("Priority Index", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for j in range(len(diseases), len(axes)):
        axes[j].set_axis_off()

    handles = [
        Patch(facecolor=SECTOR_COLOR[s], edgecolor="white", label=SECTOR_LABEL[s])
        for s in ["CK", "ico", "MSWI", "pow"]
        if s in set(d["sector"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Top-5 priority targets by disease", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "08_top5_by_disease")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig09_rr_priority(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    diseases = sorted(d["Disease_Name"].unique())
    palette = dict(zip(diseases, sns.color_palette("tab10", n_colors=len(diseases))))
    for dis in diseases:
        sub = d[d["Disease_Name"] == dis]
        ax.scatter(
            sub["RR"],
            sub["Priority_Index"],
            c=[palette[dis]],
            s=42,
            alpha=0.75,
            edgecolors="white",
            lw=0.4,
            label=dis,
        )
    ax.axvline(1.0, color="#999999", ls="--", lw=0.8)
    ax.set_xlabel("Relative risk (RR)", fontsize=10)
    ax.set_ylabel("Priority Index", fontsize=10)
    ax.set_title("Epidemiological effect size vs priority rank score", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="best")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, ls=":", lw=0.45, color="#cccccc")
    save(fig, "09_RR_vs_priority")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig10_count_tile(d: pd.DataFrame) -> None:
    ct = d.groupby(["Disease_Name", "cat_short"]).size().reset_index(name="n")
    dis_order = d.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index
    cat_order = d.groupby("cat_short")["Priority_Index"].sum().sort_values(ascending=False).index
    mat = ct.pivot(index="Disease_Name", columns="cat_short", values="n").fillna(0)
    mat = mat.reindex(index=dis_order, columns=cat_order)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    sns.heatmap(
        mat,
        ax=ax,
        cmap="Greys",
        annot=True,
        fmt=".0f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "No. of selected targets", "shrink": 0.75},
        annot_kws={"fontsize": 8},
    )
    ax.set_title("Number of top-50% priority targets", fontsize=12, fontweight="bold")
    ax.set_xlabel("Chemical class", fontsize=10)
    ax.set_ylabel("Disease", fontsize=10)
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    plt.setp(ax.get_xticklabels(), ha="right")
    save(fig, "10_count_matrix_disease_class")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig11_score_decomposition(d: pd.DataFrame) -> None:
    top = d.nlargest(15, "Priority_Index").iloc[::-1].copy()
    top["label"] = top["pol"] + " · " + top["Disease_Name"]
    y = np.arange(len(top))
    h = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(y + h, top["Unified_Bio_Score"], height=h, color="#9ecae1", label="Bio score", edgecolor="white", lw=0.3)
    ax.barh(y, top["Epi_Score"], height=h, color="#fc9272", label="Epi score", edgecolor="white", lw=0.3)
    ax.barh(y - h, top["Priority_Index"], height=h, color="#08519c", label="Priority Index", edgecolor="white", lw=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(top["label"], fontsize=8)
    ax.set_xlabel("Score", fontsize=10)
    ax.set_title("Score decomposition for top-15 priority pairs", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, "11_score_decomposition_top15")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig12_integrated_atlas(d: pd.DataFrame) -> None:
    """
    Integrated overview of Bio, Epi and Priority scores:
    A: Bio versus Epi; point size represents Priority and colour represents disease (all targets).
    B: Three-score heatmap for all exposure–disease pairs (rows=pairs; columns=Bio/Epi/Priority).
    """
    diseases = (
        d.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index.tolist()
    )
    palette = dict(zip(diseases, sns.color_palette("tab10", n_colors=len(diseases))))

    # Implementation note.
    rows = []
    for dis in diseases:
        sub = d[d["Disease_Name"] == dis].sort_values("Priority_Index", ascending=False)
        for _, r in sub.iterrows():
            rows.append(r)
    ordered = pd.DataFrame(rows)
    ordered["row_label"] = ordered["Exposure"] + " · " + ordered["Disease_Name"]
    score_mat = ordered[["Unified_Bio_Score", "Epi_Score", "Priority_Index"]].to_numpy(dtype=float)
    score_df = pd.DataFrame(
        score_mat,
        index=ordered["row_label"],
        columns=["Bio", "Epi", "Priority"],
    )

    n = len(score_df)
    fig = plt.figure(figsize=(12.5, max(8.5, 0.175 * n + 2.5)))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[1.05, 1.35],
        wspace=0.28,
        left=0.07,
        right=0.92,
        top=0.93,
        bottom=0.06,
    )
    ax_sc = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])

    # ---- A: scatter ----
    pmin, pmax = d["Priority_Index"].min(), d["Priority_Index"].max()
    for dis in diseases:
        sub = d[d["Disease_Name"] == dis]
        s = 35 + 260 * (sub["Priority_Index"] - pmin) / (pmax - pmin + 1e-9)
        ax_sc.scatter(
            sub["Unified_Bio_Score"],
            sub["Epi_Score"],
            s=s,
            c=[palette[dis]],
            alpha=0.78,
            edgecolors="white",
            linewidths=0.45,
            label=dis,
            zorder=3,
        )
    # Implementation note.
    for pi, lab in [(pmin, "Low priority"), ((pmin + pmax) / 2, "Mid"), (pmax, "High priority")]:
        s = 35 + 260 * (pi - pmin) / (pmax - pmin + 1e-9)
        ax_sc.scatter([], [], s=s, c="#888888", alpha=0.55, label=lab)

    ax_sc.set_xlabel("Biological evidence score (Bio)", fontsize=9)
    ax_sc.set_ylabel("Epidemiological score (Epi)", fontsize=9)
    ax_sc.set_title("A  Priority landscape", fontsize=11, fontweight="bold", loc="left")
    ax_sc.grid(True, ls=":", lw=0.45, color="#cccccc", zorder=0)
    for spine in ("top", "right"):
        ax_sc.spines[spine].set_visible(False)
    leg = ax_sc.legend(
        frameon=False,
        fontsize=6.5,
        loc="upper left",
        title="Disease / size",
        title_fontsize=7,
        markerscale=0.9,
    )

    # ---- B: heatmap of three scores for ALL pairs ----
    sns.heatmap(
        score_df,
        ax=ax_hm,
        cmap=CMAP_BLUE,
        vmin=0,
        vmax=max(1.0, float(np.nanmax(score_mat))),
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Score", "shrink": 0.45, "pad": 0.02},
        yticklabels=True,
    )
    ax_hm.set_title("B  Score decomposition (all pairs)", fontsize=11, fontweight="bold", loc="left")
    ax_hm.set_xlabel("")
    ax_hm.set_ylabel("")
    ax_hm.tick_params(axis="y", labelsize=5.5)
    ax_hm.tick_params(axis="x", labelsize=9, labelrotation=0)
    # Implementation note.
    counts = ordered.groupby("Disease_Name", sort=False).size()
    # ordered diseases follow `diseases` order
    ypos = 0
    for dis in diseases:
        ypos += int(counts.get(dis, 0))
        if ypos < n:
            ax_hm.axhline(ypos, color="#333333", lw=0.6)

    fig.suptitle(
        "Integrated Bio / Epi / Priority atlas (all prioritized pairs)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    save(fig, "12_integrated_bio_epi_priority_atlas")


# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig13_decomposition_all_by_disease(d: pd.DataFrame) -> None:
    diseases = (
        d.groupby("Disease_Name")["Priority_Index"].sum().sort_values(ascending=False).index.tolist()
    )
    n = len(diseases)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    # Implementation note.
    max_per = int(d.groupby("Disease_Name").size().max())
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(12.0, max(3.2, 0.28 * max_per) * nrows),
        sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()
    h = 0.22

    for ax, dis in zip(axes, diseases):
        sub = d[d["Disease_Name"] == dis].sort_values("Priority_Index", ascending=True)
        y = np.arange(len(sub))
        ax.barh(y + h, sub["Unified_Bio_Score"], height=h, color="#9ecae1", edgecolor="white", lw=0.2)
        ax.barh(y, sub["Epi_Score"], height=h, color="#fc9272", edgecolor="white", lw=0.2)
        ax.barh(y - h, sub["Priority_Index"], height=h, color="#08519c", edgecolor="white", lw=0.2)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["pol"], fontsize=15)
        ax.set_title(f"{dis} (n={len(sub)})", fontsize=15, fontweight="bold")
        ax.set_xlim(0, max(1.05, d[["Unified_Bio_Score", "Epi_Score", "Priority_Index"]].max().max() * 1.02))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, axis="x", ls=":", lw=0.4, color="#dddddd")

    for j in range(len(diseases), len(axes)):
        axes[j].set_axis_off()

    handles = [
        Patch(facecolor="#9ecae1", edgecolor="white", label="Bio score"),
        Patch(facecolor="#fc9272", edgecolor="white", label="Epi score"),
        Patch(facecolor="#08519c", edgecolor="white", label="Priority Index"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Bio / Epi / Priority decomposition — all pollutants by disease",
        fontsize=20,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save(fig, "13_decomposition_all_by_disease")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
from adjustText import adjust_text

# ---------------------------------------------------------------------------
# Implementation note.
# ---------------------------------------------------------------------------
def fig14_rr_priority_bio_size(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    diseases = sorted(d["Disease_Name"].unique())
    
    # Implementation note.
    palette = dict(zip(diseases, sns.color_palette("tab10", n_colors=len(diseases))))
    bmin, bmax = d["Unified_Bio_Score"].min(), d["Unified_Bio_Score"].max()

    # Implementation note.
    for dis in diseases:
        sub = d[d["Disease_Name"] == dis]
        # Implementation note.
        s = 50 + 400 * (sub["Unified_Bio_Score"] - bmin) / (bmax - bmin + 1e-9)
        ax.scatter(
            sub["RR"],
            sub["Priority_Index"],
            s=s,
            c=[palette[dis]],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

    # Implementation note.
    # texts = []
    # for _, r in d.nsmallest(20, "Priority_Index").iterrows():
    # Implementation note.
    #     label_text = r["Exposure"][2:] if str(r["Exposure"]).startswith("X_") else r["Exposure"]
    #     texts.append(
    #         ax.text(
    #             r["RR"], 
    #             r["Priority_Index"], 
    #             label_text, 
    #             fontsize=7.5, 
    #             color="#333333",
    #             fontweight="medium",
    #             zorder=4
    #         )
    #     )
    
    # Implementation note.
    # adjust_text(
    #     texts, 
    #     ax=ax, 
    #     expand_points=(1.5, 1.5),
    #     arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, alpha=0.8)
    # )

    # Implementation note.
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=dis,
               markerfacecolor=palette[dis], markersize=9, alpha=0.9)
        for dis in diseases
    ]
    ax.legend(
        handles=legend_elements, 
        frameon=False, 
        fontsize=8.5, 
        ncol=2, 
        loc="center left", 
        bbox_to_anchor=(1.02, 0.5), # Implementation note.
        title="Disease Category", 
        title_fontproperties={'weight':'bold', 'size':9}
    )

    # Implementation note.
    ax.axvline(1.0, color="#bbbbbb", ls="--", lw=1.2, zorder=1)
    ax.set_xlabel("Relative risk (RR)", fontsize=11, fontweight='medium', labelpad=8)
    ax.set_ylabel("Priority Index", fontsize=11, fontweight='medium', labelpad=8)
    ax.set_title("Epidemiological effect size vs Priority Index", fontsize=13, fontweight="bold", pad=15)
    
    # Implementation note.
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color("#333333")

    # Implementation note.
    # ax.text(
    #     0.98, 0.02,
    #     "Circle size encodes Biological Evidence Score",
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="bottom",
    #     fontsize=8,
    #     color="#777777",
    #     style="italic"
    # )
    
    fig.tight_layout()
    save(fig, "14_RR_vs_priority_sizeBio")


def write_readme() -> None:
    text = """# Priority ranking figures (code_4_update)

Data: code_3_results/3_Final_Top50pct_Categorized_Targets.csv

| File | Content |
|------|---------|
| 01 (existing) | Equal-size exposome matrix, fill = contribution |
| 02_stacked_contribution_by_class | Disease × chemical-class stacked % |
| 03_bio_vs_epi_priority_scatter | Biology vs epidemiology landscape |
| 04_pareto_cumulative_by_disease | Cumulative contribution / Pareto |
| 05_top_priority_lollipop | Global top priority pairs |
| 06_priority_heatmap_disease_pollutant | Disease × pollutant Priority heatmap |
| 07_stacked_contribution_by_sector | Disease × sector stacked % |
| 08_top5_by_disease | Top-5 targets per disease |
| 09_RR_vs_priority | RR versus Priority Index |
| 10_count_matrix_disease_class | Counts of selected targets |
| 11_score_decomposition_top15 | Bio / Epi / Priority bars (top 15) |
| 12_integrated_bio_epi_priority_atlas | COMPREHENSIVE: scatter + all-pair score heatmap |
| 13_decomposition_all_by_disease | COMPREHENSIVE: fig11-style bars for ALL pairs by disease |
| 14_RR_vs_priority_sizeBio | RR vs Priority, size = Bio (three metrics) |
"""
    (OUT / "README_priority_figures.txt").write_text(text, encoding="utf-8")

#%%
def main() -> None:
    set_style()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    d = prep(pd.read_csv(INPUT))
    d.to_csv(OUT / "2_priority_targets_plotready.csv", index=False)

    print(">>> plotting priority figure suite ...")
    fig02_stacked_class(d)
    fig03_bio_epi(d)
    fig04_pareto(d)
    fig05_top_lollipop(d)
    fig06_heatmap(d)
    fig07_sector_stack(d)
    fig08_top5_panels(d)
    fig09_rr_priority(d)
    fig10_count_tile(d)
    fig11_score_decomposition(d)
    fig12_integrated_atlas(d)
    fig13_decomposition_all_by_disease(d)
    fig14_rr_priority_bio_size(d)
    write_readme()
    print(f"done → {FIG}")

if __name__ == "__main__":
    main()

# %%


# =============================================================================
# PART 3: EXPOSOME PRIORITY MATRIX
# =============================================================================

#%%
'\ncode_4_update.py\n----------------\nGenerate a publication-quality exposome priority matrix without overwriting\nthe original code_4 outputs.\n- All circles have equal size.\n- Fill intensity represents within-disease contribution (%).\n- Pollutant names are displayed without percentage labels.\n- Circle-edge colour represents the primary emission sector.\nResults are written to code_4_results_update/.\n'

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "code_3_results" / "3_Final_Top50pct_Categorized_Targets.csv"
OUT_DIR = BASE_DIR / "code_4_results_update"

# Implementation note.
CAT_SHORT = {
    "Heavy metal": "Heavy metal",
    "Halogenated carbazole": "HCZ",
    "Metalloid": "Metalloid",
    "Mineral Dust": "Mineral dust",
    "Halogen": "Halogen",
    "Halogenated aliphatic hydrocarbon": "HAH",
    "Light metal": "Light metal",
    "Parent polycyclic aromatic hydrocarbon": "pPAH",
    "Methylated polycyclic aromatic hydrocarbon": "MePAH",
}

# Implementation note.
SECTOR_STYLE = {
    "pow": {"edge": "#2171b5", "label": "CK (coke oven)"},
    "ico": {"edge": "#238b45", "label": "ico (iron / industrial)"},
    "MSWI": {"edge": "#6a51a3", "label": "MSWI (waste incineration)"},
    "CK": {"edge": "#cb181d", "label": "pow (power plants)"},
    "ipr": {"edge": "#b35806", "label": "ipr (non-ferrous)"},
    "Other": {"edge": "#636363", "label": "Other"},
}

# Implementation note.
NATURE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "nature_contrib",
    ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Times new roman"],
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


ABBR_FIX = {
    "Hexachlorobutadiene": "HCBD",
    "1,3,6,8-CCZ": "1368-CCZ",
    "1,3,6-BCZ": "136-BCZ",
}


def format_pollutant_label(names: list[str], max_n: int = 3) -> str:
    'Generate the corresponding publication-quality visualization.'
    cleaned = []
    for n in names:
        n = ABBR_FIX.get(n, n)
        if n not in cleaned:
            cleaned.append(n)
    # Implementation note.
    if any(len(n) > 7 for n in cleaned):
        cleaned = cleaned[:2]
    else:
        cleaned = cleaned[:max_n]
    return "\n".join(cleaned)


def get_primary_sector(x: pd.Series) -> str:
    sectors = x.astype(str).apply(lambda s: s.split("_")[-1] if "_" in s else "Other")
    mode = sectors.mode()
    return mode.iloc[0] if not mode.empty else "Other"

def aggregate(df_raw: pd.DataFrame):
    disease_col = "Disease_Name"
    category_col = "sub category_en"

    df_raw = df_raw.sort_values(
        by=[disease_col, category_col, "Priority_Index"],
        ascending=[True, True, False],
    )

    def _top_names(g: pd.DataFrame) -> str:
        # Implementation note.
        if "name_en_abbr" in g.columns:
            names = g["name_en_abbr"].astype(str).tolist()
        else:
            names = (
                g["Exposure"]
                .astype(str)
                .str.replace(r"^X_", "", regex=True)
                .str.replace(r"_[^_]+$", "", regex=True)
                .tolist()
            )
        return format_pollutant_label(names, 3)

    rows = []
    for (dis, cat), g in df_raw.groupby([disease_col, category_col], sort=False):
        rows.append(
            {
                disease_col: dis,
                category_col: cat,
                "Priority_Index_Sum": g["Priority_Index"].sum(),
                "Contribution_Pct": g["Contribution_Pct"].sum(),
                "Top3_Unique_Pollutants": _top_names(g),
                "Primary_Sector": get_primary_sector(g["Exposure"]),
            }
        )
    df_summary = pd.DataFrame(rows)

    disease_totals = (
        df_raw.groupby(disease_col)["Priority_Index"]
        .sum()
        .reset_index(name="Group_Total_Sum")
    )
    df_summary = df_summary.merge(disease_totals, on=disease_col, how="left")
    df_summary["Contribution_Pct"] = (
        df_summary["Priority_Index_Sum"] / df_summary["Group_Total_Sum"] * 100.0
    )

    disease_order = (
        df_summary.groupby(disease_col)["Priority_Index_Sum"]
        .sum()
        .sort_values(ascending=False)
        .index
    )
    category_order = (
        df_summary.groupby(category_col)["Priority_Index_Sum"]
        .sum()
        .sort_values(ascending=False)
        .index
    )

    df_summary[disease_col] = pd.Categorical(
        df_summary[disease_col], categories=disease_order, ordered=True
    )
    df_summary[category_col] = pd.Categorical(
        df_summary[category_col], categories=category_order, ordered=True
    )
    df_summary["x_coord"] = df_summary[category_col].map(
        {c: i for i, c in enumerate(category_order)}
    )
    df_summary["y_coord"] = df_summary[disease_col].map(
        {d: i for i, d in enumerate(disease_order)}
    )
    return df_summary, list(disease_order), list(category_order)


def _text_color(rgba) -> str:
    """Automatically select black or white text according to fill luminance."""
    r, g, b = rgba[:3]
    # Implementation note.
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if lum > 0.55 else "#ffffff"


def plot_matrix(df: pd.DataFrame, disease_order: list, category_order: list) -> None:
    n_x, n_y = len(category_order), len(disease_order)
    fig = plt.figure(figsize=(11.2, 8.6), dpi=300)
    # Implementation note.
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[4.6, 1.15],
        wspace=0.08,
        left=0.10,
        right=0.98,
        top=0.92,
        bottom=0.16,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")

    ax.set_xlim(-0.72, n_x - 0.28)
    ax.set_ylim(n_y - 0.28, -0.72)
    ax.set_xticks(range(n_x))
    ax.set_yticks(range(n_y))
    ax.set_xticklabels(
        [CAT_SHORT.get(c, c) for c in category_order],
        rotation=35,
        ha="right",
        fontsize=9,
        color="#222222",
    )
    ax.set_yticklabels(disease_order, fontsize=10, color="#222222")
    ax.tick_params(length=0)
    ax.set_xlabel("Chemical class", fontsize=10, color="#333333", labelpad=8)
    ax.set_ylabel("Disease outcome", fontsize=10, color="#333333", labelpad=8)

    # Implementation note.
    ax.set_xticks(np.arange(-0.5, n_x, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_y, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.35, color="#efefef", zorder=0)
    ax.grid(False, which="major")
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.7)

    vmin, vmax = 0.0, 100.0
    cmap = NATURE_CMAP
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    bubble_size = 1850
    for _, row in df.iterrows():
        pct = float(row["Contribution_Pct"])
        face = cmap(norm(pct))
        sec = str(row["Primary_Sector"])
        edge = SECTOR_STYLE.get(sec, SECTOR_STYLE["Other"])["edge"]
        ax.scatter(
            float(row["x_coord"]),
            float(row["y_coord"]),
            s=bubble_size,
            c=[face],
            edgecolors=edge,
            linewidths=1.35,
            alpha=1.0,
            zorder=3,
        )
        labels = str(row["Top3_Unique_Pollutants"]).strip()
        if labels and labels.lower() != "nan":
            n_lines = labels.count("\n") + 1
            fs = 9.0 if n_lines >= 3 else (9 if n_lines == 2 else 9)
            ax.text(
                float(row["x_coord"]),
                float(row["y_coord"]),
                labels,
                ha="center",
                va="center",
                fontsize=fs,
                fontweight="semibold",
                color=_text_color(face),
                linespacing=1.12,
                zorder=4,
                clip_on=True,
            )

    # Implementation note.
    y0 = 0.98
    ax_leg.text(
        0.0,
        y0,
        "Contribution",
        fontsize=9,
        fontweight="bold",
        color="#222222",
        transform=ax_leg.transAxes,
        va="top",
    )
    # Implementation note.
    cax = ax_leg.inset_axes([0.08, 0.52, 0.18, 0.38])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.ax.tick_params(labelsize=7.5, length=2.5, width=0.6, colors="#333333")
    cbar.outline.set_linewidth(0.6)
    ax_leg.text(
        0.42,
        0.71,
        "(%)",
        fontsize=8,
        color="#444444",
        transform=ax_leg.transAxes,
        va="center",
    )

    ax_leg.text(
        0.0,
        0.44,
        "Primary sector\n(circle edge)",
        fontsize=9,
        fontweight="bold",
        color="#222222",
        transform=ax_leg.transAxes,
        va="top",
        linespacing=1.25,
    )

    # Implementation note.
    sector_order = ["CK", "ico", "MSWI", "pow", "ipr", "Other"]
    present = set(df["Primary_Sector"].astype(str))
    sectors_show = [s for s in sector_order if s in present]
    for s in sorted(present):
        if s not in sectors_show:
            sectors_show.append(s)

    handles = []
    labels = []
    for sec in sectors_show:
        sty = SECTOR_STYLE.get(sec, SECTOR_STYLE["Other"])
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=9,
                markerfacecolor="#f7fbff",
                markeredgecolor=sty["edge"],
                markeredgewidth=2.0,
            )
        )
        labels.append(sty["label"])

    leg = ax_leg.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.38),
        frameon=False,
        fontsize=7.5,
        handletextpad=0.5,
        borderaxespad=0.0,
        labelspacing=0.85,
    )
    for t in leg.get_texts():
        t.set_color("#222222")

    ax_leg.text(
        0.0,
        0.02,
        "Circle size equal.\nFill depth = contribution.\nLabels = top pollutants.",
        fontsize=6.5,
        color="#777777",
        transform=ax_leg.transAxes,
        va="bottom",
        linespacing=1.35,
    )

    fig.suptitle(
        "Exposome priority matrix (top 50% cumulative risk)",
        fontsize=12,
        fontweight="bold",
        color="#111111",
        y=0.97,
    )

    out_png = OUT_DIR / "1_Matrix_EqualSize_ColorByPct.png"
    out_pdf = OUT_DIR / "1_Matrix_EqualSize_ColorByPct.pdf"
    fig.savefig(out_png, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {out_png.name}")


def main() -> None:
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    print(">>> aggregate ...")
    df_raw = pd.read_csv(INPUT_FILE)
    df_summary, disease_order, category_order = aggregate(df_raw)
    df_summary.to_csv(OUT_DIR / "1_Aggregated_Matrix_Data.csv", index=False)

    print(">>> plot Nature-style matrix ...")
    plot_matrix(df_summary, disease_order, category_order)

    (OUT_DIR / "README.txt").write_text(
        "\n".join(
            [
                "# code_4_update",
                "- Equal-size circles; fill depth = contribution (%)",
                "- Bubble text = pollutant names (no %)",
                "- Edge colour = primary sector (full names in legend)",
                "- Nature-like sequential blue colormap",
            ]
        ),
        encoding="utf-8",
    )
    print(f"done → {OUT_DIR}")


if __name__ == "__main__":
    main()

#%%
