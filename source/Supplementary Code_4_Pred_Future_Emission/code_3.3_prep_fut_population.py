#%%
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from rasterio.enums import Resampling
from pathlib import Path
from tqdm import tqdm
import rioxarray as rxr
from pathlib import Path
import xarray as xr

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
POPULATION_PATH = os.path.join(BASE_PATH, 'data_raw', 'population')
HIST_POP_PATH = os.path.join(POPULATION_PATH, 'history_pop')
SSPS = ['SSP1','SSP2','SSP3','SSP5']

POP_FILES = ['2020.tif','2030.tif','2040.tif','2050.tif','2060.tif','2070.tif','2080.tif','2090.tif','2100.tif']

SHP_PATH = os.path.join(BASE_PATH, '..', 'Supplementary Code_3_Create_Emission_Index', 'create_index', 'data_raw', 'shp版', '中国_县_面.shp')
TARGET_GRID_PATH = os.path.join(BASE_PATH, 'code_3_results', 'grid_level_gdp_ssp1.csv')

# ==================== 2. Build the target spatial template ====================
print("Building the target-grid spatial template...")
target_df = pd.read_csv(TARGET_GRID_PATH)

# Extract unique longitude and latitude coordinates to create the regular grid.
target_lon = np.sort(target_df['lon'].unique())
target_lat = np.sort(target_df['lat'].unique())

# Create an empty xarray DataArray as the alignment template.
target_template = xr.DataArray(
    np.zeros((len(target_lat), len(target_lon)), dtype=np.float32),
    dims=("y", "x"),
    coords={"y": target_lat, "x": target_lon}
)
target_template.rio.write_crs("EPSG:4326", inplace=True)

# Load and dissolve the China boundary.
print("Loading shapefile boundaries...")
gdf_china = gpd.read_file(SHP_PATH)

# ==================== 3. Efficient processing function ====================
def read_and_align_raster(file_path, year, template, boundary_gdf):
    """
    1. Clip in the source projection first to reduce processing time.
    2. Align to the target grid with reproject_match (use SUM for population counts).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Use lazy loading to avoid reading all data into memory immediately.
    da = rxr.open_rasterio(file_path, masked=True).squeeze()
    
    # Optimization 1: transform the China boundary to the source TIF CRS and clip before reprojection.
    boundary_in_crs = boundary_gdf.to_crs(da.rio.crs).geometry.unary_union
    da_cropped = da.rio.clip([boundary_in_crs], drop=True)
    
    raw_total_pop = da_cropped.sum().item()
    
    # Optimization 2: align directly to the target grid; use Resampling.sum for population counts.
    da_matched = da_cropped.rio.reproject_match(
        template, 
        resampling=Resampling.sum
    )
    
    # Check totals.
    matched_total_pop = da_matched.sum().item()
    diff_ratio = abs(matched_total_pop - raw_total_pop) / (raw_total_pop + 1e-9)
    print(diff_ratio)
    print(f"Original total: {raw_total_pop:,.0f}; aligned total: {matched_total_pop:,.0f}")
        
    # Add the time dimension and assign the year.
    da_matched = da_matched.expand_dims(time=[int(year)])
    return da_matched


for ssp in SSPS:
    da_list = []
    FUT_POP_PATH = os.path.join(POPULATION_PATH, ssp)
    hist_df = None
    for f in tqdm(POP_FILES, desc="Processing spatial data and loading historical rasters"):
        file_path = os.path.join(FUT_POP_PATH, f)
        year = Path(file_path).stem[-4:]
        df_year = read_and_align_raster(file_path, year, target_template, gdf_china)
        da_list.append(df_year)
        
    ds_combined = xr.concat(da_list, dim='time')
    df_final = ds_combined.to_dataframe(name='population').reset_index()
    df_final = df_final.rename(columns={'y':'lat','x':'lon'})

    # Inner-join with target_grid to retain only the grid cells required for prediction.
    df_final = pd.merge(
        target_df[['lon', 'lat','time']], 
        df_final, 
        on=['lon', 'lat','time'], 
        how='inner'
    )
    df_final = df_final[df_final['time'] != 2020]
    df_final['population'] = df_final['population'].fillna(0)
    # Clean the output format.
    df_final = df_final.dropna(subset=['population']).reset_index(drop=True)
    df_final.groupby(['time']).sum().reset_index()[['population','time']]  # Check for duplicate grid cells.
    print(f"Processing complete. Final row count: {len(df_final)}")
    print(df_final.head())
    
    OUTPUT_DIR = os.path.join(BASE_PATH, 'code_3_results')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_final.to_csv(os.path.join(OUTPUT_DIR, f'grid_level_population_{ssp}.csv'), index=False)
# %%
