# %%
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
from pathlib import Path
from tqdm import tqdm

# ==================== 1. Paths and base configuration ====================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
POPULATION_PATH = os.path.join(BASE_PATH, 'data_raw', 'population')
HIST_POP_PATH = os.path.join(POPULATION_PATH, 'history_pop')
FUT_POP_PATH = os.path.join(POPULATION_PATH, 'SSP2')

pop_path1 = os.path.join(HIST_POP_PATH, 'pop_2010.tif')
pop_path2 = os.path.join(FUT_POP_PATH, '2020.tif')
pop_path3 = os.path.join(FUT_POP_PATH, '2030.tif')
file_paths = [pop_path1, pop_path2, pop_path3]

SHP_PATH = os.path.join(BASE_PATH, '..', 'Supplementary Code_3_Create_Emission_Index', 'create_index', 'data_raw', 'shp版', '中国_县_面.shp')
TARGET_GRID_PATH = os.path.join(BASE_PATH, 'code_1_results', 'grid_level_pollutants.csv')

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

# ==================== 4. Batch processing and temporal interpolation ====================
da_list = []
for f in tqdm(file_paths, desc="Spatial processing and aggregation"):
    year = Path(f).stem[-4:]  # Extract 2010, 2020, or 2030.
    da_matched = read_and_align_raster(f, year, target_template, gdf_china)
    da_list.append(da_matched)

# Concatenate the three years along the time dimension.
print("Concatenating data across years...")
ds_combined = xr.concat(da_list, dim='time')

# Optimization 3: interpolate time directly in xarray instead of using the slower pandas melt operation.
print("Interpolating each pixel linearly from 2010 to 2030...")
target_years = list(range(2010, 2031))
ds_interpolated = ds_combined.interp(time=target_years, method='linear')

# ==================== 5. Export the final long-format table ====================
print("Converting to a DataFrame...")
# Convert to a DataFrame and remove missing values.
df_final = ds_interpolated.to_dataframe(name='population').reset_index()
df_final = df_final.rename(columns={'x': 'lon', 'y': 'lat'})

# Inner-join with target_grid to retain only the grid cells required for prediction.
df_final = pd.merge(
    target_df[['lon', 'lat','time']], 
    df_final, 
    on=['lon', 'lat','time'], 
    how='inner'
)
df_final['population'] = df_final['population'].fillna(0)
# Clean the output format.
df_final = df_final.dropna(subset=['population']).reset_index(drop=True)
df_final.groupby(['time']).sum().reset_index()[['population','time']]  # Check for duplicate grid cells.
print(f"Processing complete. Final row count: {len(df_final)}")
print(df_final.head())
OUTPUT_DIR = os.path.join(BASE_PATH, 'code_2_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
df_final.to_csv(os.path.join(OUTPUT_DIR, 'grid_level_population.csv'), index=False)
# %%
