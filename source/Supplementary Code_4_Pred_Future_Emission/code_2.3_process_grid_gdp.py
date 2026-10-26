#%%
import xarray as xr
import os
import pandas as pd
import numpy as np
import rioxarray as rxr

# ==========================================
# Configuration parameters
# ==========================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
GDP_DATA_PATH = os.path.join(BASE_PATH, 'data_raw', 'gdp', 'GDP_025d (2000-2100)', '025d')
FILES = ['GDP2017.tif','GDP2018.tif','GDP2019.tif','GDP2020.tif','GDP2025_ssp2.tif']
TARGET_YEARS = list(range(2017, 2026)) # 2017-2025

# ==========================================
# 1. Read and resample directly to 0.5 degrees.
# ==========================================
print(">>> Reading the original 0.25-degree data and aggregating to 0.5 degrees...")
all_dfs = []

for f in FILES:
    file_path = os.path.join(GDP_DATA_PATH, f)
    
    # Extract the year.
    try:
        year = int(f.split('.')[0][3:7]) 
    except:
        print(f"Could not parse the year from filename {f}; skipping.")
        continue
        
    # Open the raster.
    ds = rxr.open_rasterio(file_path).squeeze()
    
    # Key step: coarsen the raster.
    # Since 0.5 / 0.25 = 2, combine each 2 × 2 block of pixels.
    # GDP is an extensive quantity, so aggregate with sum().
    ds_05 = ds.coarsen(x=2, y=2, boundary='trim').sum()
    
    # Convert to a DataFrame.
    df_temp = ds_05.to_dataframe(name='GDP').reset_index()
    
    # Standardize coordinate names (rioxarray commonly uses x/y; rename them to lon/lat).
    df_temp = df_temp.rename(columns={'x': 'lon', 'y': 'lat'})
    df_temp['time'] = year
    
    # Keep only the required columns.
    all_dfs.append(df_temp[['lon', 'lat', 'time', 'GDP']])
    print(f"   - Processed {f}: {len(df_temp)} 0.5-degree grid cells.")

# ==========================================
# 2. Combine data for all years.
# ==========================================
df_all_years = pd.concat(all_dfs, ignore_index=True)

# ==========================================
# 3. Interpolate the time series to fill missing years.
# ==========================================
print(">>> Linearly interpolating the time series (2017–2025)...")

# Convert to wide format for interpolation.
# Index: (lon, lat), Columns: time
df_wide = df_all_years.pivot_table(index=['lon', 'lat'], columns='time', values='GDP')

# Include all target years; missing values are filled with NaN.
df_wide = df_wide.reindex(columns=TARGET_YEARS)

# Interpolate linearly across columns (axis=1).
df_wide_interpolated = df_wide.interpolate(method='linear', axis=1, limit_direction='both')

# ==========================================
# 4. Convert back to long format.
# ==========================================
print(">>> Converting back to long format...")

df_final = df_wide_interpolated.reset_index().melt(
    id_vars=['lon', 'lat'],
    var_name='time',
    value_name='GDP'
)

# Convert data types.
df_final['time'] = df_final['time'].astype(int)

# Preview the results.
print("\n>>> Final result preview:")
print(df_final.head())
print(f"\nTotal rows: {len(df_final)}")
print(f"Coordinate-spacing check (expected to be approximately 0.5): {df_final['lon'].diff().drop_duplicates().values[:5]}")

target_grid = pd.read_csv(os.path.join(BASE_PATH, 'code_1_results', 'grid_level_pollutants.csv'))
final_result = pd.merge(target_grid,df_final,on=['lon','lat','time'],how='left').dropna()
OUTPUT_DIR = os.path.join(BASE_PATH, 'code_2_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
final_result.to_csv(os.path.join(OUTPUT_DIR, 'grid_level_gdp.csv'), index=False)

# Save the results (optional).
# df_final.to_csv('GDP_05d_panel.csv', index=False)
# %%
