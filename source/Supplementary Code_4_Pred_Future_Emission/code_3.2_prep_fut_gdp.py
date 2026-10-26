#%%
# PPP 2005 international dollars
import xarray as xr
import os
import geopandas as gpd
import pandas as pd
import glob
from pathlib import Path
import rioxarray as rxr
import numpy as np

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
GDP_DATA_PATH = os.path.join(BASE_PATH, 'data_raw', 'gdp', 'GDP_025d (2000-2100)', '025d')
SHP_PATH = os.path.join(BASE_PATH, '..', 'Supplementary Code_3_Create_Emission_Index', 'create_index', 'data_raw', 'shp版', '中国_县_面.shp')
SSPS = ['ssp1','ssp2','ssp3','ssp5']
FILES = ['GDP2030.tif','GDP2040.tif','GDP2050.tif','GDP2060.tif','GDP2070.tif','GDP2080.tif','GDP2090.tif','GDP2100.tif']

META_COL = ['lon','lat','time']
p = 'gdp'

for ssp in SSPS:
    print("Reading rasters for all years and building the spatial panel...")
    all_dfs = []

    for f in FILES:
        year = int(f.split('.')[0][3:7])  # Extract the year from a filename such as 'GDP2017.tif'.
        ds = rxr.open_rasterio(os.path.join(GDP_DATA_PATH, f'{f[:7]}_{ssp}.tif')).squeeze()
        ds_05 = ds.coarsen(x=2, y=2, boundary='trim').sum()
    
        # Convert to a DataFrame.
        df_temp = ds_05.to_dataframe(name='GDP').reset_index()
        
        # Standardize coordinate names (rioxarray commonly uses x/y; rename them to lon/lat).
        df_temp = df_temp.rename(columns={'x': 'lon', 'y': 'lat'})
        df_temp['time'] = year
        
        # Keep only the required columns.
        all_dfs.append(df_temp[['lon', 'lat', 'time', 'GDP']])
        print(f"   - Processed {f}: {len(df_temp)} 0.5-degree grid cells.")
      
    
    df_all = pd.concat(all_dfs, ignore_index=True)
        
    target_grid = pd.read_csv(os.path.join(BASE_PATH, 'code_3_results', 'grid_level_proxy_6_pols_fut_ssp245.csv'))
    final_result = pd.merge(target_grid,df_all,on=['lon','lat','time'],how='left').dropna()
    OUTPUT_DIR = os.path.join(BASE_PATH, 'code_3_results')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    final_result.to_csv(os.path.join(OUTPUT_DIR, f'grid_level_gdp_{ssp}.csv'), index=False)
    
    
    print("Processing complete. Future county-level GDP data are ready.")

# %%
