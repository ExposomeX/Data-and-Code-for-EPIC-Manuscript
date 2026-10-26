#%%
import xarray as xr
import os
import geopandas as gpd
import pandas as pd
import glob
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
GEMS_NC_PATH = os.path.join(BASE_PATH, '..', 'Supplementary Code 1_Cal_Emission', 'data_raw')
CMIP6_NC_PATH = os.path.join(BASE_PATH, 'data_raw', 'cmip6')
NC_PATH = os.path.join(GEMS_NC_PATH, 'BC_annual_total_1950_2100.nc')
SHP_PATH = os.path.join(BASE_PATH, '..', 'Supplementary Code_3_Create_Emission_Index', 'create_index', 'data_raw', 'shp版', '中国_县_面.shp')

META_COL = ['lon','lat','time']
META_DF = None
grid_longs = []
nc_files = glob.glob(os.path.join(GEMS_NC_PATH, '*.nc'))
for f in nc_files:
    p = Path(f).stem.split('_')[0]
    ds = xr.open_dataset(f)
    ds_temp = ds.sel(time=slice('2030', '2100'))
    ds_final = ds_temp.isel(time=slice(None, None, 10))  # Sample every 10 time steps.
    df = ds_final.to_dataframe().reset_index()

    df_wide = df.pivot_table(
        index=['lon', 'lat', 'time'],  # Keep these dimensions as the row index.
        columns='sector',              # Expand sector values into columns.
        values=f'{p}_emission'          # Values to populate the pivot table.
    )

    df_wide.columns = [f'{p}_emission_{col}' for col in df_wide.columns]
    df_wide = df_wide.reset_index()

    df_wide['time'] = [i.year for i in df_wide['time']]
    df_wide = df_wide.sort_values(by='time').reset_index(drop=True)
    
    if META_DF is None:
        META_DF = df_wide[META_COL]
    grid_longs.append(df_wide.iloc[:,3:])

target_grid = pd.read_csv(os.path.join(BASE_PATH, 'code_1_results', 'grid_level_pollutants.csv'))
final_grid = pd.concat(grid_longs, axis=1).reset_index()
final_grid = pd.concat([META_DF, final_grid.iloc[:, 1:]], axis=1)
target_grid = target_grid[['lon','lat']].drop_duplicates().reset_index(drop=True)
final_grid_aligned = target_grid.merge(
    final_grid, 
    on=['lon', 'lat'], 
    how='inner'  # Keep only shared grid cells, retaining the target-grid order.
)
OUTPUT_DIR = os.path.join(BASE_PATH, 'code_3_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
final_grid_aligned.to_csv(os.path.join(OUTPUT_DIR, 'grid_level_proxy_6_pols_fut_ssp245.csv'))
# %%
import xarray as xr
import os
import geopandas as gpd
import pandas as pd
import glob
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
NC_PATH = os.path.join(CMIP6_NC_PATH, 'BC_annual_total_1950_2100_ssp126.nc')
SHP_PATH = os.path.join(BASE_PATH, '..', 'Supplementary Code_3_Create_Emission_Index', 'create_index', 'data_raw', 'shp版', '中国_县_面.shp')

META_COL = ['lon','lat','time']

SSPS = ['ssp126','ssp585','ssp370']

for s in SSPS[0:1]:
    META_DF = None
    grid_longs = []
    nc_files = glob.glob(os.path.join(CMIP6_NC_PATH, f'*{s}.nc'))
    for f in nc_files:
        p = Path(f).stem.split('_')[0]
        ds = xr.open_dataset(f)
        ds_temp = ds.sel(time=slice('2030', '2100'))
        ds_final = ds_temp.isel(time=slice(None, None, 10))  # Sample every 10 time steps.
        df = ds_final.to_dataframe().reset_index()

        df_wide = df.pivot_table(
            index=['lon', 'lat', 'time'],  # Keep these dimensions as the row index.
            columns='sector',              # Expand sector values into columns.
            values=f'{p}_emission'          # Values to populate the pivot table.
        )

        df_wide.columns = [f'{p}_emission_{col}' for col in df_wide.columns]
        df_wide = df_wide.reset_index()

        df_wide['time'] = [i.year for i in df_wide['time']]
        df_wide = df_wide.sort_values(by='time').reset_index(drop=True)
        if META_DF is None:
            META_DF = df_wide[META_COL]
        grid_longs.append(df_wide.iloc[:,3:])
        
    target_grid = pd.read_csv(os.path.join(BASE_PATH, 'code_1_results', 'grid_level_pollutants.csv'))
    final_grid = pd.concat(grid_longs, axis=1).reset_index()
    final_grid = pd.concat([META_DF, final_grid.iloc[:, 1:]], axis=1)
    target_grid = target_grid[['lon','lat']].drop_duplicates().reset_index(drop=True)
    final_grid_aligned = target_grid.merge(
        final_grid, 
        on=['lon', 'lat'], 
        how='inner'  # Keep only shared grid cells, retaining the target-grid order.
    )
    final_grid_aligned.to_csv(os.path.join(OUTPUT_DIR, f'grid_level_proxy_6_pols_fut_{s}.csv'))


# %%
