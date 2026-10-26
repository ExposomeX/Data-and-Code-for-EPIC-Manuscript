#%%
# Example: Waste_BC.
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

ds = xr.open_dataset(NC_PATH)
ds_final = ds.sel(sector=6).sel(time=slice('2017', '2023'))
df = ds_final.to_dataframe().reset_index()
df['time'] = [i.year for i in df['time']]
df = df.sort_values(by='time')

grid_longs = []
nc_files = glob.glob(os.path.join(GEMS_NC_PATH, '*.nc'))
Meta_df = None
for f in nc_files:
    p = Path(f).stem.split('_')[0]
    ds = xr.open_dataset(f)
    ds_final = ds.sel(time=slice('2017', '2023'))
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
    if Meta_df is None:
        Meta_df = df_wide[['lon', 'lat', 'time']].drop_duplicates().reset_index(drop=True)
    grid_longs.append(df_wide.iloc[:,3:])

target_grid = pd.read_csv(os.path.join(BASE_PATH, 'code_1_results', 'grid_level_pollutants.csv'))
final_grid = pd.concat(grid_longs, axis=1).reset_index()
final_grid = pd.concat([Meta_df, final_grid.iloc[:, 1:]], axis=1)
final_result = pd.merge(target_grid,final_grid,on=['lon','lat','time'],how='left').dropna()

OUTPUT_DIR = os.path.join(BASE_PATH, 'code_2_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
final_result.to_csv(os.path.join(OUTPUT_DIR, 'grid_level_proxy_6_pols_ssp245.csv'))
# %%
#%%
# Example: Waste_BC.
import xarray as xr
import os
import geopandas as gpd
import pandas as pd
import glob
from pathlib import Path

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
NC_PATH = os.path.join(CMIP6_NC_PATH, 'BC_annual_total_1950_2100_ssp126.nc')

ds = xr.open_dataset(NC_PATH)
ds_final = ds.sel(sector=6).sel(time=slice('2017', '2023'))
df = ds_final.to_dataframe().reset_index()
df['time'] = [i.year for i in df['time']]
df = df.sort_values(by='time')

SSPS = ['ssp126','ssp370']
for ssp in SSPS:
    grid_longs = []

    Meta_df = None
    nc_files = glob.glob(os.path.join(CMIP6_NC_PATH, f'*{ssp}.nc'))
    for f in nc_files:
        p = Path(f).stem.split('_')[0]
        ds = xr.open_dataset(f)
        ds_final = ds.sel(time=slice('2017', '2023'))
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
        if Meta_df is None:
            Meta_df = df_wide[['lon', 'lat', 'time']].drop_duplicates().reset_index(drop=True)
        grid_longs.append(df_wide.iloc[:,3:])

    target_grid = pd.read_csv(os.path.join(BASE_PATH, 'code_1_results', 'grid_level_pollutants.csv'))
    final_grid = pd.concat(grid_longs, axis=1).reset_index()
    final_grid = pd.concat([Meta_df, final_grid.iloc[:, 1:]], axis=1)
    final_result = pd.merge(target_grid,final_grid,on=['lon','lat','time'],how='left').dropna()

    final_result.to_csv(os.path.join(OUTPUT_DIR, f'grid_level_proxy_6_pols_{ssp}.csv'))
    # %%
