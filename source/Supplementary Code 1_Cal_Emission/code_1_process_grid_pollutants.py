#%%
import pandas as pd
import os
import geopandas as gpd
import numpy as np


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH_LIU = os.path.join(BASE_PATH, 'results', 'grid_level_pollutants')
BASE_PATH_GEMS = os.path.join(BASE_PATH, 'data_raw')
BASE_PATH_GEMS_FUT = os.path.join(BASE_PATH, 'results', 'code_4.1_predict_PAH')
pollutant_list = pd.read_excel(os.path.join(BASE_PATH_GEMS, 'pollutant list.xlsx'))
SHP_PATH = os.path.join(BASE_PATH_GEMS, 'China_shp', '中国_县_面.shp')

HIST_GEMS_DF = pd.read_csv(os.path.join(BASE_PATH_GEMS,'0.5deg_gems_emission.csv'))
FUT_GEMS_DF = pd.read_csv(os.path.join(BASE_PATH_GEMS_FUT, 'Predicted_PAH_2022_2024.csv'))
SECTORS_LIU = ['BFI','CFPP','CK','COP','HWI','IOS','MSWI','PCu','SAl','SCu','SZn','EAF','SPb']
SECTORS_GEMS = ['ipr','ico','pow']

def change_position(df):
    df.columns = [
        c.replace('_pred', '') if '_pred' in c else c  # Remove _pred before processing; alternatively, include it in the regular expression.
        for c in df.columns
    ]
    new_columns = []
    for col in df.columns:
        import re
        match = re.match(r'^(\w+)_(\w+-?\w*)$', col) 

        if match:
            prefix = match.group(1)   # ipr
            pollutant = match.group(2) # BaA
            new_col = f"{pollutant}_{prefix}" # BaA_ipr_pred
            new_columns.append(new_col)
        else:
            new_columns.append(col)

    df.columns = new_columns
    return df

HIST_GEMS_DF = change_position(HIST_GEMS_DF)
FUT_GEMS_DF = change_position(FUT_GEMS_DF)
POL = pollutant_list['pollutant abbr.'].tolist()
# 3. Build pollutant–sector combinations
final_combinations = []

for index, row in pollutant_list.iterrows():
    source_type = row['source']
    # Get the pollutant identifier (the pollutant abbreviation is used here).
    # To use the full name from the 'pollutants' column for GEMS, replace this line with:
    # base_name = row['pollutants'] if source_type == 'GEMS' else row['pollutant_abbr.']
    base_name = str(row['pollutant abbr.']) 
    
    if source_type == 'L':
        # Liu inventory: combine with SECTORS_LIU.
        for sector in SECTORS_LIU:
            final_combinations.append(f"{base_name}_{sector}")
            
    elif source_type == 'GEMS':
        # GEMS inventory: combine with SECTORS_GEMS.
        for sector in SECTORS_GEMS:
            final_combinations.append(f"{base_name}_{sector}")

# 4. Report the generated combinations
print(f"Generated {len(final_combinations)} pollutant–sector combinations:\n")

META_COL = ['lon','lat','time']
META_GEMS_DF = pd.concat([HIST_GEMS_DF[META_COL],FUT_GEMS_DF[META_COL]],axis=0).reset_index(drop=True)

p = 'As_CK'
data = pd.read_csv(os.path.join(BASE_PATH_LIU,f'{p}/{p}_years_prediction.csv'))
data.rename(columns={'Year':'time'},inplace=True)
data.rename(columns={f'{p}_Total_pred':f'{p}'})

META_LIU_DF = data[META_COL]

grid_longs_gems = []
grid_longs_liu = []
for p in final_combinations:
    if p.endswith('pow') or p.endswith('ico') or p.endswith('ipr'):
        try:
            data_hist = HIST_GEMS_DF[f'{p}']
            if np.std(HIST_GEMS_DF[f'{p}'].values)==0:
                print(f'{p}: no variation in the data; skipped')
                continue
            data_fut = FUT_GEMS_DF[f'{p}']
            data = pd.concat([data_hist,data_fut],axis=0).reset_index(drop=True)
            grid_longs_gems.append(data)
        except:
            print(f'{p}: data not found; skipped')
            continue
    else:
        file_path = os.path.join(BASE_PATH_LIU,f'{p}/{p}_years_prediction.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            data.rename(columns={'Year':'time'},inplace=True)
            data.rename(columns={f'{p}_Total_pred':f'{p}'},inplace=True)
            grid_longs_liu.append(data[f'{p}'])
        
final_grid_gems = pd.concat(grid_longs_gems, axis=1).reset_index()
final_grid_gems = pd.concat([META_GEMS_DF,final_grid_gems],axis=1)
final_result_gems = final_grid_gems[final_grid_gems['time'].between(2017, 2023)].reset_index(drop=True) 

final_grid_liu = pd.concat(grid_longs_liu, axis=1).reset_index()
final_grid_liu = pd.concat([META_LIU_DF,final_grid_liu],axis=1)

final_result = pd.merge(final_grid_liu,final_result_gems,on=['time','lon','lat'],how='inner')
OUTPUT_DIR = os.path.join(BASE_PATH, 'results', 'code_1_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)
final_result.to_csv(os.path.join(OUTPUT_DIR, 'grid_level_pollutants.csv'), index=False)
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = pd.read_csv(os.path.join(OUTPUT_DIR, 'grid_level_pollutants.csv'))

def plot_single_year(df, target_year=2017, col_name='BaP_ico'):
    # Select the requested year.
    data_year = df[df['time'] == target_year]
    
    if data_year.empty:
        print(f"No data found for {target_year}.")
        return

    # Reshape the data into a grid matrix (pivot table).
    # Longitude is on the x-axis, latitude is on the y-axis, and values are taken from the target column.
    grid_data = data_year.pivot_table(index='lat', columns='lon', values=col_name)
    
    # Get coordinate vectors.
    lons = grid_data.columns.values
    lats = grid_data.index.values
    
    # 2. Create the map.
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())  # Set the map projection.
    
    # Add geographic features such as coastlines and provincial boundaries.
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=0.5)
    
    # Draw the gridded field with pcolormesh.
    mesh = ax.pcolormesh(
        lons, lats, grid_data.values,
        transform=ccrs.PlateCarree(),
        cmap='RdYlBu_r',  # Reversed red–yellow–blue palette for concentration/emission data.
        shading='auto'    # Automatically handle grid-cell edges.
    )
    
    # Add the colorbar.
    cbar = plt.colorbar(mesh, orientation='vertical', pad=0.05, shrink=0.8)
    cbar.set_label(f'{col_name} Level', fontsize=12)
    
    # Set the title and map extent.
    ax.set_title(f'Spatial Distribution of {col_name} ({target_year})', fontsize=16)
    #ax.set_extent([105, 130, 15, 55], crs=ccrs.PlateCarree())  # Approximate extent based on the reference image.
    
    plt.show()

# Generate the maps.
plot_single_year(df, target_year=2017)
# %%
