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
OUTPUT = os.path.join(base_dir,'code_4_results/code_4.1_construct_cei')
os.makedirs(OUTPUT, exist_ok=True)

HIST_POL_PATH = os.path.join(PRED_PATH,'grid_level_emission.csv')
PAIR_PATH = os.path.join(base_dir,'code_3_results/3_Final_Top50pct_Categorized_Targets.csv')
STATISIC_PATH = os.path.abspath(os.path.join(base_dir,'../../Supplementary Code_2_Epidemic_Association/demo_data/df_hospitalization_for_association_demo.csv'))
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
df_hist = pd.read_csv(os.path.join(base_dir,'data_raw/grid_level_emission.csv'))

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

print("\n>>> Aggregating grid-level data to counties...")
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
final_result['B_county_id'] = final_result['B_county_id'].astype(str)
df_year = pd.read_csv(STATISIC_PATH)
df_year['B_county_id'] = df_year['B_county_id'].astype(str)
merged_result = pd.merge(df_year,final_result[['B_county_id','B_year']+TARGET_VAL_COL],on=['B_county_id','B_year'],how='left')
merged_result.to_csv(os.path.join(OUTPUT,'2_merged_yearly_data_new_demo.csv'))
print('Saved')
# %%
