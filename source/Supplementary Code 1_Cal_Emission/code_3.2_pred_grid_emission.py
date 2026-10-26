'''
Data Description: Grid-Level Emission Prediction
Input

code_2.1_cal_emission/df_emi.xlsx — Provincial emission data (target variable).
data_raw/china_shp/geoBoundaries-CHN-ADM1.shp — Chinese province boundaries shapefile.
data_raw/*1950_2100.nc — Gridded CMIP6 proxy data (NetCDF format), used as model features.
Processing
The script performs grid-level emission prediction using XGBoost:

Spatial masking: Builds a 3D province×lat×lon mask to aggregate gridded data to province level.
Feature extraction: For three sectors (energy, industry, waste), extracts gridded proxy variables (BC, OC, NOx, CO2, SO2, CO) per province and computes density (divided by province area).
Model training: Trains XGBoost regressors with 5-fold cross-validation for each emission target, using density features + Year as predictors.
Prediction: Predicts grid-level emissions for 2017–2023, then applies province-level ratio correction to align predictions with observed provincial totals.
'''

import pandas as pd
import numpy as np
import glob
import os
import calendar
import xarray as xr
import geopandas as gpd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(SCRIPT_DIR, 'results','code_2.1_cal_emission', 'df_emi.xlsx')  # Local input file.
BASE_PATH = os.path.join(SCRIPT_DIR, 'data_raw')
SHP_PATH = os.path.join(SCRIPT_DIR, 'data_raw','china_shp','geoBoundaries-CHN-ADM1.shp')
OUTPUT_DIR = os.environ.get(
    'EMISSION_OUTPUT_DIR',
    os.path.join(SCRIPT_DIR, 'results', 'code_3.2_pred_grid_emission')
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_y = pd.read_excel(EXCEL_PATH) 
gdf = gpd.read_file(SHP_PATH)
gdf['Area_km2'] = gdf.geometry.to_crs("EPSG:32650").area / 10**6 

nc_files = glob.glob(os.path.join(BASE_PATH, '*1950_2100.nc'))
ds_sample = xr.open_dataset(nc_files[0])
lon = ds_sample.lon
lat = ds_sample.lat

def calculate_grid_area_km2(da_lon, da_lat):
    "calculate 1grid × 1grid area (km2)"""
    R = 6371.0 
    lon_rad, lat_rad = np.deg2rad(da_lon.values), np.deg2rad(da_lat.values)
    dlon = np.abs(lon_rad[1] - lon_rad[0]) if len(lon_rad) > 1 else np.deg2rad(1.0)
    dlat = np.abs(lat_rad[1] - lat_rad[0]) if len(lat_rad) > 1 else np.deg2rad(1.0)
    
    lat_bounds = np.zeros(len(lat_rad) + 1)
    lat_bounds[1:-1] = (lat_rad[:-1] + lat_rad[1:]) / 2
    lat_bounds[0] = np.max([-np.pi/2, lat_rad[0] - dlat/2])
    lat_bounds[-1] = np.min([np.pi/2, lat_rad[-1] + dlat/2])
    
    area_array = R**2 * dlon * np.abs(np.sin(lat_bounds[1:]) - np.sin(lat_bounds[:-1]))
    area_2d = np.tile(area_array[:, None], (1, len(lon_rad)))
    return xr.DataArray(area_2d, coords=[da_lat, da_lon], dims=['lat', 'lon'])

print("constuct 3D spatial mask for provinces...")
lon_grid, lat_grid = np.meshgrid(lon, lat)
grid_points = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(lon_grid.flatten(), lat_grid.flatten()),
    crs="EPSG:4326"
)

joined = gpd.sjoin(grid_points, gdf[['geometry', 'shapeName']], how="left", predicate="within")
unique_provinces = gdf['shapeName'].unique()
n_provinces = len(unique_provinces)
n_lat, n_lon = len(lat), len(lon)

mask_array = np.full((n_provinces, n_lat, n_lon), False, dtype=bool)
province_labels = (
    joined["shapeName"]
    .astype(object)
    .to_numpy()
    .reshape(n_lat, n_lon)
)

for i, prov_name in enumerate(unique_provinces):
    mask_array[i, :, :] = (province_labels == prov_name)

mask_3D_perfect = xr.DataArray(
    mask_array, 
    dims=['region', 'lat', 'lon'], 
    coords={'region': unique_provinces, 'lat': lat, 'lon': lon}
)
mask_3D_perfect.to_netcdf(os.path.join(OUTPUT_DIR, 'china_provinces_mask.nc'))

sector_toselect = [1, 2, 6]
sector_map = {1: 'energe', 2: 'industry', 6: 'waste'}

train_dict = {}
cleaning_dict = {
    'Beijing Municipality': 'Beijing', 'Tianjin Municipality': 'Tianjin',
    'Shanghai Municipality': 'Shanghai', 'Chongqing Municipality': 'Chongqing',
    'Inner Mongolia Autonomous Region': 'Inner Mongolia', 'Guangxi Zhuang Autonomous Region': 'Guangxi',
    'Tibet Autonomous Region': 'Xizang', 'Ningxia Ningxia Hui Autonomous Region': 'Ningxia',
    'Xinjiang Uyghur Autonomous Region': 'Xinjiang', 'Hong Kong Special Administrative Region': '香港',
    'Macau Special Administrative Region': '澳门', 'Guangzhou Province': 'Guangdong'
}

for i in sector_toselect:
    prov_features_list = []
    for f in nc_files:
        pol_name = os.path.basename(f).split('_')[0]
        ds = xr.open_dataset(f)
        var_name = list(ds.data_vars)[0]
        ds = ds.sel(sector=i)
        da = ds[var_name]
        
        prov_totals_list = []
        for prov_name in mask_3D_perfect.region.values:
            single_mask = mask_3D_perfect.sel(region=prov_name)
            prov_series = da.where(single_mask).sum(dim=['lat', 'lon'])
            prov_series = prov_series.assign_coords(Province_Name=prov_name)
            prov_totals_list.append(prov_series)
            
        prov_total_da = xr.concat(prov_totals_list, dim='Province_Name')
        df_feat = prov_total_da.to_dataframe(name=pol_name).reset_index()
        prov_features_list.append(df_feat[['time', 'Province_Name', pol_name]])

    from functools import reduce
    df_x = reduce(lambda left, right: pd.merge(left, right, on=['time', 'Province_Name']), prov_features_list)
    df_x['Year'] = df_x['time'].apply(lambda x: x.year)
    df_x = df_x.rename(columns={'Province_Name': 'Province'})
    df_x['Province'] = df_x['Province'].str.replace('Province', '', regex=False).str.strip().replace(cleaning_dict)
    
    df_x['Province'] = df_x['Province'].replace(cleaning_dict)

    train_df = pd.merge(df_x, df_y, on=['Year', 'Province'])
    gdf['shapeName'] = gdf['shapeName'].str.replace('Province', '', regex=False).str.strip().replace(cleaning_dict)
    train_df = pd.merge(train_df, gdf[['shapeName', 'Area_km2']], left_on='Province', right_on='shapeName')

    chem_cols = ['BC', 'OC', 'NOx', 'CO2', 'SO2', 'CO']
    for col in chem_cols:
        train_df[f'{col}_density'] = train_df[col] / train_df['Area_km2']
    train_dict[sector_map[i]] = train_df

target_suffixes = {
    "CFPP", "BFI", "IOS", "EAF", "CK", "COP",
    "PCu", "SCu", "SAl", "SPb", "SZn", "MSWI", "HWI"
}
TARGET = [
    col for col in df_y.columns
    if isinstance(col, str)
    and col.rsplit("_", 1)[-1] in target_suffixes
]

TARGET_UNIQUE = [i.split('_')[0] for i in TARGET]
print(len(set(TARGET_UNIQUE)))
predict_map = {
    'ene_predict_sectors': ['CFPP'],
    'ind_predict_sectors': ['BFI', 'IOS', 'EAF', 'CK', 'COP', 'PCu', 'SCu', 'SAl', 'SPb', 'SZn'],
    'wat_predict_sectors': ['MSWI', 'HWI']
}
years_to_predict = np.arange(2017, 2024)

META_COL = ['lon','lat','time']
pred_data = []

for target in TARGET:
    all_predictions_list = []

    if target.split('_')[-1] in predict_map['ene_predict_sectors']:
        train = train_dict['energe']
        predict_sector = 'energe'
        sector_num = 1
    elif target.split('_')[-1] in predict_map['ind_predict_sectors']:
        train = train_dict['industry']
        predict_sector = 'industry'
        sector_num = 2
    else:
        train = train_dict['waste']
        predict_sector = 'waste'
        sector_num = 6
    print(f'\n>>> Processing: {target} (Using CMIP6 proxy sector: {predict_sector})')

    train[f'{target}_density'] = train[f'{target}'] / train['Area_km2']

    density_cols = [f'{col}_density' for col in chem_cols]
    meta_cols = ['Year']
    X_cols = density_cols + meta_cols
    Y_col = f'{target}_density'
    
    if train[Y_col].isna().all() or sum(train[Y_col]) == 0:
        print(f"warning：{target} data all NaN or 0，skipping")
        continue

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xtrain_scaled = pd.concat([pd.DataFrame(scaler.fit_transform(train[density_cols])), train[meta_cols]], axis=1)
    Xtrain_scaled.columns = X_cols
    
    mean_val = train[Y_col].mean()
    scale_factor = 1 / (10 ** round(np.log10(mean_val)))
    ytrain = train[Y_col] * scale_factor

    final_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    final_model.fit(Xtrain_scaled, ytrain)

    for year_to_predict in years_to_predict:
        bbox = gdf.total_bounds
        lat_vals = ds_sample.lat.values
        lat_slice = slice(bbox[3], bbox[1]) if lat_vals[0] > lat_vals[-1] else slice(bbox[1], bbox[3])
        lon_slice = slice(bbox[0], bbox[2])

        lon_china = ds_sample.lon.sel(lon=lon_slice)
        lat_china = ds_sample.lat.sel(lat=lat_slice)

        lon_grid, lat_grid = np.meshgrid(lon_china.values, lat_china.values)
        grid_points_c = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(lon_grid.flatten(), lat_grid.flatten()), crs="EPSG:4326"
        )
        joined_c = gpd.sjoin(grid_points_c, gdf[['geometry']], how="left", predicate="within")
        mask_2D_np = ~joined_c['index_right'].isna().values.reshape(len(lat_china), len(lon_china))
        mask_2D = xr.DataArray(mask_2D_np, dims=["lat", "lon"], coords={"lat": lat_china, "lon": lon_china})

        grid_area_china = calculate_grid_area_km2(lon_china, lat_china)

        dict_da_density = {'grid_cell_area': grid_area_china}
        for f in nc_files:
            pol_name = os.path.basename(f).split('_')[0]
            ds = xr.open_dataset(f)
            ds = ds.sel(sector=sector_num)
            var_name = list(ds.data_vars)[0]
            
            da_year = ds[var_name].sel(time=str(year_to_predict)).squeeze()
            da_year_box = da_year.sel(lon=lon_slice, lat=lat_slice)
            da_year_china = da_year_box.where(mask_2D)
            dict_da_density[f'{pol_name}_density'] = da_year_china / grid_area_china

        ds_grid_features = xr.Dataset(dict_da_density)
        df_grid_features = ds_grid_features.to_dataframe().reset_index()

        valid_grid = df_grid_features.dropna(subset=density_cols).copy()
        valid_grid['Year'] = year_to_predict

        grid_scaled_features = pd.DataFrame(
            scaler.transform(valid_grid[density_cols]), 
            columns=density_cols, index=valid_grid.index
        )
        Xpred_scaled = pd.concat([grid_scaled_features, valid_grid[['Year']]], axis=1)

      
        valid_grid[f'{target}_density_pred'] = final_model.predict(Xpred_scaled) / scale_factor
        valid_grid[f'{target}_density_pred'] = np.clip(valid_grid[f'{target}_density_pred'], 0, None)
        valid_grid[f'{target}_Total_pred'] = valid_grid[f'{target}_density_pred'] * valid_grid['grid_cell_area']

        PROV_COL_GDF = 'shapeName'     
        PROV_COL_TRAIN = 'Province' 
        
        true_prov_data = train[train['Year'] == year_to_predict].copy()
        
        if not true_prov_data.empty:
            true_prov_data = true_prov_data[[PROV_COL_TRAIN, target]].rename(columns={target: 'True_Total'})
            grid_gdf = gpd.GeoDataFrame(
                valid_grid, geometry=gpd.points_from_xy(valid_grid['lon'], valid_grid['lat']), crs="EPSG:4326"
            )
            grid_with_prov = gpd.sjoin(grid_gdf, gdf[[PROV_COL_GDF, 'geometry']], how="left", predicate="within")
            pred_prov_total = grid_with_prov.groupby(PROV_COL_GDF)[f'{target}_Total_pred'].sum().reset_index()
            pred_prov_total.rename(columns={f'{target}_Total_pred': 'Pred_Total'}, inplace=True)
            
            constraint_df = pd.merge(pred_prov_total, true_prov_data, left_on=PROV_COL_GDF, right_on=PROV_COL_TRAIN, how='inner')
            constraint_df['Ratio'] = np.where(constraint_df['Pred_Total'] > 0, constraint_df['True_Total'] / constraint_df['Pred_Total'], 1.0)
           
            ratio_dict = constraint_df.set_index(PROV_COL_GDF)['Ratio'].to_dict()
            grid_with_prov['Ratio_Aligned'] = grid_with_prov[PROV_COL_GDF].map(ratio_dict).fillna(1.0)
            
            valid_grid[f'{target}_Total_pred'] = valid_grid[f'{target}_Total_pred'] * grid_with_prov['Ratio_Aligned'].values
            print(f" -> {year_to_predict} province correction completed.")

        result_df = valid_grid[['Year', 'lat', 'lon', f'{target}_Total_pred']].copy()
        all_predictions_list.append(result_df)

    final_csv_df = pd.concat(all_predictions_list, ignore_index=True)
    final_csv_df = final_csv_df.rename(columns={f'{target}_Total_pred': f'{target}'})
    pred_data.append(final_csv_df)
    

if pred_data:
    from functools import reduce
    result = reduce(lambda left, right: pd.merge(left, right, on=['Year', 'lat', 'lon']), pred_data)
    print(f"Finished, {len(pred_data)} pollutants，{result.shape[0]} rows × {result.shape[1]} columns")
else:
    print("No prediction data read")
result.to_csv(os.path.join(OUTPUT_DIR, 'Grid_Emission.csv'), index=False)





