#%%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

base_path = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_path,'code_3_results')
os.makedirs(output_dir, exist_ok=True)

tot_path = os.path.join(base_path,'code_1_tot_results/3_combined_2_pds_data.csv')
carbazole_path = os.path.join(base_path,'code_2_results/Disease_Specific_DeepTox_Scores.csv')

df_tot = pd.read_csv(tot_path)
df_carbazole = pd.read_csv(carbazole_path)

id_cols = ['smiles', 'Name_Abbr']

df_long = df_carbazole.melt(
    id_vars=id_cols,          
    var_name='Disease_Name',       
    value_name='DTL_Score'      
)

df_long.rename(columns={'Name_Abbr':'name_en_abbr'},inplace=True)
df_final = pd.merge(df_tot,df_long.iloc[:,1:],on=['name_en_abbr','Disease_Name'],how='left')

cols_to_check = ['PDS_EPPD', 'PDS_EGoD', 'DTL_Score']
nan_rows = df_final[df_final[cols_to_check].isna().all(axis=1)]
nan_rows['Exposure'].unique()
print(nan_rows)  

import matplotlib.pyplot as plt
plt.hist(df_final['PDS_EPPD'],label='EPPD')
plt.hist(df_final['PDS_EGoD'],label='EGoD')
plt.legend()
plt.show()

plt.hist(np.log1p(df_final['PDS_EPPD']),label='EPPD')
plt.hist(np.log1p(df_final['PDS_EGoD']),label='EGoD')
plt.legend()
plt.show()

df_final['log_PDS_EPPD'] = np.log1p(df_final['PDS_EPPD'])
df_final['log_PDS_EGoD'] = np.log1p(df_final['PDS_EGoD'])

data_subset = df_final[['log_PDS_EPPD', 'log_PDS_EGoD']].dropna()

corr_coef, p_value = stats.pearsonr(data_subset['log_PDS_EPPD'], data_subset['log_PDS_EGoD'])

print(f"🔢 Pearson (r): {corr_coef:.4f}")
print(f"📉 P-value: {p_value:.2e}")

df_final.to_csv(os.path.join(output_dir,'1_combined_all_method_score.csv'))
#%%
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path,'code_3_results/1_combined_all_method_score.csv')
df_final = pd.read_csv(file_path)

if 'sub category_en' in df_final.columns:
    df_final['sub category_en'] = df_final['sub category_en'].fillna('Unknown')
group_cols = ['Disease_Name', 'sub category_en']

def min_max_scale(series):
    s_valid = series.dropna()
    if len(s_valid) == 0:
        return series
    
    s_min = s_valid.min()
    s_max = s_valid.max()
    
    if s_max == s_min:
        return series.apply(lambda x: 1.0 if pd.notna(x) and x > 0 else (0.0 if pd.notna(x) else np.nan))
        
    return (series - s_min) / (s_max - s_min)

df_final['S_eppd'] = df_final.groupby(group_cols)['PDS_EPPD'].transform(min_max_scale)
df_final['S_egod'] = df_final.groupby(group_cols)['PDS_EGoD'].transform(min_max_scale)
df_final['S_dtl']  = df_final.groupby(group_cols)['DTL_Score'].transform(min_max_scale)

df_final['S_eppd'] = df_final['S_eppd'].fillna(0)
df_final['S_egod'] = df_final['S_egod'].fillna(0)
df_final['S_dtl']  = df_final['S_dtl'].fillna(0)

df_final['S_KG'] = df_final[['S_eppd', 'S_egod']].max(axis=1)

df_final['Unified_Bio_Score'] = 1 - (1 - df_final['S_KG']) * (1 - df_final['S_dtl'])

print(">>> Priority Index...")

df_final['Epi_Score'] = df_final['LogRR'] * (-np.log10(df_final['P_value'] + 1e-300))
df_final.loc[df_final['Epi_Score'] < 0, 'Epi_Score'] = 0

df_final['Priority_Index'] = df_final['Epi_Score'] * df_final['Unified_Bio_Score']

df_final = df_final.sort_values(by=['Disease_Name', 'sub category_en', 'Priority_Index'], 
                                ascending=[True, True, False])

df_priority = df_final[df_final['Priority_Index'] > 0].copy()
output_file = os.path.join(output_dir,'2_Integrated_Priority_Targets_Categorized_min_max.csv')
df_priority.to_csv(output_file, index=False)

data_subset = df_priority[['S_eppd', 'S_egod']].dropna()

corr_coef, p_value = stats.pearsonr(data_subset['S_eppd'], data_subset['S_egod'])

print(f"🔢 Pearson (r): {corr_coef:.4f}")
print(f"📉 P-value: {p_value:.2e}")

#%%
import pandas as pd
import os

file_path = os.path.join(output_dir,'2_Integrated_Priority_Targets_Categorized_min_max.csv')
df = pd.read_csv(file_path)
df = df[df.Significance_raw =='Significant Harm']

df = df.sort_values(by=['Disease_Name', 'sub category_en', 'Priority_Index'], 
                    ascending=[True, True, False])

group_cols = ['Disease_Name', 'sub category_en']
df['Group_Total_Risk'] = df.groupby(group_cols)['Priority_Index'].transform('sum')
df['Contribution_Pct'] = df['Priority_Index'] / df['Group_Total_Risk']
df['Cum_Contribution_Pct'] = df.groupby(group_cols)['Contribution_Pct'].cumsum()

def extract_top_50_percent(group):
    mask = group['Cum_Contribution_Pct'].shift(1).fillna(0) < 0.50
    return group[mask]

df_top50 = pd.concat(
    [extract_top_50_percent(group) for _, group in df.groupby(group_cols)],
    ignore_index=True,
)

output_file = os.path.join(output_dir,'3_Final_Top50pct_Categorized_Targets.csv')
df_top50.to_csv(output_file, index=False)

summary = df_top50.groupby(['Disease_Name', 'sub category_en']).size().reset_index(name='Count')


#%%
selected_pols = df_top50['Exposure'].unique()
pd.DataFrame(selected_pols).to_csv(os.path.join(output_dir,'4_selected_pols_list.csv'))


# %%
