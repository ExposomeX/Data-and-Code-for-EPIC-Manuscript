#%%
import pandas as pd
import os
base_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_excel(os.path.join(base_dir,'data_raw/disease_endpoint.xlsx'))
endpoint_selected = set(df['endpoint'])

df_raw = pd.read_csv(os.path.join(base_dir,'data_raw/carbazole_toxicity/results/df_endpoint.csv'))
endpoint_all = set(df_raw.columns[5:])

selected_in_all = endpoint_selected.issubset(endpoint_all)

endpoint_not_selected = endpoint_all - endpoint_selected
print(f"\nOrginal endpoints：{len(endpoint_all)}")
print(f"Selected endpoints：{len(endpoint_selected)}")
print(f"Unselected endpoints：{len(endpoint_not_selected)}")
print("\nUnselected endpoints list：")
for ep in sorted(endpoint_not_selected):
    print(f"  - {ep}")

selected_not_in_all = endpoint_selected - endpoint_all




# %% caculate index
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df_mapping = pd.read_excel(os.path.join(base_dir,'data_raw/disease_endpoint.xlsx'))

df_raw = pd.read_excel(os.path.join(base_dir,'data_raw/carbazole_toxicity/data_raw/carbazole_deeptoxicity.xlsx'))
df_raw = df_raw.iloc[2:, :].reset_index(drop=True)

meta_cols = ['smiles', 'Name_Abbr']

df_disease_scores = df_raw[meta_cols].copy()
diseases = df_mapping['disease'].unique()

print(">>> Calculate DTL...")

for disease in diseases:
    endpoints = df_mapping[df_mapping['disease'] == disease]['endpoint'].tolist()
    disease_prob_matrix = pd.DataFrame(index=df_raw.index)
    
    for ep in endpoints:
        if ep in df_raw.columns:
            uncert_col = f"{ep}_uncertainty"
            prob_series = pd.to_numeric(df_raw[ep], errors='coerce').fillna(0)
        
            if uncert_col in df_raw.columns:
                is_low_conf = df_raw[uncert_col].astype(str).str.contains('low', case=False, na=False)
                disease_prob_matrix[ep] = np.where(is_low_conf, 0.0, prob_series)
            else:
                disease_prob_matrix[ep] = prob_series
    
    if not disease_prob_matrix.empty:
        df_disease_scores[disease] = disease_prob_matrix.max(axis=1)
    else:
        df_disease_scores[disease] = 0.0
        print(f"  Warning ")

print(df_disease_scores.head())

output_path = os.path.join(base_dir,'code_2_results')
os.makedirs(output_path, exist_ok=True)
df_disease_scores.to_csv(os.path.join(output_path,'Disease_Specific_DeepTox_Scores.csv'), index=False)


plot_data = df_disease_scores.set_index('Name_Abbr').drop(columns=['smiles'])

plot_data = plot_data.loc[:, (plot_data != 0).any(axis=0)]

plt.figure(figsize=(14, max(8, len(plot_data) * 0.3))) 
sns.heatmap(
    plot_data, 
    cmap="YlOrRd",           
    vmin=0, vmax=1,          
    linewidths=0.5, 
    linecolor='lightgray',
    annot=False,            
    cbar_kws={'label': 'DTL (Max-Risk Probability)'}
)

plt.title('DTL (Filtered by High-Confidence)', fontsize=16, pad=20)
plt.xlabel('Endpoint', fontsize=12)
plt.ylabel('Target exposure', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

fig_path = os.path.join(output_path,'DeepTox_Heatmap.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.show()
