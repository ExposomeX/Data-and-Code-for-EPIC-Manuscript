#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
import os

# Set working directory (using relative path)
base_dir = os.path.dirname(os.path.abspath(__file__))
ebio_dir = os.path.join(os.path.dirname(base_dir),'ebio_network')
statistic_dir = os.environ.get(
    'EWAS_RESULTS_DIR',
    os.path.join(
        os.path.dirname(os.path.dirname(base_dir)),
        'Supplementary Code_2_Epidemic_Association', 'code_1_results'
    )
)
output_path = os.path.join(base_dir, 'code_1_tot_results')
os.makedirs(output_path, exist_ok=True)

#pols data
id_corrected_df = pd.read_excel(os.path.join(ebio_dir, 'id_corrected.xlsx'))
id_corrected_df.rename(columns={'unified_id': 'Exposure_ID'}, inplace=True)

#statistics results
EWAS_df = pd.read_csv(os.path.join(statistic_dir, 'EWAS_Sensitivity_Results_All.csv'))
EWAS_df = EWAS_df[EWAS_df.Model_Type == 'M4_Full'].reset_index(drop=True)

def get_abbr(name):
    parts = name.split('_')
    if len(parts) > 2:
        return '_'.join(parts[1:-1])
    else:
        return parts[-1]

EWAS_df['name_en_abbr'] = EWAS_df['Exposure'].apply(get_abbr)
EWAS_df.rename(columns={'Outcome': 'Disease_Name'}, inplace=True)
keep_exp = id_corrected_df[id_corrected_df['Exposure_ID'].str.startswith('EX:E')]
id_df = keep_exp.drop_duplicates(subset='Exposure_ID')

df_final = pd.merge(EWAS_df,id_df[['name_en_abbr','catogory_en','sub category_en','Exposure_ID']],on='name_en_abbr',how='left')
df_final.to_csv(os.path.join(output_path, '1_statistic_data_with_info.csv'), index=False)

# %% heatmaps
df = df_final.copy()
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Sans', 'Liberation Sans']
category_col = 'sub category_en'  
exposure_col = 'Exposure'         
disease_col = 'Label'             
sig_col = 'Significance_raw'      
effect_col = 'LogRR'              

df['Plot_Value'] = np.where(
    df[sig_col].astype(str).str.contains('Not', case=False, na=True), 
    np.nan, 
    df[effect_col]
)

max_val = df['Plot_Value'].abs().quantile(0.90)
if np.isnan(max_val) or max_val == 0: max_val = 1.0 
df['Plot_Value'] = df['Plot_Value'].clip(-max_val, max_val)
df = df.sort_values(by='sub category_en',ascending=True)

exp_categories = df[category_col].dropna().unique()
diseases = df[disease_col].dropna().unique()
diseases.sort()
dict_exp = {cat: df[df[category_col] == cat][exposure_col].dropna().unique() for cat in exp_categories}
exp_counts = [len(dict_exp[c]) for c in exp_categories]
total_unique_exposures = sum(exp_counts)

y_colors = sns.color_palette("Set2", len(exp_categories))
dict_cat_color_y = dict(zip(exp_categories, y_colors))
x_colors = sns.color_palette("husl", len(diseases))
dict_cat_color_x = dict(zip(diseases, x_colors))

fig_height = min(6, total_unique_exposures * 0.05) 
fig_width = 3

fig, axes = plt.subplots(
    nrows=len(exp_categories), 
    ncols=1, 
    figsize=(fig_width, fig_height),
    gridspec_kw={'height_ratios': exp_counts, 'hspace': 0.015} 
)
if len(exp_categories) == 1: axes = [axes]

custom_colors = ["#3062A5", "white", "#D0332B"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_RdBu", custom_colors)
cmap.set_bad(color='#e0e0e0') 

norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)

for i, exp_cat in enumerate(exp_categories):
    ax = axes[i]
    
    subset = df[df[category_col] == exp_cat]
    pivot_df = subset.pivot(index=exposure_col, columns=disease_col, values='Plot_Value')
    pivot_df = pivot_df.reindex(index=dict_exp[exp_cat], columns=diseases)
    
    sns.heatmap(
        pivot_df, ax=ax, cmap=cmap, norm=norm, cbar=False,
        linewidths=0,       
        yticklabels=False,  
        xticklabels=(i == len(exp_categories) - 1),
        rasterized=True 
    )
    ax.get_xaxis().set_label_text('') 
   
    rect_y = Rectangle((-0.04, 0), 0.03, 1, transform=ax.transAxes, 
                       color=dict_cat_color_y[exp_cat], clip_on=False)
    ax.add_patch(rect_y)
    ax.set_ylabel(exp_cat, fontsize=7, rotation=0, ha='right', va='center', labelpad=12)
    
    if i == len(exp_categories) - 1:
        ax.tick_params(axis='x', rotation=90, labelsize=7, pad=6) 
        ax.get_xaxis().set_label_text('') 
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for j, disease in enumerate(diseases):
            rect_x = Rectangle((j, -0.06), 1, 0.04, transform=trans, 
                               color=dict_cat_color_x[disease], clip_on=False)
            ax.add_patch(rect_x)

cbar_ax = fig.add_axes([0.91, 0.35, 0.02, 0.3]) # 让 colorbar 变修长
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Adjusted LogRR\n(Significant Only)', fontweight='bold', fontsize=7)
cbar.outline.set_visible(False) 
plt.subplots_adjust(left=0.35, right=0.88, top=0.88, bottom=0.15)

plt.savefig(os.path.join(output_path, 'plot1_Phenome_Exposome_Atlas.png'), dpi=600, bbox_inches='tight', facecolor='white')
if os.environ.get('PRIORITY_SHOW_PLOTS') == '1':
    plt.show()




# %%
import pandas as pd
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import seaborn as sns

df_sig = df_final[~df_final['Significance_raw'].astype(str).str.contains('Not', case=False, na=True)].copy()
df_sig = df_sig.dropna()
categories = df_sig['sub category_en'].dropna().unique().tolist()
exposures = df_sig['Exposure'].dropna().unique().tolist()
diseases = df_sig['Label'].dropna().unique().tolist()

all_nodes = categories + exposures + diseases
node_indices = {node: i for i, node in enumerate(all_nodes)}

exp_counts = df_sig['Exposure'].value_counts()

top_20_exposures = exp_counts.head(20).index.tolist()

display_labels = [] 
for node in all_nodes:
    if node in categories or node in diseases:
        display_labels.append(node) 
    elif node in top_20_exposures:
        display_labels.append(node) 
    else:
        display_labels.append("")   

palette = sns.color_palette("Set2", len(categories)).as_hex()
cat_color_map = dict(zip(categories, palette))

node_colors = []
for node in all_nodes:
    if node in categories:
        node_colors.append(cat_color_map[node])
    elif node in exposures:
        cat = df_sig[df_sig['Exposure'] == node]['sub category_en'].iloc[0]
        node_colors.append(cat_color_map[cat])
    else:
        node_colors.append('#2C3E50')

source, target, value, link_colors = [], [], [], []

link_1 = df_sig.groupby(['sub category_en', 'Exposure']).size().reset_index(name='count')
for _, row in link_1.iterrows():
    source.append(node_indices[row['sub category_en']])
    target.append(node_indices[row['Exposure']])
    value.append(row['count'])
    rgb = mcolors.to_rgb(cat_color_map[row['sub category_en']])
    link_colors.append(f"rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, 0.4)")

link_2 = df_sig.groupby(['Exposure', 'Label', 'Significance_raw']).size().reset_index(name='count')
for _, row in link_2.iterrows():
    source.append(node_indices[row['Exposure']])
    target.append(node_indices[row['Label']])
    value.append(row['count'])
    
    if 'Harm' in row['Significance_raw']:
        link_colors.append("rgba(208, 51, 43, 0.45)")  
    elif 'Protect' in row['Significance_raw']:
        link_colors.append("rgba(48, 98, 165, 0.45)")  
    else:
        link_colors.append("rgba(200, 200, 200, 0.3)")

fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(
        pad=1,         
        thickness=12,  
        line=dict(color="white", width=0.1),
        label=display_labels,  
        customdata=all_nodes, 
        hovertemplate="<b>%{customdata}</b><br>Connections: %{value}<extra></extra>", 
        color=node_colors
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_colors
    )
)])

fig.update_layout(
    title=dict(
        text="<b>Exposome-Phenome Network Flow</b><br><sup>Tracing significant epidemiological pathways (Top targets labeled)</sup>",
        font=dict(family="Arial", size=20, color="black"),
        x=0.01, y=0.98
    ),
    font=dict(family="Arial", size=11, color="#333333"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=1100,     
    height=800,      
    margin=dict(l=20, r=120, t=60, b=20) 
)

if os.environ.get('PRIORITY_SHOW_PLOTS') == '1':
    fig.show()
try:
    fig.write_image(os.path.join(output_path,"plot2_Network_Flow_Sankey.png"), scale=4)
except Exception as exc:
    print(f"Sankey image export skipped ({exc}); score calculation will continue.")


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
import os

# Set working directory (using relative path)
base_dir = os.path.dirname(os.path.abspath(__file__))
ebio_dir = os.path.join(os.path.dirname(base_dir),'ebio_network')
pols_data = pd.read_csv(os.path.join(base_dir,'code_1_tot_results','1_statistic_data_with_info.csv'))
pols_data['Disease_Name'] = [name[2:] for name in pols_data['Disease_Name']]
pols_data = pols_data[~(pols_data['Disease_Name']=='All_cause')].reset_index(drop=True)
for PATHWAY in ('EPPD', 'EGoD'):
    csv_file = os.path.join(
        ebio_dir, 'results_pds_score', f'path_diversity_score_{PATHWAY}',
        'exposure_disease_path_diversity_scores.csv'
    )
    df = pd.read_csv(csv_file)
    merged_df = pd.merge(pols_data, df, on=['Exposure_ID','Disease_Name'], how='left')

    df_without_ebio = merged_df[merged_df.isna().any(axis=1)]
    df_without_ebio['Exposure'].unique()
    merged_df.to_csv(os.path.join(output_path,f'2_{PATHWAY}_merged_data.csv'))

    filter_by_significant = merged_df[merged_df['Significance_raw'] == 'Significant Harm'].reset_index(drop=True)
    filter_by_significant['sub category_en'].count()
    filter_by_significant[filter_by_significant.isna().any(axis=1)].to_csv(os.path.join(output_path,f'2_{PATHWAY}_sig_harm_without_ebio.csv'))
    filter_by_significant = filter_by_significant.dropna()

#%%Combine
import pandas as pd
import os
df_EPPD = pd.read_csv(os.path.join(output_path,f'2_EPPD_merged_data.csv'))
df_EGoD = pd.read_csv(os.path.join(output_path,f'2_EGoD_merged_data.csv'))
alignment_cols = ['Model_Type', 'Exposure', 'Disease_Name']
if not df_EPPD[alignment_cols].reset_index(drop=True).equals(
    df_EGoD[alignment_cols].reset_index(drop=True)
):
    raise ValueError('EPPD and EGoD result rows are not aligned; refusing positional score combination.')
df_EPPD.rename(columns={'PDS_Weighted':'PDS_EPPD'},inplace=True)
df_EGoD.rename(columns={'PDS_Weighted':'PDS_EGoD'},inplace=True)
keep_cols = df_EPPD.columns[1:19].tolist()
scores = pd.concat([df_EPPD['PDS_EPPD'],df_EGoD['PDS_EGoD']],axis=1)
df_final = pd.concat([df_EPPD[keep_cols],scores],axis=1)
df_final.to_csv(os.path.join(output_path,'3_combined_2_pds_data.csv'))
#%%
