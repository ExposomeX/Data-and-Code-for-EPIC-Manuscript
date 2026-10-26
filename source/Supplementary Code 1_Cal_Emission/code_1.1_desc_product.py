'''Data Description
Input
data_raw/China_1990-2024_production.xlsx — Raw annual production data. Each sheet is named {Year}_yearly_production (e.g., 2020_yearly_production), with provinces as rows and product categories as columns.

Processing
The script extracts each yearly sheet, removes empty columns, adds a Year column, and merges all sheets into a single long-format dataset.

Output (saved to results/desc_product/)
df_product.xlsx — Merged dataset with columns: Province, Year, and all product production values.
production_trends.png — Provincial production trends by product category (excluding national total).
total_trends.png — National total production trends by product category (1990–2024).
'''

import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
matplotlib.rcParams['font.sans-serif'] = ['times new roman']  
matplotlib.rcParams['axes.unicode_minus'] = False  

SCRIPT_DIR = Path(__file__).parent
input_file = SCRIPT_DIR / 'data_raw' / 'China_1990-2024_production.xlsx'
xl = pd.ExcelFile(input_file)

production_sheets = [sheet for sheet in xl.sheet_names if '_yearly_production' in sheet]
print(f"find {len(production_sheets)} sheet")

all_dfs = []
reference_columns = None

for sheet_name in production_sheets:
    
    year_match = re.match(r'(\d{4})', sheet_name)
    if not year_match:
        print(f"warning: unable to extract year from {sheet_name}, skipping")
        continue
    
    year = int(year_match.group(1))
    df = pd.read_excel(xl, sheet_name=sheet_name)
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    if reference_columns is None:
        reference_columns = df.columns.tolist()
        print(f"reference columns (from {sheet_name}): {reference_columns}")
    else:
        if df.columns.tolist() != reference_columns:
            print(f"error: {sheet_name} has different column names than the reference!")
            print(f"  reference columns: {reference_columns}")
            print(f"  current columns: {df.columns.tolist()}")
            raise ValueError(f"Sheet {sheet_name} has different column names than the reference.")
    
    df.insert(df.columns.get_loc('Province') + 1, 'Year', year)
    
    all_dfs.append(df)
    print(f"reading {sheet_name}: {len(df)} rows, year={year}")

df_merged = pd.concat(all_dfs, axis=0, ignore_index=True)
print(f"\nmerged total rows: {len(df_merged)}")
print(f"merged column names: {df_merged.columns.tolist()}")


output_dir = SCRIPT_DIR / 'results' / 'code_1.1_desc_product'
os.makedirs(output_dir, exist_ok=True)

output_file = output_dir / 'df_product.xlsx'
df_merged.to_excel(output_file, index=False)
print(f"\nresults saved to: {output_file}")

print("\nstarting to plot production trends...")

production_columns = [col for col in df_merged.columns if col not in ['省份', 'Province', 'Year']]
print(f"production columns: {production_columns}")

df_total = df_merged[df_merged['Province'] == 'Total'].copy()
df_provinces = df_merged[df_merged['Province'] != 'Total'].copy()

print(f"Total data rows: {len(df_total)}")
print(f"Province data rows: {len(df_provinces)}")

# 1. plot provincial production trends
n_cols = len(production_columns)
n_rows = (n_cols + 2) // 3  
fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
axes = axes.flatten() if n_cols > 1 else [axes]

provinces = df_provinces['Province'].unique()
colors = plt.cm.tab20(range(len(provinces))) 

for idx, col in enumerate(production_columns):
    ax = axes[idx]
    
    for i, province in enumerate(provinces):
        province_data = df_provinces[df_provinces['Province'] == province].sort_values('Year')
        ax.plot(province_data['Year'], province_data[col], 
                label=province, color=colors[i], marker='o', markersize=2, linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(col, fontsize=10)
    ax.set_title(f'{col} provincial production trends', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncol=1)

for idx in range(n_cols, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plot_file = output_dir / 'production_trends.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"provincial production trends plot saved to: {plot_file}")
plt.close()

# 2. plot national total trends
print("\nstarting to plot Total trends...")
fig2, axes2 = plt.subplots(n_rows, 3, figsize=(18, 5*n_rows))
axes2 = axes2.flatten() if n_cols > 1 else [axes2]

for idx, col in enumerate(production_columns):
    ax = axes2[idx]
    
    total_data = df_total.sort_values('Year')
    ax.plot(total_data['Year'], total_data[col], 
            marker='o', linewidth=2.5, markersize=5, color='steelblue')
    ax.fill_between(total_data['Year'], total_data[col], alpha=0.3, color='steelblue')
    
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(col, fontsize=10)
    ax.set_title(f'{col} national total trends', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)


for idx in range(n_cols, len(axes2)):
    axes2[idx].set_visible(False)

plt.tight_layout()
plot_file2 = output_dir / 'total_trends.png'
plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
print(f"national total trends plot saved to: {plot_file2}")
plt.close()

print("\ndone！")

# %%
