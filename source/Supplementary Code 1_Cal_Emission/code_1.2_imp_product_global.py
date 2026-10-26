#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Output directory: results/imp_product_global.
# Apply the same procedure as the provincial script, imputing each of 204 countries by GBD_NAME.
# Input: results/desc_product/df_product_global.xlsx.
#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

INPUT_FILE = os.path.join(SCRIPT_DIR, 'results', 'code_1.1_desc_product', 'df_product_global.xlsx')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'code_1.2_imp_product_global')
META_COLUMNS = ['GBD_NAME', 'Year']
TOP_N_COUNTRIES_PLOT = 15

print("=" * 80)
print("IterativeImputer for Global 204 Countries' Production Data")
print("=" * 80)

print(f"\nreading file: {INPUT_FILE}")
df_original = pd.read_excel(INPUT_FILE)
df_original = df_original.sort_values(['GBD_NAME', 'Year']).reset_index(drop=True)

production_columns = [c for c in df_original.columns if c not in META_COLUMNS]
print(f"data shape: {df_original.shape}")
print(f"Number of countries/regions: {df_original['GBD_NAME'].nunique()}")
print(f"Production columns: {production_columns}")
print(f"Total missing values before imputation: {df_original[production_columns].isnull().sum().sum()}")

df_imputed = df_original.copy()
imputer = IterativeImputer(max_iter=100, random_state=42, verbose=0, min_value=0)

regions = df_original['GBD_NAME'].unique()
print(f"\nImputing missing values for {len(regions)} countries/regions...")

imputed_region_count = 0
for region in regions:
    mask = df_imputed['GBD_NAME'] == region
    region_data = df_imputed.loc[mask, production_columns].values
    if np.isnan(region_data).all():
        continue
    if np.isnan(region_data).any():
        n_missing = int(np.isnan(region_data).sum())
        block = df_imputed.loc[mask, production_columns]
        valid_cols = block.columns[block.notna().any()].tolist()
        if not valid_cols:
            continue
        if block[valid_cols].isnull().any().any():
            imputed_data = imputer.fit_transform(block[valid_cols].values)
            df_imputed.loc[mask, valid_cols] = imputed_data
        imputed_region_count += 1
        if imputed_region_count <= 20 or imputed_region_count % 50 == 0:
            print(f"  Imputed missing values for {region}: {n_missing}")

print(f"\nTotal regions imputed: {imputed_region_count}")
remaining_missing = df_imputed[production_columns].isnull().sum().sum()
print(f"Total missing values after imputation: {remaining_missing}")

# Global matrix imputation: Utilize correlations between industrial sources to fill cells with no observations for a single country
if remaining_missing > 0:
    print("\nPerforming second round of IterativeImputer on the global country×year matrix...")
    global_imputer = IterativeImputer(max_iter=50, random_state=42, verbose=0, min_value=0)
    mat = df_imputed[production_columns].values.astype(float)
    all_nan_rows = np.isnan(mat).all(axis=1)
    mat_filled = global_imputer.fit_transform(mat)
    mat_filled[all_nan_rows] = 0
    df_imputed[production_columns] = mat_filled
    remaining_missing = df_imputed[production_columns].isnull().sum().sum()
    print(f"Total missing values after global imputation: {remaining_missing} (Rows with no observations set to 0, total {all_nan_rows.sum()} rows)")

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, 'df_product_imp.xlsx')
df_imputed.to_excel(output_file, index=False)
print(f"\nImputed data saved: {output_file}")

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Plotting before and after imputation...")
print("=" * 80)

viz_dir = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(viz_dir, exist_ok=True)

missing_before = df_original[production_columns].isnull().sum()
missing_after = df_imputed[production_columns].isnull().sum()

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(production_columns))
w = 0.35
ax.bar(x - w / 2, missing_before.values, w, label='Before Imputation', color='steelblue', alpha=0.85)
ax.bar(x + w / 2, missing_after.values, w, label='After Imputation', color='darkgreen', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(production_columns, rotation=45, ha='right')
ax.set_ylabel('Number of Missing Values')
ax.set_title('Missing Values in Production Data (204 Countries × Years)')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'missing_by_source.png'), dpi=200, bbox_inches='tight')
plt.close()


def global_total_by_year(df):
    return df.groupby('Year')[production_columns].sum(min_count=1).reset_index()


total_before = global_total_by_year(df_original)
total_after = global_total_by_year(df_imputed)

n_cols = len(production_columns)
n_rows = (n_cols + 2) // 3
fig, axes = plt.subplots(n_rows, 6, figsize=(36, 5 * n_rows))
if n_rows == 1:
    axes = np.array([axes])

for idx, col in enumerate(production_columns):
    row, col_idx = idx // 3, (idx % 3) * 2
    ax_b = axes[row, col_idx]
    ax_a = axes[row, col_idx + 1]
    tb = total_before.sort_values('Year')
    ta = total_after.sort_values('Year')
    ax_b.plot(tb['Year'], tb[col], marker='o', linewidth=2, color='steelblue')
    ax_b.set_title(f'{col}\nGlobal Total (Before Imputation)', fontsize=10, fontweight='bold')
    ax_a.plot(ta['Year'], ta[col], marker='o', linewidth=2, color='darkgreen')
    ax_a.set_title(f'{col}\nGlobal Total (After Imputation)', fontsize=10, fontweight='bold')
    for ax in (ax_b, ax_a):
        ax.set_xlabel('Year')
        ax.grid(True, alpha=0.3)

for idx in range(n_cols, n_rows * 3):
    row, col_pos = idx // 3, (idx % 3) * 2
    axes[row, col_pos].set_visible(False)
    axes[row, col_pos + 1].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(viz_dir, 'comparison_global_total.png'), dpi=200, bbox_inches='tight')
plt.close()


country_mean = df_imputed.groupby('GBD_NAME')[production_columns].mean().sum(axis=1).sort_values(ascending=False)
top_countries = country_mean.head(TOP_N_COUNTRIES_PLOT).index.tolist()
colors = plt.cm.tab20(np.linspace(0, 1, len(top_countries)))

fig, axes = plt.subplots(n_rows, 3, figsize=(24, 5 * n_rows))
if n_rows == 1:
    axes = np.array([axes])

for idx, col in enumerate(production_columns):
    row, col_idx = idx // 3, idx % 3
    ax = axes[row, col_idx]
    for i, country in enumerate(top_countries):
        sub = df_imputed[df_imputed['GBD_NAME'] == country].sort_values('Year')
        ax.plot(sub['Year'], sub[col], label=country, color=colors[i], linewidth=1, alpha=0.75)
    ax.set_title(f'{col}（Top{TOP_N_COUNTRIES_PLOT}，After Imputation）', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

for idx in range(n_cols, n_rows * 3):
    row, col_idx = idx // 3, idx % 3
    axes[row, col_idx].set_visible(False)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(os.path.join(viz_dir, 'comparison_top_countries.png'), dpi=200, bbox_inches='tight')
plt.close()

print(f"\nVisualization directory: {viz_dir}")
print("  - missing_by_source.png")
print("  - comparison_global_total.png")
print("  - comparison_top_countries.png")
print("\nImputation process completed.")

# %%
