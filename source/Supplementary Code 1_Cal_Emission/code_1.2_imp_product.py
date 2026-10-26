'''Missing Value Imputation (Low-Rank Matrix Completion)

Input
results/desc_product/df_product.xlsx — The merged production dataset from the previous step, containing Province, Year, and all product production columns with missing values.

Processing
The script uses IterativeImputer (sklearn) to perform Low-Rank Matrix Completion-based imputation. It iterates through each province/region separately, fitting the imputer on the product columns and filling missing values while constraining imputed values to be non-negative (min_value=0).

Output (saved to results/imp_product/)
df_product_imp.xlsx — Imputed dataset with all missing values filled. Same structure as the input.
comparison_provinces.png — Side-by-side comparison of each product column before and after imputation, broken down by province.
comparison_total.png — Side-by-side comparison of each product column before and after imputation for the national total.
'''
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.experimental import enable_iterative_imputer  # Must import before IterativeImputer
from sklearn.impute import IterativeImputer
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']  
matplotlib.rcParams['axes.unicode_minus'] = False  

print("=" * 80)
print("Low-Rank Matrix Completion for Missing Value Imputation")
print("=" * 80)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

input_file = os.path.join(SCRIPT_DIR, 'results', 'desc_product', 'df_product.xlsx')
print(f"\nread file: {input_file}")
df_original = pd.read_excel(input_file)

print(f"data shape: {df_original.shape}")
print(f"\nmissing value statistics:")
missing_stats = df_original.isnull().sum()
print(missing_stats[missing_stats > 0])
print(f"\ntotal missing values: {df_original.isnull().sum().sum()}")

production_columns = [col for col in df_original.columns if col not in ['省份', 'Province', 'Year']]
print(f"production columns: {production_columns}")

df_imputed = df_original.copy()

print("\nbegin Low-Rank Matrix Completion imputation...")
print("Using IterativeImputer")

imputer = IterativeImputer(
    max_iter=100,
    random_state=42,
    verbose=1,
    min_value=0  
)

provinces = df_original['Province'].unique()
print(f"\n Imputing missing values for {len(provinces)} provinces/regions...")

for province in provinces:
    mask = df_imputed['Province'] == province
    province_data = df_imputed.loc[mask, production_columns].values
    
    if np.isnan(province_data).any():
        imputed_data = imputer.fit_transform(province_data)
        df_imputed.loc[mask, production_columns] = imputed_data
        print(f"  Imputed missing values for {province}: {np.isnan(province_data).sum()}")

print("\nImputation completed!")
print(f"Missing values after imputation: {df_imputed.isnull().sum().sum()}")

output_dir = os.path.join(SCRIPT_DIR, 'results', 'code_1.2_imp_product')
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'df_product_imp.xlsx')
df_imputed.to_excel(output_file, index=False)
print(f"\nImputed data saved to: {output_file}")

print("\n" + "=" * 80)
print("Plotting before and after imputation...")
print("=" * 80)

df_original_total = df_original[df_original['Province'] == 'Total'].copy()
df_original_provinces = df_original[df_original['Province'] != 'Total'].copy()
df_imputed_total = df_imputed[df_imputed['Province'] == 'Total'].copy()
df_imputed_provinces = df_imputed[df_imputed['Province'] != 'Total'].copy()

provinces = df_original_provinces['Province'].unique()
colors = plt.cm.tab20(range(len(provinces)))

print("\nPlotting province-wise comparison...")
n_cols = len(production_columns)
n_rows = (n_cols + 2) // 3

fig, axes = plt.subplots(n_rows, 6, figsize=(36, 5*n_rows))  # Six columns: two before/after comparison plots per metric.

for idx, col in enumerate(production_columns):
    row = idx // 3
    col_pos = (idx % 3) * 2
    
    ax_before = axes[row, col_pos] if n_rows > 1 else axes[col_pos]
    for i, province in enumerate(provinces):
        province_data = df_original_provinces[df_original_provinces['Province'] == province].sort_values('Year')
        ax_before.plot(province_data['Year'], province_data[col], 
                      label=province, color=colors[i], marker='o', markersize=1.5, linewidth=1, alpha=0.6)
    ax_before.set_xlabel('Year', fontsize=9)
    ax_before.set_ylabel(col, fontsize=9)
    ax_before.set_title(f'{col}\nBefore Imputation', fontsize=10, fontweight='bold')
    ax_before.grid(True, alpha=0.3)
    
    ax_after = axes[row, col_pos + 1] if n_rows > 1 else axes[col_pos + 1]
    for i, province in enumerate(provinces):
        province_data = df_imputed_provinces[df_imputed_provinces['Province'] == province].sort_values('Year')
        ax_after.plot(province_data['Year'], province_data[col], 
                     label=province, color=colors[i], marker='o', markersize=1.5, linewidth=1, alpha=0.6)
    ax_after.set_xlabel('Year', fontsize=9)
    ax_after.set_ylabel(col, fontsize=9)
    ax_after.set_title(f'{col}\nAfter Imputation', fontsize=10, fontweight='bold')
    ax_after.grid(True, alpha=0.3)
    ax_after.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=5, ncol=1)

for idx in range(n_cols, n_rows * 3):
    row = idx // 3
    col_pos = (idx % 3) * 2
    if n_rows > 1:
        axes[row, col_pos].set_visible(False)
        axes[row, col_pos + 1].set_visible(False)
    else:
        axes[col_pos].set_visible(False)
        axes[col_pos + 1].set_visible(False)

plt.tight_layout()
plot_file = os.path.join(output_dir, 'comparison_provinces.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Province-wise comparison plot saved to: {plot_file}")
plt.close()

print("\nPlotting Total data comparison...")
fig2, axes2 = plt.subplots(n_rows, 6, figsize=(36, 5*n_rows))

for idx, col in enumerate(production_columns):
    row = idx // 3
    col_pos = (idx % 3) * 2
    
    ax_before = axes2[row, col_pos] if n_rows > 1 else axes2[col_pos]
    total_data_before = df_original_total.sort_values('Year')
    ax_before.plot(total_data_before['Year'], total_data_before[col], 
                  marker='o', linewidth=2, markersize=4, color='steelblue')
    ax_before.fill_between(total_data_before['Year'], total_data_before[col], alpha=0.3, color='steelblue')
    ax_before.set_xlabel('Year', fontsize=9)
    ax_before.set_ylabel(col, fontsize=9)
    ax_before.set_title(f'{col}\nBefore Imputation (Total)', fontsize=10, fontweight='bold')
    ax_before.grid(True, alpha=0.3)
    
    ax_after = axes2[row, col_pos + 1] if n_rows > 1 else axes2[col_pos + 1]
    total_data_after = df_imputed_total.sort_values('Year')
    ax_after.plot(total_data_after['Year'], total_data_after[col], 
                 marker='o', linewidth=2, markersize=4, color='darkgreen')
    ax_after.fill_between(total_data_after['Year'], total_data_after[col], alpha=0.3, color='darkgreen')
    ax_after.set_xlabel('Year', fontsize=9)
    ax_after.set_ylabel(col, fontsize=9)
    ax_after.set_title(f'{col}\nAfter Imputation (Total)', fontsize=10, fontweight='bold')
    ax_after.grid(True, alpha=0.3)

for idx in range(n_cols, n_rows * 3):
    row = idx // 3
    col_pos = (idx % 3) * 2
    if n_rows > 1:
        axes2[row, col_pos].set_visible(False)
        axes2[row, col_pos + 1].set_visible(False)
    else:
        axes2[col_pos].set_visible(False)
        axes2[col_pos + 1].set_visible(False)

plt.tight_layout()
plot_file2 = os.path.join(output_dir, 'comparison_total.png')
plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
print(f"Total comparison plot saved to: {plot_file2}")
plt.close()

print("\n" + "=" * 80)
print("All plots completed!")
print("=" * 80)
print(f"\nOutput directory: {output_dir}")
print(f"  - df_product_imp.xlsx: Imputed data")
print(f"  - comparison_provinces.png: Province-wise before and after imputation")
print(f"  - comparison_total.png: Total before and after imputation")

# %%
