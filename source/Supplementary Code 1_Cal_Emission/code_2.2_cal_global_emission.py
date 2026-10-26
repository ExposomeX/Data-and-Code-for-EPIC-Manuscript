#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Output directory: results/cal_emission_global
# Calculate PM and pollutant emissions based on imputed global production data
# (same formula as China code_3_cal_emission)
# Input production data: results/imp_product_global/df_product_imp.xlsx

import os
import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

EMISSION_DATA_FILE = os.path.join(SCRIPT_DIR, 'data_raw', 'Industrial_EF_conc.xlsx')
PRODUCT_IMP_FILE = os.path.join(SCRIPT_DIR, 'results', 'code_1.2_imp_product_global', 'df_product_imp.xlsx')
DESC_PRODUCT_FILE = os.path.join(SCRIPT_DIR, 'results', 'code_1.1_desc_product', 'df_product_global.xlsx')
RAW_PRODUCT_FILE = os.path.join(SCRIPT_DIR, 'data_raw', 'Global_1990-2021_production.xlsx')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results', 'code_2.2_cal_emission_global')
VIZ_DIR = os.path.join(OUTPUT_DIR, 'figures')
META_COLUMNS = ['GBD_NAME', 'Year']
TOP_N_COUNTRIES_PLOT = 15
SOURCE_SUFFIXES = ['BFI', 'CFPP', 'CK', 'COP', 'EAF', 'HWI', 'IOS', 'MSWI', 'PCu', 'SAl', 'SCu', 'SPb', 'SZn']


def ensure_desc_product_global():
    """If df_product_global has not been generated yet, merge from raw Excel."""
    if os.path.exists(DESC_PRODUCT_FILE):
        return
    print("desc_product not found, merging from raw global production Excel...")
    xl = pd.ExcelFile(RAW_PRODUCT_FILE)
    production_sheets = [s for s in xl.sheet_names if '年产量数据' in s]
    all_dfs = []
    reference_columns = None
    for sheet_name in production_sheets:
        year_match = re.match(r'(\d{4})', sheet_name)
        if not year_match:
            continue
        year = int(year_match.group(1))
        df = pd.read_excel(xl, sheet_name=sheet_name)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        if reference_columns is None:
            reference_columns = df.columns.tolist()
        elif df.columns.tolist() != reference_columns:
            raise ValueError(f'Sheet {sheet_name} column names inconsistent')
        df.insert(df.columns.get_loc('GBD_NAME') + 1, 'Year', year)
        all_dfs.append(df)
    df_merged = pd.concat(all_dfs, axis=0, ignore_index=True)
    os.makedirs(os.path.dirname(DESC_PRODUCT_FILE), exist_ok=True)
    df_merged.to_excel(DESC_PRODUCT_FILE, index=False)
    print(f"Saved: {DESC_PRODUCT_FILE}")


def get_conversion_factor(pollutant_name):
    if 'μg/g' in pollutant_name or 'µg/g' in pollutant_name:
        return 1 / 1_000_000_000
    if 'ng/g' in pollutant_name or 'pg/g' in pollutant_name:
        return 1 / 1_000_000_000
    if 'nmol/min/µg' in pollutant_name or 'nmol/min/μg' in pollutant_name:
        return 1 / 1_000_000_000_000
    if 'Particles/g' in pollutant_name:
        return 1 / 1_000_000_000
    return 1


ensure_desc_product_global()

if not os.path.exists(PRODUCT_IMP_FILE):
    raise FileNotFoundError(
        f'Imputed production file not found {PRODUCT_IMP_FILE}, please run code_1.2_imp_product_global.py first'
    )

print("=" * 80)
print("Global Emission Calculation (204 Countries)")
print("=" * 80)

print(f"\nReading imputed production: {PRODUCT_IMP_FILE}")
df_product_imp = pd.read_excel(PRODUCT_IMP_FILE)
production_columns = [c for c in df_product_imp.columns if c not in META_COLUMNS]
print(f"Production data shape: {df_product_imp.shape}, number of countries: {df_product_imp['GBD_NAME'].nunique()}")

print("\nReading emission factors...")
df_ef = pd.read_excel(EMISSION_DATA_FILE, sheet_name='EF')

df_PM_g = df_product_imp[META_COLUMNS].copy()
print("\nCalculating PM emissions (g)...")
for col in production_columns:
    keyword = col.split('(')[0].strip()
    match = df_ef[df_ef['Industrial source abbr.'] == keyword]
    if match.empty:
        print(f"  Warning: emission factor not found for {keyword}")
        continue
    ef_value = match['EF (kg/t)'].values[0]
    # Production column unit consistent with national script: values in 10^4 t
    df_PM_g[keyword] = df_product_imp[col] * ef_value * 10_000_000

os.makedirs(OUTPUT_DIR, exist_ok=True)
pm_path = os.path.join(OUTPUT_DIR, 'df_PM_g_global.xlsx')
df_PM_g.to_excel(pm_path, index=False)
print(f"PM emissions saved: {pm_path}")

# ---------------------------------------------------------------------------
# Pollutant-specific emissions
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Calculating pollutant-specific emissions...")
print("=" * 80)

df_conc_mean = pd.read_excel(EMISSION_DATA_FILE, sheet_name='Pols mean Conc. per source')
df_conc_mean = df_conc_mean[~df_conc_mean['Pollutants'].str.contains('1,3,6,8-BCZ', na=False)]
industrial_sources = [c for c in df_conc_mean.columns if c != 'Pollutants']

df_emi = df_PM_g[META_COLUMNS].copy()
for _, row in df_conc_mean.iterrows():
    pollutant_full = row['Pollutants']
    pollutant_name = pollutant_full.split('(')[0].strip()
    conversion_factor = get_conversion_factor(pollutant_full)
    for source in industrial_sources:
        if source not in df_PM_g.columns:
            continue
        concentration = row[source]
        new_col = f'{pollutant_name}_{source}'
        df_emi[new_col] = df_PM_g[source] * concentration * conversion_factor

pollutant_groups = {}
for col in df_emi.columns:
    if col in META_COLUMNS:
        continue
    last_u = col.rfind('_')
    if last_u <= 0:
        continue
    pname = col[:last_u]
    pollutant_groups.setdefault(pname, []).append(col)

new_columns = OrderedDict()
for col in META_COLUMNS:
    new_columns[col] = df_emi[col]
for pollutant_name in sorted(pollutant_groups.keys()):
    cols = pollutant_groups[pollutant_name]
    for col in sorted(cols):
        new_columns[col] = df_emi[col]
    new_columns[f'{pollutant_name}_Sum'] = df_emi[cols].sum(axis=1)

df_emi = pd.DataFrame(new_columns)
emi_path = os.path.join(OUTPUT_DIR, 'df_emi_global.xlsx')
df_emi.to_excel(emi_path, index=False)
print(f"Pollutant emissions saved: {emi_path}, shape={df_emi.shape}")

# ---------------------------------------------------------------------------
# Visualization (global total + top countries, to avoid 204 overlapping lines)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("Plotting emission trends...")
print("=" * 80)

os.makedirs(VIZ_DIR, exist_ok=True)
sum_columns = [c for c in df_emi.columns if c.endswith('_Sum')]

global_sum = df_emi.groupby('Year')[sum_columns].sum(min_count=1).reset_index()

n_poll = len(sum_columns)
n_rows, n_cols_grid = 14, 4
fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(24, 56))
axes_flat = axes.flatten()
fig.suptitle('Global Total Emission Trends by Pollutant (204 Countries)', fontsize=22, y=0.995)

for idx, sum_col in enumerate(sum_columns):
    ax = axes_flat[idx]
    pname = sum_col.replace('_Sum', '')
    ax.plot(global_sum['Year'], global_sum[sum_col], marker='o', markersize=2, linewidth=1.5, color='darkgreen')
    ax.set_title(pname, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45, labelsize=6)

for i in range(len(sum_columns), len(axes_flat)):
    axes_flat[i].set_visible(False)

plt.tight_layout(rect=[0, 0.01, 1, 0.99])
all_trends_path = os.path.join(VIZ_DIR, 'all_pollutants_global_total_trends.png')
plt.savefig(all_trends_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {all_trends_path}")

country_total = df_emi.groupby('GBD_NAME')[sum_columns].sum().sum(axis=1).sort_values(ascending=False)
top_countries = country_total.head(TOP_N_COUNTRIES_PLOT).index.tolist()
colors = plt.cm.tab20(np.linspace(0, 1, len(top_countries)))

key_pollutants = sum_columns[: min(8, len(sum_columns))]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
fig.suptitle(f'Major Pollutant Emission Trends (Top {TOP_N_COUNTRIES_PLOT} Countries)', fontsize=16)

for idx, sum_col in enumerate(key_pollutants):
    ax = axes[idx]
    pname = sum_col.replace('_Sum', '')
    for i, country in enumerate(top_countries):
        sub = df_emi[df_emi['GBD_NAME'] == country].sort_values('Year')
        ax.plot(sub['Year'], sub[sum_col], label=country, color=colors[i], linewidth=1, alpha=0.8)
    ax.set_title(pname, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.06, 1, 0.96])
top_path = os.path.join(VIZ_DIR, 'top_countries_key_pollutants.png')
plt.savefig(top_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {top_path}")

# Emission source contribution (global total)
pollutant_data = {}
pollutants_with_data = []
for sum_col in sum_columns:
    pname = sum_col.replace('_Sum', '')
    source_totals = {}
    for suffix in SOURCE_SUFFIXES:
        col_name = f'{pname}_{suffix}'
        source_totals[suffix] = df_emi[col_name].sum() if col_name in df_emi.columns else 0
    total_emission = sum(source_totals.values())
    if total_emission == 0 or np.isnan(total_emission):
        continue
    pollutant_data[pname] = {s: (source_totals[s] / total_emission) * 100 for s in SOURCE_SUFFIXES}
    pollutants_with_data.append(pname)

if pollutants_with_data:
    source_avg = {
        s: np.mean([pollutant_data[p][s] for p in pollutants_with_data]) for s in SOURCE_SUFFIXES
    }
    sorted_sources = [x[0] for x in sorted(source_avg.items(), key=lambda x: x[1], reverse=True)]
    gradient_colors = [
        '#CD5C5C', '#E9967A', '#FF8C00', '#FFD700', '#F0E68C', '#BDB76B', '#9ACD32',
        '#32CD32', '#20B2AA', '#4682B4', '#4169E1', '#6A5ACD', '#483D8B',
    ]
    source_colors = {s: gradient_colors[i] for i, s in enumerate(sorted_sources)}

    fig, ax = plt.subplots(figsize=(max(16, len(pollutants_with_data) * 0.45), 10))
    x_pos = np.arange(len(pollutants_with_data))
    bottom = np.zeros(len(pollutants_with_data))
    for suffix in sorted_sources:
        pct = [pollutant_data[p][suffix] for p in pollutants_with_data]
        ax.bar(x_pos, pct, 0.8, bottom=bottom, label=suffix, color=source_colors[suffix], edgecolor='white', linewidth=0.4)
        bottom += pct
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pollutants_with_data, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Contribution Percentage (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Global Pollutant Emission Source Contribution (%) (204 Countries)', fontsize=14, fontweight='bold')
    ax.legend(title='Emission Source', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    stack_path = os.path.join(VIZ_DIR, 'global_source_contribution_stacked.png')
    plt.savefig(stack_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {stack_path}")

print("\n" + "=" * 80)
print("Global emission calculation and visualization completed")
print("=" * 80)
print(f"Data output: {OUTPUT_DIR}")
print(f"  - df_PM_g_global.xlsx")
print(f"  - df_emi_global.xlsx")
print(f"Chart output: {VIZ_DIR}")
