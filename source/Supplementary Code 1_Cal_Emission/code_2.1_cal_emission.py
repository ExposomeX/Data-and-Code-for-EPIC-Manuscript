#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Output directory: results/cal_emission
# Calculate pollutant emissions: formula = production(10^4 t) x PM emission factor EF(kg/t) x 10000000 x concentration
# df_emi: 13 industrial sources x 55 pollutants + sum values across all sources = 770 columns total
# df_PM_g: PM emissions for 13 industrial sources, 16 columns total (Province, Province_en, Year + 13 industrial sources), unit: g
# Plot trend lines for each pollutant summed across all industrial sources (55 total), each subplot shows trends for all provinces

import os
import pandas as pd
import numpy as np
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMISSION_DATA_FILE = os.path.join(SCRIPT_DIR, 'data_raw', 'Industrial_EF_conc.xlsx')

print("=" * 80)
print("Calculate PM Emissions from Product Data and Emission Factors")
print("=" * 80)

# Read imputed production data
print("\nreading imputed production data...")
df_product_imp = pd.read_excel(os.path.join(SCRIPT_DIR, 'results', 'code_1.2_imp_product', 'df_product_imp.xlsx'))
print(f"Imputed production data shape: {df_product_imp.shape}")
print(f"Column names: {df_product_imp.columns.tolist()}")

# Read emission factor data
print("\nreading emission factor data...")
df_conc = pd.read_excel(EMISSION_DATA_FILE, sheet_name='EF')
print(f"Emission factor data shape: {df_conc.shape}")
print(f"Emission factor data:\n{df_conc}")

# Extract production columns (from BFI (10^4 t) to SZn (10^4 t))
production_columns = [col for col in df_product_imp.columns if col not in ['省份', 'Province', 'Year']]
print(f"\nImputed production columns: {production_columns}")

# Create result DataFrame, keeping Province, Province_en, Year columns
df_PM_g = df_product_imp[['省份', 'Province', 'Year']].copy()

# Calculate PM emissions for each production column
print("\ncalculating PM emissions...")
for col in production_columns:
    # Extract keyword before "("
    keyword = col.split('(')[0].strip()
    print(f"\nCalculating emissions for: {col}")
    print(f"  Keyword: {keyword}")
    
    # Find matching Industrial source in emission factor table
    match = df_conc[df_conc['Industrial source abbr.'] == keyword]
    
    if not match.empty:
        # Get corresponding emission factor EF (kg/t)
        ef_value = match['EF (kg/t)'].values[0]
        print(f"  Found emission factor: {ef_value} kg/t")
        
        # Calculate PM emissions: production(10^4 t) x EF(kg/t) x 10000000
        # Production unit: 10^4 t = 10000 t
        # 10000 t × kg/t × 10000000 = 10000 × 10000000 kg = 10^11 kg = 10^8 t = 10^5 × 10^3 kg = 10^5 kg × 10^3 = g × 10^11
        # Actually: 10^4 t = 10^4 t, 10^4 t x kg/t = 10^4 kg, 10^4 kg x 10^7 = 10^11 g
        df_PM_g[keyword] = df_product_imp[col] * ef_value * 10000000
        print(f"  already calculated: {keyword}")
    else:
        print(f"  Warning: No matching emission factor found!")

print(f"\nCalculation completed!")
print(f"Result data shape: {df_PM_g.shape}")
print(f"Result column names: {df_PM_g.columns.tolist()}")

# Create output directory
output_dir = os.path.join(SCRIPT_DIR, 'results', 'code_2.1_cal_emission')
os.makedirs(output_dir, exist_ok=True)

# Save results
output_file = os.path.join(output_dir, 'df_PM_g.xlsx')
df_PM_g.to_excel(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Display result sample
print(f"\nResult data sample:")
print(df_PM_g.head())

print("\n" + "=" * 80)
print("Finished PM emission calculation!")
print("=" * 80)

print("\n" + "=" * 80)
print("Starting calculation of pollutant emissions...")
print("=" * 80)

# Read pollutant concentration mean data
df_conc_mean = pd.read_excel(EMISSION_DATA_FILE, 
                              sheet_name='Pols mean Conc. per source')
print(f"Number of pollutants: {len(df_conc_mean)}")
print(f"Industrial source columns: {df_conc_mean.columns.tolist()}")

# Create result DataFrame, keeping Province, Province_en, Year columns
df_emi = df_PM_g[['省份', 'Province', 'Year']].copy()

# Get industrial source columns (excluding Pollutants column)
industrial_sources = [col for col in df_conc_mean.columns if col != 'Pollutants']
print(f"\nIndustrial sources: {industrial_sources}")

# Define unit conversion factor function
def get_conversion_factor(pollutant_name):
    """Determine conversion factor based on pollutant unit"""
    if 'μg/g' in pollutant_name or 'µg/g' in pollutant_name:
        return 1 / 1000000000  # μg/g divided by 10^9
    elif 'ng/g' in pollutant_name:
        return 1 / 1000000000  # ng/g divided by 10^9
    elif 'pg/g' in pollutant_name:
        return 1 / 1000000000  # pg/g divided by 10^9
    elif 'nmol/min/µg' in pollutant_name or 'nmol/min/μg' in pollutant_name:
        return 1 / 1000000000000  # nmol/min/µg divided by 10^12
    elif 'Particles/g' in pollutant_name:
        return 1 / 1000000000  # Particles/g divided by 10^9
    else:
        return 1  # default: no conversion

# Iterate through each pollutant
print("\nStarting calculation of each pollutant's emissions...")
for idx, row in df_conc_mean.iterrows():
    pollutant_full = row['Pollutants']
    
    # Extract pollutant name (part before "(")
    pollutant_name = pollutant_full.split('(')[0].strip()
    
    # Get conversion factor
    conversion_factor = get_conversion_factor(pollutant_full)
    
    print(f"\nProcessing pollutant  {idx+1}/{len(df_conc_mean)}: {pollutant_full}")
    print(f"  Pollutant name: {pollutant_name}")
    print(f"  Conversion factor: {conversion_factor}")
    
    # Iterate through each industrial source
    for source in industrial_sources:
        if source in df_PM_g.columns:
            # Get concentration of this pollutant for this industrial source
            concentration = row[source]
            
            # Create new column name: pollutant_name_source
            new_col_name = f"{pollutant_name}_{source}"
            
            # Calculate emissions: PM emissions(g) x concentration x conversion factor
            df_emi[new_col_name] = df_PM_g[source] * concentration * conversion_factor
            
            if idx < 3:  # Only print detailed info for first few
                print(f"    {source}: conc={concentration}, new_col={new_col_name}")

print(f"\nPollutant emissions calculation completed!")
print(f"Current data shape: {df_emi.shape}")

print("\n" + "=" * 80)
print("Starting calculation of total emissions for each pollutant (sum of 13 industrial sources)...")
print("=" * 80)
    
# Get all pollutant names (extracted from column names)
# Column name format: pollutant_name_source
# Extract characters before the last "_" as pollutant name
pollutant_groups = {}
for col in df_emi.columns:
    if col not in ['省份', 'Province', 'Year']:
        # Find the position of the last "_"
        last_underscore_idx = col.rfind('_')
        if last_underscore_idx > 0:
            pollutant_name = col[:last_underscore_idx]
            source_name = col[last_underscore_idx+1:]
            
            if pollutant_name not in pollutant_groups:
                pollutant_groups[pollutant_name] = []
            pollutant_groups[pollutant_name].append(col)

print(f"\nIdentified {len(pollutant_groups)} types of pollutants")
print(f"Number of industrial sources: {len(industrial_sources)}")

# Create new DataFrame with base columns, all emission columns, and sum columns
# Use OrderedDict to maintain column order
from collections import OrderedDict
new_columns = OrderedDict()

# First add base columns
for col in ['省份', 'Province', 'Year']:
    new_columns[col] = df_emi[col]

# Add columns grouped by pollutant
for pollutant_name in sorted(pollutant_groups.keys()):
    cols = pollutant_groups[pollutant_name]
    
    # Add all industrial source columns for this pollutant
    for col in sorted(cols):
        new_columns[col] = df_emi[col]
    
    # Calculate and add sum column
    sum_col_name = f"{pollutant_name}_Sum"
    sum_values = df_emi[cols].sum(axis=1)
    new_columns[sum_col_name] = sum_values
    
    print(f"  {pollutant_name}: {len(cols)} sources -> {sum_col_name}")

# Create new DataFrame
df_emi = pd.DataFrame(new_columns)

print(f"\nSummation calculation completed!")
print(f"Result data shape: {df_emi.shape}")
print(f"Total columns: {len(df_emi.columns)} (3 base columns + {len(pollutant_groups)} pollutants × 14 columns)")

# ============================================================================
# Retain all columns (including all-zero or all-missing columns) to ensure consistent data structure for 13 emission sources
# ============================================================================
print("\n" + "=" * 80)
print("Checking column value states (retaining all columns to ensure data structure consistency)...")
print("=" * 80)

# Get non-base columns
data_columns = [col for col in df_emi.columns if col not in ['省份', 'Province', 'Year']]
columns_with_zeros = []
columns_with_missing = []

for col in data_columns:
    # Check if column is all missing or all zero
    col_data = df_emi[col]
    
    # All missing values
    if col_data.isnull().all():
        columns_with_missing.append(col)
        print(f"  Retaining (all missing values): {col}")
    # All zeros (including NaN treated as 0)
    elif (col_data.fillna(0) == 0).all():
        columns_with_zeros.append(col)
        print(f"  Retaining (all zeros): {col}")

# Do not delete any columns, maintain complete data structure for 13 emission sources
print(f"\nRetaining all columns to ensure data structure consistency for 13 emission sources")
print(f"  Columns with all missing values: {len(columns_with_missing)}")
print(f"  Columns with all zeros: {len(columns_with_zeros)}")
print(f"Data shape: {df_emi.shape}")

# Save results
output_file_emi = os.path.join(output_dir, 'df_emi.xlsx')
df_emi.to_excel(output_file_emi, index=False)
print(f"\nEmission data saved to: {output_file_emi}")

# Display result sample
print(f"\nResult data sample (first few columns):")
print(df_emi.iloc[:3, :10])

print("\n" + "=" * 80)
print("All calculations completed!")
print("=" * 80)
print(f"\nOutput files:")
print(f"  1. {os.path.join(output_dir, 'df_PM_g.xlsx')}: PM emissions data")
print(f"  2. {output_file_emi}: Pollutant emissions data")

print("\n" + "=" * 80)
print("Starting to plot pollutant emission trends...")
print("=" * 80)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']  # Set font to Times New Roman
matplotlib.rcParams['axes.unicode_minus'] = False

# Get all _Sum columns (re-extracted after potential column removal)
sum_columns = [col for col in df_emi.columns if col.endswith('_Sum')]
print(f"\nNumber of pollutants to plot: {len(sum_columns)}")
print(f"Note: Columns with all zeros or missing values have been excluded")

# Get all provinces (excluding Total)
provinces = df_emi['Province'].unique()
provinces = [p for p in provinces if p != 'Total']
print(f"Number of provinces: {len(provinces)}")

# Create color mapping, assign different colors to each province
colors = plt.cm.tab20(range(20))
colors = list(colors) + list(plt.cm.tab20b(range(20)))
colors = list(colors) + list(plt.cm.tab20c(range(20)))

fig, axes = plt.subplots(14, 4, figsize=(24, 56))
fig.suptitle('Trends in Pollutant Emissions Over Time (Grouped by Province)', fontsize=24, y=0.9995)

# Flatten axes to 1D array for easier indexing
axes_flat = axes.flatten()

# Plot subplot for each pollutant
for idx, sum_col in enumerate(sum_columns):
    pollutant_name = sum_col.replace('_Sum', '')
    print(f"  plot {idx+1}/{len(sum_columns)}: {pollutant_name}")
    
    ax = axes_flat[idx]
    
    # Plot trend line for each province
    for i, province in enumerate(provinces):
        # Filter data for this province
        province_data = df_emi[df_emi['Province'] == province].sort_values('Year')
        
        # Plot trend line with different color for each province
        ax.plot(province_data['Year'], province_data[sum_col], 
                marker='o', markersize=2, linewidth=1, 
                label=province, color=colors[i % len(colors)], alpha=0.7)
    
    # Set subplot title and labels
    ax.set_title(pollutant_name, fontsize=10, fontweight='bold')
    ax.set_xlabel('Year', fontsize=7)
    ax.set_ylabel('Emission', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels to avoid overlap
    ax.tick_params(axis='x', rotation=45)

# Hide extra subplots (if any)
for i in range(len(sum_columns), 56):
    axes_flat[i].set_visible(False)

# Add legend (only on first subplot, including all provinces)
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=10, 
           fontsize=8, bbox_to_anchor=(0.5, -0.01))

# Adjust layout
plt.tight_layout(rect=[0, 0.02, 1, 0.999])

# Save image
plot_filename = "all_pollutants_trends.png"
plot_path = os.path.join(output_dir, plot_filename)
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "=" * 80)
print("Starting to plot individual pollutant trends...")
print("=" * 80)

# Create individual figure for each pollutant
for idx, sum_col in enumerate(sum_columns):
    pollutant_name = sum_col.replace('_Sum', '')
    print(f"  Plotting {idx+1}/{len(sum_columns)}: {pollutant_name}")
    
    # Create individual plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot trend line for each province
    for i, province in enumerate(provinces):
        # Filter data for this province
        province_data = df_emi[df_emi['Province'] == province].sort_values('Year')
        
        # Plot trend line with different color for each province
        ax.plot(province_data['Year'], province_data[sum_col], 
                marker='o', markersize=4, linewidth=1.5, 
                label=province, color=colors[i % len(colors)], alpha=0.7)
    
    # Set chart title and labels
    ax.set_title(f'{pollutant_name} Emission Trends Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Emission', fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='best', ncol=3, fontsize=8, framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    individual_plot_filename = f"{pollutant_name}_individual_trend.png"
    individual_plot_path = os.path.join(output_dir, individual_plot_filename)
    plt.savefig(individual_plot_path, dpi=150, bbox_inches='tight')
    plt.close()


print("\n" + "=" * 80)
print("Starting to plot average concentration trends for each pollutant across all provinces (8×7 layout)...")
print("=" * 80)

# Read pollutant concentration data
df_conc_mean_full = pd.read_excel(EMISSION_DATA_FILE, 
                                   sheet_name='Pols mean Conc. per source')

# Filter out 1,3,6,8-BCZ pollutants
df_conc_mean_full = df_conc_mean_full[~df_conc_mean_full['Pollutants'].str.contains('1,3,6,8-BCZ', na=False)]

# Extract pollutant names
pollutants_list = []
for pollutant_full in df_conc_mean_full['Pollutants']:
    pollutant_name = pollutant_full.split('(')[0].strip()
    pollutants_list.append(pollutant_name)

print(f"\nPollutant count: {len(pollutants_list)}")
print(f"Industrial source columns: {industrial_sources}")

# Calculate average concentration for each pollutant across all provinces (across industrial sources)
print("\nCalculating average concentrations for each pollutant across all provinces...")

# Create figure with 8x7 layout (56 subplots)
fig, axes = plt.subplots(8, 7, figsize=(28, 32))
fig.suptitle('Average Concentration Trends for Each Pollutant Across All Provinces (8×7 Layout)', fontsize=28, fontweight='bold', y=0.995)

# Flatten axes to 1D array
axes_flat = axes.flatten()

# Plot subplot for each pollutant
for idx, pollutant_name in enumerate(pollutants_list):
    if idx >= 56:  # max 56 subplots
        break
        
    print(f"  Plotting subplot {idx+1}/{len(pollutants_list)}: {pollutant_name}")
    
    ax = axes_flat[idx]
    
    # Find all columns corresponding to this pollutant (all industrial sources)
    pollutant_cols = [col for col in df_emi.columns 
                      if col.startswith(f"{pollutant_name}_") and not col.endswith('_Sum')]
    
    if len(pollutant_cols) == 0:
        # If no matching columns found, display empty subplot
        ax.text(0.5, 0.5, f'{pollutant_name}\n(No data)', 
                ha='center', va='center', fontsize=10, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        continue
    
    # Group by year, calculate average concentration across all provinces for each year
    years = sorted(df_emi['Year'].unique())
    avg_concentrations = []
    std_concentrations = []
    
    for year in years:
        year_data = df_emi[df_emi['Year'] == year]
        
        # Calculate average concentration across all provinces and industrial sources for this year
        # For each province, first calculate average across all industrial sources, then average across all provinces
        province_avgs = []
        for _, row in year_data.iterrows():
            # Average concentration of this province across all industrial sources
            province_avg = row[pollutant_cols].mean()
            if not np.isnan(province_avg):
                province_avgs.append(province_avg)
        
        if len(province_avgs) > 0:
            avg_concentrations.append(np.mean(province_avgs))
            std_concentrations.append(np.std(province_avgs))
        else:
            avg_concentrations.append(np.nan)
            std_concentrations.append(np.nan)
    
    # Plot average concentration trend line
    avg_concentrations = np.array(avg_concentrations)
    std_concentrations = np.array(std_concentrations)
    
    # Main trend line
    ax.plot(years, avg_concentrations, marker='o', markersize=3, 
            linewidth=2, color='steelblue', label='Average Concentration')
    
    # Add standard deviation shaded area
    ax.fill_between(years, 
                    avg_concentrations - std_concentrations,
                    avg_concentrations + std_concentrations,
                    alpha=0.2, color='steelblue', label='±1 Standard Deviation')
    
    # Set subplot title and labels
    ax.set_title(pollutant_name, fontsize=11, fontweight='bold', pad=5)
    ax.set_xlabel('Year', fontsize=8)
    ax.set_ylabel('Average Concentration', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)
    
    # Add legend only on first subplot
    if idx == 0:
        ax.legend(fontsize=7, loc='upper left')

# Hide extra subplots
for i in range(len(pollutants_list), 56):
    axes_flat[i].set_visible(False)

# Adjust layout
plt.tight_layout(rect=[0, 0.01, 1, 0.995])

# Save image
avg_conc_plot_filename = "average_concentration_trends_8x7.png"
avg_conc_plot_path = os.path.join(output_dir, avg_conc_plot_filename)
plt.savefig(avg_conc_plot_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"\nAverage concentration trends plot saved: {avg_conc_plot_filename}")

print("\n" + "=" * 80)
print("All trend plots completed!")
print("=" * 80)

print("\n" + "=" * 80)
print("Starting to plot contribution percentage charts for each emission source...")
print("=" * 80)

# Find all _Sum columns from 1,3,6,8-CCZ_Sum onwards
sum_columns_all = [col for col in df_emi.columns if col.endswith('_Sum')]
print(f"\nAll _Sum columns count: {len(sum_columns_all)}")

# Find position of 1,3,6,8-CCZ_Sum
ccz_sum_col = '1,3,6,8-CCZ_Sum'
if ccz_sum_col in sum_columns_all:
    ccz_index = sum_columns_all.index(ccz_sum_col)
    sum_columns_for_contribution = sum_columns_all[ccz_index:]
    print(f"Starting from {ccz_sum_col}, total {len(sum_columns_for_contribution)} _Sum columns to plot contribution charts for")
else:
    print(f"Warning: {ccz_sum_col} not found, will plot contribution charts for all _Sum columns")
    sum_columns_for_contribution = sum_columns_all

# Get industrial source suffix list (13 sources)
# Dynamically obtained from pollutant concentration data to ensure consistency with data file
source_suffixes = ['BFI', 'CFPP', 'CK', 'COP', 'EAF', 'HWI', 'IOS', 'MSWI', 'PCu', 'SAl', 'SCu', 'SPb', 'SZn']
print(f"\nEmission source suffixes (total {len(source_suffixes)}): {source_suffixes}")

# Prepare data: calculate contribution percentage of each industrial source for each pollutant
pollutant_data = {}
pollutants_with_data = []

for sum_col in sum_columns_for_contribution:
    pollutant_name = sum_col.replace('_Sum', '')
    
    # Find all industrial source columns for this pollutant
    source_totals = {}
    for suffix in source_suffixes:
        col_name = f"{pollutant_name}_{suffix}"
        if col_name in df_emi.columns:
            total = df_emi[col_name].sum()
            source_totals[suffix] = total
        else:
            source_totals[suffix] = 0
    
    # Calculate total emissions
    total_emission = sum(source_totals.values())
    
    if total_emission == 0 or np.isnan(total_emission):
        print(f"  Skipping {pollutant_name} (Total emission is 0 or NaN)")
        continue
    
    # Calculate contribution percentage of each industrial source
    source_percentages = {}
    for suffix in source_suffixes:
        percentage = (source_totals[suffix] / total_emission) * 100
        source_percentages[suffix] = percentage
    
    pollutant_data[pollutant_name] = source_percentages
    pollutants_with_data.append(pollutant_name)
    print(f"  Processing pollutant: {pollutant_name}")

print(f"\n {len(pollutants_with_data)} pollutants with valid data for contribution percentage calculation")

# Calculate average contribution percentage of each industrial source
source_avg_contribution = {}
for suffix in source_suffixes:
    avg_pct = np.mean([pollutant_data[pollutant][suffix] for pollutant in pollutants_with_data])
    source_avg_contribution[suffix] = avg_pct

# Sort industrial sources by average contribution percentage (descending)
sorted_sources = sorted(source_avg_contribution.items(), key=lambda x: x[1], reverse=True)
sorted_source_suffixes = [item[0] for item in sorted_sources]

print(f"\nEmission sources sorted by average contribution percentage (from highest to lowest):")
for suffix, avg_pct in sorted_sources:
    print(f"  {suffix}: {avg_pct:.2f}%")

# Create stacked bar chart
fig, ax = plt.subplots(figsize=(max(16, len(pollutants_with_data) * 0.5), 10))

# Use gradient color scheme: from dark (high contribution) to light (low contribution)
# Use blue-to-red gradient to enhance visual hierarchy
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# Create red-to-blue gradient color scheme (13 levels)
# Increase color contrast for first 4 colors, reduce saturation for darkest red
gradient_colors = [
    '#CD5C5C',  # Indian red - highest contribution (reduced saturation)
    '#E9967A',  # Deep salmon - 2nd highest (increased contrast)
    '#FF8C00',  # Dark orange - 3rd highest (increased contrast)
    '#FFD700',  # Gold - 4th highest (increased contrast)
    '#F0E68C',  # Khaki
    '#BDB76B',  # Dark khaki
    '#9ACD32',  # Yellow green
    '#32CD32',  # Lime green
    '#20B2AA',  # Sea green
    '#4682B4',  # Steel blue
    '#4169E1',  # Royal blue
    '#6A5ACD',  # Slate blue
    '#483D8B'   # Dark slate blue - lowest contribution
]

# Assign colors to sorted industrial sources (by contribution descending)
source_colors = {}
for i, suffix in enumerate(sorted_source_suffixes):
    source_colors[suffix] = gradient_colors[i]

# X-axis positions
x_pos = np.arange(len(pollutants_with_data))
bar_width = 0.8

# Plot stacked bar chart (sorted by average contribution descending, stacked from bottom to top)
bottom = np.zeros(len(pollutants_with_data))

for suffix in sorted_source_suffixes:
    percentages = [pollutant_data[pollutant][suffix] for pollutant in pollutants_with_data]
    ax.bar(x_pos, percentages, bar_width, bottom=bottom, 
           label=suffix, color=source_colors[suffix], edgecolor='white', linewidth=0.5)
    bottom += percentages

# Set x-axis labels
ax.set_xticks(x_pos)
ax.set_xticklabels(pollutants_with_data, rotation=45, ha='right', fontsize=9)

# Set y-axis
ax.set_ylabel('Contribution Percentage (%)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks(np.arange(0, 101, 10))

# Set title
ax.set_title('Contribution Percentages of Emission Sources for Each Pollutant', fontsize=16, fontweight='bold', pad=20)

# Add grid lines
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add legend
ax.legend(title='Emission Sources', bbox_to_anchor=(1.02, 1), loc='upper left', 
          fontsize=10, title_fontsize=11, frameon=True, shadow=True)

# Adjust layout
plt.tight_layout()

# Save image
stacked_plot_filename = "all_pollutants_source_contribution_stacked.png"
stacked_plot_path = os.path.join(output_dir, stacked_plot_filename)
plt.savefig(stacked_plot_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"\nStacked bar chart saved: {stacked_plot_filename}")

print("\n" + "=" * 80)
print("All analysis and plotting completed!")
print("=" * 80)
print(f"\nOutput files:")
print(f"  1. {os.path.join(output_dir, 'df_PM_g.xlsx')}: PM emission data")
print(f"  2. {output_file_emi}: Pollutant emission data")
print(f"  3. {plot_path}: Comprehensive trend chart for all pollutants (14×4 layout)")
print(f"  4. results/cal_emission/*_individual_trend.png: {len(sum_columns)} individual pollutant trend charts")
print(f"  5. {avg_conc_plot_path}: Average concentration trend chart for all substances (8×7 layout)")
print(f"  6. {stacked_plot_path}: Stacked contribution percentage chart for all pollutants")

