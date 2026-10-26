#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
"""
Calculate Path Diversity Score for Exposure-Disease Associations
Network Structure: Exposure (E) -> GO -> Disease (D)
Fixed path length = 2
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set font
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# Ignore warnings
warnings.filterwarnings('ignore')

# Set working directory (using relative path)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Auto-detect folder starting with 'ebio_summary_'
ebio_folders = [f for f in os.listdir(base_dir) if f.startswith('ebio_summary_') and os.path.isdir(os.path.join(base_dir, f))]
if not ebio_folders:
    raise FileNotFoundError("Folder starting with 'ebio_summary_' not found")
ebio_folder = ebio_folders[0]  # Use the first matching folder
print(f"Data folder found: {ebio_folder}")

data_dir = os.path.join(base_dir, ebio_folder, 'buildModel')
results_dir = os.path.join(base_dir, 'results_pds_score', 'path_diversity_score_EGoD')

# Create results directory
os.makedirs(results_dir, exist_ok=True)

print("=" * 80)
print("Path Diversity Score Calculation for Exposure-Disease Associations")
print("=" * 80)

# Read node and edge data
print("\n1. Loading data...")
nodes_file = os.path.join(data_dir, 'EGoD_nodes_corrected.csv')
edges_file = os.path.join(data_dir, 'EGoD_edges_corrected.csv')

nodes_df = pd.read_csv(nodes_file)
edges_df = pd.read_csv(edges_file)

print(f"   Number of nodes: {len(nodes_df)}")
print(f"   Number of edges: {len(edges_df)}")

# Load ID correction file for unified naming
id_correction_file = os.path.join(base_dir, 'id_corrected.xlsx')
if os.path.exists(id_correction_file):
    id_correction_df = pd.read_excel(id_correction_file)
    # Create mapping from exid to unified_id and unified_name
    exid_to_unified_id = dict(zip(id_correction_df['exid'], id_correction_df['unified_id']))
    exid_to_unified_name = dict(zip(id_correction_df['exid'], id_correction_df['unified_name']))
    
    # Create mapping from numeric disease ID to unified_id and unified_name
    # Filter for disease rows (exid starts with EX:D)
    disease_rows = id_correction_df[id_correction_df['exid'].str.startswith('EX:D', na=False)]
    disease_numeric_to_unified_id = {}
    disease_numeric_to_unified_name = {}
    for _, row in disease_rows.iterrows():
        # Convert unified_id to string to match edge data format
        # Handle both numeric and string unified_id
        unified_id_val = row['unified_id']
        if pd.notna(unified_id_val):
            try:
                disease_id = str(int(float(unified_id_val)))
            except (ValueError, TypeError):
                disease_id = str(unified_id_val)
        else:
            continue
        disease_numeric_to_unified_id[disease_id] = row['unified_id']
        disease_numeric_to_unified_name[disease_id] = row['unified_name']
    print(f"   Loaded {len(exid_to_unified_id)} exid to unified_id mappings")
    print(f"   Loaded {len(disease_numeric_to_unified_id)} disease numeric ID mappings")
else:
    exid_to_unified_id = {}
    exid_to_unified_name = {}
    disease_numeric_to_unified_id = {}
    disease_numeric_to_unified_name = {}
    print("   Warning: id_corrected.xlsx not found, using original names")

# Count nodes by type
exposure_nodes = nodes_df[nodes_df['EXID'].str.startswith('EX:E')]
go_nodes = nodes_df[nodes_df['EXID'].str.startswith('GO:')]
# In corrected files, disease nodes are identified by numeric IDs in edges
print(f"\n   Exposure nodes: {len(exposure_nodes)}")
print(f"   GO nodes: {len(go_nodes)}")

# Create node name mapping, using unified_name for exposure and disease nodes
node_names = {}
for _, row in nodes_df.iterrows():
    node_id = row['EXID']
    # Use unified_name for exposure and disease nodes if available
    if node_id in exid_to_unified_name:
        node_names[node_id] = exid_to_unified_name[node_id]
    else:
        node_names[node_id] = row['Preferred name']

# 2. Build network structure
print("\n2. Building network structure...")

# Exposure -> GO edges (Edge type = 1)
exposure_go_edges = edges_df[
    (edges_df['Source'].str.startswith('EX:E')) & 
    (edges_df['Target'].str.startswith('GO:'))
]

# GO -> Disease edges (Edge type = 2)
# In corrected files, disease IDs are numeric (not EX:D prefix)
go_disease_edges = edges_df[
    (edges_df['Source'].str.startswith('GO:')) & 
    (edges_df['Edge type'] == 2)
]

print(f"   Exposure->GO edges: {len(exposure_go_edges)}")
print(f"   GO->Disease edges: {len(go_disease_edges)}")

# Build adjacency dictionaries
# exposure_to_gos: {exposure_id: [go_id1, go_id2, ...]}
exposure_to_gos = defaultdict(list)
for _, row in exposure_go_edges.iterrows():
    exposure_to_gos[row['Source']].append(row['Target'])

# go_to_diseases: {go_id: [disease_id1, disease_id2, ...]}
go_to_diseases = defaultdict(list)
for _, row in go_disease_edges.iterrows():
    # Target is now numeric disease ID
    disease_id = str(row['Target'])
    go_to_diseases[row['Source']].append(disease_id)

# 3. Calculate Path Diversity Score
print("\n3. Calculating Path Diversity Scores for Exposure-Disease pairs...")

# Store results
path_scores = []

# Iterate through all exposures
for exposure_id in exposure_to_gos.keys():
    exposure_name = node_names.get(exposure_id, exposure_id)
    
    # Get all GO terms connected to this exposure
    connected_gos = exposure_to_gos[exposure_id]
    
    # Find all reachable diseases through these GO terms
    # Use dictionary to record paths for each disease
    disease_paths = defaultdict(list)
    
    for go_id in connected_gos:
        go_name = node_names.get(go_id, go_id)
        
        # Get all diseases connected to this GO term
        connected_diseases = go_to_diseases.get(go_id, [])
        
        for disease_id in connected_diseases:
            # Record path: exposure -> GO -> disease
            disease_paths[disease_id].append(go_id)
    
    # Calculate Path Diversity Score for each exposure-disease pair
    for disease_id, proteins_in_paths in disease_paths.items():
        # Get unified_id and unified_name for exposure
        exposure_unified_id = exid_to_unified_id.get(exposure_id, exposure_id)
        exposure_unified_name = exid_to_unified_name.get(exposure_id, exposure_name)
        
        # Get unified_id and unified_name for disease (disease_id is numeric)
        disease_unified_id = disease_numeric_to_unified_id.get(disease_id, disease_id)
        disease_unified_name = disease_numeric_to_unified_name.get(disease_id, disease_id)
        
        # Number of paths
        num_paths = len(proteins_in_paths)
        
        # Path Diversity Score (PDS)
        # Method 1: Simple count - number of paths
        pds_simple = num_paths
        
        # Method 2: Weighted by GO importance - using GO degree
        # Get degree for each mediating GO term
        go_degrees = []
        for go_id in proteins_in_paths:
            degree = nodes_df[nodes_df['EXID'] == go_id]['Degree'].values
            if len(degree) > 0:
                go_degrees.append(degree[0])
            else:
                go_degrees.append(1)  # Default value
        
        # Weighted score: considering GO importance
        pds_weighted = sum(go_degrees)
        
        # Normalized weighted score
        pds_normalized = sum(go_degrees) / num_paths if num_paths > 0 else 0
        
        path_scores.append({
            'Exposure_ID': exposure_unified_id,
            'Exposure_Name': exposure_unified_name,
            'Disease_ID': disease_unified_id,
            'Disease_Name': disease_unified_name,
            'Num_Paths': num_paths,
            'PDS_Simple': pds_simple,
            'PDS_Weighted': pds_weighted,
            'PDS_Normalized': pds_normalized,
            'Mediating_GOs': '|'.join(proteins_in_paths),
            'GO_Names': '|'.join([str(node_names.get(p, p)) if pd.notna(node_names.get(p, p)) else p for p in proteins_in_paths])
        })

# Convert to DataFrame
results_df = pd.DataFrame(path_scores)

# Calculate PDS_Scaled: each value divided by max PDS_Weighted
max_pds_weighted = results_df['PDS_Weighted'].max()
results_df['PDS_Scaled'] = results_df['PDS_Weighted'] / max_pds_weighted

# Reorder columns to place PDS_Scaled right after PDS_Weighted
column_order = ['Exposure_ID', 'Exposure_Name', 'Disease_ID', 'Disease_Name', 
                'Num_Paths', 'PDS_Simple', 'PDS_Weighted', 'PDS_Scaled', 'PDS_Normalized',
                'Mediating_GOs', 'GO_Names']
results_df = results_df[column_order]

print(f"   Calculated {len(results_df)} exposure-disease pairs")
print(f"   Max PDS_Weighted: {max_pds_weighted:.2f}")

# 4. Statistical analysis
print("\n4. Statistical analysis...")
print(f"\n   Path count statistics:")
print(f"   - Min paths: {results_df['Num_Paths'].min()}")
print(f"   - Max paths: {results_df['Num_Paths'].max()}")
print(f"   - Mean paths: {results_df['Num_Paths'].mean():.2f}")
print(f"   - Median paths: {results_df['Num_Paths'].median():.2f}")

print(f"\n   PDS_Simple statistics:")
print(f"   - Range: {results_df['PDS_Simple'].min()} - {results_df['PDS_Simple'].max()}")
print(f"   - Mean: {results_df['PDS_Simple'].mean():.2f}")

print(f"\n   PDS_Weighted statistics:")
print(f"   - Range: {results_df['PDS_Weighted'].min():.2f} - {results_df['PDS_Weighted'].max():.2f}")
print(f"   - Mean: {results_df['PDS_Weighted'].mean():.2f}")

# 5. Save results
print("\n5. Saving results...")

# Sort by PDS_Weighted in descending order
results_sorted = results_df.sort_values('PDS_Weighted', ascending=False)

# Save complete results
output_file = os.path.join(results_dir, 'exposure_disease_path_diversity_scores.csv')
results_sorted.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"   Complete results saved: {output_file}")

# Save simplified results (Exposure_ID to PDS_Scaled)
simple_columns = ['Exposure_ID', 'Exposure_Name', 'Disease_ID', 'Disease_Name', 
                  'Num_Paths', 'PDS_Simple', 'PDS_Weighted', 'PDS_Scaled', 'PDS_Normalized']
output_file_simple = os.path.join(results_dir, 'exposure_disease_path_diversity_scores_simple.csv')
results_sorted[simple_columns].to_csv(output_file_simple, index=False, encoding='utf-8-sig')
print(f"   Simplified results saved: {output_file_simple}")


# 6. Display Top 20 results
print("\n6. Top 20 Exposure-Disease pairs (sorted by PDS_Weighted):")
print("=" * 120)
print(f"{'Rank':<6}{'Exposure':<30}{'Disease':<30}{'Paths':<10}{'PDS_Weighted':<15}")
print("=" * 120)

for idx, row in results_sorted.head(20).iterrows():
    rank = results_sorted.index.get_loc(idx) + 1
    print(f"{rank:<6}{row['Exposure_Name'][:28]:<30}{row['Disease_Name'][:28]:<30}"
          f"{row['Num_Paths']:<10}{row['PDS_Weighted']:<15.2f}")

# 7. Visualization
print("\n7. Generating visualization plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Path count distribution
ax1 = axes[0, 0]
ax1.hist(results_df['Num_Paths'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Number of Paths', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of Path Counts (Exposure-Disease)', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: PDS_Weighted distribution
ax2 = axes[0, 1]
ax2.hist(results_df['PDS_Weighted'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
ax2.set_xlabel('PDS Weighted Score', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of PDS Weighted Scores', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Top 15 Exposure-Disease pairs
ax3 = axes[1, 0]
top15 = results_sorted.head(15)
y_pos = np.arange(len(top15))
labels = [f"{row['Exposure_Name'][:15]}→{row['Disease_Name'][:15]}" 
          for _, row in top15.iterrows()]
ax3.barh(y_pos, top15['PDS_Weighted'], color='mediumseagreen', alpha=0.8)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(labels, fontsize=9)
ax3.set_xlabel('PDS Weighted Score', fontsize=12)
ax3.set_title('Top 15 Exposure-Disease Pairs', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

# Plot 4: Path count vs PDS_Weighted scatter plot
ax4 = axes[1, 1]
scatter = ax4.scatter(results_df['Num_Paths'], results_df['PDS_Weighted'], 
                     c=results_df['PDS_Normalized'], cmap='viridis', 
                     alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax4.set_xlabel('Number of Paths', fontsize=12)
ax4.set_ylabel('PDS Weighted Score', fontsize=12)
ax4.set_title('Path Count vs PDS Weighted Score', fontsize=14, fontweight='bold')
ax4.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('PDS Normalized', fontsize=10)

plt.tight_layout()
plot_file = os.path.join(results_dir, 'viz_path_diversity_score_analysis.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"   Visualization plots saved: {plot_file}")

# Generate cumulative percentage stacked bar chart
print("\n   Generating cumulative percentage stacked bar chart...")

# Group by Disease_Name and Exposure_Name, sum PDS_Weighted
disease_exposure_scores = results_df.groupby(['Disease_Name', 'Exposure_Name'])['PDS_Weighted'].sum().reset_index()

# Calculate total PDS_Weighted for each disease
disease_totals = disease_exposure_scores.groupby('Disease_Name')['PDS_Weighted'].sum()

# Calculate proportions
disease_exposure_scores['Proportion'] = disease_exposure_scores.apply(
    lambda row: row['PDS_Weighted'] / disease_totals[row['Disease_Name']] * 100, 
    axis=1
)

# Pivot data for stacked bar chart
pivot_data = disease_exposure_scores.pivot(
    index='Disease_Name', 
    columns='Exposure_Name', 
    values='Proportion'
).fillna(0)

# Sort by total score (descending)
disease_order = disease_totals.sort_values(ascending=False).index
pivot_data = pivot_data.loc[disease_order]

# Create figure for cumulative percentage chart
fig_cum, ax_cum = plt.subplots(figsize=(20, 8))

# Define colors for exposures
colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2']
exposure_names = pivot_data.columns.tolist()

# Create stacked bar chart
pivot_data.plot(
    kind='bar',
    stacked=True,
    ax=ax_cum,
    color=colors[:len(exposure_names)],
    width=0.85,
    edgecolor='white',
    linewidth=0.3
)

# Customize plot
ax_cum.set_xlabel('Disease Name', fontsize=12, fontweight='bold')
ax_cum.set_ylabel('Relative Proportion (%)', fontsize=12, fontweight='bold')
ax_cum.set_title('Cumulative Percentage of PDS_Weighted by Disease and Exposure', fontsize=14, fontweight='bold', pad=20)
ax_cum.set_ylim(0, 100)
ax_cum.legend(title='Exposure', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=10)
ax_cum.tick_params(axis='x', rotation=90, labelsize=6)
ax_cum.tick_params(axis='y', labelsize=10)
ax_cum.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

# Save cumulative percentage chart
cumulative_plot_file = os.path.join(results_dir, 'viz_cumulative_percentage_by_disease.png')
plt.savefig(cumulative_plot_file, dpi=300, bbox_inches='tight')
print(f"   Cumulative percentage chart saved: {cumulative_plot_file}")
plt.close()

# Generate total score bar chart for each disease
print("\n   Generating total score bar chart by disease...")

# Create figure for total score chart
fig_total, ax_total = plt.subplots(figsize=(20, 8))

# Use the same disease order (already sorted by total score)
disease_totals_sorted = disease_totals.loc[disease_order]

# Create bar chart
bars = ax_total.bar(range(len(disease_totals_sorted)), disease_totals_sorted.values, 
                     color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.5, width=0.85)

# Customize plot
ax_total.set_xlabel('Disease Name', fontsize=12, fontweight='bold')
ax_total.set_ylabel('Total PDS_Weighted Score', fontsize=12, fontweight='bold')
ax_total.set_title('Total PDS_Weighted Score by Disease (All Exposures Combined)', fontsize=14, fontweight='bold', pad=20)
ax_total.set_xticks(range(len(disease_totals_sorted)))
ax_total.set_xticklabels(disease_totals_sorted.index, rotation=90, ha='right', fontsize=6)
ax_total.set_xlim(-0.5, len(disease_totals_sorted) - 0.5)
ax_total.tick_params(axis='y', labelsize=10)
ax_total.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

# Save total score chart
total_score_plot_file = os.path.join(results_dir, 'viz_total_score_by_disease.png')
plt.savefig(total_score_plot_file, dpi=300, bbox_inches='tight')
print(f"   Total score chart saved: {total_score_plot_file}")
plt.close()

# Generate combined figure with both charts (2x1 layout)
print("\n   Generating combined chart (2x1 layout)...")

# Create figure with 2 subplots
fig_combined, (ax_comb_cum, ax_comb_total) = plt.subplots(2, 1, figsize=(20, 16))

# Reduced font size for disease names (10% reduction: 6 * 0.9 = 5.4)
disease_fontsize = 10

# Top subplot: Cumulative percentage stacked bar chart
pivot_data.plot(
    kind='bar',
    stacked=True,
    ax=ax_comb_cum,
    color=colors[:len(exposure_names)],
    width=0.85,
    edgecolor='white',
    linewidth=0.3
)
ax_comb_cum.set_xlabel('Disease Name', fontsize=12, fontweight='bold')
ax_comb_cum.set_ylabel('Relative Proportion (%)', fontsize=12, fontweight='bold')
ax_comb_cum.set_title('Cumulative Percentage of PDS_Weighted by Disease and Exposure', fontsize=14, fontweight='bold', pad=20)
ax_comb_cum.set_ylim(0, 100)
ax_comb_cum.legend(title='Exposure', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=10)
ax_comb_cum.tick_params(axis='x', rotation=90, labelsize=disease_fontsize)
ax_comb_cum.tick_params(axis='y', labelsize=10)
ax_comb_cum.grid(axis='y', alpha=0.3, linestyle='--')

# Bottom subplot: Total score bar chart
bars = ax_comb_total.bar(range(len(disease_totals_sorted)), disease_totals_sorted.values, 
                          color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.5, width=0.85)
ax_comb_total.set_xlabel('Disease Name', fontsize=12, fontweight='bold')
ax_comb_total.set_ylabel('Total PDS_Weighted Score', fontsize=12, fontweight='bold')
ax_comb_total.set_title('Total PDS_Weighted Score by Disease (All Exposures Combined)', fontsize=14, fontweight='bold', pad=20)
ax_comb_total.set_xticks(range(len(disease_totals_sorted)))
ax_comb_total.set_xticklabels(disease_totals_sorted.index, rotation=90, ha='right', fontsize=disease_fontsize)
ax_comb_total.set_xlim(-0.5, len(disease_totals_sorted) - 0.5)
ax_comb_total.tick_params(axis='y', labelsize=10)
ax_comb_total.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

# Save combined chart
combined_plot_file = os.path.join(results_dir, 'viz_combined_percentage_and_total_score.png')
plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
print(f"   Combined chart saved: {combined_plot_file}")
plt.close()

# 8. Generate summary statistics by exposure and disease
print("\n8. Generating summary statistics for exposures and diseases...")

# Aggregate by exposure
exposure_summary = results_df.groupby(['Exposure_ID', 'Exposure_Name']).agg({
    'Disease_ID': 'count',
    'Num_Paths': 'sum',
    'PDS_Weighted': 'sum',
    'PDS_Scaled': 'sum'
}).rename(columns={
    'Disease_ID': 'Num_Diseases',
    'Num_Paths': 'Total_Paths',
    'PDS_Weighted': 'Total_PDS_Weighted',
    'PDS_Scaled': 'Total_PDS_Scaled'
}).reset_index()

exposure_summary = exposure_summary.sort_values('Total_PDS_Weighted', ascending=False)
exposure_summary_file = os.path.join(results_dir, 'exposure_summary_scores.csv')
exposure_summary.to_csv(exposure_summary_file, index=False, encoding='utf-8-sig')
print(f"   Exposure summary saved: {exposure_summary_file}")

# Aggregate by disease
disease_summary = results_df.groupby(['Disease_ID', 'Disease_Name']).agg({
    'Exposure_ID': 'count',
    'Num_Paths': 'sum',
    'PDS_Weighted': 'sum',
    'PDS_Scaled': 'sum'
}).rename(columns={
    'Exposure_ID': 'Num_Exposures',
    'Num_Paths': 'Total_Paths',
    'PDS_Weighted': 'Total_PDS_Weighted',
    'PDS_Scaled': 'Total_PDS_Scaled'
}).reset_index()

disease_summary = disease_summary.sort_values('Total_PDS_Weighted', ascending=False)
disease_summary_file = os.path.join(results_dir, 'disease_summary_scores.csv')
disease_summary.to_csv(disease_summary_file, index=False, encoding='utf-8-sig')
print(f"   Disease summary saved: {disease_summary_file}")

# Display Top 10 exposures
print("\n   Top 10 Exposures (by Total_PDS_Weighted):")
print("   " + "-" * 80)
for idx, row in exposure_summary.head(10).iterrows():
    print(f"   {row['Exposure_Name'][:40]:<42} | Diseases: {row['Num_Diseases']:<3} | "
          f"Paths: {row['Total_Paths']:<4} | Score: {row['Total_PDS_Weighted']:.2f}")

# Display Top 10 diseases
print("\n   Top 10 Diseases (by Total_PDS_Weighted):")
print("   " + "-" * 80)
for idx, row in disease_summary.head(10).iterrows():
    print(f"   {row['Disease_Name'][:40]:<42} | Exposures: {row['Num_Exposures']:<3} | "
          f"Paths: {row['Total_Paths']:<4} | Score: {row['Total_PDS_Weighted']:.2f}")

print("\n" + "=" * 80)
print("Calculation completed!")
print("=" * 80)
print("\nOutput files:")
print(f"  1. {output_file}")
print(f"  2. {output_file_simple}")
print(f"  3. {exposure_summary_file}")
print(f"  4. {disease_summary_file}")
print(f"  5. {plot_file}")
print("\nScore descriptions:")
print("  - Num_Paths: Number of pathways from exposure to disease")
print("  - PDS_Simple: Simple path count score (equals Num_Paths)")
print("  - PDS_Weighted: Weighted score (considering protein degree)")
print("  - PDS_Scaled: Scaled score (PDS_Weighted / max(PDS_Weighted))")
print("  - PDS_Normalized: Normalized score (PDS_Weighted / Num_Paths)")
print("=" * 80)

# %%
