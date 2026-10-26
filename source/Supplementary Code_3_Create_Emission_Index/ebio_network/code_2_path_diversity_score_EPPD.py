#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate Path Diversity Score for Exposure-Protein-Protein-Disease (EPPD) Associations
Network Structure: 
  - PPI path: Exposure (E) -> Protein_1 (P1) -> Protein_2 (P2) -> Disease (D) [length=3]
"""
#%%
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
results_dir = os.path.join(base_dir, 'results_pds_score', 'path_diversity_score_EPPD')

# Create results directory
os.makedirs(results_dir, exist_ok=True)

print("=" * 80)
print("EPPD Path Diversity Score Calculation")
print("(Exposure-Protein-Protein-Disease Network with PPI)")
print("=" * 80)

# Read node and edge data
print("\n1. Loading data...")
nodes_file = os.path.join(data_dir, 'EPPD_nodes_corrected.csv')
edges_file = os.path.join(data_dir, 'EPPD_edges_corrected.csv')

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
protein_nodes = nodes_df[nodes_df['EXID'].str.startswith('EX:P')]
# In corrected files, disease nodes are identified by numeric IDs in edges
print(f"\n   Exposure nodes: {len(exposure_nodes)}")
print(f"   Protein nodes: {len(protein_nodes)}")

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
print("\n2. Building EPPD network structure...")

# Identify edge types based on Source and Target groups
# Edge type 1: Exposure -> Protein_1
exposure_protein1_edges = edges_df[
    (edges_df['Source'].str.startswith('EX:E')) & 
    (edges_df['Target'].str.startswith('EX:P')) &
    (edges_df['Target group'] == 'Protein_1')
]

# Edge type 2: Protein_1 -> Protein_2 (protein-protein interaction)
protein_protein_edges = edges_df[
    (edges_df['Source'].str.startswith('EX:P')) & 
    (edges_df['Target'].str.startswith('EX:P'))
]

# Edge type 3: Protein_2 -> Disease
# In corrected files, disease IDs are numeric (not EX:D prefix)
protein_disease_edges = edges_df[
    (edges_df['Source'].str.startswith('EX:P')) & 
    (edges_df['Edge type'] == 3)
]

print(f"   Exposure->Protein_1 edges: {len(exposure_protein1_edges)}")
print(f"   Protein_1->Protein_2 edges (PPI): {len(protein_protein_edges)}")
print(f"   Protein_2->Disease edges: {len(protein_disease_edges)}")

# Build adjacency dictionaries
# exposure_to_protein1: {exposure_id: [protein1_id1, protein1_id2, ...]}
exposure_to_protein1 = defaultdict(list)
for _, row in exposure_protein1_edges.iterrows():
    exposure_to_protein1[row['Source']].append(row['Target'])

# protein1_to_protein2: {protein1_id: [protein2_id1, protein2_id2, ...]}
protein1_to_protein2 = defaultdict(list)
for _, row in protein_protein_edges.iterrows():
    protein1_to_protein2[row['Source']].append(row['Target'])

# protein2_to_diseases: {protein2_id: [disease_id1, disease_id2, ...]}
protein2_to_diseases = defaultdict(list)
for _, row in protein_disease_edges.iterrows():
    # Target is now numeric disease ID
    disease_id = str(row['Target'])
    protein2_to_diseases[row['Source']].append(disease_id)

# 3. Calculate Path Diversity Score
print("\n3. Calculating Path Diversity Scores for Exposure-Disease pairs...")
print("   Only considering PPI-mediated pathways (E->P1->P2->D)...")

# Store results
path_scores = []

# Iterate through all exposures
for exposure_id in exposure_to_protein1.keys():
    exposure_name = node_names.get(exposure_id, exposure_id)
    
    # Dictionary to store all paths to each disease
    # Structure: {disease_id: [(protein1_id, protein2_id), ...]}
    disease_paths = defaultdict(list)
    
    # Get all Protein_1 connected to this exposure
    protein1_list = exposure_to_protein1.get(exposure_id, [])
    
    # For each Protein_1, find Protein_2 through PPI
    for protein1_id in protein1_list:
        # Get Protein_2 that interact with Protein_1
        protein2_interactors = protein1_to_protein2.get(protein1_id, [])
        
        for protein2_id in protein2_interactors:
            # Check if Protein_2 connects to any disease
            connected_diseases = protein2_to_diseases.get(protein2_id, [])
            
            for disease_id in connected_diseases:
                # Record path: E -> P1 -> P2 -> D
                disease_paths[disease_id].append((protein1_id, protein2_id))
    
    # Calculate Path Diversity Score for each exposure-disease pair
    for disease_id, paths in disease_paths.items():
        # Get unified_id and unified_name for exposure
        exposure_unified_id = exid_to_unified_id.get(exposure_id, exposure_id)
        exposure_unified_name = exid_to_unified_name.get(exposure_id, exposure_name)
        
        # Get unified_id and unified_name for disease (disease_id is numeric)
        disease_unified_id = disease_numeric_to_unified_id.get(disease_id, disease_id)
        disease_unified_name = disease_numeric_to_unified_name.get(disease_id, disease_id)
        
        # Total number of paths
        num_ppi_paths = len(paths)
        
        # Collect all unique proteins (both P1 and P2)
        all_protein1 = [p[0] for p in paths]
        all_protein2 = [p[1] for p in paths]
        all_proteins = all_protein1 + all_protein2
        
        # Remove duplicates while preserving order
        unique_proteins = list(dict.fromkeys(all_proteins))
        unique_protein1 = list(dict.fromkeys(all_protein1))
        unique_protein2 = list(dict.fromkeys(all_protein2))
        
        # Get degree for each unique protein
        protein_degrees = []
        for protein_id in unique_proteins:
            degree = nodes_df[nodes_df['EXID'] == protein_id]['Degree'].values
            if len(degree) > 0:
                protein_degrees.append(degree[0])
            else:
                protein_degrees.append(1)  # Default value
        
        # Calculate scores
        pds_simple = num_ppi_paths
        pds_weighted = sum(protein_degrees)
        pds_normalized = pds_weighted / len(unique_proteins) if len(unique_proteins) > 0 else 0
        
        # Path length is always 3 for E->P1->P2->D
        path_length = 3
        
        path_scores.append({
            'Exposure_ID': exposure_unified_id,
            'Exposure_Name': exposure_unified_name,
            'Disease_ID': disease_unified_id,
            'Disease_Name': disease_unified_name,
            'Num_Paths_PPI': num_ppi_paths,
            'Num_Unique_Proteins': len(unique_proteins),
            'Num_Unique_Protein1': len(unique_protein1),
            'Num_Unique_Protein2': len(unique_protein2),
            'Path_Length': path_length,
            'PDS_Simple': pds_simple,
            'PDS_Weighted': pds_weighted,
            'PDS_Normalized': pds_normalized,
            'Mediating_Protein1': '|'.join(unique_protein1),
            'Mediating_Protein2': '|'.join(unique_protein2),
            'Protein1_Names': '|'.join([node_names.get(p, p) for p in unique_protein1]),
            'Protein2_Names': '|'.join([node_names.get(p, p) for p in unique_protein2])
        })

# Convert to DataFrame
results_df = pd.DataFrame(path_scores)

# Calculate PDS_Scaled: each value divided by max PDS_Weighted
max_pds_weighted = results_df['PDS_Weighted'].max()
results_df['PDS_Scaled'] = results_df['PDS_Weighted'] / max_pds_weighted

# Reorder columns to place PDS_Scaled right after PDS_Weighted
column_order = ['Exposure_ID', 'Exposure_Name', 'Disease_ID', 'Disease_Name', 
                'Num_Paths_PPI', 'Num_Unique_Proteins', 'Num_Unique_Protein1', 'Num_Unique_Protein2',
                'Path_Length', 'PDS_Simple', 'PDS_Weighted', 'PDS_Scaled', 'PDS_Normalized',
                'Mediating_Protein1', 'Mediating_Protein2', 'Protein1_Names', 'Protein2_Names']
results_df = results_df[column_order]

print(f"   Calculated {len(results_df)} exposure-disease pairs")
print(f"   Max PDS_Weighted: {max_pds_weighted:.2f}")

# 4. Statistical analysis
print("\n4. Statistical analysis...")
print(f"\n   PPI path statistics:")
print(f"   - Min paths: {results_df['Num_Paths_PPI'].min()}")
print(f"   - Max paths: {results_df['Num_Paths_PPI'].max()}")
print(f"   - Mean paths: {results_df['Num_Paths_PPI'].mean():.2f}")
print(f"   - Median paths: {results_df['Num_Paths_PPI'].median():.2f}")

print(f"\n   Protein statistics:")
print(f"   - Mean unique proteins (total): {results_df['Num_Unique_Proteins'].mean():.2f}")
print(f"   - Mean unique Protein_1: {results_df['Num_Unique_Protein1'].mean():.2f}")
print(f"   - Mean unique Protein_2: {results_df['Num_Unique_Protein2'].mean():.2f}")

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
                  'Num_Paths_PPI', 'Num_Unique_Proteins', 'Num_Unique_Protein1', 
                  'Num_Unique_Protein2', 'Path_Length', 'PDS_Simple', 'PDS_Weighted', 'PDS_Scaled', 'PDS_Normalized']
output_file_simple = os.path.join(results_dir, 'exposure_disease_path_diversity_scores_simple.csv')
results_sorted[simple_columns].to_csv(output_file_simple, index=False, encoding='utf-8-sig')
print(f"   Simplified results saved: {output_file_simple}")


# 6. Display Top 20 results
print("\n6. Top 20 Exposure-Disease pairs (sorted by PDS_Weighted):")
print("=" * 130)
print(f"{'Rank':<6}{'Exposure':<25}{'Disease':<25}{'PPI_Paths':<10}{'Protein1':<10}{'Protein2':<10}{'PDS_W':<10}")
print("=" * 130)

for idx, row in results_sorted.head(20).iterrows():
    rank = results_sorted.index.get_loc(idx) + 1
    print(f"{rank:<6}{str(row['Exposure_Name'])[:23]:<25}{str(row['Disease_Name'])[:23]:<25}"
          f"{row['Num_Paths_PPI']:<10}{row['Num_Unique_Protein1']:<10}{row['Num_Unique_Protein2']:<10}"
          f"{row['PDS_Weighted']:<10.2f}")

# 7. Visualization
print("\n7. Generating visualization plots...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Plot 1: PPI path count distribution
ax1 = axes[0, 0]
ax1.hist(results_df['Num_Paths_PPI'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Number of PPI Paths', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of PPI Path Counts', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Protein count comparison
ax2 = axes[0, 1]
protein_types = ['Protein_1', 'Protein_2', 'Total']
protein_means = [
    results_df['Num_Unique_Protein1'].mean(),
    results_df['Num_Unique_Protein2'].mean(),
    results_df['Num_Unique_Proteins'].mean()
]
ax2.bar(protein_types, protein_means, color=['steelblue', 'lightcoral', 'mediumseagreen'], alpha=0.8)
ax2.set_ylabel('Average Number of Unique Proteins', fontsize=12)
ax2.set_title('Average Unique Proteins by Type', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Plot 3: PDS_Weighted distribution
ax3 = axes[0, 2]
ax3.hist(results_df['PDS_Weighted'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
ax3.set_xlabel('PDS Weighted Score', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Distribution of PDS Weighted Scores', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Top 15 Exposure-Disease pairs
ax4 = axes[1, 0]
top15 = results_sorted.head(15)
y_pos = np.arange(len(top15))
labels = [f"{str(row['Exposure_Name'])[:12]}→{str(row['Disease_Name'])[:12]}" 
          for _, row in top15.iterrows()]
ax4.barh(y_pos, top15['PDS_Weighted'], color='mediumseagreen', alpha=0.8)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(labels, fontsize=9)
ax4.set_xlabel('PDS Weighted Score', fontsize=12)
ax4.set_title('Top 15 Exposure-Disease Pairs', fontsize=14, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

# Plot 5: Path count vs PDS_Weighted scatter plot
ax5 = axes[1, 1]
scatter = ax5.scatter(results_df['Num_Paths_PPI'], results_df['PDS_Weighted'], 
                     c=results_df['Num_Unique_Proteins'], cmap='viridis', 
                     alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax5.set_xlabel('Number of PPI Paths', fontsize=12)
ax5.set_ylabel('PDS Weighted Score', fontsize=12)
ax5.set_title('PPI Path Count vs PDS Weighted', fontsize=14, fontweight='bold')
ax5.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax5)
cbar.set_label('Unique Proteins', fontsize=10)

# Plot 6: Protein1 vs Protein2 count
ax6 = axes[1, 2]
ax6.scatter(results_df['Num_Unique_Protein1'], results_df['Num_Unique_Protein2'], 
           alpha=0.5, s=30, color='mediumpurple', edgecolors='black', linewidth=0.5)
ax6.set_xlabel('Number of Unique Protein_1', fontsize=12)
ax6.set_ylabel('Number of Unique Protein_2', fontsize=12)
ax6.set_title('Protein_1 vs Protein_2 Distribution', fontsize=14, fontweight='bold')
ax6.grid(alpha=0.3)

plt.tight_layout()
plot_file = os.path.join(results_dir, 'viz_path_diversity_score_analysis_EPPD.png')
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
ax_total.set_xticklabels(disease_totals_sorted.index, rotation=90, ha='right', fontsize=12)
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
    'Num_Paths_PPI': 'sum',
    'PDS_Weighted': 'sum',
    'PDS_Scaled': 'sum'
}).rename(columns={
    'Disease_ID': 'Num_Diseases',
    'Num_Paths_PPI': 'Total_PPI_Paths',
    'PDS_Weighted': 'Total_PDS_Weighted',
    'PDS_Scaled': 'Total_PDS_Scaled'
}).reset_index()

exposure_summary = exposure_summary.sort_values('Total_PDS_Weighted', ascending=False)
exposure_summary_file = os.path.join(results_dir, 'exposure_summary_scores_EPPD.csv')
exposure_summary.to_csv(exposure_summary_file, index=False, encoding='utf-8-sig')
print(f"   Exposure summary saved: {exposure_summary_file}")

# Aggregate by disease
disease_summary = results_df.groupby(['Disease_ID', 'Disease_Name']).agg({
    'Exposure_ID': 'count',
    'Num_Paths_PPI': 'sum',
    'PDS_Weighted': 'sum',
    'PDS_Scaled': 'sum'
}).rename(columns={
    'Exposure_ID': 'Num_Exposures',
    'Num_Paths_PPI': 'Total_PPI_Paths',
    'PDS_Weighted': 'Total_PDS_Weighted',
    'PDS_Scaled': 'Total_PDS_Scaled'
}).reset_index()

disease_summary = disease_summary.sort_values('Total_PDS_Weighted', ascending=False)
disease_summary_file = os.path.join(results_dir, 'disease_summary_scores_EPPD.csv')
disease_summary.to_csv(disease_summary_file, index=False, encoding='utf-8-sig')
print(f"   Disease summary saved: {disease_summary_file}")

# Display Top 10 exposures
print("\n   Top 10 Exposures (by Total_PDS_Weighted):")
print("   " + "-" * 90)
for idx, row in exposure_summary.head(10).iterrows():
    print(f"   {str(row['Exposure_Name'])[:35]:<37} | Diseases: {row['Num_Diseases']:<3} | "
          f"PPI Paths: {row['Total_PPI_Paths']:<5} | Score: {row['Total_PDS_Weighted']:.2f}")

# Display Top 10 diseases
print("\n   Top 10 Diseases (by Total_PDS_Weighted):")
print("   " + "-" * 90)
for idx, row in disease_summary.head(10).iterrows():
    print(f"   {str(row['Disease_Name'])[:35]:<37} | Exposures: {row['Num_Exposures']:<3} | "
          f"PPI Paths: {row['Total_PPI_Paths']:<5} | Score: {row['Total_PDS_Weighted']:.2f}")

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
print("  - Num_Paths_PPI: Number of E->P1->P2->D pathways (with protein-protein interaction)")
print("  - Num_Unique_Protein1: Number of unique Protein_1 in pathways")
print("  - Num_Unique_Protein2: Number of unique Protein_2 in pathways")
print("  - PDS_Weighted: Weighted score (sum of protein degrees)")
print("  - PDS_Scaled: Scaled score (PDS_Weighted / max(PDS_Weighted))")
print("  - PDS_Normalized: Normalized score (PDS_Weighted / Num_Unique_Proteins)")
print("  - Path_Length: Fixed at 3 (E->P1->P2->D)")
print("=" * 80)
