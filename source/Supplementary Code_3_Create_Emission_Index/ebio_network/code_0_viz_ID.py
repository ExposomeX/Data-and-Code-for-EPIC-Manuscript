#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
"""
Visualize the hierarchical relationship between unified_name and preferred_name
from id_corrected.xlsx, categorized by unified_id prefix (EX:E and EX:D)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set matplotlib to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_hierarchy_visualization(df, category_name, output_dir):
    """
    Create a hierarchical visualization showing preferred_name belonging to unified_name
    
    Args:
        df: DataFrame with unified_name and preferred_name columns
        category_name: 'Exposure' or 'Disease'
        output_dir: Directory to save the output
    """
    # Group by unified_name to get all preferred_names
    grouped = df.groupby('unified_name')['preferred_name'].apply(list).reset_index()
    grouped['count'] = grouped['preferred_name'].apply(len)
    grouped = grouped.sort_values('count', ascending=False)
    
    print(f"\n{category_name} category:")
    print(f"  Total unified_names: {len(grouped)}")
    print(f"  Total preferred_names: {len(df)}")
    print(f"  Average preferred_names per unified_name: {len(df)/len(grouped):.2f}")
    
    # Create a detailed text file with the mapping
    output_file = os.path.join(output_dir, f'{category_name}_hierarchy_mapping.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{category_name} Hierarchy Mapping\n")
        f.write("=" * 80 + "\n\n")
        for idx, row in grouped.iterrows():
            f.write(f"Unified Name: {row['unified_name']} ({row['count']} items)\n")
            for pref_name in row['preferred_name']:
                f.write(f"  └─ {pref_name}\n")
            f.write("\n")
    print(f"  Saved mapping to: {output_file}")
    
    # Create visualization - sunburst-style or tree diagram
    # For simplicity, create a bar chart showing the distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Top unified_names by count
    top_n = min(20, len(grouped))
    top_unified = grouped.head(top_n)
    
    y_pos = np.arange(len(top_unified))
    ax1.barh(y_pos, top_unified['count'], color='steelblue', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_unified['unified_name'], fontsize=9)
    ax1.set_xlabel('Number of Preferred Names', fontsize=11)
    ax1.set_title(f'Top {top_n} {category_name} Unified Names by Count', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add count labels on bars
    for i, v in enumerate(top_unified['count']):
        ax1.text(v + 0.1, i, str(v), va='center', fontsize=8)
    
    # Right plot: Distribution histogram
    ax2.hist(grouped['count'], bins=30, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Preferred Names per Unified Name', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'{category_name} Distribution of Preferred Names', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text = f"Total Unified Names: {len(grouped)}\n"
    stats_text += f"Total Preferred Names: {len(df)}\n"
    stats_text += f"Mean: {grouped['count'].mean():.2f}\n"
    stats_text += f"Median: {grouped['count'].median():.0f}\n"
    stats_text += f"Max: {grouped['count'].max()}\n"
    stats_text += f"Min: {grouped['count'].min()}"
    ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=9, family='monospace')
    
    plt.tight_layout()
    output_fig = os.path.join(output_dir, f'{category_name}_hierarchy_visualization.png')
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization to: {output_fig}")
    plt.close()
    
    # Create a detailed tree diagram for top unified_names
    create_tree_diagram(grouped.head(10), category_name, output_dir)

def create_tree_diagram(grouped_df, category_name, output_dir):
    """
    Create a tree diagram showing the hierarchical relationship
    
    Args:
        grouped_df: DataFrame with unified_name and preferred_name list
        category_name: 'Exposure' or 'Disease'
        output_dir: Directory to save the output
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(grouped_df) + 1)
    ax.axis('off')
    
    y_position = len(grouped_df)
    
    for idx, row in grouped_df.iterrows():
        unified_name = row['unified_name']
        preferred_names = row['preferred_name']
        count = len(preferred_names)
        
        # Draw unified_name box
        unified_box = FancyBboxPatch((0.5, y_position - 0.4), 3, 0.8,
                                      boxstyle="round,pad=0.1",
                                      edgecolor='darkblue', facecolor='lightblue',
                                      linewidth=2)
        ax.add_patch(unified_box)
        ax.text(2, y_position, unified_name, ha='center', va='center',
                fontsize=9, fontweight='bold', wrap=True)
        
        # Draw count badge
        circle = mpatches.Circle((3.7, y_position), 0.25, color='coral', ec='darkred', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(3.7, y_position, str(count), ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')
        
        # Draw preferred_names (show first few)
        max_show = min(5, count)
        pref_text = ', '.join(preferred_names[:max_show])
        if count > max_show:
            pref_text += f', ... (+{count - max_show} more)'
        
        # Draw connection line
        ax.plot([3.5, 4.5], [y_position, y_position], 'k-', linewidth=1, alpha=0.5)
        
        # Draw preferred_names box
        pref_box = FancyBboxPatch((4.5, y_position - 0.35), 5, 0.7,
                                   boxstyle="round,pad=0.05",
                                   edgecolor='darkgreen', facecolor='lightgreen',
                                   linewidth=1, alpha=0.6)
        ax.add_patch(pref_box)
        ax.text(7, y_position, pref_text, ha='center', va='center',
                fontsize=7, wrap=True)
        
        y_position -= 1
    
    # Add title
    ax.text(5, len(grouped_df) + 0.7, f'{category_name} Hierarchy: Unified Name → Preferred Names',
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add legend
    ax.text(0.5, 0.3, 'Unified Name', ha='left', va='center',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='darkblue'))
    ax.text(5, 0.3, 'Preferred Names', ha='left', va='center',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='darkgreen'))
    
    plt.tight_layout()
    output_fig = os.path.join(output_dir, f'{category_name}_tree_diagram_top10.png')
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"  Saved tree diagram to: {output_fig}")
    plt.close()

def main():
    print("="*80)
    print("Visualizing ID Hierarchy from id_corrected.xlsx")
    print("="*80)
    
    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Read id_corrected.xlsx
    id_file = os.path.join(base_dir, 'id_corrected.xlsx')
    if not os.path.exists(id_file):
        raise FileNotFoundError(f"File not found: {id_file}")
    
    print(f"\nReading: {id_file}")
    df = pd.read_excel(id_file)
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check required columns
    required_cols = ['unified_id', 'unified_name', 'preferred_name']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {id_file}")
    
    # Filter out rows with missing values
    df_clean = df[required_cols].dropna()
    print(f"Rows after removing NaN: {len(df_clean)}")
    
    # Categorize by unified_id prefix
    df_exposure = df_clean[df_clean['unified_id'].astype(str).str.startswith('EX:E')].copy()
    df_disease = df_clean[df_clean['unified_id'].astype(str).str.startswith('EX:D')].copy()
    
    print(f"\nCategorization:")
    print(f"  Exposure (EX:E): {len(df_exposure)} rows")
    print(f"  Disease (EX:D): {len(df_disease)} rows")
    print(f"  Other: {len(df_clean) - len(df_exposure) - len(df_disease)} rows")
    
    # Create output directory
    output_dir = os.path.join(base_dir, 'results_viz_ID')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Create visualizations for Exposure category
    if len(df_exposure) > 0:
        print("\n" + "="*80)
        print("Processing Exposure Category (EX:E)")
        print("="*80)
        create_hierarchy_visualization(df_exposure, 'Exposure', output_dir)
    else:
        print("\nWarning: No Exposure data (EX:E) found")
    
    # Create visualizations for Disease category
    if len(df_disease) > 0:
        print("\n" + "="*80)
        print("Processing Disease Category (EX:D)")
        print("="*80)
        create_hierarchy_visualization(df_disease, 'Disease', output_dir)
    else:
        print("\nWarning: No Disease data (EX:D) found")
    
    print("\n" + "="*80)
    print("Visualization completed successfully!")
    print("="*80)

if __name__ == '__main__':
    main()

# %%
