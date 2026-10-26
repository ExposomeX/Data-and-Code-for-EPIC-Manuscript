#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
"""
Tidy EGoD_nodes.csv by correcting EXID and Preferred name based on id_corrected.xlsx
"""

import os
import sys
import pandas as pd

# Set UTF-8 encoding for output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    # Find the ebio_summary folder
    current_dir = os.getcwd()
    ebio_folders = [f for f in os.listdir(current_dir) if f.startswith('ebio_summary_') and os.path.isdir(f)]
    
    if not ebio_folders:
        raise FileNotFoundError("No folder starting with 'ebio_summary_' found in current directory")
    
    if len(ebio_folders) > 1:
        print(f"Warning: Multiple ebio_summary folders found: {ebio_folders}")
        print(f"Using the first one: {ebio_folders[0]}")
    
    ebio_folder = ebio_folders[0]
    buildmodel_path = os.path.join(ebio_folder, 'buildModel')
    
    if not os.path.exists(buildmodel_path):
        raise FileNotFoundError(f"buildModel folder not found in {ebio_folder}")
    
    print(f"Using folder: {ebio_folder}")
    print(f"BuildModel path: {buildmodel_path}\n")
    
    # File paths
    id_corrected_file = 'id_corrected.xlsx'
    EGoD_nodes_file = os.path.join(buildmodel_path, 'EGoD_nodes.csv')
    EGoD_edges_file = os.path.join(buildmodel_path, 'EGoD_edges.csv')
    nodes_output_file = os.path.join(buildmodel_path, 'EGoD_nodes_corrected.csv')
    edges_output_file = os.path.join(buildmodel_path, 'EGoD_edges_corrected.csv')
    
    # Read id_corrected.xlsx
    print(f"Reading {id_corrected_file}...")
    id_corrected = pd.read_excel(id_corrected_file)
    print(f"Loaded {len(id_corrected)} rows from id_corrected.xlsx")
    
    # Create mapping dictionaries
    # exid -> unified_id mapping
    exid_to_unified = dict(zip(id_corrected['exid'], id_corrected['unified_id']))
    # unified_id -> unified_name mapping (for updating Preferred name based on corrected EXID)
    unified_id_to_name = dict(zip(id_corrected['unified_id'], id_corrected['unified_name']))
    
    print(f"Created {len(exid_to_unified)} EXID mappings")
    print(f"Created {len(unified_id_to_name)} unified_id to name mappings")
    
    # ========== Process EGoD_nodes.csv ==========
    print(f"\n{'='*60}")
    print("Processing EGoD_nodes.csv")
    print('='*60)
    
    # Read EGoD_nodes.csv
    print(f"Reading {EGoD_nodes_file}...")
    EGoD_nodes = pd.read_csv(EGoD_nodes_file)
    print(f"Loaded {len(EGoD_nodes)} rows from EGoD_nodes.csv")
    
    # Replace EXID column values
    print("\nReplacing EXID values...")
    original_exid_count = EGoD_nodes['EXID'].nunique()
    EGoD_nodes['EXID'] = EGoD_nodes['EXID'].map(exid_to_unified).fillna(EGoD_nodes['EXID'])
    new_exid_count = EGoD_nodes['EXID'].nunique()
    print(f"EXID unique values: {original_exid_count} -> {new_exid_count}")
    
    # Replace Preferred name column values based on corrected EXID
    print("Replacing Preferred name values based on corrected EXID...")
    original_name_count = EGoD_nodes['Preferred name'].nunique()
    # Update Preferred name using the corrected EXID (unified_id)
    EGoD_nodes['Preferred name'] = EGoD_nodes['EXID'].map(unified_id_to_name).fillna(EGoD_nodes['Preferred name'])
    new_name_count = EGoD_nodes['Preferred name'].nunique()
    print(f"Preferred name unique values: {original_name_count} -> {new_name_count}")
    
    # Merge duplicate nodes by summing their Degree values
    print("\nMerging duplicate nodes...")
    print(f"Before merge: {len(EGoD_nodes)} rows")
    duplicates_before = EGoD_nodes.duplicated(subset=['EXID'], keep=False).sum()
    print(f"Duplicate EXID rows: {duplicates_before}")
    
    # Group by EXID and aggregate
    # For Degree: sum all degrees
    # For other columns: take the first value (they should be the same after ID correction)
    agg_dict = {
        'Preferred name': 'first',
        'Degree': 'sum',  # Sum all degrees for the same node
        'Group name': 'first',
        'ID': 'first',
        'Synonyms': 'first'
    }
    
    EGoD_nodes = EGoD_nodes.groupby('EXID', as_index=False).agg(agg_dict)
    
    print(f"After merge: {len(EGoD_nodes)} rows")
    print(f"Unique EXID count: {EGoD_nodes['EXID'].nunique()}")
    
    # Save corrected nodes file
    print(f"\nSaving corrected nodes file to {nodes_output_file}...")
    EGoD_nodes.to_csv(nodes_output_file, index=False)
    print(f"Successfully saved {len(EGoD_nodes)} rows")
    
    # Display sample of corrected data
    print("\nFirst 5 rows of corrected nodes data:")
    print(EGoD_nodes[['EXID', 'Preferred name']].head())
    
    # ========== Process EGoD_edges.csv ==========
    print(f"\n{'='*60}")
    print("Processing EGoD_edges.csv")
    print('='*60)
    
    # Read EGoD_edges.csv
    print(f"Reading {EGoD_edges_file}...")
    EGoD_edges = pd.read_csv(EGoD_edges_file)
    print(f"Loaded {len(EGoD_edges)} rows from EGoD_edges.csv")
    
    # Replace Source column values
    print("\nReplacing Source values...")
    original_source_count = EGoD_edges['Source'].nunique()
    EGoD_edges['Source'] = EGoD_edges['Source'].map(exid_to_unified).fillna(EGoD_edges['Source'])
    new_source_count = EGoD_edges['Source'].nunique()
    print(f"Source unique values: {original_source_count} -> {new_source_count}")
    
    # Replace Source preferred name column values based on corrected Source EXID
    print("Replacing Source preferred name values based on corrected Source...")
    original_source_name_count = EGoD_edges['Source preferred name'].nunique()
    EGoD_edges['Source preferred name'] = EGoD_edges['Source'].map(unified_id_to_name).fillna(EGoD_edges['Source preferred name'])
    new_source_name_count = EGoD_edges['Source preferred name'].nunique()
    print(f"Source preferred name unique values: {original_source_name_count} -> {new_source_name_count}")
    
    # Replace Target column values
    print("\nReplacing Target values...")
    original_target_count = EGoD_edges['Target'].nunique()
    EGoD_edges['Target'] = EGoD_edges['Target'].map(exid_to_unified).fillna(EGoD_edges['Target'])
    new_target_count = EGoD_edges['Target'].nunique()
    print(f"Target unique values: {original_target_count} -> {new_target_count}")
    
    # Replace Target preferred name column values based on corrected Target EXID
    print("Replacing Target preferred name values based on corrected Target...")
    original_target_name_count = EGoD_edges['Target preferred name'].nunique()
    EGoD_edges['Target preferred name'] = EGoD_edges['Target'].map(unified_id_to_name).fillna(EGoD_edges['Target preferred name'])
    new_target_name_count = EGoD_edges['Target preferred name'].nunique()
    print(f"Target preferred name unique values: {original_target_name_count} -> {new_target_name_count}")
    
    # Save corrected edges file
    print(f"\nSaving corrected edges file to {edges_output_file}...")
    EGoD_edges.to_csv(edges_output_file, index=False)
    print(f"Successfully saved {len(EGoD_edges)} rows")
    
    # Display sample of corrected data
    print("\nFirst 5 rows of corrected edges data:")
    print(EGoD_edges[['Source', 'Source preferred name', 'Target', 'Target preferred name']].head())
    
    print(f"\n{'='*60}")
    print("Processing completed successfully!")
    print('='*60)

if __name__ == '__main__':
    main()

# %%
