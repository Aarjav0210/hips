#!/usr/bin/env python3
"""
Quick test to check the number and names of donor metadata columns
"""

import pandas as pd
import sys
import h5py

sys.path.append('/oscar/data/rsingh47/ajain181/hips/hvg-selection')
from load_donor_data import load_donor_as_dataframe, get_file_info

HDF5_FILE = '/oscar/data/rsingh47/ajain181/hips/data/hvg-transformer/hvg_expression_with_metadata.h5'

print("Loading sample data...")
file_info = get_file_info(HDF5_FILE)
first_donor = file_info['donor_ids'][0]
print(f"Loading first donor: {first_donor}")

# Load just one donor to test
df = load_donor_as_dataframe(HDF5_FILE, first_donor)
df['Donor ID'] = first_donor

print(f"\nInitial columns: {len(df.columns)}")
print(f"Initial shape: {df.shape}")

# Apply same preprocessing as hvg_trial.py
print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)

# Identify HVG columns (they start with 'hvg_' in the HDF5 file)
hvg_cols = [col for col in df.columns if col.startswith('hvg_')]
print(f"\nHVG columns: {len(hvg_cols)}")

# 1. Ordinal encoding (skip, these are targets)
ordinal_maps = {
    'ADNC':  {"Not AD": 0, "Low": 1, "Intermediate": 2, "High": 3},
    'Braak': {"Braak 0": 0, "Braak I": 1, "Braak II": 2,
              "Braak III": 3, "Braak IV": 4, "Braak V": 5, "Braak VI": 6},
    'Thal':  {"Thal 0": 0, "Thal 1": 1, "Thal 2": 2, "Thal 3": 3, "Thal 4": 4, "Thal 5": 5},
    'CERAD': {"Absent": 0, "Sparse": 1, "Moderate": 2, "Frequent": 3},
}

for col, mapping in ordinal_maps.items():
    if col in df:
        df[col] = df[col].map(mapping).fillna(0).astype(int)

# 2. One-hot encode categorical variables
one_hot_cat = ['Sex', 'Hispanic/Latino', 'APOE Genotype', 'region']
one_hot_cat_present = [col for col in one_hot_cat if col in df.columns]
print(f"\nOne-hot encoding: {one_hot_cat_present}")
if one_hot_cat_present:
    df = pd.get_dummies(df, columns=one_hot_cat_present, drop_first=True)

print(f"Columns after one-hot encoding: {len(df.columns)}")

# 3. Identify feature columns
print("\n" + "="*60)
print("IDENTIFYING FEATURES")
print("="*60)

# Define target columns
y1_cols_all = ['percent 6e10 positive area', 'percent AT8 positive area',
               'percent NeuN positive area', 'percent GFAP positive area']
y2_cols_all = ['Thal', 'Braak', 'CERAD', 'ADNC']
target_cols = y1_cols_all + y2_cols_all

# Identify x_scvi latent columns
x_scvi_cols = [col for col in df.columns if col.startswith('x_scvi_')]
print(f"\nx_scvi columns: {len(x_scvi_cols)}")

# Columns to explicitly drop from metadata
drop_cols = [
    'library_prep',
    'Method',
    'Highest Lewy Body Disease',
    'LATE',
    'percent aSyn positive area',
    'percent pTDP43 positive area',
    'Class',
    'Subclass'
]

# Define columns to exclude
exclude_cols = set(['Donor ID', 'Supertype'] + target_cols + x_scvi_cols + drop_cols)

# Get region columns
region_cols = [col for col in df.columns if col.startswith('region_')]
print(f"Region columns: {len(region_cols)} - {region_cols}")

# Get donor metadata columns
donor_metadata_cols = [col for col in df.columns
                       if col not in exclude_cols
                       and not col.startswith('hvg_')  # Exclude HVG genes
                       and not col.startswith('x_scvi_')
                       and not col.startswith('region_')]

print(f"\n" + "="*60)
print("DONOR METADATA COLUMNS")
print("="*60)
print(f"\nTotal donor metadata columns: {len(donor_metadata_cols)}")
print(f"\nColumn names:")
for i, col in enumerate(donor_metadata_cols, 1):
    print(f"  {i:2}. {col}")

# Show data types
print(f"\n" + "="*60)
print("DATA TYPES")
print("="*60)
for col in donor_metadata_cols:
    print(f"  {col}: {df[col].dtype}")

# Summary
print(f"\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Cell-level features: {len(hvg_cols)} HVGs + {len(region_cols)} region = {len(hvg_cols) + len(region_cols)}")
print(f"Donor-level features: {len(donor_metadata_cols)}")
print(f"Total model input (cell-level): {len(hvg_cols) + len(region_cols)}")
print(f"Added at donor aggregation: {len(donor_metadata_cols)}")
print(f"Combined dim for prediction heads: 512 (embedding) + {len(donor_metadata_cols)} (metadata) = {512 + len(donor_metadata_cols)}")
