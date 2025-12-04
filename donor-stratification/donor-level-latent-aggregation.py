import os
import pandas as pd
import numpy as np

# --------------------------------------------------
# Load donor-by-class latent vectors
# --------------------------------------------------
DATA_DIR = "./data"
df = pd.read_csv(os.path.join(DATA_DIR, "donor_latents_per_class_scvi.csv"))

print("Loaded:", df.shape)
print("Original columns:", df.columns.tolist())
print("\nSample data:")
print(df.head())

print("\nUnique regions:", df['region'].unique())
print("Donors with multiple regions:")
print(df.groupby('donor')['region'].nunique().value_counts())

# --------------------------------------------------
# First aggregation: concatenate two regions per (donor, class)
# --------------------------------------------------
latent_cols = [f'z{i}' for i in range(20)]

df_sorted = df.sort_values(['donor', 'Class', 'region'])
grouped = df_sorted.groupby(['donor', 'Class'])

aggregated_rows = []

for (donor, cell_class), group in grouped:
    if len(group) != 2:
        print(f"Warning: Donor {donor}, Class {cell_class} has {len(group)} regions (expected 2)")
        continue

    rows = group.reset_index(drop=True)
    row1 = rows.iloc[0]
    row2 = rows.iloc[1]

    if row1['ADNC'] != row2['ADNC']:
        print(f"Warning: Donor {donor}, Class {cell_class} has different ADNC values: {row1['ADNC']} vs {row2['ADNC']}")

    agg_row = {
        'donor': donor,
        'Class': cell_class,
        'ADNC': row1['ADNC']
    }

    for i in range(20):
        agg_row[f'y{i}'] = row1[f'z{i}']

    for i in range(20):
        agg_row[f'z{i}'] = row2[f'z{i}']

    aggregated_rows.append(agg_row)

df_intermediate = pd.DataFrame(aggregated_rows)

y_cols = [f'y{i}' for i in range(20)]
z_cols = [f'z{i}' for i in range(20)]
df_intermediate = df_intermediate[['donor', 'Class', 'ADNC'] + y_cols + z_cols]

print("\n" + "="*60)
print("Intermediate DataFrame (donor-class level):")
print("Shape:", df_intermediate.shape)
print("\nColumns:", df_intermediate.columns.tolist())
print("\nFirst few rows:")
print(df_intermediate.head(10))

# --------------------------------------------------
# Second aggregation: concatenate three classes per donor
# --------------------------------------------------
donor_grouped = df_intermediate.groupby('donor')
final_rows = []

for donor, group in donor_grouped:
    if len(group) != 3:
        print(f"Warning: Donor {donor} has {len(group)} classes (expected 3)")
        continue

    group_sorted = group.sort_values('Class').reset_index(drop=True)

    adnc_values = group_sorted['ADNC'].unique()
    if len(adnc_values) > 1:
        print(f"Warning: Donor {donor} has different ADNC values: {adnc_values}")

    final_row = {
        'donor': donor,
        'ADNC': group_sorted.iloc[0]['ADNC']
    }

    for class_idx in range(3):
        for i in range(20):
            final_row[f'y{i}_class{class_idx+1}'] = group_sorted.iloc[class_idx][f'y{i}']

    for class_idx in range(3):
        for i in range(20):
            final_row[f'z{i}_class{class_idx+1}'] = group_sorted.iloc[class_idx][f'z{i}']

    final_rows.append(final_row)

df_final = pd.DataFrame(final_rows)

# --------------------------------------------------
# Reorder columns and save
# --------------------------------------------------
y_class_cols = []
for class_idx in range(1, 4):
    for i in range(20):
        y_class_cols.append(f'y{i}_class{class_idx}')

z_class_cols = []
for class_idx in range(1, 4):
    for i in range(20):
        z_class_cols.append(f'z{i}_class{class_idx}')

df_final = df_final[['donor', 'ADNC'] + y_class_cols + z_class_cols]

print("\n" + "="*60)
print("Final DataFrame (donor level with all classes aggregated):")
print("Shape:", df_final.shape)
print("\nColumns:", df_final.columns.tolist()[:10], "...", df_final.columns.tolist()[-10:])
print("\nFirst few rows:")
print(df_final.head())

output_path = os.path.join(DATA_DIR, "donor_latents_aggregated.csv")
df_final.to_csv(output_path, index=False)
print(f"\nSaved fully aggregated data to: {output_path}")

