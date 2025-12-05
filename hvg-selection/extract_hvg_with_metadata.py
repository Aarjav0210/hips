"""
Extract HVG expression values + x_scvi latents + metadata for transformer model input
Output: HDF5 file organized by donor for memory-efficient loading
"""

import os
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_DIR = "./data"

# --------------------------------------------------
# Paths
# --------------------------------------------------
HVG_FILE = os.path.join(DATA_DIR, "hvg-selection", "combined_hvg_genes_h5py.csv")
A9_H5AD = os.path.join(DATA_DIR, "SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad")
MTG_H5AD = os.path.join(DATA_DIR, "SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad")
OUTPUT_FILE = os.path.join(DATA_DIR, "hvg-transformer", "hvg_expression_with_metadata.h5")

# --------------------------------------------------
# Load top k HVG gene names
# --------------------------------------------------
print("Loading HVG gene list...")
hvg_df = pd.read_csv(HVG_FILE)
hvg_genes = hvg_df['gene'].tolist()
k = len(hvg_genes)
print(f"Loaded {k} HVG genes")

def extract_region_data(h5ad_path, hvg_genes, region_name):
    """Extract HVG expression + x_scvi latents + metadata for one region"""
    print(f"\nProcessing {region_name}...")

    with h5py.File(h5ad_path, 'r') as f:
        # --------------------------------------------------
        # Get all gene names
        # --------------------------------------------------
        all_genes = f['var']['_index'][:].astype(str)
        gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}

        # --------------------------------------------------
        # Find indices for HVG genes
        # --------------------------------------------------
        hvg_indices = []
        missing_genes = []
        for gene in hvg_genes:
            if gene in gene_to_idx:
                hvg_indices.append(gene_to_idx[gene])
            else:
                missing_genes.append(gene)

        if missing_genes:
            print(f"Warning: {len(missing_genes)} HVG genes not found in {region_name}")

        print(f"Extracting expression for {len(hvg_indices)} HVG genes...")

        # --------------------------------------------------
        # Get CSR sparse matrix components
        # --------------------------------------------------
        data = f['X/data']
        indices = f['X/indices']
        indptr = f['X/indptr']

        n_cells = len(indptr) - 1
        n_genes = len(all_genes)
        print(f"Total cells: {n_cells}")

        # --------------------------------------------------
        # Extract HVG expression in chunks
        # --------------------------------------------------
        chunk_size = 10000
        hvg_expression_chunks = []

        for start_cell in tqdm(range(0, n_cells, chunk_size), desc=f"Extracting {region_name} expression"):
            end_cell = min(start_cell + chunk_size, n_cells)

            start_idx = indptr[start_cell]
            end_idx = indptr[end_cell]

            # Extract chunk data
            chunk_data = data[start_idx:end_idx]
            chunk_indices = indices[start_idx:end_idx]
            chunk_indptr = indptr[start_cell:end_cell+1] - start_idx

            chunk_cells = end_cell - start_cell
            chunk_hvg_expression = np.zeros((chunk_cells, len(hvg_indices)), dtype=np.float32)

            gene_to_hvg_pos = {gene_idx: hvg_pos for hvg_pos, gene_idx in enumerate(hvg_indices)}

            for cell_in_chunk in range(chunk_cells):
                cell_start = chunk_indptr[cell_in_chunk]
                cell_end = chunk_indptr[cell_in_chunk + 1]

                for i in range(cell_start, cell_end):
                    gene_idx = chunk_indices[i]
                    if gene_idx in gene_to_hvg_pos:
                        hvg_pos = gene_to_hvg_pos[gene_idx]
                        chunk_hvg_expression[cell_in_chunk, hvg_pos] = chunk_data[i]

            hvg_expression_chunks.append(chunk_hvg_expression)

        hvg_expression = np.vstack(hvg_expression_chunks)
        print(f"HVG expression matrix shape: {hvg_expression.shape}")

        # --------------------------------------------------
        # Extract x_scvi latents
        # --------------------------------------------------
        print("Extracting x_scvi latents...")
        x_scvi = f['obsm']['X_scVI'][:]
        print(f"x_scvi latents shape: {x_scvi.shape}")

        # --------------------------------------------------
        # Extract metadata
        # --------------------------------------------------
        print("Extracting metadata...")
        obs_keys = list(f['obs'].keys())
        metadata_dict = {}

        for key in tqdm(obs_keys, desc=f"Extracting {region_name} metadata"):
            if key == '_index':
                continue

            obj = f['obs'][key]

            # Check if this is a categorical column (has 'codes' and 'categories')
            if isinstance(obj, h5py.Group):
                if 'codes' in obj and 'categories' in obj:
                    # Categorical data - decode using codes and categories
                    codes = obj['codes'][:]
                    categories = obj['categories'][:].astype('U')
                    data = categories[codes]
                else:
                    # Group without codes/categories - check if it has a single nested item
                    subkeys = list(obj.keys())
                    if len(subkeys) == 1:
                        subobj = obj[subkeys[0]]
                        if isinstance(subobj, h5py.Dataset):
                            # Extract the nested dataset
                            data = subobj[:]
                            # Combine group name and subkey name for full column name
                            key = f"{key}{subkeys[0]}"
                            if data.dtype.kind in ['O', 'S', 'U']:
                                data = data.astype(str)
                        elif isinstance(subobj, h5py.Group) and 'codes' in subobj and 'categories' in subobj:
                            # Nested categorical data (e.g., Hispanic/Latino)
                            codes = subobj['codes'][:]
                            categories = subobj['categories'][:].astype('U')
                            data = categories[codes]
                            # Combine group name and subkey name for full column name
                            key = f"{key}{subkeys[0]}"
                        else:
                            print(f"Warning: Skipping nested group '{key}/{subkeys[0]}'")
                            continue
                    else:
                        print(f"Warning: Skipping complex group '{key}' with {len(subkeys)} subkeys")
                        continue
            elif isinstance(obj, h5py.Dataset):
                # Regular dataset
                data = obj[:]
                # Decode bytes to strings if necessary
                if data.dtype.kind in ['O', 'S', 'U']:
                    data = data.astype(str)
            else:
                print(f"Warning: Skipping unknown type '{key}': {type(obj)}")
                continue

            metadata_dict[key] = data

        metadata_df = pd.DataFrame(metadata_dict)
        print(f"Metadata shape: {metadata_df.shape}")

        # --------------------------------------------------
        # Combine expression + latents + metadata
        # --------------------------------------------------
        hvg_columns = [f"hvg_{gene}" for gene in [hvg_genes[i] for i, _ in enumerate(hvg_indices)]]
        latent_columns = [f"x_scvi_{i}" for i in range(x_scvi.shape[1])]

        expression_df = pd.DataFrame(hvg_expression, columns=hvg_columns)
        latent_df = pd.DataFrame(x_scvi, columns=latent_columns)

        combined_df = pd.concat([expression_df, latent_df, metadata_df], axis=1)
        combined_df['region'] = region_name

        print(f"Combined data shape for {region_name}: {combined_df.shape}")

        return combined_df

# --------------------------------------------------
# Process both regions
# --------------------------------------------------
print("="*60)
print("Extracting HVG expression + x_scvi latents + metadata from h5ad files")
print("="*60)

a9_data = extract_region_data(A9_H5AD, hvg_genes, 'A9')
mtg_data = extract_region_data(MTG_H5AD, hvg_genes, 'MTG')

# --------------------------------------------------
# Combine both regions
# --------------------------------------------------
print("\nCombining A9 and MTG data...")
combined_data = pd.concat([a9_data, mtg_data], axis=0, ignore_index=True)
print(f"Final combined shape: {combined_data.shape}")

# --------------------------------------------------
# Save to HDF5 organized by donor
# --------------------------------------------------
print(f"\nSaving to HDF5: {OUTPUT_FILE}...")
print("Organizing data by donor for memory-efficient loading...")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

donors = combined_data['Donor ID'].unique()
print(f"Found {len(donors)} unique donors")

hvg_cols = [col for col in combined_data.columns if col.startswith('hvg_')]
latent_cols = [col for col in combined_data.columns if col.startswith('x_scvi_')]
metadata_cols = [col for col in combined_data.columns if col not in hvg_cols + latent_cols]

print(f"Column breakdown:")
print(f"  - HVG genes: {len(hvg_cols)}")
print(f"  - x_scvi latents: {len(latent_cols)}")
print(f"  - Metadata: {len(metadata_cols)}")

# --------------------------------------------------
# Create HDF5 file with donor-level organization
# --------------------------------------------------
with h5py.File(OUTPUT_FILE, 'w') as hf:
    hf.create_dataset('hvg_columns', data=np.array(hvg_cols, dtype='S'))
    hf.create_dataset('latent_columns', data=np.array(latent_cols, dtype='S'))
    hf.create_dataset('metadata_columns', data=np.array(metadata_cols, dtype='S'))

    # Store each donor's data in a separate group
    for donor_id in tqdm(donors, desc="Saving donor data"):
        donor_data = combined_data[combined_data['Donor ID'] == donor_id]

        # Create group for this donor
        donor_group = hf.create_group(str(donor_id))

        # Store gene expression (HVG)
        hvg_data = donor_data[hvg_cols].values.astype(np.float32)
        donor_group.create_dataset('hvg_expression', data=hvg_data, compression='gzip', compression_opts=4)

        # Store x_scvi latents
        latent_data = donor_data[latent_cols].values.astype(np.float32)
        donor_group.create_dataset('x_scvi_latents', data=latent_data, compression='gzip', compression_opts=4)

        # Store metadata
        for meta_col in metadata_cols:
            meta_data = donor_data[meta_col].values
            # Handle string columns
            if meta_data.dtype == object or meta_data.dtype.kind in ['U', 'S']:
                meta_data = meta_data.astype(str)
                donor_group.create_dataset(f'metadata/{meta_col}', data=meta_data.astype('S'))
            else:
                donor_group.create_dataset(f'metadata/{meta_col}', data=meta_data)

        donor_group.attrs['n_cells'] = len(donor_data)

print("Done!")

# --------------------------------------------------
# Summary
# --------------------------------------------------
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total cells: {len(combined_data):,}")
print(f"Total donors: {len(donors)}")
print(f"HVG genes: {len(hvg_cols)}")
print(f"x_scvi latent dimensions: {len(latent_cols)}")
print(f"Metadata columns: {len(metadata_cols)}")
print(f"Output file: {OUTPUT_FILE}")
print("\nTo load data for a specific donor:")
print("  with h5py.File('{}', 'r') as f:".format(OUTPUT_FILE))
print("      donor_data = f['<donor_id>']")
print("      hvg = donor_data['hvg_expression'][:]")
print("      latents = donor_data['x_scvi_latents'][:]")
