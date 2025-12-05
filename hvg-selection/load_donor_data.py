"""
Utility script to load donor data from the HDF5 file
Demonstrates memory-efficient loading of specific donors
"""

import os
import h5py
import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_DIR = "./data"
HDF5_FILE = os.path.join(DATA_DIR, "hvg-transformer", "hvg_expression_with_metadata.h5")


def list_donors(h5_file_path):
    """List all donor IDs in the HDF5 file"""
    with h5py.File(h5_file_path, 'r') as f:
        exclude_keys = {'hvg_columns', 'latent_columns', 'metadata_columns'}
        donors = [k for k in f.keys() if k not in exclude_keys]
    return donors


def load_donor_data(h5_file_path, donor_id):
    """
    Load all data for a specific donor without loading entire dataset into memory

    Parameters:
    -----------
    h5_file_path : str
        Path to the HDF5 file
    donor_id : str
        Donor ID to load

    Returns:
    --------
    dict with keys:
        - 'hvg_expression': np.array of shape (n_cells, n_hvg_genes)
        - 'x_scvi_latents': np.array of shape (n_cells, n_latent_dims)
        - 'metadata': dict of metadata arrays
        - 'n_cells': int
    """
    with h5py.File(h5_file_path, 'r') as f:
        if str(donor_id) not in f:
            raise ValueError(f"Donor {donor_id} not found in file")

        donor_group = f[str(donor_id)]

        # --------------------------------------------------
        # Load expression data
        # --------------------------------------------------
        hvg_expression = donor_group['hvg_expression'][:]

        # --------------------------------------------------
        # Load latent embeddings
        # --------------------------------------------------
        x_scvi_latents = donor_group['x_scvi_latents'][:]

        # --------------------------------------------------
        # Load metadata
        # --------------------------------------------------
        metadata = {}
        if 'metadata' in donor_group:
            for key in donor_group['metadata'].keys():
                data = donor_group['metadata'][key][:]
                if data.dtype.kind == 'S':
                    data = data.astype(str)
                metadata[key] = data

        n_cells = donor_group.attrs['n_cells']

    return {
        'hvg_expression': hvg_expression,
        'x_scvi_latents': x_scvi_latents,
        'metadata': metadata,
        'n_cells': n_cells
    }


def load_donor_as_dataframe(h5_file_path, donor_id):
    """
    Load donor data and combine into a pandas DataFrame

    Parameters:
    -----------
    h5_file_path : str
        Path to the HDF5 file
    donor_id : str
        Donor ID to load

    Returns:
    --------
    pd.DataFrame with HVG expression, x_scvi latents, and metadata
    """
    with h5py.File(h5_file_path, 'r') as f:
        if str(donor_id) not in f:
            raise ValueError(f"Donor {donor_id} not found in file")

        # --------------------------------------------------
        # Get column names
        # --------------------------------------------------
        hvg_cols = f['hvg_columns'][:].astype(str)
        latent_cols = f['latent_columns'][:].astype(str)

        donor_group = f[str(donor_id)]

        # --------------------------------------------------
        # Load data
        # --------------------------------------------------
        hvg_expression = donor_group['hvg_expression'][:]
        x_scvi_latents = donor_group['x_scvi_latents'][:]

        # --------------------------------------------------
        # Create dataframes
        # --------------------------------------------------
        hvg_df = pd.DataFrame(hvg_expression, columns=hvg_cols)
        latent_df = pd.DataFrame(x_scvi_latents, columns=latent_cols)

        # --------------------------------------------------
        # Load metadata
        # --------------------------------------------------
        metadata_dict = {}
        if 'metadata' in donor_group:
            for key in donor_group['metadata'].keys():
                data = donor_group['metadata'][key][:]
                if data.dtype.kind == 'S':
                    data = data.astype(str)
                metadata_dict[key] = data

        metadata_df = pd.DataFrame(metadata_dict)

        # --------------------------------------------------
        # Combine all
        # --------------------------------------------------
        combined_df = pd.concat([hvg_df, latent_df, metadata_df], axis=1)

    return combined_df


def get_file_info(h5_file_path):
    """Get summary information about the HDF5 file"""
    with h5py.File(h5_file_path, 'r') as f:
        # --------------------------------------------------
        # Get donor list
        # --------------------------------------------------
        exclude_keys = {'hvg_columns', 'latent_columns', 'metadata_columns'}
        donors = [k for k in f.keys() if k not in exclude_keys]
        n_donors = len(donors)

        # --------------------------------------------------
        # Get dimensions from first donor
        # --------------------------------------------------
        first_donor = f[donors[0]]
        n_hvg = first_donor['hvg_expression'].shape[1]
        n_latents = first_donor['x_scvi_latents'].shape[1]

        # --------------------------------------------------
        # Get total cells
        # --------------------------------------------------
        total_cells = sum(f[donor].attrs['n_cells'] for donor in donors)

        # --------------------------------------------------
        # Get column names
        # --------------------------------------------------
        hvg_cols = f['hvg_columns'][:].astype(str).tolist()
        latent_cols = f['latent_columns'][:].astype(str).tolist()
        metadata_cols = f['metadata_columns'][:].astype(str).tolist()

    return {
        'n_donors': n_donors,
        'total_cells': total_cells,
        'n_hvg_genes': n_hvg,
        'n_latent_dims': n_latents,
        'n_metadata_cols': len(metadata_cols),
        'hvg_columns': hvg_cols,
        'latent_columns': latent_cols,
        'metadata_columns': metadata_cols,
        'donor_ids': donors
    }


if __name__ == "__main__":
    # --------------------------------------------------
    # Example usage
    # --------------------------------------------------
    print("HDF5 File Information")
    print("=" * 60)

    try:
        # --------------------------------------------------
        # Get file information
        # --------------------------------------------------
        info = get_file_info(HDF5_FILE)
        print(f"Number of donors: {info['n_donors']}")
        print(f"Total cells: {info['total_cells']:,}")
        print(f"HVG genes: {info['n_hvg_genes']}")
        print(f"x_scvi latent dimensions: {info['n_latent_dims']}")
        print(f"Metadata columns: {info['n_metadata_cols']}")

        # --------------------------------------------------
        # Load first donor
        # --------------------------------------------------
        print("\n" + "=" * 60)
        print("Example: Loading first donor")
        print("=" * 60)

        first_donor = info['donor_ids'][0]
        print(f"Loading donor: {first_donor}")

        donor_dict = load_donor_data(HDF5_FILE, first_donor)
        print(f"\nDonor has {donor_dict['n_cells']} cells")
        print(f"HVG expression shape: {donor_dict['hvg_expression'].shape}")
        print(f"x_scvi latents shape: {donor_dict['x_scvi_latents'].shape}")
        print(f"Metadata keys: {list(donor_dict['metadata'].keys())}")

        # --------------------------------------------------
        # Load donor as DataFrame
        # --------------------------------------------------
        print(f"\nLoading donor as DataFrame...")
        donor_df = load_donor_as_dataframe(HDF5_FILE, first_donor)
        print(f"DataFrame shape: {donor_df.shape}")
        print(f"DataFrame columns: {list(donor_df.columns[:5])} ... {list(donor_df.columns[-5:])}")

    except FileNotFoundError:
        print(f"Error: File not found at {HDF5_FILE}")
        print("Please run extract_hvg_with_metadata.py first to generate the HDF5 file")
    except Exception as e:
        print(f"Error: {e}")
