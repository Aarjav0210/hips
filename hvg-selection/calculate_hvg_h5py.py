"""
Calculate highly variable genes (HVG) using h5py for memory-efficient processing.

This script uses h5py to directly access h5ad files without loading them fully
into memory via scanpy, making it much faster for iterative development.

Key features:
- Processes CSR sparse matrices in cell chunks
- Accumulates gene statistics incrementally
- Implements Seurat v3 scoring for robust HVG selection
- Handles multiple datasets with combined statistics
- Memory-efficient (constant memory usage)

Expected speedup: ~10-15 minutes vs ~25 minutes for scanpy load
"""

import os
import h5py
import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_DIR = "./data"
OUTPUT_DIR = "./hvg-selection"
K_HVG = 5000
CHUNK_SIZE = 10000

# --------------------------------------------------
# File paths
# --------------------------------------------------
A9_FILE = os.path.join(DATA_DIR, "SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad")
MTG_FILE = os.path.join(DATA_DIR, "SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_gene_names(h5_file: str) -> np.ndarray:
    """
    Extract gene names from h5ad file.

    Parameters
    ----------
    h5_file : str
        Path to h5ad file

    Returns
    -------
    gene_names : np.ndarray
        Array of gene names (strings)
    """
    with h5py.File(h5_file, 'r') as f:
        gene_names = f['var/_index'][:]

        # Decode if stored as bytes
        if gene_names.dtype.kind == 'S' or gene_names.dtype.kind == 'O':
            if hasattr(gene_names[0], 'decode'):
                gene_names = np.array([g.decode() for g in gene_names])

    return gene_names


def compute_gene_stats_chunked(
    h5_file: str,
    chunk_size: int = 10000,
    gene_sum: np.ndarray = None,
    gene_sum_sq: np.ndarray = None,
    total_cells_so_far: int = 0
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Compute gene statistics from CSR sparse matrix in chunks.

    This function processes the CSR matrix in cell chunks and accumulates
    statistics (sum and sum of squares) for each gene. It supports
    incremental accumulation across multiple files.

    Parameters
    ----------
    h5_file : str
        Path to h5ad file
    chunk_size : int
        Number of cells to process at once
    gene_sum : np.ndarray, optional
        Existing gene sum array to accumulate into
    gene_sum_sq : np.ndarray, optional
        Existing gene sum of squares array to accumulate into
    total_cells_so_far : int
        Number of cells processed so far (for multi-file processing)

    Returns
    -------
    gene_sum : np.ndarray
        Sum of expression values for each gene
    gene_sum_sq : np.ndarray
        Sum of squared expression values for each gene
    total_cells : int
        Total number of cells processed
    n_genes : int
        Number of genes
    """
    print(f"\nProcessing: {os.path.basename(h5_file)}")

    with h5py.File(h5_file, 'r') as f:
        # Get CSR components
        data = f['X/data']
        indices = f['X/indices']
        indptr = f['X/indptr']

        n_cells = len(indptr) - 1
        n_genes = len(f['var/_index'])

        print(f"  Cells: {n_cells:,}")
        print(f"  Genes: {n_genes:,}")
        print(f"  Non-zero elements: {len(data):,}")

        # Initialize accumulators if not provided
        if gene_sum is None:
            gene_sum = np.zeros(n_genes, dtype=np.float64)
        if gene_sum_sq is None:
            gene_sum_sq = np.zeros(n_genes, dtype=np.float64)

        # Process cells in chunks
        n_chunks = (n_cells + chunk_size - 1) // chunk_size
        print(f"  Processing in {n_chunks} chunks of {chunk_size} cells...")

        for chunk_idx in tqdm(range(n_chunks), desc=f"  Chunks"):
            start_cell = chunk_idx * chunk_size
            end_cell = min(start_cell + chunk_size, n_cells)

            # Get data indices for this chunk of cells
            start_idx = indptr[start_cell]
            end_idx = indptr[end_cell]

            # Extract chunk data
            chunk_data = data[start_idx:end_idx]
            chunk_indices = indices[start_idx:end_idx]

            # Accumulate statistics per gene
            # Use numpy's bincount for efficient accumulation
            gene_sum += np.bincount(
                chunk_indices,
                weights=chunk_data,
                minlength=n_genes
            )

            gene_sum_sq += np.bincount(
                chunk_indices,
                weights=chunk_data ** 2,
                minlength=n_genes
            )

    total_cells = total_cells_so_far + n_cells

    return gene_sum, gene_sum_sq, total_cells, n_genes


def seurat_v3_hvg_scoring(
    means: np.ndarray,
    variances: np.ndarray,
    n_cells: int,
    clip_max: float = None
) -> np.ndarray:
    """
    Compute Seurat v3 standardized variance scores.

    The Seurat v3 method accounts for the mean-variance
    relationship in count data by:
    1. Fitting a curve relating log(variance) to log(mean)
    2. Computing expected variance for each mean
    3. Standardizing variance by dividing by expected variance

    Parameters
    ----------
    means : np.ndarray
        Mean expression for each gene
    variances : np.ndarray
        Variance for each gene
    n_cells : int
        Total number of cells
    clip_max : float, optional
        Maximum value for clipping standardized variance

    Returns
    -------
    scores : np.ndarray
        Standardized variance scores for ranking genes
    """
    # Filter out genes with zero or very low variance/mean
    # Use a small threshold to avoid log(0)
    min_mean = 1e-10
    min_var = 1e-10

    valid = (means > min_mean) & (variances > min_var)

    if not np.any(valid):
        raise ValueError("No genes with sufficient mean and variance")

    print(f"\nSeurat v3 scoring:")
    print(f"  Valid genes (mean > {min_mean}, var > {min_var}): {np.sum(valid):,} / {len(means):,}")

    # Work with log-transformed values
    log_mean = np.log10(means[valid])
    log_var = np.log10(variances[valid])

    # Fit a polynomial (degree 2) to log(var) vs log(mean)
    # This captures the mean-variance relationship
    print("  Fitting mean-variance relationship (polynomial degree 2)...")
    coeffs = np.polyfit(log_mean, log_var, deg=2)
    expected_log_var = np.polyval(coeffs, log_mean)
    expected_var = 10 ** expected_log_var

    # Compute standardized variance
    std_variance = variances[valid] / expected_var

    # Clip if specified
    if clip_max is not None:
        print(f"  Clipping standardized variance to max={clip_max}")
        std_variance = np.clip(std_variance, 0, clip_max)

    # Create output array with scores
    scores = np.zeros(len(means), dtype=np.float64)
    scores[valid] = std_variance

    print(f"  Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
    print(f"  Mean score: {np.mean(scores[valid]):.4f}")

    return scores


def select_top_hvg(
    gene_names: np.ndarray,
    scores: np.ndarray,
    k: int = 5000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top k genes by score.

    Parameters
    ----------
    gene_names : np.ndarray
        Array of gene names
    scores : np.ndarray
        Scores for each gene
    k : int
        Number of top genes to select

    Returns
    -------
    top_genes : np.ndarray
        Names of top k genes
    top_scores : np.ndarray
        Scores of top k genes
    """
    # Get indices of top k scores
    top_indices = np.argsort(scores)[::-1][:k]

    # Return top gene names and their scores
    top_genes = gene_names[top_indices]
    top_scores = scores[top_indices]

    return top_genes, top_scores


def compute_combined_gene_stats(
    file_paths: List[str],
    chunk_size: int = 10000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Process multiple h5ad files and accumulate combined gene statistics.

    Parameters
    ----------
    file_paths : List[str]
        List of paths to h5ad files
    chunk_size : int
        Number of cells to process at once

    Returns
    -------
    gene_names : np.ndarray
        Array of gene names
    combined_means : np.ndarray
        Mean expression across all cells from all files
    combined_variances : np.ndarray
        Variance across all cells from all files
    total_cells : int
        Total number of cells across all files
    """
    print("="*80)
    print("Computing Combined Gene Statistics")
    print("="*80)

    # --------------------------------------------------
    # Verify gene consistency across files
    # --------------------------------------------------
    print("\nVerifying gene consistency across files...")
    gene_names = None
    n_genes = None

    for fpath in file_paths:
        curr_genes = extract_gene_names(fpath)
        if gene_names is None:
            gene_names = curr_genes
            n_genes = len(gene_names)
            print(f"  Reference genes from {os.path.basename(fpath)}: {n_genes:,}")
        else:
            if not np.array_equal(gene_names, curr_genes):
                raise ValueError(
                    f"Gene names/order do not match between files!\n"
                    f"  First file had {len(gene_names)} genes\n"
                    f"  {os.path.basename(fpath)} has {len(curr_genes)} genes"
                )
            print(f"  {os.path.basename(fpath)}: ✓ genes match")

    # --------------------------------------------------
    # Initialize combined accumulators
    # --------------------------------------------------
    combined_sum = np.zeros(n_genes, dtype=np.float64)
    combined_sum_sq = np.zeros(n_genes, dtype=np.float64)
    total_cells = 0

    # --------------------------------------------------
    # Process each file
    # --------------------------------------------------
    print("\nProcessing files and accumulating statistics...")
    for fpath in file_paths:
        combined_sum, combined_sum_sq, total_cells, _ = compute_gene_stats_chunked(
            fpath,
            chunk_size=chunk_size,
            gene_sum=combined_sum,
            gene_sum_sq=combined_sum_sq,
            total_cells_so_far=total_cells
        )

    # --------------------------------------------------
    # Compute combined mean and variance
    # --------------------------------------------------
    print(f"\nComputing final statistics across {total_cells:,} total cells...")
    combined_means = combined_sum / total_cells

    # Variance: E[X^2] - E[X]^2
    # For sparse data, zeros are implicit in the calculation
    combined_variances = (combined_sum_sq / total_cells) - (combined_means ** 2)

    print(f"  Mean expression range: [{np.min(combined_means):.4f}, {np.max(combined_means):.4f}]")
    print(f"  Variance range: [{np.min(combined_variances):.4f}, {np.max(combined_variances):.4f}]")

    return gene_names, combined_means, combined_variances, total_cells


def process_two_datasets(
    a9_path: str,
    mtg_path: str,
    k: int = 5000,
    chunk_size: int = 10000,
    output_dir: str = os.path.join(DATA_DIR, "hvg-selection")
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process two datasets and select top k HVGs using Seurat v3 method.

    Parameters
    ----------
    a9_path : str
        Path to A9 h5ad file
    mtg_path : str
        Path to MTG h5ad file
    k : int
        Number of HVGs to select
    chunk_size : int
        Number of cells to process at once
    output_dir : str
        Directory to save results

    Returns
    -------
    top_genes : np.ndarray
        Names of top k HVGs
    top_scores : np.ndarray
        Scores of top k HVGs
    """
    # --------------------------------------------------
    # Compute combined statistics
    # --------------------------------------------------
    gene_names, means, variances, n_cells = compute_combined_gene_stats(
        [a9_path, mtg_path],
        chunk_size=chunk_size
    )

    # --------------------------------------------------
    # Apply Seurat v3 scoring
    # --------------------------------------------------
    print("\n" + "="*80)
    print("Applying Seurat v3 HVG Scoring")
    print("="*80)

    scores = seurat_v3_hvg_scoring(means, variances, n_cells)

    # --------------------------------------------------
    # Select top k
    # --------------------------------------------------
    print(f"\nSelecting top {k} highly variable genes...")
    top_genes, top_scores = select_top_hvg(gene_names, scores, k=k)

    print(f"\nTop 10 HVGs:")
    for i in range(min(10, len(top_genes))):
        print(f"  {i+1}. {top_genes[i]}: score={top_scores[i]:.4f}")

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)

    hvg_df = pd.DataFrame({
        'gene': top_genes,
        'score': top_scores,
        'rank': np.arange(1, len(top_genes) + 1)
    })
    hvg_file = os.path.join(output_dir, "combined_hvg_genes_h5py.csv")
    hvg_df.to_csv(hvg_file, index=False)
    print(f"\nSaved HVG gene list to: {hvg_file}")

    # Save full statistics for all genes
    stats_df = pd.DataFrame({
        'gene': gene_names,
        'mean': means,
        'variance': variances,
        'seurat_v3_score': scores
    })
    # Sort by score for easier inspection
    stats_df = stats_df.sort_values('seurat_v3_score', ascending=False)

    stats_file = os.path.join(output_dir, "combined_gene_stats_h5py.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved full gene statistics to: {stats_file}")

    # Save summary
    summary_data = {
        'metric': [
            'Total cells (combined)',
            'Total genes analyzed',
            'HVGs selected',
            'Min HVG score',
            'Max HVG score',
            'Median HVG score'
        ],
        'value': [
            n_cells,
            len(gene_names),
            k,
            np.min(top_scores),
            np.max(top_scores),
            np.median(top_scores)
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "hvg_summary_h5py.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary to: {summary_file}")

    return top_genes, top_scores


# --------------------------------------------------
# Main execution
# --------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print("H5PY-BASED HIGHLY VARIABLE GENE (HVG) SELECTION")
    print("="*80)
    print("\nThis script uses h5py for memory-efficient HVG calculation.")
    print("It processes large h5ad files in chunks without loading them")
    print("fully into memory, making it much faster than scanpy.read_h5ad().")
    print("\nMethod: Seurat v3 (accounts for mean-variance relationship)")
    print(f"Target: Top {K_HVG} HVGs from combined A9 + MTG datasets")
    print(f"Chunk size: {CHUNK_SIZE} cells")

    # --------------------------------------------------
    # Check files exist
    # --------------------------------------------------
    print("\n" + "="*80)
    print("Checking Input Files")
    print("="*80)

    for fpath in [A9_FILE, MTG_FILE]:
        if os.path.exists(fpath):
            size_gb = os.path.getsize(fpath) / (1024**3)
            print(f"✓ {os.path.basename(fpath)}: {size_gb:.1f} GB")
        else:
            print(f"✗ NOT FOUND: {fpath}")
            exit(1)

    # --------------------------------------------------
    # Process datasets
    # --------------------------------------------------
    import time
    start_time = time.time()

    top_genes, top_scores = process_two_datasets(
        A9_FILE,
        MTG_FILE,
        k=K_HVG,
        chunk_size=CHUNK_SIZE,
        output_dir=OUTPUT_DIR
    )

    elapsed_time = time.time() - start_time

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nTotal runtime: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f} seconds)")
    print(f"\nSelected {len(top_genes)} highly variable genes")
    print(f"\nOutput files in: {OUTPUT_DIR}/")
    print("  - combined_hvg_genes_h5py.csv (list of HVG names)")
    print("  - combined_gene_stats_h5py.csv (full statistics)")
    print("  - hvg_summary_h5py.csv (summary metrics)")
    print("\n" + "="*80)
