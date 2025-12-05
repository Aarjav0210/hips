# HVG Transformer Model

This directory contains the transformer-based model for Alzheimer's Disease prediction using **5000 highly variable genes (HVGs)** directly as input features.

## Overview

### Key Differences from scGPT Version

| Aspect | scGPT Version ([tf_trial.py](../scgpt-transformer-model/tf_trial.py)) | HVG Version ([hvg_trial.py](hvg_trial.py)) |
|--------|----------------------------------|---------------------------|
| **Input Features** | 512-dim scGPT embeddings | 5000-dim HVG + cell-type metadata (cell-level) + donor metadata (donor-level) |
| **Data Source** | CSV file with pre-computed embeddings | HDF5 file with raw gene expression |
| **Input Projection** | None (direct 512-dim input) | Linear projection: ~5000+ → 512 dimensions |
| **Model Input** | Cell-level scGPT embeddings | Cell-level raw gene expression |
| **Data Split** | Random 70/15/15 split | Pre-defined stratified splits from CSV |

## Architecture Flow

### 1. Data Loading
- Loads data from HDF5 file: `data/hvg-transformer/hvg_expression_with_metadata.h5`
- Uses utility functions from `load_donor_data.py` in [hvg-selection](../hvg-selection/) directory
- Combines all donor data into single dataframe with metadata

### 2. Preprocessing
- **Ordinal encoding**: ADNC, Braak, Thal, CERAD scores
- **One-hot encoding**:
  - Cell-level: **region**, **Supertype**, **Class**, **Subclass** (drop_first=True)
  - Donor-level: Sex, APOE Genotype
- **Boolean conversion**: Race categories
- **Standardization**: Age at Death, Years of education, PMI
- **Missing values**: Median imputation within donors
- **Feature separation**:
  - **Cell-level**: HVG genes (5000) + region + Supertype + Class + Subclass (varies by data, ~5000-5100 total)
  - **Donor-level**: Age, Sex, APOE, Race, Education, PMI (~15-20 features)

### 3. Data Split
- **By donor** (not by cells) to prevent data leakage
- Uses **pre-defined splits** from `donor_clusters_aggregated_splits.csv`
- This ensures stratified splitting based on ADNC clusters
- Splits are deterministic and reproducible across runs
- Regression targets are scaled using StandardScaler

### 4. Model Architecture

```
CELL-LEVEL PROCESSING:
Input: [batch_size, ~5000+] HVG expression (5000) + cell-type metadata (region, Supertype, Class, Subclass)
   ↓
Input Projection: Linear(~5000+ → 512) + LayerNorm + ReLU + Dropout
   ↓
Add sequence dimension: [batch_size, 1, 512]
   ↓
Transformer Encoder (n_layers=1, n_heads=8, d_model=512, hidden=128)
   ↓
Cell-level embeddings: [batch_size, 512]
   ↓
DONOR-LEVEL AGGREGATION:
Attention-weighted pooling by donor
   ↓
Donor embeddings: [n_donors, 512]
   ↓
ADD DONOR METADATA:
Donor metadata: [n_donors, ~15-20] → MLP → [n_donors, 64]
Concatenate: [donor_embeddings, donor_metadata_proj] → [n_donors, 576]
   ↓
PREDICTION HEADS:
  - Thal: Linear(576 → 6)
  - Braak: Linear(576 → 7)
  - CERAD: Linear(576 → 4)
  - ADNC: Linear(576 → 4)
  - Regression: Linear(576 → 4)
```

### 5. Training
- **Loss functions**:
  - Classification: CrossEntropyLoss with label smoothing (0.05)
  - Regression: CCC loss + 0.25 × SmoothL1Loss
- **Optimizer**: AdamW (lr=1e-4, weight_decay=5e-3)
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)
- **Epochs**: 30
- **Batch size**: 128

### 6. Evaluation
- **Classification metrics**: Quadratic Weighted Kappa (QWK)
- **Regression metrics**: Concordance Correlation Coefficient (CCC)
- **Attention analysis**: Identifies which cell types receive highest attention

## Target Variables

### Classification (Ordinal)
- **Thal**: Thal phase (0-5)
- **Braak**: Braak stage (0-VI)
- **CERAD**: Neuritic plaque density (Absent, Sparse, Moderate, Frequent)
- **ADNC**: AD neuropathologic change (Not AD, Low, Intermediate, High)

### Regression (Continuous)
- **percent 6e10 positive area**: Amyloid-β pathology
- **percent AT8 positive area**: Tau pathology
- **percent NeuN positive area**: Neuronal density
- **percent GFAP positive area**: Astrocyte reactivity

## Usage

```bash
# Run the training script
python hvg_trial.py
```

The script will:
1. Load all donor data from HDF5
2. Preprocess features and targets
3. Split data by donor
4. Train transformer model
5. Evaluate on test set
6. Save best model as `best_hvg_transformer_model.pt`
7. Save predictions as `hvg_test_predictions.csv`
8. Print attention analysis results

## Key Insights

### Why HVGs instead of scGPT embeddings?

1. **Interpretability**: Direct gene expression is more interpretable than learned embeddings
2. **Flexibility**: Can apply different normalization/preprocessing strategies
3. **Baseline comparison**: Establishes performance using raw features
4. **Feature importance**: Attention weights can reveal which genes matter most

### Cell-Type Metadata as Cell-Level Features

The model includes cell-type information (region, Supertype, Class, Subclass) at the cell level because:
1. **Cell-specific context**: These vary per cell and provide important biological context
2. **Heterogeneity**: Different cell types may have different disease-related expression patterns
3. **Region effects**: Brain region (A9 vs MTG) influences gene expression and pathology
4. **Attention analysis**: Enables identification of which cell types are most predictive

Donor-level metadata (Age, Sex, APOE, etc.) is added after aggregation since it's constant for all cells from a donor.

### Input Projection Layer

The input projection layer is critical for this architecture:
- Reduces dimensionality from ~5000+ (HVG + cell-type metadata) → 512
- Applies non-linear transformation (ReLU)
- Includes normalization and dropout for stability
- Bridges raw gene expression and cell-type information to transformer-compatible embeddings

## Files

- [hvg_trial.py](hvg_trial.py) - Main training script
- `best_hvg_transformer_model.pt` - Saved model checkpoint (generated)
- `hvg_test_predictions.csv` - Test set predictions (generated)

## Dependencies

- torch
- pandas
- numpy
- scikit-learn
- h5py
- [load_donor_data.py](../hvg-selection/load_donor_data.py) - Data loading utilities
