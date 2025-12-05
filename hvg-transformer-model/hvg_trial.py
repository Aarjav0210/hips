# -*- coding: utf-8 -*-
"""
Transformer-based Model for Alzheimer's Disease Prediction
Cell-level: HVG expression (5000 genes) + cell-type metadata (region, Supertype, Class, Subclass)
Donor-level: Demographics (Age, Sex, APOE, etc.) added after aggregation
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import h5py
import os
import sys

# Add path to load_donor_data utility
sys.path.append('/oscar/data/rsingh47/ajain181/hips/hvg-selection')
from load_donor_data import load_donor_as_dataframe, list_donors, get_file_info

# ============================================================
# SET SEED FOR REPRODUCIBILITY
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# LOAD DATA FROM HDF5
# ============================================================

print("Loading HVG data from HDF5...")
HDF5_FILE = '/oscar/data/rsingh47/ajain181/hips/data/hvg-transformer/hvg_expression_with_metadata.h5'

# Get file information
file_info = get_file_info(HDF5_FILE)
print(f"Found {file_info['n_donors']} donors with {file_info['total_cells']:,} total cells")
print(f"HVG genes: {file_info['n_hvg_genes']}")
print(f"Metadata columns: {file_info['n_metadata_cols']}")

# Exclude donors that only have single region data (missing either A9 or MTG)
# These 4 donors only have MTG data, missing A9
EXCLUDE_DONORS = ['H20.33.043', 'H21.33.009', 'H21.33.010', 'H21.33.039']
donor_ids = [d for d in file_info['donor_ids'] if d not in EXCLUDE_DONORS]
print(f"Excluding {len(EXCLUDE_DONORS)} donors with single-region data: {EXCLUDE_DONORS}")
print(f"Using {len(donor_ids)} donors with both A9 and MTG regions")

# Load all donors and combine into single dataframe
print("\nLoading all donor data...")
all_donor_dfs = []

for i, donor_id in enumerate(donor_ids):
    if (i + 1) % 10 == 0:
        print(f"  Loading donor {i+1}/{len(donor_ids)}...")

    donor_df = load_donor_as_dataframe(HDF5_FILE, donor_id)
    donor_df['Donor ID'] = donor_id  # Add donor ID column
    all_donor_dfs.append(donor_df)

df = pd.concat(all_donor_dfs, axis=0, ignore_index=True)
print(f"\nLoaded {len(df)} cells from {df['Donor ID'].nunique()} donors")
print(f"Data shape: {df.shape}")

# Identify HVG columns (gene expression columns)
# Note: HVG columns are prefixed with 'hvg_' in the HDF5 file
hvg_cols = [col for col in df.columns if col.startswith('hvg_')]
print(f"Found {len(hvg_cols)} HVG gene expression features")

# ============================================================
# DATA PREPROCESSING
# ============================================================

print("\nPreprocessing data...")

# 1. Ordinal label mappings
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
# Cell-level categorical: Supertype, Class, Subclass, region (vary per cell)
# Donor-level categorical: Sex, APOE Genotype (same for all cells from donor)
one_hot_cat = ['Sex', 'APOE Genotype', 'region', 'Supertype', 'Class', 'Subclass']
# Only encode columns that exist in the dataframe
one_hot_cat_present = [col for col in one_hot_cat if col in df.columns]
if one_hot_cat_present:
    df = pd.get_dummies(df, columns=one_hot_cat_present, drop_first=True)
    print(f"One-hot encoded: {one_hot_cat_present}")

# 3. Convert boolean features to int
boolean_features = [
    'Race (choice=White)', 'Race (choice=Black/ African American)',
    'Race (choice=Asian)', 'Race (choice=American Indian/ Alaska Native)',
    'Race (choice=Native Hawaiian or Pacific Islander)',
    'Race (choice=Unknown or unreported)', 'Race (choice=Other)'
]
for col in [c for c in boolean_features if c in df]:
    df[col] = df[col].astype(int)

# 4. Scale numerical features
numerical_features_to_scale = ['Age at Death', 'Years of education', 'PMI']
numerical_features_present = [col for col in numerical_features_to_scale if col in df.columns]
if numerical_features_present:
    scaler = StandardScaler()
    df[numerical_features_present] = scaler.fit_transform(df[numerical_features_present])

# 5. Handle missing values (median imputation within donors)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df.groupby('Donor ID')[numeric_cols].transform(
    lambda x: x.fillna(x.median())
)

# Fill any remaining NaNs with global median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

print("\nIdentifying feature columns...")

# Define target columns (will be excluded from features)
y1_cols_all = ['percent 6e10 positive area', 'percent AT8 positive area',
               'percent NeuN positive area', 'percent GFAP positive area']
y2_cols_all = ['Thal', 'Braak', 'CERAD', 'ADNC']
target_cols = y1_cols_all + y2_cols_all

# Identify x_scvi latent columns (exclude these)
x_scvi_cols = [col for col in df.columns if col.startswith('x_scvi_')]

# Columns to explicitly drop from metadata
drop_cols = [
    'library_prep',
    'Method',
    'Highest Lewy Body Disease',
    'LATE',
    'percent aSyn positive area',
    'percent pTDP43 positive area',
    'HispanicLatino'
]

# Define columns to exclude
exclude_cols = set(['Donor ID'] + target_cols + x_scvi_cols + drop_cols)

# CELL-LEVEL FEATURES: HVG + region + cell-type metadata (Supertype, Class, Subclass)
# Get one-hot encoded cell-level categorical columns
region_cols = [col for col in df.columns if col.startswith('region_')]
supertype_cols = [col for col in df.columns if col.startswith('Supertype_')]
class_cols = [col for col in df.columns if col.startswith('Class_')]
subclass_cols = [col for col in df.columns if col.startswith('Subclass_')]

# Combine all cell-level features
cell_feature_cols = hvg_cols + region_cols + supertype_cols + class_cols + subclass_cols
print(f"HVG columns: {len(hvg_cols)}")
print(f"Region columns: {len(region_cols)} - {region_cols}")
print(f"Supertype columns: {len(supertype_cols)}")
print(f"Class columns: {len(class_cols)}")
print(f"Subclass columns: {len(subclass_cols)}")
print(f"Cell-level features: {len(cell_feature_cols)} (HVG: {len(hvg_cols)} + Cell metadata: {len(region_cols) + len(supertype_cols) + len(class_cols) + len(subclass_cols)})")

# DONOR-LEVEL FEATURES: All other metadata (Age, Sex, APOE, etc.)
# These will be added at donor aggregation step
donor_metadata_cols = [col for col in df.columns
                       if col not in exclude_cols
                       and not col.startswith('hvg_')  # Exclude HVG genes
                       and not col.startswith('x_scvi_')
                       and not col.startswith('region_')  # Exclude cell-level region
                       and not col.startswith('Supertype_')  # Exclude cell-level cell type
                       and not col.startswith('Class_')  # Exclude cell-level class
                       and not col.startswith('Subclass_')]  # Exclude cell-level subclass

print(f"Donor-level metadata columns: {len(donor_metadata_cols)}")
print(f"Donor metadata features: {donor_metadata_cols}")

# ============================================================
# TRAIN/VAL/TEST SPLIT (By Donor - Using Pre-defined Splits)
# ============================================================

print("\nLoading pre-defined donor splits...")

# Load the split assignments
SPLITS_FILE = '/oscar/data/rsingh47/ajain181/hips/data/donor_clusters_aggregated_splits.csv'
splits_df = pd.read_csv(SPLITS_FILE)
print(f"Loaded splits for {len(splits_df)} donors")

# Create mapping of donor ID to split
donor_to_split = dict(zip(splits_df['donor'], splits_df['split']))

# Get donors for each split
train_donors = splits_df[splits_df['split'] == 'train']['donor'].tolist()
val_donors = splits_df[splits_df['split'] == 'val']['donor'].tolist()
test_donors = splits_df[splits_df['split'] == 'test']['donor'].tolist()

print(f"Split assignment: {len(train_donors)} train, {len(val_donors)} val, {len(test_donors)} test donors")

# Split the dataframe
train_df = df[df['Donor ID'].isin(train_donors)].copy()
val_df = df[df['Donor ID'].isin(val_donors)].copy()
test_df = df[df['Donor ID'].isin(test_donors)].copy()

print(f"Train: {len(train_donors)} donors, {len(train_df)} cells")
print(f"Val:   {len(val_donors)} donors, {len(val_df)} cells")
print(f"Test:  {len(test_donors)} donors, {len(test_df)} cells")

# Check for donors in data but not in splits file
donors_in_data = set(df['Donor ID'].unique())
donors_in_splits = set(splits_df['donor'])
missing_in_splits = donors_in_data - donors_in_splits
if missing_in_splits:
    print(f"WARNING: {len(missing_in_splits)} donors in data but not in splits file: {missing_in_splits}")
missing_in_data = donors_in_splits - donors_in_data
if missing_in_data:
    print(f"INFO: {len(missing_in_data)} donors in splits file but not in data")

# ============================================================
# DEFINE TARGET COLUMNS
# ============================================================

y1_cols = ['percent 6e10 positive area', 'percent AT8 positive area',
           'percent NeuN positive area', 'percent GFAP positive area']
y2_cols = ['Thal', 'Braak', 'CERAD', 'ADNC']

# Verify all target columns exist
y1_cols_present = [col for col in y1_cols if col in df.columns]
y2_cols_present = [col for col in y2_cols if col in df.columns]

print(f"\nRegression targets: {len(y1_cols_present)}/{len(y1_cols)}")
print(f"Classification targets: {len(y2_cols_present)}/{len(y2_cols)}")

# Scale regression targets
if y1_cols_present:
    print("\nScaling regression targets...")
    y1_scaler = StandardScaler()
    train_df[y1_cols_present] = y1_scaler.fit_transform(train_df[y1_cols_present].values)
    val_df[y1_cols_present] = y1_scaler.transform(val_df[y1_cols_present].values)
    test_df[y1_cols_present] = y1_scaler.transform(test_df[y1_cols_present].values)

# ============================================================
# DATASET CLASS
# ============================================================

class CellLevelDataset(torch.utils.data.Dataset):
    """Dataset that works at cell level with HVG expression + cell-type metadata (region, Supertype, Class, Subclass) + donor metadata"""
    def __init__(self, df, cell_feature_cols, donor_metadata_cols, y1_cols, y2_cols):
        self.donor_ids = df['Donor ID'].values
        self.X_cell = df[cell_feature_cols].values.astype(np.float32)  # HVG + region + Supertype + Class + Subclass
        self.X_donor = df[donor_metadata_cols].values.astype(np.float32) if donor_metadata_cols else None  # Age, Sex, APOE, etc.
        self.y1 = df[y1_cols].values.astype(np.float32) if y1_cols else np.zeros((len(df), 4), dtype=np.float32)
        self.y2 = df[y2_cols].values.astype(np.int64) if y2_cols else np.zeros((len(df), 4), dtype=np.int64)

    def __len__(self):
        return len(self.X_cell)

    def __getitem__(self, i):
        return (torch.tensor(self.X_cell[i]),
                torch.tensor(self.X_donor[i]) if self.X_donor is not None else torch.tensor([]),
                torch.tensor(self.y1[i]),
                torch.tensor(self.y2[i]),
                self.donor_ids[i])

def collate_with_donors(batch):
    """Custom collate function"""
    X_cell_batch = torch.stack([item[0] for item in batch])
    X_donor_batch = torch.stack([item[1] for item in batch]) if item[1].numel() > 0 else None
    y1_batch = torch.stack([item[2] for item in batch])
    y2_batch = torch.stack([item[3] for item in batch])
    donor_ids = np.array([item[4] for item in batch])
    return X_cell_batch, X_donor_batch, y1_batch, y2_batch, donor_ids

# Create datasets
train_ds = CellLevelDataset(train_df, cell_feature_cols, donor_metadata_cols, y1_cols_present, y2_cols_present)
val_ds = CellLevelDataset(val_df, cell_feature_cols, donor_metadata_cols, y1_cols_present, y2_cols_present)
test_ds = CellLevelDataset(test_df, cell_feature_cols, donor_metadata_cols, y1_cols_present, y2_cols_present)

# Create dataloaders
BATCH_SIZE = 128
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_with_donors)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_with_donors)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_with_donors)

print(f"\nDataloaders created with batch_size={BATCH_SIZE}")

# ============================================================
# LOSS FUNCTIONS
# ============================================================

def ccc_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Concordance Correlation Coefficient loss"""
    pred = pred.float()
    target = target.float()
    mx = pred.mean(dim=0)
    my = target.mean(dim=0)
    vx = pred.var(dim=0, unbiased=False)
    vy = target.var(dim=0, unbiased=False)
    cov = ((pred - mx) * (target - my)).mean(dim=0)
    ccc = (2 * cov) / (vx + vy + (mx - my).pow(2) + eps)
    loss = 1.0 - ccc
    return loss.mean()

def ccc_np(y_true, y_pred, eps=1e-12):
    """CCC for numpy arrays (evaluation)"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mt, mp = y_true.mean(), y_pred.mean()
    vt, vp = y_true.var(), y_pred.var()
    cov = ((y_true - mt) * (y_pred - mp)).mean()
    return (2*cov) / (vt + vp + (mt - mp)**2 + eps)

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class TransformerNet(nn.Module):
    def __init__(self, input_dim=5000, donor_metadata_dim=0, embed_dim=512, hidden=256, n_heads=8, n_layers=2, dropout=0.4):
        super().__init__()

        # Input projection layer to project HVG + cell-type metadata to embed_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Attention pooling mechanism
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

        # Donor metadata projection (optional, if donor_metadata_dim > 0)
        self.donor_metadata_dim = donor_metadata_dim
        if donor_metadata_dim > 0:
            self.donor_metadata_proj = nn.Sequential(
                nn.Linear(donor_metadata_dim, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            combined_dim = embed_dim + hidden // 2
        else:
            combined_dim = embed_dim

        # Classification heads (take combined embedding + donor metadata)
        self.thal_head = nn.Linear(combined_dim, 6)
        self.braak_head = nn.Linear(combined_dim, 7)
        self.cerad_head = nn.Linear(combined_dim, 4)
        self.adnc_head = nn.Linear(combined_dim, 4)
        self.reg_head = nn.Linear(combined_dim, 4)

    def attention_weighted_pooling(self, cell_features):
        """Apply learned attention weights to pool cells"""
        attn_scores = self.attention_pool(cell_features)  # [n_cells, 1]
        attn_weights = torch.softmax(attn_scores, dim=0)  # [n_cells, 1]
        pooled = (cell_features * attn_weights).sum(dim=0)  # [embed_dim]
        return pooled, attn_weights.squeeze()

    def forward(self, x, donor_ids=None, donor_metadata=None, aggregate=True):
        """
        x: [batch_size, input_dim] HVG expression + cell-type metadata (region, Supertype, Class, Subclass)
        donor_ids: [batch_size] donor ID for each cell
        donor_metadata: [batch_size, donor_metadata_dim] donor-level metadata (Age, Sex, APOE, etc.)
        aggregate: whether to aggregate to donor level
        """
        # Project input from (HVG + cell-type metadata)-dim to embed_dim
        x = self.input_projection(x)  # [batch, 512]

        # Add sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, 512]

        # Transform cell embeddings
        cell_features = self.transformer(x).squeeze(1)  # [batch, 512]

        if not aggregate or donor_ids is None:
            return cell_features

        # Aggregate cells to donor level with attention
        unique_donors = np.unique(donor_ids)
        donor_embeddings = []
        donor_metadata_list = []
        donor_ids_out = []

        for donor in unique_donors:
            mask = (donor_ids == donor)
            donor_cells = cell_features[mask]  # [n_cells_for_donor, 512]

            # Attention-weighted pooling
            donor_emb, _ = self.attention_weighted_pooling(donor_cells)

            donor_embeddings.append(donor_emb)
            donor_ids_out.append(donor)

            # Get donor-level metadata (same for all cells from this donor)
            if donor_metadata is not None:
                donor_meta = donor_metadata[mask][0]  # Take first cell's metadata (all same for donor)
                donor_metadata_list.append(donor_meta)

        donor_embeddings = torch.stack(donor_embeddings)  # [n_donors, 512]

        # Combine donor embeddings with donor-level metadata
        if self.donor_metadata_dim > 0 and donor_metadata is not None:
            donor_metadata_tensor = torch.stack(donor_metadata_list)  # [n_donors, donor_metadata_dim]
            donor_metadata_proj = self.donor_metadata_proj(donor_metadata_tensor)  # [n_donors, hidden//2]
            combined_features = torch.cat([donor_embeddings, donor_metadata_proj], dim=1)  # [n_donors, 512 + hidden//2]
        else:
            combined_features = donor_embeddings

        # Predictions
        thal_out = self.thal_head(combined_features)
        braak_out = self.braak_head(combined_features)
        cerad_out = self.cerad_head(combined_features)
        adnc_out = self.adnc_head(combined_features)
        reg_out = self.reg_head(combined_features)

        return thal_out, braak_out, cerad_out, adnc_out, reg_out, donor_ids_out

# ============================================================
# MODEL INITIALIZATION
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

model = TransformerNet(
    input_dim=len(cell_feature_cols),  # HVG genes + region
    donor_metadata_dim=len(donor_metadata_cols),  # Age, Sex, APOE, etc.
    embed_dim=512,
    hidden=128,
    n_heads=8,
    n_layers=1,
    dropout=0.5
).to(device)

print(f"Model created with:")
print(f"  Cell-level input: {len(cell_feature_cols)} ({len(hvg_cols)} HVGs + {len(region_cols)} region + {len(supertype_cols)} Supertype + {len(class_cols)} Class + {len(subclass_cols)} Subclass)")
print(f"  Donor-level metadata: {len(donor_metadata_cols)} features")

loss_y1 = ccc_loss
loss_y2 = nn.CrossEntropyLoss(label_smoothing=0.05)
smooth_l1_loss = nn.SmoothL1Loss()

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

# ============================================================
# TRAINING LOOP
# ============================================================

NUM_EPOCHS = 30
best_val_loss = float('inf')

print(f"\nStarting training for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for i, (inputs, donor_metadata, y1, y2, donor_ids) in enumerate(train_dl):
        inputs, y1, y2 = inputs.to(device), y1.to(device), y2.to(device)
        donor_metadata = donor_metadata.to(device) if donor_metadata is not None else None

        opt.zero_grad(set_to_none=True)

        # Forward pass with donor aggregation
        thal_out, braak_out, cerad_out, adnc_out, reg_out, batch_donors = model(
            inputs, donor_ids=donor_ids, donor_metadata=donor_metadata, aggregate=True
        )

        # Get unique donor indices
        unique_donors_np = np.array(batch_donors)
        donor_indices = []
        for donor in unique_donors_np:
            idx = np.where(donor_ids == donor)[0][0]
            donor_indices.append(idx)

        donor_indices = torch.tensor(donor_indices, device=device)

        # Get ground truth for unique donors
        y1_donors = y1[donor_indices]
        y2_donors = y2[donor_indices]

        # Compute losses
        L_thal = loss_y2(thal_out, y2_donors[:, 0])
        L_braak = loss_y2(braak_out, y2_donors[:, 1])
        L_cerad = loss_y2(cerad_out, y2_donors[:, 2])
        L_adnc = loss_y2(adnc_out, y2_donors[:, 3])
        L_reg = loss_y1(reg_out, y1_donors) + 0.25 * smooth_l1_loss(reg_out, y1_donors)

        total_loss = L_thal + L_braak + L_cerad + L_adnc + L_reg

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running_loss += total_loss.item()
        n_batches += 1

        if (i + 1) % 50 == 0:
            avg_loss = running_loss / n_batches
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{i+1}/{len(train_dl)}] Loss: {avg_loss:.4f}")

    model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for inputs, donor_metadata, y1, y2, donor_ids in val_dl:
            inputs, y1, y2 = inputs.to(device), y1.to(device), y2.to(device)
            donor_metadata = donor_metadata.to(device) if donor_metadata is not None else None

            thal_out, braak_out, cerad_out, adnc_out, reg_out, batch_donors = model(
                inputs, donor_ids=donor_ids, donor_metadata=donor_metadata, aggregate=True
            )

            unique_donors_np = np.array(batch_donors)
            donor_indices = []
            for donor in unique_donors_np:
                idx = np.where(donor_ids == donor)[0][0]
                donor_indices.append(idx)
            donor_indices = torch.tensor(donor_indices, device=device)

            y1_donors = y1[donor_indices]
            y2_donors = y2[donor_indices]

            L_thal_val = loss_y2(thal_out, y2_donors[:, 0])
            L_braak_val = loss_y2(braak_out, y2_donors[:, 1])
            L_cerad_val = loss_y2(cerad_out, y2_donors[:, 2])
            L_adnc_val = loss_y2(adnc_out, y2_donors[:, 3])
            L_reg_val = loss_y1(reg_out, y1_donors) + 0.25 * smooth_l1_loss(reg_out, y1_donors)

            total_val_loss = L_thal_val + L_braak_val + L_cerad_val + L_adnc_val + L_reg_val
            val_loss += total_val_loss.item()
            val_batches += 1

    val_loss /= max(1, val_batches)
    sched.step(val_loss)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'model_state': model.state_dict()}, 'best_hvg_transformer_model.pt')
        print(f"  → Best model saved!")

print("\nTraining complete!")

# ============================================================
# EVALUATION ON TEST SET
# ============================================================

print("\nEvaluating on test set...")

# Load best model
checkpoint = torch.load('best_hvg_transformer_model.pt')
model.load_state_dict(checkpoint['model_state'])
model.eval()

donor_predictions = {}
donor_targets = {}

with torch.no_grad():
    for donor_id in test_df['Donor ID'].unique():
        # Get all cells for this donor
        donor_mask = test_df['Donor ID'] == donor_id
        donor_data = test_df[donor_mask]

        X_donor = torch.tensor(donor_data[cell_feature_cols].values, dtype=torch.float32).to(device)
        X_donor_meta = torch.tensor(donor_data[donor_metadata_cols].values[0:1], dtype=torch.float32).to(device) if donor_metadata_cols else None
        y1_donor = donor_data[y1_cols_present].values[0] if y1_cols_present else np.zeros(4)
        y2_donor = donor_data[y2_cols_present].values[0] if y2_cols_present else np.zeros(4)

        # Forward pass through cell-level layers
        X_projected = model.input_projection(X_donor)
        cell_features = model.transformer(X_projected.unsqueeze(1)).squeeze(1)
        donor_emb, _ = model.attention_weighted_pooling(cell_features)
        donor_emb = donor_emb.unsqueeze(0)  # [1, 512]

        # Combine donor embedding with donor metadata
        if model.donor_metadata_dim > 0 and X_donor_meta is not None:
            donor_meta_proj = model.donor_metadata_proj(X_donor_meta)  # [1, hidden//2]
            combined_features = torch.cat([donor_emb, donor_meta_proj], dim=1)  # [1, 512 + hidden//2]
        else:
            combined_features = donor_emb

        # Get predictions
        thal_pred = model.thal_head(combined_features).argmax(dim=1).cpu().item()
        braak_pred = model.braak_head(combined_features).argmax(dim=1).cpu().item()
        cerad_pred = model.cerad_head(combined_features).argmax(dim=1).cpu().item()
        adnc_pred = model.adnc_head(combined_features).argmax(dim=1).cpu().item()
        reg_pred = model.reg_head(combined_features).cpu().numpy()[0]

        donor_predictions[donor_id] = {
            'thal': thal_pred,
            'braak': braak_pred,
            'cerad': cerad_pred,
            'adnc': adnc_pred,
            'reg': reg_pred
        }

        donor_targets[donor_id] = {
            'thal': y2_donor[0],
            'braak': y2_donor[1],
            'cerad': y2_donor[2],
            'adnc': y2_donor[3],
            'reg': y1_donor
        }

# Extract predictions and targets
test_thal_preds = [donor_predictions[d]['thal'] for d in donor_predictions]
test_thal_targets = [donor_targets[d]['thal'] for d in donor_targets]

test_braak_preds = [donor_predictions[d]['braak'] for d in donor_predictions]
test_braak_targets = [donor_targets[d]['braak'] for d in donor_targets]

test_cerad_preds = [donor_predictions[d]['cerad'] for d in donor_predictions]
test_cerad_targets = [donor_targets[d]['cerad'] for d in donor_targets]

test_adnc_preds = [donor_predictions[d]['adnc'] for d in donor_predictions]
test_adnc_targets = [donor_targets[d]['adnc'] for d in donor_targets]

test_reg_preds = np.array([donor_predictions[d]['reg'] for d in donor_predictions])
test_reg_targets = np.array([donor_targets[d]['reg'] for d in donor_targets])

# Calculate QWK for classification tasks
thal_qwk = cohen_kappa_score(test_thal_targets, test_thal_preds, weights='quadratic')
braak_qwk = cohen_kappa_score(test_braak_targets, test_braak_preds, weights='quadratic')
cerad_qwk = cohen_kappa_score(test_cerad_targets, test_cerad_preds, weights='quadratic')
adnc_qwk = cohen_kappa_score(test_adnc_targets, test_adnc_preds, weights='quadratic')

# Calculate CCC for regression tasks
reg_ccc_scores = []
for i in range(test_reg_preds.shape[1]):
    ccc = ccc_np(test_reg_targets[:, i], test_reg_preds[:, i])
    reg_ccc_scores.append(ccc)

print("\n" + "="*60)
print("TEST SET EVALUATION RESULTS")
print("="*60)
print(f"\nClassification Tasks (Quadratic Weighted Kappa):")
print(f"  Thal QWK:  {thal_qwk:.4f}")
print(f"  Braak QWK: {braak_qwk:.4f}")
print(f"  CERAD QWK: {cerad_qwk:.4f}")
print(f"  ADNC QWK:  {adnc_qwk:.4f}")
print(f"  Mean QWK:  {np.mean([thal_qwk, braak_qwk, cerad_qwk, adnc_qwk]):.4f}")

print(f"\nRegression Tasks (Concordance Correlation Coefficient):")
if y1_cols_present:
    for i, col in enumerate(y1_cols_present):
        print(f"  {col}: {reg_ccc_scores[i]:.4f}")
    print(f"  Mean CCC: {np.mean(reg_ccc_scores):.4f}")

print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)

# Save results to CSV
results_df = pd.DataFrame({
    'Donor_ID': list(donor_predictions.keys()),
    'Thal_Pred': test_thal_preds,
    'Thal_True': test_thal_targets,
    'Braak_Pred': test_braak_preds,
    'Braak_True': test_braak_targets,
    'CERAD_Pred': test_cerad_preds,
    'CERAD_True': test_cerad_targets,
    'ADNC_Pred': test_adnc_preds,
    'ADNC_True': test_adnc_targets,
})

results_df.to_csv('hvg_test_predictions.csv', index=False)
print("\nPredictions saved to 'hvg_test_predictions.csv'")
