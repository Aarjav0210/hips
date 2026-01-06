# -*- coding: utf-8 -*-
"""
Transformer-based Model for Alzheimer's Disease Prediction
Input: Sequence of Cell Embeddings (scGPT) -> Output: Pathology Staging and Quantification
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from collections import defaultdict
import random
import os
import datetime

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
# LOAD DATA
# ============================================================

print("Loading data...")
# Update path as necessary
df = pd.read_csv('/users/cpschult/scratch/hip/cell_level_scgpt.csv')
print(f"Loaded {len(df)} cells from {df['Donor ID'].nunique()} donors")
print(f"Data shape: {df.shape}")

# Identify scGPT columns
scgpt_cols = [col for col in df.columns if col.startswith('scGPT_')]
print(f"Found {len(scgpt_cols)} scGPT embedding dimensions")

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
one_hot_cat = ['Sex', 'Hispanic/Latino', 'APOE Genotype']
df = pd.get_dummies(df, columns=one_hot_cat, drop_first=True)

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
scaler = StandardScaler()
df[numerical_features_to_scale] = scaler.fit_transform(df[numerical_features_to_scale])

# 5. Handle missing values (Median Imputation per Donor is safer)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df.groupby('Donor ID')[numeric_cols].transform(
    lambda x: x.fillna(x.median())
)
# Global fill for any remaining NaNs (e.g., if a whole donor is missing a col)
df = df.fillna(df.median(numeric_only=True))

# ============================================================
# TRAIN/VAL/TEST SPLIT (By Donor)
# ============================================================

print("\nSplitting data by donor...")

TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.15, 0.15

donors = list(df['Donor ID'].unique())
# Sort then shuffle to ensure deterministic split with seed
donors.sort()
random.shuffle(donors)

n_donors = len(donors)
train_split_idx = int(n_donors * TRAIN_SPLIT)
val_split_idx = int(n_donors * (TRAIN_SPLIT + VAL_SPLIT))

train_donors = donors[:train_split_idx]
val_donors = donors[train_split_idx:val_split_idx]
test_donors = donors[val_split_idx:]

train_df = df[df['Donor ID'].isin(train_donors)].copy()
val_df = df[df['Donor ID'].isin(val_donors)].copy()
test_df = df[df['Donor ID'].isin(test_donors)].copy()

print(f"Train: {len(train_donors)} donors")
print(f"Val:   {len(val_donors)} donors")
print(f"Test:  {len(test_donors)} donors")

# ============================================================
# DEFINE TARGET COLUMNS
# ============================================================

y1_cols = ['percent 6e10 positive area', 'percent AT8 positive area',
           'percent NeuN positive area', 'percent GFAP positive area']
y2_cols = ['Thal', 'Braak', 'CERAD', 'ADNC']

# Scale regression targets
print("\nScaling regression targets...")
y1_scaler = StandardScaler()
train_df[y1_cols] = y1_scaler.fit_transform(train_df[y1_cols].values)
val_df[y1_cols] = y1_scaler.transform(val_df[y1_cols].values)
test_df[y1_cols] = y1_scaler.transform(test_df[y1_cols].values)

# ============================================================
# DATASET & COLLATE (WITH SUBSAMPLING)
# ============================================================

class DonorLevelDataset(torch.utils.data.Dataset):
    def __init__(self, df, scgpt_cols, y1_cols, y2_cols, max_cells=1024, mode='train'):
        """
        max_cells: Maximum number of cells to sample per donor per epoch.
        mode: 'train' (random subsampling) or 'eval' (deterministic subsampling).
        """
        self.grouped = df.groupby('Donor ID')
        self.donor_ids = list(self.grouped.groups.keys())
        self.scgpt_cols = scgpt_cols
        self.y1_cols = y1_cols
        self.y2_cols = y2_cols
        self.max_cells = max_cells
        self.mode = mode 

    def __len__(self):
        return len(self.donor_ids)

    def __getitem__(self, idx):
        donor = self.donor_ids[idx]
        group = self.grouped.get_group(donor)
        
        # 1. Get all available cells
        all_cells = group[self.scgpt_cols].values
        n_available = len(all_cells)
        
        # 2. Subsampling Logic (Prevents Overfitting)
        if n_available > self.max_cells:
            if self.mode == 'train':
                # Random sample (Data Augmentation)
                indices = np.random.choice(n_available, self.max_cells, replace=False)
            else:
                # Deterministic sample (Stable Validation)
                rng = np.random.RandomState(seed=42 + idx)
                indices = rng.choice(n_available, self.max_cells, replace=False)
            x_data = all_cells[indices]
            
            # Keep track of cell types for analysis
            cell_types = group['Supertype'].values[indices] if 'Supertype' in group.columns else []
        else:
            # Take all cells
            x_data = all_cells
            cell_types = group['Supertype'].values if 'Supertype' in group.columns else []

        # 3. Prepare Tensors
        x = torch.tensor(x_data, dtype=torch.float32)
        y1 = torch.tensor(group[self.y1_cols].values[0], dtype=torch.float32)
        y2 = torch.tensor(group[self.y2_cols].values[0], dtype=torch.long)
        
        return x, y1, y2, donor, cell_types

def collate_donors(batch):
    """
    Batches donors. Since donors have different cell counts (up to max_cells),
    we pad the sequences and create a mask.
    """
    xs, y1s, y2s, ids, cell_types_list = zip(*batch)
    
    # Pad sequences to longest in batch
    # Shape: [Batch_Size, Max_Seq_Len, Features]
    X_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    
    # Create Padding Mask (True = Ignore this position)
    # Shape: [Batch_Size, Max_Seq_Len]
    pad_mask = torch.zeros((len(xs), X_padded.size(1)), dtype=torch.bool)
    for i, x in enumerate(xs):
        pad_mask[i, len(x):] = True 
        
    y1_batch = torch.stack(y1s)
    y2_batch = torch.stack(y2s)
    
    return X_padded, pad_mask, y1_batch, y2_batch, ids, cell_types_list

# Instantiate Datasets
# Max cells = 1024 provides good context. Reduce to 512 if OOM occurs.
MAX_CELLS = 1024 
train_ds = DonorLevelDataset(train_df, scgpt_cols, y1_cols, y2_cols, max_cells=MAX_CELLS, mode='train')
val_ds = DonorLevelDataset(val_df, scgpt_cols, y1_cols, y2_cols, max_cells=MAX_CELLS, mode='eval')
test_ds = DonorLevelDataset(test_df, scgpt_cols, y1_cols, y2_cols, max_cells=MAX_CELLS, mode='eval')

# Instantiate Loaders
# BATCH_SIZE represents number of DONORS. 
# 8 Donors * 1024 Cells * 512 Features is manageable on most GPUs.
BATCH_SIZE = 8
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_donors)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_donors)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_donors)

print(f"\nDataloaders created. Max Cells: {MAX_CELLS}, Batch Size (Donors): {BATCH_SIZE}")

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
# TRANSFORMER MODEL
# ============================================================

class TransformerNet(nn.Module):
    def __init__(self, embed_dim=512, hidden=256, n_heads=4, n_layers=2, dropout=0.4):
        super().__init__()
        
        # 1. Transformer Encoder (Cell-to-Cell Context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 2. Attention Pooling (Aggregation)
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        
        # 3. Heads (Diagnosis)
        self.thal_head = nn.Linear(embed_dim, 6)
        self.braak_head = nn.Linear(embed_dim, 7)
        self.cerad_head = nn.Linear(embed_dim, 4)
        self.adnc_head = nn.Linear(embed_dim, 4)
        self.reg_head = nn.Linear(embed_dim, 4)
    
    def forward(self, x, mask=None, return_attn=False):
        """
        x: [Batch, Seq_Len, 512]
        mask: [Batch, Seq_Len] (True where padding exists)
        """
        # Step 1: Contextualize Cells
        # src_key_padding_mask=mask ensures we don't attend to padding
        h = self.transformer(x, src_key_padding_mask=mask) 
        
        # Step 2: Calculate Attention Scores for Pooling
        attn_scores = self.attention_pool(h) # [Batch, Seq, 1]
        
        # Mask padding in pooling layer (set to -inf so softmax makes them 0)
        if mask is not None:
            # mask is [Batch, Seq], unsqueeze to [Batch, Seq, 1]
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=1) # [Batch, Seq, 1]
        
        # Step 3: Aggregate to Donor Vector
        # Sum(Features * Weights). Padding has 0 weight.
        donor_embedding = (h * attn_weights).sum(dim=1) # [Batch, 512]
        
        # Step 4: Predict
        thal = self.thal_head(donor_embedding)
        braak = self.braak_head(donor_embedding)
        cerad = self.cerad_head(donor_embedding)
        adnc = self.adnc_head(donor_embedding)
        reg = self.reg_head(donor_embedding)
        
        if return_attn:
            return thal, braak, cerad, adnc, reg, attn_weights
            
        return thal, braak, cerad, adnc, reg

# ============================================================
# TRAINING SETUP
# ============================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

time = datetime.datetime.now()
time = time.strftime("%Y-%m-%d_%H_%M")

SAVE_DIR = '/users/cpschult/scratch/hip/'  
os.makedirs(SAVE_DIR, exist_ok=True)

model = TransformerNet(embed_dim=512, hidden=256, n_heads=4, n_layers=2, dropout=0.4).to(device)

loss_y1 = ccc_loss
loss_y2 = nn.CrossEntropyLoss(label_smoothing=0.1)
smooth_l1_loss = nn.SmoothL1Loss()

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

NUM_EPOCHS = 30
best_val_loss = float('inf')

print(f"\nStarting training for {NUM_EPOCHS} epochs...")

# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    n_batches = 0
    
    # Iterate over Batches of DONORS
    for x, mask, y1, y2, ids, _ in train_dl:
        x, mask = x.to(device), mask.to(device)
        y1, y2 = y1.to(device), y2.to(device)
        
        opt.zero_grad()
        
        # Forward Pass (Pass Mask!)
        thal_out, braak_out, cerad_out, adnc_out, reg_out = model(x, mask=mask)
        
        # Compute Loss (Per Donor)
        L_thal = loss_y2(thal_out, y2[:, 0])
        L_braak = loss_y2(braak_out, y2[:, 1])
        L_cerad = loss_y2(cerad_out, y2[:, 2])
        L_adnc = loss_y2(adnc_out, y2[:, 3])
        L_reg = loss_y1(reg_out, y1) + 0.25 * smooth_l1_loss(reg_out, y1)
        
        total_loss = L_thal + L_braak + L_cerad + L_adnc + L_reg
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        running_loss += total_loss.item()
        n_batches += 1
        
        if n_batches % 10 == 0:
            print(f"  Batch {n_batches} Loss: {total_loss.item():.4f}")

    avg_train_loss = running_loss / n_batches
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f}")

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for x, mask, y1, y2, ids, _ in val_dl:
            x, mask = x.to(device), mask.to(device)
            y1, y2 = y1.to(device), y2.to(device)
            
            thal_out, braak_out, cerad_out, adnc_out, reg_out = model(x, mask=mask)
            
            L_thal = loss_y2(thal_out, y2[:, 0])
            L_braak = loss_y2(braak_out, y2[:, 1])
            L_cerad = loss_y2(cerad_out, y2[:, 2])
            L_adnc = loss_y2(adnc_out, y2[:, 3])
            L_reg = loss_y1(reg_out, y1) + 0.25 * smooth_l1_loss(reg_out, y1)
            
            val_loss += (L_thal + L_braak + L_cerad + L_adnc + L_reg).item()
            val_batches += 1
    
    val_loss /= max(1, val_batches)
    sched.step(val_loss)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        save_path = os.path.join(SAVE_DIR, f'best_transformer_model_{time}.pt')

        torch.save({'model_state': model.state_dict()}, save_path)
        print(f"  -> Best model saved to {save_path}")

print("\nTraining complete!")

# ============================================================
# EVALUATION ON TEST SET
# ============================================================

print("\nEvaluating on test set...")

checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Containers for results
test_results = {
    'ids': [],
    'thal_pred': [], 'thal_true': [],
    'braak_pred': [], 'braak_true': [],
    'cerad_pred': [], 'cerad_true': [],
    'adnc_pred': [], 'adnc_true': [],
    'reg_pred': [], 'reg_true': []
}

donor_attention_weights = {}

with torch.no_grad():
    for x, mask, y1, y2, ids, cell_types_list in test_dl:
        x, mask = x.to(device), mask.to(device)
        
        # Forward with attention weights
        thal, braak, cerad, adnc, reg, attn = model(x, mask=mask, return_attn=True)
        
        # Store batch results
        for i, donor_id in enumerate(ids):
            test_results['ids'].append(donor_id)
            
            # Classification
            test_results['thal_pred'].append(thal[i].argmax().item())
            test_results['thal_true'].append(y2[i, 0].item())
            
            test_results['braak_pred'].append(braak[i].argmax().item())
            test_results['braak_true'].append(y2[i, 1].item())
            
            test_results['cerad_pred'].append(cerad[i].argmax().item())
            test_results['cerad_true'].append(y2[i, 2].item())
            
            test_results['adnc_pred'].append(adnc[i].argmax().item())
            test_results['adnc_true'].append(y2[i, 3].item())
            
            # Regression
            test_results['reg_pred'].append(reg[i].cpu().numpy())
            test_results['reg_true'].append(y1[i].cpu().numpy())
            
            # Attention Analysis Storage
            # attn[i] is [Seq_Len, 1]. Remove padding using mask.
            valid_len = (~mask[i]).sum().item()
            raw_weights = attn[i, :valid_len, 0].cpu().numpy()
            
            if cell_types_list[i] is not None and len(cell_types_list[i]) > 0:
                # Ensure length matches (it should if valid_len logic is correct)
                donor_attention_weights[donor_id] = {
                    'weights': raw_weights,
                    'cell_types': cell_types_list[i][:valid_len]
                }

# Convert regression lists to arrays
test_results['reg_pred'] = np.array(test_results['reg_pred'])
test_results['reg_true'] = np.array(test_results['reg_true'])

# Calculate Metrics
thal_qwk = cohen_kappa_score(test_results['thal_true'], test_results['thal_pred'], weights='quadratic')
braak_qwk = cohen_kappa_score(test_results['braak_true'], test_results['braak_pred'], weights='quadratic')
cerad_qwk = cohen_kappa_score(test_results['cerad_true'], test_results['cerad_pred'], weights='quadratic')
adnc_qwk = cohen_kappa_score(test_results['adnc_true'], test_results['adnc_pred'], weights='quadratic')

reg_ccc_scores = []
for i in range(test_results['reg_pred'].shape[1]):
    ccc = ccc_np(test_results['reg_true'][:, i], test_results['reg_pred'][:, i])
    reg_ccc_scores.append(ccc)

print("\n" + "="*60)
print("TEST SET EVALUATION RESULTS")
print("="*60)
print(f"Thal QWK:  {thal_qwk:.4f}")
print(f"Braak QWK: {braak_qwk:.4f}")
print(f"CERAD QWK: {cerad_qwk:.4f}")
print(f"ADNC QWK:  {adnc_qwk:.4f}")
print(f"Mean QWK:  {np.mean([thal_qwk, braak_qwk, cerad_qwk, adnc_qwk]):.4f}")

print("\nRegression CCC:")
for i, col in enumerate(y1_cols):
    print(f"  {col}: {reg_ccc_scores[i]:.4f}")
print(f"Mean CCC: {np.mean(reg_ccc_scores):.4f}")

# ============================================================
# ATTENTION ANALYSIS
# ============================================================

print("\n" + "="*60)
print("ATTENTION WEIGHT ANALYSIS")
print("="*60)

def analyze_attention(donor_weights):
    cell_type_stats = defaultdict(list)
    
    for d_id, data in donor_weights.items():
        w = data['weights']
        c = data['cell_types']
        # Normalize weights to sum to 1 per donor just to be safe for comparison
        # (Though softmax already did this)
        for val, c_type in zip(w, c):
            cell_type_stats[c_type].append(val)
            
    summary = []
    for c_type, values in cell_type_stats.items():
        summary.append({
            'Cell Type': c_type,
            'Mean Attention': np.mean(values),
            'Max Attention': np.max(values),
            'Count': len(values)
        })
    
    return pd.DataFrame(summary).sort_values('Mean Attention', ascending=False)

if len(donor_attention_weights) > 0:
    attn_df = analyze_attention(donor_attention_weights)
    print("\nTop 10 Cell Types by Attention Importance:")
    print(attn_df.head(10).to_string(index=False))
else:
    print("No cell type information available for attention analysis.")

# Save Predictions
res_df = pd.DataFrame({
    'Donor_ID': test_results['ids'],
    'Thal_Pred': test_results['thal_pred'],
    'Thal_True': test_results['thal_true'],
    'Braak_Pred': test_results['braak_pred'],
    'Braak_True': test_results['braak_true']
})

save_path = os.path.join(SAVE_DIR, 'test_predictions_transformer.csv')

res_df.to_csv(save_path, index=False)
print("\nPredictions saved.")