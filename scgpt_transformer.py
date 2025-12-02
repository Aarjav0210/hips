# -*- coding: utf-8 -*-
"""
Transformer-based Model for Alzheimer's Disease Prediction
Cell-level scGPT embeddings → Donor-level predictions
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random

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
df = pd.read_csv('/users/aiyer51/scratch/cell_level_scgpt.csv')
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



# 5. Handle missing values (median imputation)
# print(f"Missing values before imputation: {df.isna().sum().sum()}")
# median_values = df.drop(columns=['Donor ID']).median()
# df = df.fillna(median_values)
# print(f"Missing values after imputation: {df.isna().sum().sum()}")

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df.groupby('Donor ID')[numeric_cols].transform(
    lambda x: x.fillna(x.median())
)

# ============================================================
# TRAIN/VAL/TEST SPLIT (By Donor)
# ============================================================

print("\nSplitting data by donor...")

TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.15, 0.15

donors = list(df['Donor ID'].unique())
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

print(f"Train: {len(train_donors)} donors, {len(train_df)} cells")
print(f"Val:   {len(val_donors)} donors, {len(val_df)} cells")
print(f"Test:  {len(test_donors)} donors, {len(test_df)} cells")

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
# DATASET CLASS
# ============================================================

class CellLevelDataset(torch.utils.data.Dataset):
    """Dataset that works at cell level"""
    def __init__(self, df, scgpt_cols, y1_cols, y2_cols):
        self.donor_ids = df['Donor ID'].values
        self.cell_types = df['Supertype'].values if 'Supertype' in df.columns else None
        self.X = df[scgpt_cols].values.astype(np.float32)
        self.y1 = df[y1_cols].values.astype(np.float32)
        self.y2 = df[y2_cols].values.astype(np.int64)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        return (torch.tensor(self.X[i]), 
                torch.tensor(self.y1[i]), 
                torch.tensor(self.y2[i]),
                self.donor_ids[i],
                self.cell_types[i] if self.cell_types is not None else '')

def collate_with_donors(batch):
    """Custom collate function"""
    X_batch = torch.stack([item[0] for item in batch])
    y1_batch = torch.stack([item[1] for item in batch])
    y2_batch = torch.stack([item[2] for item in batch])
    donor_ids = np.array([item[3] for item in batch])
    cell_types = [item[4] for item in batch]
    return X_batch, y1_batch, y2_batch, donor_ids, cell_types

# Create datasets
train_ds = CellLevelDataset(train_df, scgpt_cols, y1_cols, y2_cols)
val_ds = CellLevelDataset(val_df, scgpt_cols, y1_cols, y2_cols)
test_ds = CellLevelDataset(test_df, scgpt_cols, y1_cols, y2_cols)

# Create dataloaders
#changed batch size from 256 to 128
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_with_donors)
val_dl = DataLoader(val_ds, batch_size=128, collate_fn=collate_with_donors)
test_dl = DataLoader(test_ds, batch_size=128, collate_fn=collate_with_donors)

print(f"\nDataloaders created with batch_size=128")

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
# TRANSFORMER MODEL WITH ATTENTION POOLING
# ============================================================

class TransformerNet(nn.Module):
    def __init__(self, embed_dim=512, hidden=256, n_heads=4, n_layers=2, dropout=0.4):
        super().__init__()
        
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
        
        # Classification heads
        self.thal_head = nn.Linear(embed_dim, 6)
        self.braak_head = nn.Linear(embed_dim, 7)
        self.cerad_head = nn.Linear(embed_dim, 4)
        self.adnc_head = nn.Linear(embed_dim, 4)
        self.reg_head = nn.Linear(embed_dim, 4)
    
    def attention_weighted_pooling(self, cell_features):
        """Apply learned attention weights to pool cells"""
        attn_scores = self.attention_pool(cell_features)  # [n_cells, 1]
        attn_weights = torch.softmax(attn_scores, dim=0)  # [n_cells, 1]
        pooled = (cell_features * attn_weights).sum(dim=0)  # [embed_dim]
        return pooled, attn_weights.squeeze()
    
    def forward(self, x, donor_ids=None, aggregate=True):
        """
        x: [batch_size, 512] cell embeddings
        donor_ids: [batch_size] donor ID for each cell
        aggregate: whether to aggregate to donor level
        """
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
        donor_ids_out = []
        
        for donor in unique_donors:
            mask = (donor_ids == donor)
            donor_cells = cell_features[mask]  # [n_cells_for_donor, 512]
            
            # Attention-weighted pooling
            donor_emb, _ = self.attention_weighted_pooling(donor_cells)
            
            donor_embeddings.append(donor_emb)
            donor_ids_out.append(donor)
        
        donor_embeddings = torch.stack(donor_embeddings)  # [n_donors, 512]
        
        # Predictions
        thal_out = self.thal_head(donor_embeddings)
        braak_out = self.braak_head(donor_embeddings)
        cerad_out = self.cerad_head(donor_embeddings)
        adnc_out = self.adnc_head(donor_embeddings)
        reg_out = self.reg_head(donor_embeddings)
        
        return thal_out, braak_out, cerad_out, adnc_out, reg_out, donor_ids_out



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {device}")

model = TransformerNet(embed_dim=512, hidden=128, n_heads=4, n_layers=1, dropout=0.5).to(device)#(embed_dim=512, hidden=256, n_heads=4, n_layers=2, dropout=0.4).to(device)

loss_y1 = ccc_loss
loss_y2 = nn.CrossEntropyLoss(label_smoothing=0.05)
smooth_l1_loss = nn.SmoothL1Loss()

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)#(model.parameters(), lr=1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

NUM_EPOCHS = 30
best_val_loss = float('inf')

print(f"\nStarting training for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    n_batches = 0
    
    for i, (inputs, y1, y2, donor_ids, cell_types) in enumerate(train_dl):
        inputs, y1, y2 = inputs.to(device), y1.to(device), y2.to(device)
        
        opt.zero_grad(set_to_none=True)
        
        # Forward pass with donor aggregation
        thal_out, braak_out, cerad_out, adnc_out, reg_out, batch_donors = model(
            inputs, donor_ids=donor_ids, aggregate=True
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
        for inputs, y1, y2, donor_ids, cell_types in val_dl:
            inputs, y1, y2 = inputs.to(device), y1.to(device), y2.to(device)
            
            thal_out, braak_out, cerad_out, adnc_out, reg_out, batch_donors = model(
                inputs, donor_ids=donor_ids, aggregate=True
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
        torch.save({'model_state': model.state_dict()}, 'best_transformer_model.pt')
        print(f"  → Best model saved!")

print("\nTraining complete!")

# ============================================================
# EVALUATION ON TEST SET
# ============================================================

print("\nEvaluating on test set...")

# Load best model
checkpoint = torch.load('best_transformer_model.pt')
model.load_state_dict(checkpoint['model_state'])
model.eval()

donor_predictions = {}
donor_targets = {}
donor_attention_weights = {}

with torch.no_grad():
    for donor_id in test_df['Donor ID'].unique():
        # Get all cells for this donor
        donor_mask = test_df['Donor ID'] == donor_id
        donor_data = test_df[donor_mask]
        
        X_donor = torch.tensor(donor_data[scgpt_cols].values, dtype=torch.float32).to(device)
        y1_donor = donor_data[y1_cols].values[0]
        y2_donor = donor_data[y2_cols].values[0]
        cell_types_donor = donor_data['Supertype'].values if 'Supertype' in donor_data.columns else None
        
        # Forward pass
        cell_features = model.transformer(X_donor.unsqueeze(1)).squeeze(1)
        donor_emb, attn_weights = model.attention_weighted_pooling(cell_features)
        donor_emb = donor_emb.unsqueeze(0)  # [1, 512]
        
        # Store attention weights
        donor_attention_weights[donor_id] = {
            'weights': attn_weights.cpu().numpy(),
            'cell_types': cell_types_donor
        }
        
        # Get predictions
        thal_pred = model.thal_head(donor_emb).argmax(dim=1).cpu().item()
        braak_pred = model.braak_head(donor_emb).argmax(dim=1).cpu().item()
        cerad_pred = model.cerad_head(donor_emb).argmax(dim=1).cpu().item()
        adnc_pred = model.adnc_head(donor_emb).argmax(dim=1).cpu().item()
        reg_pred = model.reg_head(donor_emb).cpu().numpy()[0]
        
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
for i, col in enumerate(y1_cols):
    print(f"  {col}: {reg_ccc_scores[i]:.4f}")
print(f"  Mean CCC: {np.mean(reg_ccc_scores):.4f}")

# ============================================================
# ATTENTION ANALYSIS
# ============================================================

print("\n" + "="*60)
print("ATTENTION WEIGHT ANALYSIS")
print("="*60)

def analyze_attention_by_cell_type(donor_attention_weights):
    """Analyze which cell types receive highest attention"""
    cell_type_attentions = defaultdict(list)
    
    for donor_id, data in donor_attention_weights.items():
        weights = data['weights']
        cell_types = data['cell_types']
        
        if cell_types is not None:
            for cell_type, weight in zip(cell_types, weights):
                cell_type_attentions[cell_type].append(weight)
    
    # Average attention per cell type
    avg_attention = {
        ct: (np.mean(weights), np.std(weights), len(weights))
        for ct, weights in cell_type_attentions.items()
    }
    
    # Sort by average attention
    sorted_attention = sorted(avg_attention.items(), key=lambda x: x[1][0], reverse=True)
    
    print("\nCell Types Ranked by Average Attention Weight:")
    print(f"{'Rank':<6}{'Cell Type':<40}{'Mean':<10}{'Std':<10}{'Count':<10}")
    print("-" * 76)
    
    for rank, (cell_type, (mean_attn, std_attn, count)) in enumerate(sorted_attention[:20], 1):
        print(f"{rank:<6}{cell_type:<40}{mean_attn:<10.6f}{std_attn:<10.6f}{count:<10}")
    
    return avg_attention, cell_type_attentions

avg_attention, cell_type_attentions = analyze_attention_by_cell_type(donor_attention_weights)

# Analyze a sample donor
sample_donor = list(donor_attention_weights.keys())[0]
sample_data = donor_attention_weights[sample_donor]

print(f"\n\nExample: Top 15 Most Attended Cells for Donor {sample_donor}")
print(f"{'Rank':<6}{'Cell Type':<40}{'Attention Weight':<20}")
print("-" * 66)

if sample_data['cell_types'] is not None:
    sorted_indices = np.argsort(sample_data['weights'])[::-1][:15]
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"{rank:<6}{sample_data['cell_types'][idx]:<40}{sample_data['weights'][idx]:<20.6f}")

# Summary statistics
print(f"\n\nAttention Weight Statistics Across All Test Donors:")
all_weights = np.concatenate([data['weights'] for data in donor_attention_weights.values()])
print(f"  Mean:   {np.mean(all_weights):.6f}")
print(f"  Median: {np.median(all_weights):.6f}")
print(f"  Std:    {np.std(all_weights):.6f}")
print(f"  Min:    {np.min(all_weights):.6f}")
print(f"  Max:    {np.max(all_weights):.6f}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
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

results_df.to_csv('test_predictions.csv', index=False)
print("\nPredictions saved to 'test_predictions.csv'")
