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
import itertools

# ============================================================
# 1. SETUP & CONFIGURATION
# ============================================================

# --- Define Paths ---
SAVE_DIR = '/users/cpschult/scratch/hip/'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Outputs will be saved to: {SAVE_DIR}")

# --- Set Seed for Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================
# 2. DATA LOADING & PREPROCESSING
# ============================================================

print("\nLoading data...")
# Ensure input file exists at this path or update accordingly
input_csv_path = os.path.join(SAVE_DIR, 'cell_level_scgpt.csv') 
# If file is not in scratch yet, use original path: '/users/cpschult/scratch/hip/cell_level_scgpt.csv'

df = pd.read_csv(input_csv_path)
print(f"Loaded {len(df)} cells from {df['Donor ID'].nunique()} donors")

scgpt_cols = [col for col in df.columns if col.startswith('scGPT_')]
print(f"Found {len(scgpt_cols)} scGPT embedding dimensions")

print("\nPreprocessing...")

# Ordinal label mappings
ordinal_maps = {
    'ADNC':  {"Not AD": 0, "Low": 1, "Intermediate": 2, "High": 3},
    'Braak': {"Braak 0": 0, "Braak I": 1, "Braak II": 2, "Braak III": 3, 
              "Braak IV": 4, "Braak V": 5, "Braak VI": 6},
    'Thal':  {"Thal 0": 0, "Thal 1": 1, "Thal 2": 2, "Thal 3": 3, "Thal 4": 4, "Thal 5": 5},
    'CERAD': {"Absent": 0, "Sparse": 1, "Moderate": 2, "Frequent": 3},
}

for col, mapping in ordinal_maps.items():
    if col in df:
        df[col] = df[col].map(mapping).fillna(0).astype(int)

# Numeric Scaling
numerical_features = ['Age at Death', 'Years of education', 'PMI']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Handle Missing Values (Donor Median Imputation)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df.groupby('Donor ID')[numeric_cols].transform(lambda x: x.fillna(x.median()))
df = df.fillna(df.median(numeric_only=True))

# ============================================================
# 3. SPLIT & WEIGHTS
# ============================================================

print("\nSplitting data by donor...")
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.15, 0.15
donors = list(df['Donor ID'].unique())
donors.sort()
random.shuffle(donors)

n_donors = len(donors)
train_idx = int(n_donors * TRAIN_SPLIT)
val_idx = int(n_donors * (TRAIN_SPLIT + VAL_SPLIT))

train_donors = donors[:train_idx]
val_donors = donors[train_idx:val_idx]
test_donors = donors[val_idx:]

train_df = df[df['Donor ID'].isin(train_donors)].copy()
val_df = df[df['Donor ID'].isin(val_donors)].copy()
test_df = df[df['Donor ID'].isin(test_donors)].copy()

print(f"Train: {len(train_donors)} | Val: {len(val_donors)} | Test: {len(test_donors)}")

# Target Columns
y1_cols = ['percent 6e10 positive area', 'percent AT8 positive area',
           'percent NeuN positive area', 'percent GFAP positive area']
y2_cols = ['Thal', 'Braak', 'CERAD', 'ADNC']

# Scale Regression Targets
y1_scaler = StandardScaler()
train_df[y1_cols] = y1_scaler.fit_transform(train_df[y1_cols].values)
val_df[y1_cols] = y1_scaler.transform(val_df[y1_cols].values)
test_df[y1_cols] = y1_scaler.transform(test_df[y1_cols].values)

# Calculate Class Weights (Per Donor)
def get_class_weights(df, target_col):
    donor_df = df.drop_duplicates(subset=['Donor ID'])
    counts = donor_df[target_col].value_counts().sort_index()
    weights = len(donor_df) / (len(counts) * counts)
    
    max_idx = int(donor_df[target_col].max())
    full_weights = torch.ones(max_idx + 1)
    for idx, w in weights.items():
        full_weights[int(idx)] = w
    return full_weights.float().to(device)

print("\nCalculating Class Weights...")
w_thal = get_class_weights(train_df, 'Thal')
w_braak = get_class_weights(train_df, 'Braak')
w_cerad = get_class_weights(train_df, 'CERAD')
w_adnc = get_class_weights(train_df, 'ADNC')

# ============================================================
# 4. DATASET & MODEL
# ============================================================

class DonorLevelDataset(torch.utils.data.Dataset):
    def __init__(self, df, scgpt_cols, y1_cols, y2_cols, max_cells=1024, mode='train'):
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
        all_cells = group[self.scgpt_cols].values
        
        # Subsampling
        n_avail = len(all_cells)
        if n_avail > self.max_cells:
            if self.mode == 'train':
                indices = np.random.choice(n_avail, self.max_cells, replace=False)
            else:
                rng = np.random.RandomState(seed=42+idx)
                indices = rng.choice(n_avail, self.max_cells, replace=False)
            x_data = all_cells[indices]
        else:
            x_data = all_cells

        x = torch.tensor(x_data, dtype=torch.float32)
        y1 = torch.tensor(group[self.y1_cols].values[0], dtype=torch.float32)
        y2 = torch.tensor(group[self.y2_cols].values[0], dtype=torch.long)
        return x, y1, y2, donor, []

def collate_donors(batch):
    xs, y1s, y2s, ids, _ = zip(*batch)
    X_padded = pad_sequence(xs, batch_first=True, padding_value=0.0)
    pad_mask = torch.zeros((len(xs), X_padded.size(1)), dtype=torch.bool)
    for i, x in enumerate(xs):
        pad_mask[i, len(x):] = True 
    return X_padded, pad_mask, torch.stack(y1s), torch.stack(y2s), ids, _

MAX_CELLS = 1024
train_ds = DonorLevelDataset(train_df, scgpt_cols, y1_cols, y2_cols, max_cells=MAX_CELLS, mode='train')
val_ds = DonorLevelDataset(val_df, scgpt_cols, y1_cols, y2_cols, max_cells=MAX_CELLS, mode='eval')
test_ds = DonorLevelDataset(test_df, scgpt_cols, y1_cols, y2_cols, max_cells=MAX_CELLS, mode='eval')

class TransformerNet(nn.Module):
    def __init__(self, embed_dim=512, hidden=128, n_heads=4, n_layers=1, dropout=0.5):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=hidden,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.Tanh(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )
        # Heads
        self.thal_head = nn.Linear(embed_dim, 6)
        self.braak_head = nn.Linear(embed_dim, 7)
        self.cerad_head = nn.Linear(embed_dim, 4)
        self.adnc_head = nn.Linear(embed_dim, 4)
        self.reg_head = nn.Linear(embed_dim, 4)
    
    def forward(self, x, mask=None):
        h = self.transformer(x, src_key_padding_mask=mask) 
        attn_scores = self.attention_pool(h)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        emb = (h * attn_weights).sum(dim=1)
        return (self.thal_head(emb), self.braak_head(emb), 
                self.cerad_head(emb), self.adnc_head(emb), self.reg_head(emb))

# ============================================================
# 5. METRIC HELPERS
# ============================================================

def compute_ccc(x, y):
    """Numpy Concordance Correlation Coefficient"""
    if len(x) < 2: return 0.0
    x_mean, y_mean = np.mean(x), np.mean(y)
    cov = np.mean((x - x_mean) * (y - y_mean))
    return (2 * cov) / (np.var(x) + np.var(y) + (x_mean - y_mean)**2 + 1e-8)

def ccc_loss(pred, target):
    """PyTorch CCC Loss"""
    mx, my = pred.mean(0), target.mean(0)
    vx, vy = pred.var(0, unbiased=False), target.var(0, unbiased=False)
    cov = ((pred - mx) * (target - my)).mean(0)
    ccc = (2 * cov) / (vx + vy + (mx - my).pow(2) + 1e-8)
    return 1.0 - ccc.mean()

def evaluate_model(model, dataloader, device):
    """Runs inference and calculates aggregated metrics."""
    model.eval()
    preds = defaultdict(list)
    trues = defaultdict(list)
    val_loss = 0
    
    # Loss functions for reporting
    crit_cls = nn.CrossEntropyLoss()
    crit_reg = nn.SmoothL1Loss()
    
    with torch.no_grad():
        for x, mask, y1, y2, ids, _ in dataloader:
            x, mask, y1, y2 = x.to(device), mask.to(device), y1.to(device), y2.to(device)
            thal, braak, cerad, adnc, reg = model(x, mask)
            
            # Simple unweighted loss for tracking
            loss = (crit_cls(thal, y2[:,0]) + crit_cls(braak, y2[:,1]) + 
                    crit_cls(cerad, y2[:,2]) + crit_cls(adnc, y2[:,3]) + crit_reg(reg, y1))
            val_loss += loss.item()

            # Store
            preds['thal'].extend(thal.argmax(1).cpu().numpy())
            trues['thal'].extend(y2[:,0].cpu().numpy())
            preds['braak'].extend(braak.argmax(1).cpu().numpy())
            trues['braak'].extend(y2[:,1].cpu().numpy())
            preds['cerad'].extend(cerad.argmax(1).cpu().numpy())
            trues['cerad'].extend(y2[:,2].cpu().numpy())
            preds['adnc'].extend(adnc.argmax(1).cpu().numpy())
            trues['adnc'].extend(y2[:,3].cpu().numpy())
            preds['reg'].extend(reg.cpu().numpy())
            trues['reg'].extend(y1.cpu().numpy())

    # Calculate Aggregated Metrics
    res = {'loss': val_loss / len(dataloader)}
    
    # QWK
    for k in ['thal', 'braak', 'cerad', 'adnc']:
        res[f'{k}_k'] = cohen_kappa_score(trues[k], preds[k], weights='quadratic')
    res['mean_qwk'] = np.mean([res['thal_k'], res['braak_k'], res['cerad_k'], res['adnc_k']])
    
    # CCC
    reg_trues = np.array(trues['reg'])
    reg_preds = np.array(preds['reg'])
    cccs = [compute_ccc(reg_trues[:,i], reg_preds[:,i]) for i in range(reg_trues.shape[1])]
    res['mean_ccc'] = np.mean(cccs)
    
    res['raw_preds'], res['raw_trues'] = preds, trues
    return res

# ============================================================
# 6. GRID SEARCH
# ============================================================

param_grid = {
    'lr': [1e-4, 5e-5],
    'weight_decay': [1e-2, 0.1],
    'hidden_dim': [64, 128],
    'n_layers': [1],
    'dropout': [0.5],
    'batch_size': [8]
}

keys, values = zip(*param_grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"\nStarting Grid Search ({len(experiments)} experiments)...")
print(f"{'ID':<3} | {'LR':<8} | {'WD':<6} | {'Hid':<4} | {'Val Loss':<8} | {'Mean QWK':<8} | {'Mean CCC':<8}")
print("-" * 75)

best_overall_val_loss = float('inf')
best_overall_params = None
best_model_path = ""
timestamp = datetime.datetime.now().strftime("%m%d_%H%M")

for i, params in enumerate(experiments):
    # Setup
    train_dl = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_donors)
    val_dl = DataLoader(val_ds, batch_size=params['batch_size'], collate_fn=collate_donors)
    
    model = TransformerNet(embed_dim=512, hidden=params['hidden_dim'], n_layers=params['n_layers'], dropout=params['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    # Weighted Losses
    loss_fns = [nn.CrossEntropyLoss(weight=w, label_smoothing=0.1) for w in [w_thal, w_braak, w_cerad, w_adnc]]
    smooth_l1 = nn.SmoothL1Loss()
    
    # Train Loop
    best_exp_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(25):
        model.train()
        for x, mask, y1, y2, _, _ in train_dl:
            x, mask, y1, y2 = x.to(device), mask.to(device), y1.to(device), y2.to(device)
            opt.zero_grad()
            thal, braak, cerad, adnc, reg = model(x, mask)
            
            loss = (loss_fns[0](thal, y2[:,0]) + loss_fns[1](braak, y2[:,1]) + 
                    loss_fns[2](cerad, y2[:,2]) + loss_fns[3](adnc, y2[:,3]) + 
                    ccc_loss(reg, y1) + 0.5 * smooth_l1(reg, y1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        
        # Validation
        val_res = evaluate_model(model, val_dl, device)
        
        if val_res['loss'] < best_exp_loss:
            best_exp_loss = val_res['loss']
            patience_counter = 0
            
            if best_exp_loss < best_overall_val_loss:
                best_overall_val_loss = best_exp_loss
                best_overall_params = params
                best_model_path = os.path.join(SAVE_DIR, f'best_model_{timestamp}.pt')
                torch.save({'model_state': model.state_dict(), 'params': params}, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= 5: break
            
    print(f"{i+1:<3} | {params['lr']:<8.1e} | {params['weight_decay']:<6} | {params['hidden_dim']:<4} | "
          f"{best_exp_loss:<8.4f} | {val_res['mean_qwk']:<8.4f} | {val_res['mean_ccc']:<8.4f}")

# ============================================================
# 7. FINAL TEST EVALUATION
# ============================================================

print("\n" + "="*60)
print("FINAL TEST EVALUATION")
print(f"Best Params: {best_overall_params}")
print("="*60)

# Load Best
checkpoint = torch.load(best_model_path)
final_model = TransformerNet(embed_dim=512, hidden=checkpoint['params']['hidden_dim'], 
                             n_layers=checkpoint['params']['n_layers'], dropout=checkpoint['params']['dropout']).to(device)
final_model.load_state_dict(checkpoint['model_state'])

test_dl = DataLoader(test_ds, batch_size=8, collate_fn=collate_donors)
res = evaluate_model(final_model, test_dl, device)

print(f"\nOverall Metrics:")
print(f"  Mean QWK: {res['mean_qwk']:.4f}")
print(f"  Mean CCC: {res['mean_ccc']:.4f}")

print("\nClassification (QWK):")
print(f"  Thal:  {res['thal_k']:.4f}")
print(f"  Braak: {res['braak_k']:.4f}")
print(f"  CERAD: {res['cerad_k']:.4f}")
print(f"  ADNC:  {res['adnc_k']:.4f}")

print("\nRegression (CCC):")
reg_trues = np.array(res['raw_trues']['reg'])
reg_preds = np.array(res['raw_preds']['reg'])
for i, col in enumerate(y1_cols):
    score = compute_ccc(reg_trues[:, i], reg_preds[:, i])
    print(f"  {col:<28}: {score:.4f}")

# Save CSV
csv_path = os.path.join(SAVE_DIR, 'final_predictions.csv')
out_df = pd.DataFrame()
for k in ['Thal', 'Braak', 'CERAD', 'ADNC']:
    out_df[f'{k}_True'] = res['raw_trues'][k.lower()]
    out_df[f'{k}_Pred'] = res['raw_preds'][k.lower()]

for i, col in enumerate(y1_cols):
    out_df[f'True_{col}'] = reg_trues[:, i]
    out_df[f'Pred_{col}'] = reg_preds[:, i]

out_df.to_csv(csv_path, index=False)
print(f"\nSaved results to: {csv_path}")