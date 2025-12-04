import os
import pandas as pd
import numpy as np
import random
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import umap
import math

# --------------------------------------------------
# Load donor-by-class latent vectors (long format)
# --------------------------------------------------
DATA_DIR = "./data"
df = pd.read_csv(os.path.join(DATA_DIR, "donor_clusters_k3_aggregated.csv"))

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
# TRAIN/VAL/TEST SPLIT (stratified by cluster)
# ============================================================
df["sample_id"] = df["donor"]

print("\nSplitting data by donor, stratified by cluster...")

TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.15, 0.15

DONOR_COL = "donor"
CLUSTER_COL = "cluster"

train_ids = []
val_ids = []
test_ids = []

for cl, cl_df in df.groupby(CLUSTER_COL):

    ids = cl_df["sample_id"].tolist()
    random.shuffle(ids)
    n = len(ids)

    n_train = math.floor(TRAIN_SPLIT * n)
    remaining = n - n_train

    n_val = max(1, math.floor(VAL_SPLIT * n))

    # Ensure we leave AT LEAST 1 for test
    if n_val > remaining - 1:
        n_val = remaining - 1

    n_test = remaining - n_val

    cl_train = ids[:n_train]
    cl_val   = ids[n_train:n_train+n_val]
    cl_test  = ids[n_train+n_val:n_train+n_val+n_test]

    train_ids.extend(cl_train)
    val_ids.extend(cl_val)
    test_ids.extend(cl_test)

    print(f"Cluster {cl}: n={n}, train={n_train}, val={n_val}, test={n_test}")

# ============================================================
# ASSIGN SPLIT LABELS BACK TO FULL DF
# ============================================================

df["split"] = "none"
df.loc[df["sample_id"].isin(train_ids), "split"] = "train"
df.loc[df["sample_id"].isin(val_ids),   "split"] = "val"
df.loc[df["sample_id"].isin(test_ids),  "split"] = "test"

# ============================================================
# PRINT SUMMARIES
# ============================================================

print("\n=== Final Split Counts ===")
print(df["split"].value_counts())

print("\n=== Cluster counts in Train ===")
print(df[df.split=="train"][CLUSTER_COL].value_counts().sort_index())

print("\n=== Cluster counts in Val ===")
print(df[df.split=="val"][CLUSTER_COL].value_counts().sort_index())

print("\n=== Cluster counts in Test ===")
print(df[df.split=="test"][CLUSTER_COL].value_counts().sort_index())

out_path = os.path.join(DATA_DIR, "donor_clusters_aggregated_splits.csv")
df.to_csv(out_path, index=False)
print("\nSaved: donor_clusters_aggregated_splits.csv")