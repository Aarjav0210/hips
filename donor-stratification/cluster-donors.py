import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import umap

# --------------------------------------------------
# Load pre-aggregated donor-level latent vectors
# --------------------------------------------------
DATA_DIR = "./data"
df = pd.read_csv(os.path.join(DATA_DIR, "donor_latents_aggregated.csv"))

print("Loaded:", df.shape)
print("Columns:", df.columns.tolist()[:10], "...", df.columns.tolist()[-10:])
print("\nFirst few rows:")
print(df.head())

# --------------------------------------------------
# Extract feature matrix
# --------------------------------------------------
y_class_cols = [f'y{i}_class{j}' for j in range(1, 4) for i in range(20)]
z_class_cols = [f'z{i}_class{j}' for j in range(1, 4) for i in range(20)]
feature_cols = y_class_cols + z_class_cols

X = df[feature_cols].values

print(f"\nFeature matrix shape: {X.shape}")

# --------------------------------------------------
# Standardize
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# PCA
# --------------------------------------------------
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained variance (10 PCs):", pca.explained_variance_ratio_)
print(f"Cumulative variance explained: {pca.explained_variance_ratio_.sum():.4f}")

df[[f"PC{i}" for i in range(1, 11)]] = X_pca[:, :10]

# --------------------------------------------------
# Save PCA scree plot
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(pca.explained_variance_ratio_, marker='o')
plt.xlabel("PC index")
plt.ylabel("Variance explained ratio")
plt.title("PCA Scree Plot")
plt.grid(True)

plot_path = os.path.join(DATA_DIR, "pca_scree_plot_aggregated.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Scree plot saved to: {plot_path}")

# --------------------------------------------------
# Calculate Silhouette Scores for k = 2 to 12
# --------------------------------------------------
scores = {}

print("\nComputing silhouette scores...")
for k in range(2, 13):
    clustering = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = clustering.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    scores[k] = score
    print(f"k={k}: silhouette={score:.4f}")

plt.figure(figsize=(6, 4))
plt.plot(list(scores.keys()), list(scores.values()), marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Score vs k")
plt.grid(True)
plt.savefig(os.path.join(DATA_DIR, "silhouette_scores_aggregated.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"Silhouette plot saved to: {os.path.join(DATA_DIR, 'silhouette_scores_aggregated.png')}")

# --------------------------------------------------
# Clustering (Ward hierarchical)
# --------------------------------------------------
clustering = AgglomerativeClustering(
    n_clusters=3, linkage="ward"
)
df["cluster"] = clustering.fit_predict(X_pca)

print("\nCluster counts:")
print(df["cluster"].value_counts())

# --------------------------------------------------
# Cluster x ADNC table
# --------------------------------------------------
print("\nCluster x ADNC:")
cluster_adnc_table = pd.crosstab(df["cluster"], df["ADNC"])
print(cluster_adnc_table)

cluster_adnc_path = os.path.join(DATA_DIR, "cluster_adnc_table_aggregated.csv")
cluster_adnc_table.to_csv(cluster_adnc_path)
print(f"Saved ADNC contingency table to: {cluster_adnc_path}")

# --------------------------------------------------
# PCA scatter plot colored by cluster
# --------------------------------------------------
plt.figure(figsize=(6, 5))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["cluster"],
    cmap="tab10",
    alpha=0.9,
    edgecolors='none',
    s=40
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Donor PCA colored by Ward clusters (k=3)")
plt.grid(True)

pca_cluster_plot_path = os.path.join(DATA_DIR, "pca_clusters_k3_aggregated.png")
plt.savefig(pca_cluster_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"PCA cluster plot saved to: {pca_cluster_plot_path}")

# --------------------------------------------------
# PCA scatter focusing on PC2 vs PC3
# --------------------------------------------------
plt.figure(figsize=(6, 5))

scatter = plt.scatter(
    X_pca[:, 1],   # PC2
    X_pca[:, 2],   # PC3
    c=df["cluster"],
    cmap="tab10",
    s=40,
    alpha=0.9,
    edgecolors="none",
)

plt.xlabel("PC2")
plt.ylabel("PC3")
plt.title("PCA (PC2 vs PC3) highlighting subtle cluster structure (k=3)")
plt.grid(True)

pca_pc2_pc3_path = os.path.join(DATA_DIR, "pca_pc2_pc3_clusters_k3_aggregated.png")
plt.savefig(pca_pc2_pc3_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"PCA (PC2 vs PC3) plot saved to: {pca_pc2_pc3_path}")

# --------------------------------------------------
# UMAP visualization colored by cluster
# --------------------------------------------------
print("\nComputing UMAP...")
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)
X_umap = reducer.fit_transform(X_pca)

plt.figure(figsize=(6, 5))
plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=df["cluster"],
    cmap="tab10",
    alpha=0.9,
    s=40
)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title("UMAP colored by donor cluster (k=3)")
plt.grid(True)

umap_plot_path = os.path.join(DATA_DIR, "umap_clusters_k3_aggregated.png")
plt.savefig(umap_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"UMAP cluster plot saved to: {umap_plot_path}")

# --------------------------------------------------
# Save donor --> cluster assignments
# --------------------------------------------------
cluster_assignments_path = os.path.join(DATA_DIR, "donor_clusters_k3_aggregated.csv")

df[["donor", "ADNC", "cluster"]].to_csv(
    cluster_assignments_path,
    index=False
)

print(f"\nSaved donor cluster assignments to: {cluster_assignments_path}")
