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
# Load donor-by-class latent vectors (long format)
# --------------------------------------------------
DATA_DIR = "./data"
df = pd.read_csv(os.path.join(DATA_DIR, "donor_latents_per_class_scvi.csv"))

print("Loaded:", df.shape)
print(df.head())

# --------------------------------------------------
# Pivot to wide format
# --------------------------------------------------
pivoted = df.pivot_table(
    index=["donor", "region", "ADNC"],
    columns="Class",
    values=[f"z{i}" for i in range(20)],
)

pivoted.columns = [
    f"{cls.replace(':','').replace(' ', '_')}_{z}"
    for (z, cls) in pivoted.columns.to_flat_index()
]

pivoted = pivoted.reset_index()

pivoted_A9 = pivoted[pivoted["region"] == "A9"].copy()
pivoted_MTG = pivoted[pivoted["region"] == "MTG"].copy()

print("Pivoted A9 shape:", pivoted_A9.shape)
print("Pivoted MTG shape:", pivoted_MTG.shape)

# --------------------------------------------------
# Extract feature matrix
# --------------------------------------------------
feature_cols_A9 = [c for c in pivoted_A9.columns if "_z" in c]
feature_cols_MTG = [c for c in pivoted_MTG.columns if "_z" in c]
X_A9 = pivoted_A9[feature_cols_A9].values
X_MTG = pivoted_MTG[feature_cols_MTG].values

# --------------------------------------------------
# Standardize
# --------------------------------------------------
scaler_A9 = StandardScaler()
X_A9_scaled = scaler_A9.fit_transform(X_A9)
scaler_MTG = StandardScaler()
X_MTG_scaled = scaler_MTG.fit_transform(X_MTG)

# --------------------------------------------------
# PCA
# --------------------------------------------------
pca_A9 = PCA(n_components=10)
X_A9_pca = pca_A9.fit_transform(X_A9_scaled)

print("\nExplained variance (10 PCs) [A9]:", pca_A9.explained_variance_ratio_)

pivoted_A9[[f"PC{i}" for i in range(1, 11)]] = X_A9_pca[:, :10]


pca_MTG = PCA(n_components=10)
X_MTG_pca = pca_MTG.fit_transform(X_MTG_scaled)

print("\nExplained variance (10 PCs) [MTG]:", pca_MTG.explained_variance_ratio_)

pivoted_MTG[[f"PC{i}" for i in range(1, 11)]] = X_MTG_pca[:, :10]

# # --------------------------------------------------
# # Save PCA scree plot
# # --------------------------------------------------

# plt.figure(figsize=(8, 5))
# plt.plot(pca_A9.explained_variance_ratio_, marker='o')
# plt.xlabel("PC index")
# plt.ylabel("Variance explained ratio")
# plt.title("A9 PCA Scree Plot")
# plt.grid(True)

# plot_path = os.path.join("data", "pca_scree_plot_A9.png")
# plt.savefig(plot_path, dpi=300, bbox_inches='tight')
# plt.close()

# print(f"A9 Scree plot saved to: {plot_path}")


# plt.figure(figsize=(8, 5))
# plt.plot(pca_MTG.explained_variance_ratio_, marker='o')
# plt.xlabel("PC index")
# plt.ylabel("Variance explained ratio")
# plt.title("MTG PCA Scree Plot")
# plt.grid(True)

# plot_path = os.path.join("data", "pca_scree_plot_MTG.png")
# plt.savefig(plot_path, dpi=300, bbox_inches='tight')
# plt.close()

# print(f"MTG Scree plot saved to: {plot_path}")

# # --------------------------------------------------
# # Calculate Silhouette Scores for k = 2 to 12
# # --------------------------------------------------
# scores = {}

# print("Computing silhouette scores...")
# for k in range(2, 13):
#     clustering = AgglomerativeClustering(n_clusters=k, linkage="ward")
#     labels = clustering.fit_predict(X_A9_pca)
#     score = silhouette_score(X_A9_pca, labels)
#     scores[k] = score
#     print(f"k={k}: silhouette={score:.4f}")

# plt.figure(figsize=(6,4))
# plt.plot(list(scores.keys()), list(scores.values()), marker='o')
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Silhouette score")
# plt.title("A9 Silhouette Score vs k")
# plt.grid(True)
# plt.savefig("data/silhouette_scores_A9.png", dpi=300, bbox_inches='tight')
# plt.close()

# print("A9 Silhouette plot saved to data/silhouette_scores_A9.png")


# scores = {}

# print("Computing silhouette scores...")
# for k in range(2, 13):
#     clustering = AgglomerativeClustering(n_clusters=k, linkage="ward")
#     labels = clustering.fit_predict(X_MTG_pca)
#     score = silhouette_score(X_MTG_pca, labels)
#     scores[k] = score
#     print(f"k={k}: silhouette={score:.4f}")

# plt.figure(figsize=(6,4))
# plt.plot(list(scores.keys()), list(scores.values()), marker='o')
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Silhouette score")
# plt.title("MTG Silhouette Score vs k")
# plt.grid(True)
# plt.savefig("data/silhouette_scores_MTG.png", dpi=300, bbox_inches='tight')
# plt.close()

# print("MTG Silhouette plot saved to data/silhouette_scores_MTG.png")

# --------------------------------------------------
# Clustering (Ward hierarchical)
# --------------------------------------------------
clustering_A9 = AgglomerativeClustering(
    n_clusters=2, linkage="ward"
)
pivoted_A9["cluster"] = clustering_A9.fit_predict(X_A9_pca)

print("\nCluster counts:")
print(pivoted_A9["cluster"].value_counts())


clustering_MTG = AgglomerativeClustering(
    n_clusters=2, linkage="ward"
)
pivoted_MTG["cluster"] = clustering_MTG.fit_predict(X_MTG_pca)

print("\nCluster counts:")
print(pivoted_MTG["cluster"].value_counts())

# --------------------------------------------------
# Create unified donor cluster table + global labels
# --------------------------------------------------

df_clusters = pd.concat([pivoted_A9, pivoted_MTG], ignore_index=True)

df_clusters["region_cluster"] = df_clusters["region"] + "_" + df_clusters["cluster"].astype(str)

global_mapping = {
    "A9_0": 0,
    "A9_1": 1,
    "MTG_0": 2,
    "MTG_1": 3,
}

df_clusters["global_cluster"] = df_clusters["region_cluster"].map(global_mapping)

df_clusters = df_clusters[["donor", "region", "ADNC", "cluster", "region_cluster", "global_cluster"]]

out_path = os.path.join(DATA_DIR, "donor_clusters_global.csv")
df_clusters.to_csv(out_path, index=False)

print(f"\nSaved donor cluster table with global labels to: {out_path}")

print("\nGlobal cluster counts:")
print(df_clusters["global_cluster"].value_counts())


# # --------------------------------------------------
# # Cluster x ADNC table
# # --------------------------------------------------
# print("\nA9 Cluster × ADNC:")
# cluster_adnc_table_A9 = pd.crosstab(pivoted_A9["cluster"], pivoted_A9["ADNC"])
# print(cluster_adnc_table_A9)

# cluster_adnc_path = os.path.join(DATA_DIR, "cluster_adnc_table_A9.csv")
# cluster_adnc_table_A9.to_csv(cluster_adnc_path)
# print(f"Saved A9 ADNC contingency table to: {cluster_adnc_path}")


# print("\nMTG Cluster × ADNC:")
# cluster_adnc_table_MTG = pd.crosstab(pivoted_MTG["cluster"], pivoted_MTG["ADNC"])
# print(cluster_adnc_table_MTG)

# cluster_adnc_path = os.path.join(DATA_DIR, "cluster_adnc_table_MTG.csv")
# cluster_adnc_table_MTG.to_csv(cluster_adnc_path)
# print(f"Saved MTG ADNC contingency table to: {cluster_adnc_path}")


# # --------------------------------------------------
# # PCA scatter plot colored by cluster
# # --------------------------------------------------
# plt.figure(figsize=(6, 5))
# plt.scatter(
#     X_pca[:, 0],
#     X_pca[:, 1],
#     c=pivoted["cluster"],
#     cmap="tab10",
#     alpha=0.9,
#     edgecolors='none',
#     s=40
# )
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("Donor PCA colored by Ward clusters (k=3)")
# plt.grid(True)

# pca_cluster_plot_path = os.path.join(DATA_DIR, "pca_clusters_k3.png")
# plt.savefig(pca_cluster_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"PCA cluster plot saved to: {pca_cluster_plot_path}")


# # --------------------------------------------------
# # PCA scatter focusing on PC2 vs PC3 (with correct legend)
# # --------------------------------------------------

# plt.figure(figsize=(6, 5))

# scatter = plt.scatter(
#     X_pca[:, 1],   # PC2
#     X_pca[:, 2],   # PC3
#     c=pivoted["cluster"],
#     cmap="tab10",
#     s=40,
#     alpha=0.9,
#     edgecolors="none",
# )

# plt.xlabel("PC2")
# plt.ylabel("PC3")
# plt.title("PCA (PC2 vs PC3) highlighting subtle cluster structure (k=3)")
# plt.grid(True)

# pca_pc2_pc3_path = os.path.join(DATA_DIR, "pca_pc2_pc3_clusters_k3.png")
# plt.savefig(pca_pc2_pc3_path, dpi=300, bbox_inches='tight')
# plt.close()

# print(f"PCA (PC2 vs PC3) plot saved to: {pca_pc2_pc3_path}")




# # --------------------------------------------------
# # UMAP visualization colored by cluster
# # --------------------------------------------------
# print("\nComputing UMAP (optional)...")
# reducer = umap.UMAP(
#     n_neighbors=15,
#     min_dist=0.1,
#     metric="euclidean",
#     random_state=42
# )
# X_umap = reducer.fit_transform(X_pca)

# plt.figure(figsize=(6, 5))
# plt.scatter(
#     X_umap[:, 0],
#     X_umap[:, 1],
#     c=pivoted["cluster"],
#     cmap="tab10",
#     alpha=0.9,
#     s=40
# )
# plt.xlabel("UMAP-1")
# plt.ylabel("UMAP-2")
# plt.title("UMAP colored by donor cluster (k=3)")
# plt.grid(True)

# umap_plot_path = os.path.join(DATA_DIR, "umap_clusters_k3.png")
# plt.savefig(umap_plot_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"UMAP cluster plot saved to: {umap_plot_path}")


# # --------------------------------------------------
# # Save donor --> cluster assignments
# # --------------------------------------------------
# cluster_assignments_path = os.path.join(DATA_DIR, "donor_clusters_k3.csv")

# pivoted[["donor", "region", "ADNC", "cluster"]].to_csv(
#     cluster_assignments_path,
#     index=False
# )

# print(f"\nSaved donor cluster assignments to: {cluster_assignments_path}")























