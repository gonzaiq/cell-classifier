import numpy as np
import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

data_pth = "data/cells.npy"
figs_dir = "figs"

def main():

    # UMI: Unique Molecular Identifier

    cells = np.load(data_pth, allow_pickle=True).ravel()[0]

    for k, v in cells.items():
        print(f"{k} type: {type(v)} shape: {v.shape}")
    
    # crete an AnnData object
    adata = ad.AnnData(cells["UMI"])
    adata.obs_names = [f"cell_{i}" for i, _ in enumerate(cells["classes"])]
    adata.var_names = cells["gene_ids"]
    adata.obs["class"] = cells["classes"]
    print(adata)
    print(f"Min value: {adata.X.min()}, Max value: {adata.X.max()}")

    # Apply log1p transformation
    adata_log = sc.pp.log1p(adata, copy=True)
    print(f"After log1p - Min value: {adata_log.X.min()}, Max value: {adata_log.X.max()}")

    # Compute QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")  # Identify mitochondrial genes
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True)
    print(adata)

    # Print summary of the new QC metrics
    print(adata.obs.head())

    print(adata.var.head())

    # visualize genes and cells
    plt.figure(figsize=(8, 6))
    sc.pl.heatmap(adata_log, var_names=adata.var_names, groupby="class", cmap="viridis")
    plt.xlabel("Genes")
    plt.ylabel("Cells")
    plt.savefig(os.path.join(figs_dir, "genes_cells_heatmap_log.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sc.pl.heatmap(adata, var_names=adata.var_names, groupby="class", cmap="viridis")
    plt.xlabel("Genes")
    plt.ylabel("Cells")
    plt.savefig(os.path.join(figs_dir, "genes_cells_heatmap.png"))
    plt.close()

    # highly expressive genes
    plt.figure(figsize=(6,4))
    sc.pl.highest_expr_genes(adata, n_top=20)
    plt.savefig(os.path.join(figs_dir, "highly_expressive_genes.png"), bbox_inches="tight")
    plt.close()

    # highly expressive genes
    plt.figure(figsize=(6,4))
    sc.pl.highly_variable_genes(adata, n_top=20)
    plt.savefig(os.path.join(figs_dir, "highly_variable_genes.png"), bbox_inches="tight")
    plt.close()

    # cell label histogram
    label_encoder = LabelEncoder()
    encoded_names = label_encoder.fit_transform(adata.obs["class"])
    plt.figure(figsize=(6,4))
    sns.histplot(encoded_names, bins=50)
    plt.xlabel("Cell classes")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Cell class")
    # Save the plot
    plt.savefig(os.path.join(figs_dir, "cell_classes.png"), bbox_inches="tight")
    plt.close()

    # cells per gene
    plt.figure(figsize=(6,4))
    sns.histplot(adata.var["n_cells_by_counts"], bins=50)
    plt.xlabel("Number of Celles Detected per Gene")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Distribution of Cells Detected per Gene")
    # Save the plot
    plt.savefig(os.path.join(figs_dir, "n_cells_by_counts_histogram.png"), bbox_inches="tight")
    plt.close()

    # genes per cell
    plt.figure(figsize=(6,4))
    sns.histplot(adata.obs["n_genes_by_counts"], bins=50)
    plt.xlabel("Number of Genes Detected per Cell")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Distribution of Genes Detected per Cell")
    # Save the plot
    plt.savefig(os.path.join(figs_dir, "n_genes_by_counts_histogram.png"), bbox_inches="tight")
    plt.close()

    sns.jointplot(
    data=adata.obs,
    x="log1p_total_counts",
    y="log1p_n_genes_by_counts",
    kind="hex")
    plt.xlim([5, 9])
    plt.ylim([5, 9])
    ax = plt.gca()
    ax.plot([5, 9], [5, 9], "--", color="black", linewidth=1)
    plt.savefig(os.path.join(figs_dir, "total_counts_n_genes_by_count_joint.png"))
    plt.close()

    # pca plot
    #pca = PCA(n_components=2)
    sc.pp.pca(adata, n_comps=2)
    #X_red = pca.fit_transform(adata.X.toarray())
    #df = pd.DataFrame(X_red, columns=["PC1", "PC2"])
    df = pd.DataFrame(adata.obsm['X_pca'], columns=["PC1", "PC2"])
    df["class"] = adata.obs["class"].values
    # Plot with seaborn
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="PC1", y="PC2", hue="class", data=df, alpha=0.7, palette="Set2")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Projection")
    plt.savefig(os.path.join(figs_dir, "pca.png"))
    plt.close()

    # t-SNE plot
    sc.tl.tsne(adata, n_pcs=2)
    df_tsne = pd.DataFrame(adata.obsm['X_tsne'], columns=["TSNE1", "TSNE2"])
    df_tsne["class"] = adata.obs["class"].values  # Assuming 'class' is stored in adata.obs
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="TSNE1", y="TSNE2", hue="class", data=df_tsne, alpha=0.7, palette="Set2")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Projection Colored by Class")
    plt.savefig(os.path.join(figs_dir, "tsne.png"))
    plt.close()

    print(adata)

if __name__ == "__main__":
    main()