import numpy as np
import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import os

data_pth = "data/cells.npy"
figs_dir = "figs"

def main():

    # UMI: Unique Molecular Identifier

    cells = np.load(data_pth, allow_pickle=True).ravel()[0]

    for k, v in cells.items():
        print(f"{k} type: {type(v)} shape: {v.shape}")
    
    #print(cells["UMI"])

    # crete an AnnData object
    adata = ad.AnnData(cells["UMI"])
    adata.obs_names = [f"cell_{i}" for i, _ in enumerate(cells["classes"])]
    adata.var_names = cells["gene_ids"]
    adata.obs["class"] = cells["classes"]
    print(adata)

    # Compute QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")  # Identify mitochondrial genes
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=True, inplace=True)
    print(adata)

    # Print summary of the new QC metrics
    print(adata.obs.head())

    print(adata.var.head())

    # genes per cell
    plt.figure(figsize=(6,4))
    sns.histplot(adata.obs["n_genes_by_counts"], bins=50)
    plt.xlabel("Number of Genes Detected per Cell")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Distribution of Genes Detected per Cell")
    # Save the plot
    plt.savefig(os.path.join(figs_dir, "n_genes_by_counts_histogram.png"), dpi=300, bbox_inches="tight")

    sns.jointplot(
    data=adata.obs,
    x="log1p_total_counts",
    y="log1p_n_genes_by_counts",
    kind="hex")
    plt.xlim([5, 9])
    plt.ylim([5, 9])
    ax = plt.gca()
    ax.plot([5, 9], [5, 9], "--", color="black", linewidth=1)
    plt.savefig(os.path.join(figs_dir, "total_counts_n_genes_by_count_joint.png"), dpi=300)  # Save as a high-resolution PNG

if __name__ == "__main__":
    main()