import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import preprocess

data_dir = "data"
figs_dir = "figs"

def main():
    """
    Perform data analysis with of an annData and a given pre-processing. Plots are sabed in the figs_dir directory.
    """

    # read data
    data_pth = os.path.join(data_dir, 'cell_w_split.h5ad')
    adata = ad.read_h5ad(data_pth)
    train_set = adata[adata.obs["split"] == "train"]
    valid_set = adata[adata.obs["split"] == "valid"]

    # initialize datasets
    args_preproc = {
        "log_transform": True, 
        "normalize_per_cell": False, 
        "normalize_per_gene": False, 
        "filter_top_genes": True,
        "filter_min_counts": True,
        "min_counts": 3, 
        "n_top_genes": 1000
    }   

    # preprocess train set
    mean_per_gene, std_per_gene = preprocess(train_set, is_train_set=True, **args_preproc)

    # preprocess valid set
    preprocess(valid_set, is_train_set=False, **args_preproc, mean_train=mean_per_gene, std_train=std_per_gene, genes_to_keep=train_set.var_names)

    print(train_set)

    # visualize genes and cells
    plt.figure(figsize=(8, 6))
    sc.pl.heatmap(train_set, var_names=train_set.var_names, groupby="class", cmap="viridis")
    plt.xlabel("Genes")
    plt.ylabel("Cells")
    plt.savefig(os.path.join(figs_dir, "genes_cells_heatmap_train_preproc.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sc.pl.heatmap(valid_set, var_names=valid_set.var_names, groupby="class", cmap="viridis")
    plt.xlabel("Genes")
    plt.ylabel("Cells")
    plt.savefig(os.path.join(figs_dir, "genes_cells_heatmap_valid_preproc.png"))
    plt.close()


if __name__ == "__main__":
    main()