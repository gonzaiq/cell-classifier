import numpy as np
import anndata as ad
from sklearn.model_selection import train_test_split
import os

data_dir = "data"
test_ratio = 0.2
valid_ratio = 0.2

def main():

    # open cells file
    cells = np.load(os.path.join(data_dir, "cells.npy"), allow_pickle=True).ravel()[0]
    
    # crete an AnnData object
    adata = ad.AnnData(cells["UMI"])
    adata.obs_names = [f"cell_{i}" for i, _ in enumerate(cells["classes"])]
    adata.var_names = cells["gene_ids"]
    adata.obs["class"] = cells["classes"]
    print(adata)

    # split indices for train, validation, and test
    train_valid_obs, test_obs = train_test_split(
        adata.obs_names, test_size=test_ratio, stratify=adata.obs["class"]
    )
    train_valid_subset = adata[adata.obs_names.isin(train_valid_obs)]
    train_obs, valid_obs = train_test_split(
        train_valid_obs, test_size=valid_ratio, stratify=train_valid_subset.obs["class"]
    )

    adata.obs["split"] = "unassigned"
    adata.obs.loc[adata.obs_names.isin(train_obs), "split"] = "train" 
    adata.obs.loc[adata.obs_names.isin(valid_obs), "split"] = "valid" 
    adata.obs.loc[adata.obs_names.isin(test_obs), "split"] = "test" 

    # print for verifications
    print(adata)
    print(adata.obs["split"])
    print("Training set: ", len(adata.obs[adata.obs["split"] == "train"]))
    print("Validation set: ", len(adata.obs[adata.obs["split"] == "valid"]))
    print("Test set: ", len(adata.obs[adata.obs["split"] == "test"]))
    print("Unassigned: ", len(adata.obs[adata.obs["split"] == "teunassignedst"]))
    print("Total: ", len(adata))

    # save the AnnData object to a file
    adata.write(os.path.join(data_dir, "cell_w_split.h5ad"))

if __name__ == "__main__":
    main()