from typing import List, Optional
import scanpy as sc
import anndata as ad
import numpy as np
from numpy.typing import NDArray
import sys
import warnings
from scipy.sparse import issparse
from operator import truediv
import scipy.sparse as sp
import os
from sklearn.model_selection import train_test_split

def scale_array(
    X: np.ndarray,
    *,
    zero_center: bool = True,
    max_value: float | None = None,
    copy: bool = False,
    mean: np.ndarray,
    std: np.ndarray
    ) -> np.ndarray:
    """
    Scale the input array with a given mean and std.

    Args:
        X (np.ndarray): Input data array with shape (n_samples, n_features). If the array has an integer dtype, it will be cast to float.
        zero_center (bool): If True, subtract the mean (provided by `mean`) from each feature (column) of X.
        max_value (float): If provided, values in X will be clipped to the range [-max_value, max_value] after scaling.
        copy (bool): If True, a copy of X is made before performing any operations.
        mean (np.ndarray): 1D array of means used for zero-centering each feature.
        std (np.ndarray): 1D array of standard deviations used to scale each feature.
    Returns
        (np.ndarray) The scaled (and optionally clipped) array.
    """
    if copy:
        X = X.copy()

    if np.issubdtype(X.dtype, np.integer):
        X = X.astype(float)

    if zero_center:
        X -= mean

    X = sc._utils.axis_mul_or_truediv(
        X,
        std,
        op=truediv,
        out=X if isinstance(X, np.ndarray) or issparse(X) else None,
        axis=1,
    )

    # do the clipping
    if max_value is not None:
        if issparse(X):
            X.data = sc.pp._scale.clip_array(X.data, max_value=max_value, zero_center=False)
        else:
            X = sc.pp._scale.clip_array(X, max_value=max_value, zero_center=zero_center)

    return X

def preprocess_train(
        adata: ad.AnnData, 
        log_transform: bool = True, 
        normalize_per_cell: bool = True, 
        normalize_per_gene: bool = True,
        filter_min_counts: bool= True,
        min_counts: int = 1, 
        filter_top_genes: bool = True, 
        n_top_genes: int=1000):
    """
    Preprocess an AnnData object for training by performing filtering, normalization, log transformation,
    and scaling.

    This function applies several preprocessing steps on a training dataset:
      - Optionally filters out genes with fewer than `min_counts` counts.
      - Identifies highly variable genes and subsets the data accordingly.
      - Normalizes the data per cell by total counts.
      - Applies log1p transformation (log(x + 1)).
      - Scales each gene (feature) to unit variance and zero mean, returning the computed mean and std for use in evaluation.

    Args:
        adata (ad.AnnData): Annotated data matrix to preprocess.  
        log_transform (bool, optional): If True, apply log1p transformation to the data. Defaults to True.  
        normalize_per_cell (bool, optional): If True, normalize counts per cell (total-count normalization). Defaults to True.  
        normalize_per_gene (bool, optional): If True, scale genes to zero mean and unit variance. Defaults to True.  
        filter_min_counts (bool, optional): If True, filter out genes that do not have at least `min_counts` counts. Defaults to True.  
        min_counts (int, optional): Minimum number of counts required for a gene to be kept. Defaults to 1.  
        filter_top_genes (bool, optional): If True, select the top highly-variable genes. Defaults to True.  
        n_top_genes (int, optional): Number of top highly-variable genes to select. Defaults to 1000.  


    Returns
        _mean (np.ndarray): The per-gene mean computed during scaling (if `normalize_per_gene` is True). Otherwise, None.
        _std : (np.ndarray): The per-gene standard deviation computed during scaling (if `normalize_per_gene` is True). Otherwise, None.
    """

    # filter out genes with less than 1 count in the train set
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=min_counts)

    # select highly-variable genes
    if filter_top_genes:
        #filter_result = sc.pp.filter_genes_dispersion(adata.X, flavor='cell_ranger', n_top_genes=n_top_genes, log=False)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3')

        # subset the genes
        adata._inplace_subset_var(adata.var["highly_variable"])

        #adata.var_names = adata.var_names[adata.var["highly_variable"]].copy()
        adata.var_names = adata.var_names[adata.var["highly_variable"]]

    # normalize with total UMI count per cell
    if normalize_per_cell:
        sc.pp.normalize_total(adata)

    # log transform: adata.X = log(adata.X + 1)
    if log_transform:
        sc.pp.log1p(adata)
    
    # scale to unit variance and shift to zero mean. Return mean and std for normalizing eval set
    if normalize_per_gene:
        X, _mean, _std = sc.pp._scale.scale_array(adata.X, return_mean_std=True)
        adata.X = sp.csr_matrix(X) 
    else:
        _mean, _std = None, None

    return _mean, _std

def preprocess_eval(
        adata: ad.AnnData, 
        genes_to_keep: List[str], 
        log_transform: bool = True, 
        normalize_per_cell: bool = True, 
        normalize_per_gene: bool = True,
        mean_train: Optional[np.ndarray] = None, 
        std_train: Optional[np.ndarray] = None):
    """
    Preprocess an AnnData object for evaluation using the gene selection and scaling parameters from training.

    This function preprocesses an evaluation dataset by:
      - Subsetting the genes to those provided in `genes_to_keep` (which should match the training set).
      - Normalizing counts per cell.
      - Applying log1p transformation.
      - Scaling each gene using the provided training set mean and standard deviation.

    Args
        adata (ad.AnnData): Annotated data matrix to preprocess.  
        genes_to_keep (List[str]): List of gene names to retain; these should correspond to genes selected during training.  
        log_transform (bool, optional): If True, apply log1p transformation to the data. Defaults to True.  
        normalize_per_cell (bool, optional): If True, normalize counts per cell (total-count normalization). Defaults to True.  
        normalize_per_gene (bool, optional): If True, scale genes using the provided training statistics. Defaults to True.  
        mean_train (Optional[np.ndarray], optional): The per-gene mean from the training set. Must be provided if `normalize_per_gene` is True. Defaults to None.  
        std_train (Optional[np.ndarray], optional): The per-gene standard deviation from the training set. Must be provided if `normalize_per_gene` is True. Defaults to None.  

    Returns: None, None. this function modifies `adata` in place.
    """

    # assertions nad verifications
    assert np.all([gene in adata.var_names for gene in genes_to_keep]), "preprocess_eval: all genes_to_keep must be included in the data"

    if normalize_per_gene:
        assert mean_train is not None, "if normalize_per_gene, mean_train must be given"
        assert std_train is not None, "if normalize_per_gene, std_train must be given"

    # filter out the same genes removed during training
    adata._inplace_subset_var(genes_to_keep)

    # normalize per cell 
    if normalize_per_cell:
        sc.pp.normalize_total(adata)

    # log transform: adata.X = log(adata.X + 1)
    if log_transform:
        sc.pp.log1p(adata)

    # scale to unit variance and shift to zero mean
    if normalize_per_gene:         
        X = scale_array(adata.X, mean=mean_train, std=std_train)
        adata.X = sp.csr_matrix(X) 

    return None, None

def preprocess(
        adata: ad.AnnData, 
        is_train_set: bool, 
        normalize_per_cell: bool = True, 
        normalize_per_gene: bool = True,
        genes_to_keep: Optional[List[str]] = None, 
        log_transform: bool = True, 
        filter_min_counts: bool= True,
        min_counts: int = 1, 
        filter_top_genes: bool = True, 
        n_top_genes: int=1000,
        mean_train: Optional[np.ndarray] = None, 
        std_train: Optional[np.ndarray] = None
        ):
    """
    Unified preprocessing function for both training and evaluation datasets.

    This wrapper function delegates preprocessing to either the training or evaluation preprocessing routines
    based on the `is_train_set` flag.
    """

    if is_train_set:
        return preprocess_train(
            adata, 
            normalize_per_cell=normalize_per_cell, 
            normalize_per_gene=normalize_per_gene, 
            log_transform=log_transform, 
            filter_min_counts=filter_min_counts,
            min_counts=min_counts, 
            filter_top_genes=filter_top_genes,
            n_top_genes=n_top_genes
            )
    else:
        return preprocess_eval(
            adata, 
            genes_to_keep, 
            normalize_per_cell=normalize_per_cell, 
            normalize_per_gene=normalize_per_gene, 
            log_transform=log_transform, 
            mean_train=mean_train, 
            std_train=std_train)
