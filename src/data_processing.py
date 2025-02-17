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
