import numpy as np
import anndata as ad
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import scanpy as sc
import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_processing import preprocess

data_dir = "data"
models_dir = "models"
keep_ratio = 1

def train_and_validate(max_iter, args_normalization):

    """
    Perform training and validation of a linear classifier (logistic regression).
    """

    # read data
    adata = ad.read_h5ad(os.path.join(data_dir, 'cell_w_split.h5ad'))

    train_set = adata[adata.obs["split"] == "train"]
    valid_set = adata[adata.obs["split"] == "valid"]

    if keep_ratio < 1.0:
        _, train_obs = train_test_split(train_set.obs_names, test_size=keep_ratio, stratify=train_set.obs["class"])
        train_set = adata[adata.obs_names.isin(train_obs)]

    # pre-process data
    print("number of genes: ", len(train_set.var_names))
    print("number of genes: ", train_set)
    mean_train, std_train = preprocess(train_set, is_train_set=True, **args_normalization)
    preprocess(valid_set, is_train_set=False, genes_to_keep=train_set.var_names, mean_train=mean_train, std_train=std_train, **args_normalization)

    X_train, y_train = train_set.X, train_set.obs["encoded_class"]
    X_valid, y_valid = valid_set.X, valid_set.obs["encoded_class"]

    # train the linear classifier
    classifier = LogisticRegression(max_iter=max_iter)
    print("Fitting classifier...")
    classifier.fit(X_train, y_train)

    # evaluate the classifier
    print("Evaluating classifier...")
    y_valid_pred = classifier.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_valid_pred)
    print(f"Accuracy val: {accuracy * 100:.2f}%")

    y_train_pred = classifier.predict(X_train)
    accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy train: {accuracy * 100:.2f}%")

    # save the model
    with open(os.path.join(models_dir, 'log_reg_classifier.pkl'), 'wb') as f:
        pickle.dump(classifier, f)

if __name__ == "__main__":

    args_normalization = {
        "log_transform": True, 
        "normalize_per_cell": True, 
        "normalize_per_gene": False, 
        "filter_top_genes": True,
        "filter_min_counts": True,
        "min_counts": 3, 
        "n_top_genes": 1000
    }   

    train_and_validate(max_iter=1000, args_normalization=args_normalization)