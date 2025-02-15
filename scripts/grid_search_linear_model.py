from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import pickle
import anndata as ad
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from data_processing import preprocess

# Custom transformer to include the normalization parameters in the grid search
class PreprocessWithNormalization(BaseEstimator, TransformerMixin):
    def __init__(self, **args_normalization):
        self.args_normalization = args_normalization
        
    def fit(self, X, y=None):
        # Fit function, no transformation needed
        return self
    
    def transform(self, X):
        # Preprocess data with the provided normalization args
        adata = ad.AnnData(X)  # Convert the data to AnnData
        mean_train, std_train = preprocess(adata, is_develop_set=True, **self.args_normalization)
        preprocess(adata, is_develop_set=False, genes_to_keep=adata.var_names, mean_train=mean_train, std_train=std_train, **self.args_normalization)
        return adata.X

def grid_search_cv(param_grid, n_splits, max_iter, args_normalization, data_dir, models_dir, keep_ratio=1.0):
    # read data
    adata = ad.read_h5ad(os.path.join(data_dir, 'cell_w_split.h5ad'))
    develop_set = adata[adata.obs["split"].isin(["train", "valid"])]  # development subset
    
    X = develop_set.X
    y = develop_set.obs["class"].values
    
    # label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Create a pipeline with preprocessing and classifier
    preprocess_with_norm = PreprocessWithNormalization(**args_normalization)
    classifier = LogisticRegression(max_iter=max_iter)
    
    # Create the pipeline
    pipeline = Pipeline([
        ('preprocess', preprocess_with_norm),
        ('classifier', classifier)
    ])
    
    # Define the parameter grid for the classifier
    grid_search = GridSearchCV(pipeline, param_grid, cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), scoring='accuracy', n_jobs=-1)
    
    # Run grid search
    grid_search.fit(X, y_encoded)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%")
    
    # save the best model
    best_classifier = grid_search.best_estimator_
    with open(os.path.join(models_dir, 'log_reg_classifier_grid.pkl'), 'wb') as f:
        pickle.dump(best_classifier, f)
    print("Best model saved.")

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
    
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    }
    
    grid_search_cv(param_grid=param_grid, n_splits=5, max_iter=1000, args_normalization=args_normalization, data_dir="data", models_dir="models")
