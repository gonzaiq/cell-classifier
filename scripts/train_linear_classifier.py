import numpy as np
import anndata as ad
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import os

data_dir = "data"
down_ratio = 0.1

def main():

    # read data
    adata = ad.read_h5ad(os.path.join(data_dir, 'cell_w_split.h5ad'))

    train_set = adata[adata.obs["split"] == "train"]
    valid_set = adata[adata.obs["split"] == "valid"]

    X_train, y_train = train_set.X, train_set.obs_names
    X_valid, y_valid = valid_set.X, valid_set.obs_names

    if down_ratio < 1.0:
        # downsample to 50% of the original dataset size
        X_train, y_train = resample(X_train, y_train, replace=False, n_samples=int(X_train.shape[0]*down_ratio))

    # train the linear classifier
    classifier = LogisticRegression()
    print("Fitting classifier...")
    classifier.fit(X_train, y_train)

    # make predictions on the validation set
    y_valid_pred = classifier.predict(X_valid)

    # evaluate the classifier
    print("Evaluating classifier...")
    accuracy = accuracy_score(y_valid, y_valid_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")



if __name__ == "__main__":
    main()