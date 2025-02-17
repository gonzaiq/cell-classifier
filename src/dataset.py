import torch
from torch.utils.data import Dataset
import anndata as ad
from typing import List, Optional
import numpy as np
from src.data_processing import preprocess
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import Dict

class CustomDataset(Dataset):
    def __init__(
            self, 
            data_pth: str, 
            split: str, 
            args_normalization: Dict,
            genes_to_keep: Optional[List[str]] = None, 
            mean_per_gene:  Optional[np.ndarray] = None, 
            std_per_gene:  Optional[np.ndarray] = None,
            max_value: int = 260,
            num_bins: int = 10,
            keep_ratio: float = 1.0
            ):

        assert split in ["train", "valid", "test"], f"Split {split} is not supported"

        if split == "train":
            assert genes_to_keep is None, "genes_to_keep should not be provided for train split"
            assert mean_per_gene is None, "mean_per_gene should not be provided for train split"
            assert std_per_gene is None, "std_per_gene should not be provided for train split"
        elif args_normalization["filter_top_genes"] or args_normalization["filter_min_counts"]:
            assert genes_to_keep is not None, "genes_to_keep should be provided for validation or test sets"
        elif args_normalization["normalize_per_gene"]:
            assert mean_per_gene is not None, "mean_per_gene should be provided for validation or test sets"
            assert std_per_gene is not None, "std_per_gene should be provided for validation or test sets"

        self.split = split

        # red data
        adata = ad.read_h5ad(data_pth)

        self.data = adata[adata.obs["split"] == split]

        if keep_ratio < 1.0:
            _, obs_kept = train_test_split(self.data.obs_names, test_size=keep_ratio, stratify=self.data.obs["class"])
            self.data = self.data[self.data.obs_names.isin(obs_kept)]

        # pre-process data
        if self.split == "train":
            self.mean_per_gene, self.std_per_gene = preprocess(self.data, is_train_set=True, **args_normalization)
            self.genes_to_keep = self.data.var_names
        else:
            preprocess(self.data, is_train_set=False, genes_to_keep=genes_to_keep, mean_train=mean_per_gene, std_train=std_per_gene, **args_normalization)
            self.genes_to_keep = genes_to_keep 
            self.mean_per_gene = mean_per_gene 
            self.std_per_gene = std_per_gene 

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        
        self.num_classes = self.data.uns["num_classes"]
        # map to bins
        self.num_bins = num_bins
        self.max_value = max_value


        # Calculate class frequencies
        self.labels = self.data.obs["encoded_class"].values
        class_counts = torch.bincount(torch.tensor(self.labels))
        self.class_weights = 1. / class_counts.float()
        self.sample_weights = self.class_weights[torch.tensor(self.labels)]  

    def __len__(self):
        """Returns the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a single sample from the dataset"""
        x = self.data[idx].X.toarray()
        y = self.data[idx].obs["encoded_class"].iloc[0]
        if self.transform:
            x = self.transform(x).squeeze()
    
        # map to bins
        x = (x / self.max_value * (self.num_bins-1) ).to(torch.int64)
        x = torch.clamp(x, min=0, max=self.num_bins - 1)  # Ensure valid range

        return x, y
