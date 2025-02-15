import numpy as np
import anndata as ad
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
import scanpy as sc
import os
import pickle
import sys

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import preprocess
from src.transformer_classifier import TransformerClassifier
from src.trainer import CellCLassificationTrainer
from src.dataset import CustomDataset

data_dir = "data"
models_dir = "models"
keep_ratio = 0.1

def train_and_validate(max_epochs):

    train_bs = 8
    valid_bs = 8
    num_workers = 4

    # read data
    data_pth = os.path.join(data_dir, 'cell_w_split.h5ad')

    # settings
    num_expression_levels = 10
    embed_dim = 64

    # initialize Comet experiment
    project_name = "cell-classif"
    workspace="gonzaq94"
    comet_api_key = os.getenv("COMET_API_KEY")
    experiment = Experiment(
        api_key=comet_api_key,
        project_name=project_name,
        workspace=workspace
        
    )
    comet_logger = CometLogger(api_key=comet_api_key, project_name=project_name, workspace=workspace)
    progress_bar = TQDMProgressBar()

    # initialize datasets
    args_normalization = {
        "log_transform": False, 
        "normalize_per_cell": False, 
        "normalize_per_gene": False, 
        "filter_top_genes": True,
        "filter_min_counts": True,
        "min_counts": 3, 
        "n_top_genes": 1000
    }   
    
    train_dataset = CustomDataset(data_pth, "train", args_normalization, max_value=260, num_bins=num_expression_levels, keep_ratio=keep_ratio)
    valid_dataset = CustomDataset(
        data_pth, 
        "valid", 
        args_normalization,
        genes_to_keep=train_dataset.genes_to_keep, 
        mean_per_gene=train_dataset.mean_per_gene, 
        std_per_gene=train_dataset.std_per_gene,
        max_value=260,
        num_bins=num_expression_levels,
        keep_ratio=keep_ratio
        )

    # initialize dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=num_workers)

    # Model initialization
    model = TransformerClassifier(
        num_expression_levels=num_expression_levels,
        embed_dim=embed_dim, 
        num_classes=train_dataset.num_classes, 
        num_layers=2, 
        fc_hidden_dim=embed_dim, 
        num_heads=8)
    
    trainer_module = CellCLassificationTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        base_lr=1e-3
    )

    trainer = Trainer(
        max_epochs=max_epochs, 
        val_check_interval=100,
        logger=comet_logger,
        callbacks=[progress_bar],
        )
    trainer.fit(trainer_module, train_dataloader, valid_dataloader)

    # Save the model
    torch.save(model.state_dict(), os.path.join(models_dir, 'transformer_classifier.pth'))

    # enc comet experiment
    experiment.end()

if __name__ == "__main__":

    train_and_validate(max_epochs=10)
