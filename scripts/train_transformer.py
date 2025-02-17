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
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from torch.utils.data import DataLoader, WeightedRandomSampler
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
keep_ratio = 1.0

def train_and_validate(max_epochs):

    train_bs = 256 # 256
    valid_bs = 128
    num_workers = 8

    # read data
    data_pth = os.path.join(data_dir, 'cell_w_split.h5ad')

    # settings
    num_expression_levels = 260 #20 #10
    embed_dim = 512 # 200 64
    num_layers = 1 # 2
    base_lr = 1e-3
    weighted_loss = False
    use_custom_attention = True
    num_heads = 8
    fc_hidden_dim = embed_dim * 4

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
        "log_transform": True, 
        "normalize_per_cell": False, 
        "normalize_per_gene": False, 
        "filter_top_genes": True,
        "filter_min_counts": False,
        "min_counts": 3, 
        "n_top_genes": 1000
    }   

    max_value = 6 if args_normalization["log_transform"] else 260
    
    train_dataset = CustomDataset(
        data_pth, 
        "train", 
        args_normalization, 
        max_value=max_value, 
        num_bins=num_expression_levels, 
        keep_ratio=keep_ratio, 
        use_precomputed_stats_and_features=False
        )
        
    valid_dataset = CustomDataset(
        data_pth, 
        "valid", 
        args_normalization,
        genes_to_keep=train_dataset.genes_to_keep, 
        mean_per_gene=train_dataset.mean_per_gene, 
        std_per_gene=train_dataset.std_per_gene,
        max_value=max_value,
        num_bins=num_expression_levels,
        keep_ratio=keep_ratio,
        use_precomputed_stats_and_features=True
        )

    # create balanced dataloader for training
    sampler = WeightedRandomSampler(weights=train_dataset.sample_weights, num_samples=len(train_dataset.sample_weights), replacement=True)

    # initialize dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_bs, sampler=sampler, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=num_workers)

    # Model initialization
    model = TransformerClassifier(
        num_expression_levels=num_expression_levels,
        embed_dim=embed_dim, 
        num_classes=train_dataset.num_classes, 
        num_layers=num_layers, 
        fc_hidden_dim=fc_hidden_dim, 
        num_heads=num_heads,
        use_custom_attention=use_custom_attention
        )
    
    print(model)

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1_macro",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints/",
        filename=f"best_model_embed_dim_{embed_dim}_num_layers_{num_layers}_base_lr_{base_lr}_num_expression_levels_{num_expression_levels}_bs_{train_bs}_valid_init"
    )

    # Trainer initialization
    
    loss_args = {}
    if weighted_loss: 
        loss_args["weight"] = train_dataset.class_weights

    trainer_module = CellCLassificationTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(**loss_args),
        base_lr=base_lr
    )

    trainer = Trainer(
        max_epochs=max_epochs, 
        val_check_interval=1.0,
        logger=comet_logger,
        callbacks=[progress_bar, checkpoint_callback],
        )
    trainer.fit(trainer_module, train_dataloader, valid_dataloader)

    # Save the model
    torch.save(model.state_dict(), os.path.join(models_dir, 'transformer_classifier.pth'))

    # enc comet experiment
    experiment.end()

if __name__ == "__main__":

    train_and_validate(max_epochs=1000)
