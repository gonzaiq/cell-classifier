import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from typing import Callable
import numpy as np

class CellCLassificationTrainer(pl.LightningModule):
    def __init__(self, model, criterion: Callable = F.cross_entropy, base_lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.base_lr = base_lr
        self.criterion = criterion

        # initialize metrics
        self.acc_train = []
        self.acc_val = []
        self.loss_val = []
        self.acc_test = []
        self.loss_test = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.acc_train.append(acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()

        self.acc_val.append(acc)
        self.loss_val.append(loss)

    def on_train_epoch_end(self):

        acc = np.mean(self.acc_train)
        self.log("train_acc", acc.item())

    def on_validation_epoch_end(self):

        acc = np.mean(self.acc_val)
        loss = np.mean(self.loss_val)
    
        self.log("val_loss", loss.item())
        self.log("val_acc", acc.item())

    def on_test_epoch_end(self):

        acc = torch.cat(self.acc_test).mean()
        loss = torch.cat(self.loss_test).mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.base_lr)
