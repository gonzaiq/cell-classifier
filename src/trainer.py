import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from typing import Callable
import numpy as np
from sklearn.metrics import confusion_matrix

class CellCLassificationTrainer(pl.LightningModule):
    def __init__(self, model, criterion: Callable = F.cross_entropy, base_lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.base_lr = base_lr
        self.criterion = criterion

        # initialize metrics
        self.init_train_metrics()
        self.init_validation_metrics()
        self.init_test_metrics()

    def init_train_metrics(self):

        # accuracy
        self.acc_train = []

        # confusion matrix
        self.train_y = []
        self.train_y_pred = []

    def init_validation_metrics(self):

        # loss
        self.loss_val = []

        # accuracy
        self.acc_val = []

        # confusion matrix
        self.val_y = []
        self.val_y_pred = []

    def init_test_metrics(self):

        # loss
        self.loss_test = []

        # accuracy
        self.acc_test = []

        # confusion matrix
        self.test_y = []
        self.test_y_pred = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)

        # loss
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)

        # prediction
        y_hat = y_hat.argmax(dim=1)
    
        self.train_y.append(y.cpu().detach())
        self.train_y_pred.append(y_hat.cpu().detach())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # loss
        loss = self.criterion(y_hat, y)
        self.loss_val.append(loss)

        # prediction
        y_hat = y_hat.argmax(dim=1)

        self.val_y.append(y.cpu().detach())
        self.val_y_pred.append(y_hat.cpu().detach())

    def on_train_epoch_end(self):
        
        y_pred = torch.cat(self.train_y_pred)
        y = torch.cat(self.train_y)

        acc = (y_pred == y).float().mean()    
        self.log("train_acc", acc.item())

        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred).tolist()
        self.logger.experiment.log_confusion_matrix(matrix=cm, step=self.current_epoch, title="train_cm")

        # reset metrics
        self.init_train_metrics()

    def on_validation_epoch_end(self):

        y_pred = torch.cat(self.val_y_pred)
        y = torch.cat(self.val_y)

        acc = (y_pred == y).float().mean()    
        loss = np.mean(self.loss_val)
    
        self.log("val_loss", loss.item())
        self.log("val_acc", acc.item())

        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred).tolist()
        self.logger.experiment.log_confusion_matrix(matrix=cm, step=self.current_epoch, title="val_cm")

        # reset metrics
        self.init_validation_metrics()

    def on_test_epoch_end(self):

        acc = (self.test_y_pred == self.test_y).mean()    
        loss = torch.cat(self.loss_test).mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)
    
        # Calculate confusion matrix
        cm = confusion_matrix(self.test_y, self.test_y_pred).tolist()
        self.logger.experiment.log_confusion_matrix(matrix=cm, step=self.current_epoch, title="test_cm")

        # reset metrics
        self.init_test_metrics()

    def test_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)

        # loss
        loss = self.criterion(y_hat, y)
        self.loss_test.append(loss)

        # acc
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.acc_test.append(acc)

        # confusion matrix
        self.test_y.extend(y.cpu().numpy().tolist())
        self.test_y_pred.extend(y_hat.cpu().numpy().tolist())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.base_lr)
