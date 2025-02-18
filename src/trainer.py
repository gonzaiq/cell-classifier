import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from typing import Callable
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from typing import Dict, Any
from pytorch_lightning.loggers import CometLogger

class CellCLassificationTrainer(pl.LightningModule):
    """
    A PyTorch Lightning module for training, validating, and testing a cell classification model.
    
    Attributes:
        model (nn.Module): The neural network model to be trained.
        criterion (Callable): The loss function used for training.
        base_lr (float): The learning rate for the optimizer.
        weight_decay (float): weight decay parameter.
    """

    def __init__(self, model: nn.Module, criterion: Callable = F.cross_entropy, base_lr: float = 1e-3, weight_decay: float = 0.0):

        super().__init__()
        self.model = model
        self.base_lr = base_lr
        self.criterion = criterion
        self.weight_decay = weight_decay

        # initialize metrics
        self.init_train_metrics()
        self.init_validation_metrics()
        self.init_test_metrics()

    def init_train_metrics(self):
        """Initializes training metrics such as accuracy and confusion matrix components."""

        # accuracy
        self.acc_train = []

        # accumulate for metrics
        self.train_y = []
        self.train_y_pred = []

    def init_validation_metrics(self):
        """Initializes validation metrics including loss, accuracy, and confusion matrix components."""

        # loss
        self.loss_val = []

        # accumulate for metrics
        self.val_y = []
        self.val_y_pred = []

    def init_test_metrics(self):
        """Initializes test metrics including loss, accuracy, and confusion matrix components."""

        # loss
        self.loss_test = []

        # accumulate for metrics
        self.test_y = []
        self.test_y_pred = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.
        
        Args:
            x: The input tensor.
        
        Returns:
            The model output.
        """
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step.
        
        Args:
            batch: A batch of training data.
            batch_idx: The index of the batch.
        
        Returns:
            The computed loss.
        """
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

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Performs a single validation step.
        
        Args:
            batch: A batch of validation data.
            batch_idx: The index of the batch.
        """
        x, y = batch
        y_hat = self(x)

        # loss
        loss = self.criterion(y_hat, y)
        self.loss_val.append(loss.detach().cpu())

        # prediction
        y_hat = y_hat.argmax(dim=1)

        self.val_y.append(y.cpu().detach())
        self.val_y_pred.append(y_hat.cpu().detach())
    
    def get_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, Any]:
        """
        Computes accuracy, F1 scores, and the confusion matrix.
        
        Args:
            y_pred: Predicted labels.
            y_true: Ground truth labels.
        
        Returns:
            A dictionary containing accuracy, F1-macro, F1-weighted, and confusion matrix.
        """

        # accuracy
        acc = (y_pred == y_true).float().mean().item()

        # f1 scores
        f1_macro = f1_score(y_true, y_pred, average="macro")
        f1_weighted = f1_score(y_true, y_pred, average="weighted")

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred).tolist()

        out = {
            "acc": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "cm": cm
        }

        return out

    def on_train_epoch_end(self):
        """Computes and logs training metrics at the end of an epoch."""
        
        y_pred = torch.cat(self.train_y_pred)
        y_true = torch.cat(self.train_y)

        metrics = self.get_metrics(y_pred, y_true)

        for name, value in metrics.items():
            if name == "cm":
                pass
                # TODO: differentiate train and valid cm in comet interface (title is not enough)
                self.train_cm = value
            else:
                self.log(f"train_{name}", value)

        # reset metrics
        self.init_train_metrics()

    def on_validation_epoch_end(self):
        """Computes and logs validation metrics at the end of an epoch."""

        y_pred = torch.cat(self.val_y_pred)
        y_true = torch.cat(self.val_y)

        loss = np.mean(self.loss_val)
        self.log(f"val_loss", loss)

        metrics = self.get_metrics(y_pred, y_true)

        for name, value in metrics.items():
            if name == "cm":
                if isinstance(self.logger, CometLogger):
                    self.logger.experiment.log_confusion_matrix(matrix=value, step=self.current_epoch, title="val_cm")
                self.val_cm = value
            else:
                self.log(f"val_{name}", value)

        # reset metrics
        self.init_validation_metrics()

    def on_test_epoch_end(self):
        """Computes and logs test metrics at the end of an epoch."""

        y_pred = torch.cat(self.test_y_pred)
        y_true = torch.cat(self.test_y)

        loss = np.mean(self.loss_test)
        self.log(f"test_loss", loss)

        metrics = self.get_metrics(y_pred, y_true)
        
        for name, value in metrics.items():
            if name == "cm":
                self.test_cm = value
            else:
                self.log(f"test_{name}", value)

        # reset metrics
        self.init_test_metrics()

        return metrics

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        """
        Performs a single test step.
        
        Args:
            batch: A batch of test data.
            batch_idx: The index of the batch.
        """
        x, y = batch
        y_hat = self(x)

        # loss
        loss = self.criterion(y_hat, y)
        self.loss_test.append(loss)
    
        # prediction
        y_hat = y_hat.argmax(dim=1)

        self.test_y.append(y.cpu().detach())
        self.test_y_pred.append(y_hat.cpu().detach())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer for training."""
        return torch.optim.AdamW(self.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)