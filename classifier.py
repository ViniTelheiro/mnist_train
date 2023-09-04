from typing import Any, Optional
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


class TrainerClassifier(pl.LightningModule):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

        self.losses = {"train": [], "val": []}
        self.acc = []
        self.validation_step_outputs = {"loss": [], "acc": []}
        self.train_step_output = []
        self.test_output = 0

    def training_step(self, batch, batch_idx):
        features, labels = batch
        output = self.model(features)
        loss = F.cross_entropy(output, labels)
        self.train_step_output.append(loss.item())

        self.log("train_loss", loss, True)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        return optimizer

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        output = self.model(features)

        loss = F.cross_entropy(output, labels)
        self.validation_step_outputs["loss"].append(loss.item())

        predicted = torch.argmax(output, -1)
        acc = accuracy_score(labels, predicted)
        self.validation_step_outputs["acc"].append(acc)

        self.log("val_loss", loss, True)

        return loss

    def test_step(self, batch, batch_idx):
        features, labels = batch
        output = self.model(features)
        predicted = torch.argmax(output, -1)

        acc = accuracy_score(labels, predicted)
        cm = confusion_matrix(labels, predicted, labels=range(1, 10))
        self.test_output += cm

        self.log("accuracy", acc, True)

        return acc

    def on_train_epoch_end(self) -> None:
        avg_loss = np.mean(self.train_step_output.copy())
        self.losses["train"].append(avg_loss)
        self.train_step_output.clear()

    def on_validation_epoch_end(self) -> None:
        avg_loss = np.mean(self.validation_step_outputs["loss"].copy())
        avg_acc = np.mean(self.validation_step_outputs["acc"].copy())

        self.losses["val"].append(avg_loss)
        self.acc.append(avg_acc)

        self.validation_step_outputs["loss"].clear()
        self.validation_step_outputs["acc"].clear()
