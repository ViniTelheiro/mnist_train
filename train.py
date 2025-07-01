import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import os
from torch.utils.data import DataLoader
from typing import Dict, Any
import yaml
import torch

from classifier import TrainerClassifier
from utils import plot_graph
from dataset import get_train_dataset
from model import CNN

torch.set_float32_matmul_precision("medium")


def trainer(config: Dict[str, Any]):
    train_dataset, val_dataset = get_train_dataset()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch-size"],
        shuffle=True,
        num_workers=config["n-workers"],
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch-size"],
        shuffle=False,
        num_workers=config["n-workers"],
    )

    checkpoint_dir = "./checkpoints"
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=config["checkpoint"]["monitor"],
        mode=config["checkpoint"]["mode"],
        filename="best",
        save_last=False,
    )
    callbacks.append(checkpoint_callback)

    if config["checkpoint"]["patience"] > 0:
        early_stop_callback = EarlyStopping(
            monitor=config["checkpoint"]["monitor"],
            patience=config["checkpoint"]["patience"],
            verbose=True,
            mode=config["checkpoint"]["mode"],
        )
        callbacks.append(early_stop_callback)

    model = CNN()

    train_classifier = TrainerClassifier(model=model, config=config)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        logger=False,
        callbacks=callbacks,
        enable_progress_bar=True,
        min_epochs=config["epochs"]["min"],
        max_epochs=config["epochs"]["max"],
    )

    trainer.fit(
        model=train_classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    save_dir = "./log"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    plot_graph(
        os.path.join(save_dir, "loss_graph.jpg"),
        y_label="Loss",
        x_label="Epoch",
        train_loss=train_classifier.losses["train"],
        val_loss=train_classifier.losses["val"],
    )

    plot_graph(
        os.path.join(save_dir, "acc_graph.jpg"),
        y_label="Accuracy",
        x_label="Epoch",
        val_acc=train_classifier.acc,
    )


if __name__ == "__main__":
    with open("./config.yaml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    trainer(config=config)
