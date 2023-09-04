from lightning.pytorch.utilities.types import STEP_OUTPUT
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import os
from torch.utils.data import DataLoader

from classifier import TrainerClassifier
from utils import plot_graph
from dataset import get_train_dataset
from model import CNN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="size of the trainning and validation batch.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Dataloader num_worker value"
    )

    return parser.parse_args()


def trainer(batch_size: int, num_workers: int):
    train_dataset, val_dataset = get_train_dataset()

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    checkpoint_dir = "./checkpoints"
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        mode="min",
        filename="best",
        save_last=False,
    )

    model = CNN()

    train_classifier = TrainerClassifier(model=model)

    trainer = pl.Trainer(
        logger=False,
        callbacks=[checkpoint_callback, early_stop_callback],
        enable_progress_bar=True,
        min_epochs=10,
        max_epochs=100,
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
    args = get_args()

    trainer(batch_size=args.batch_size, num_workers=args.num_workers)
