import argparse
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import yaml

from utils import get_confusion_matrix
from classifier import TrainerClassifier
from dataset import get_test_dataset
from model import CNN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        type=str,
        required=True,
        help="Set the checkpoint path to test the model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    with open("./config.yaml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.SafeLoader)

    dataset = get_test_dataset()
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch-size"],
        shuffle=False,
        num_workers=config["n-workers"],
    )

    model = TrainerClassifier(model=CNN(), config=config)

    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
    )

    acc = trainer.test(model=model, dataloaders=dataloader, ckpt_path=args.checkpoint)
    get_confusion_matrix(model.test_output, range(1, 10), "./log/confusion_matrix.jpeg")
