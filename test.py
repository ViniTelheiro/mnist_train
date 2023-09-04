import argparse
from torch.utils.data import DataLoader
from utils import get_confusion_matrix

import lightning.pytorch as pl
from classifier import TrainerClassifier
from dataset import get_test_dataset
from model import CNN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", "-b", type=int, default=32, help="set the test batch size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="dataloader num_workers"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        required=True,
        help="Set the checkpoint path to test the model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    dataset = get_test_dataset()
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = TrainerClassifier(model=CNN())

    trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=True,
        min_epochs=10,
        max_epochs=100,
    )

    acc = trainer.test(model=model, dataloaders=dataloader, ckpt_path=args.checkpoint)
    get_confusion_matrix(model.test_output, range(1, 10), "./log/confusion_matrix.jpeg")
