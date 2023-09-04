import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import os
from sklearn.model_selection import train_test_split


class CNN_Dataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        labels = []
        features = []

        for i in range(0, len(dataset)):
            features.append(dataset[i][0])
            labels.append((dataset[i][1]))

        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img = self.features[index]
        label = self.labels[index]

        return img, label


def get_train_dataset() -> list:
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])

    download = False
    if not os.path.isdir("./data"):
        os.makedirs("./data")
        download = True

    dataset = MNIST(root="./data", train=True, transform=transform, download=download)


    train_dataset, val_dataset = train_test_split(
        dataset, test_size=0.1, shuffle=True
    )
    train_dataset = CNN_Dataset(train_dataset)
    val_dataset = CNN_Dataset(val_dataset)
    return [train_dataset, val_dataset]

def get_test_dataset() -> CNN_Dataset:
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])

    download = False
    if not os.path.isdir("./data"):
        os.makedirs("./data")
        download = True

    dataset = MNIST(root="./data", train=False, transform=transform, download=download)
    dataset = CNN_Dataset(dataset)

    return dataset
