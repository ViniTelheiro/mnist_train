import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class New_Dataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        
        features = []
        labels = []
        for idx in range(0, len(dataset)):
            features.append(dataset[idx][0])
            labels.append(dataset[idx][1])
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        img = self.features[index][:]
        img = np.reshape(img.numpy(), (28,28,1))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = torch.Tensor(np.reshape(img, (3,28,28)))
        return img, self.labels[index]
