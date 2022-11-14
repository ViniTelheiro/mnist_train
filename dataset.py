import torch

class CNN_Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset) -> None:
      super().__init__()
      labels= []
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
    lbl = self.labels[index]
    
    return img, lbl