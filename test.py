import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CNN_Dataset
from model import CNN
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import json
import os
import cv2
import argparse
import itertools
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='path of the checkpoint file.')
    
    return parser.parse_args()


def get_confusion_matrix(cm, classes, path:str, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    ticks_mark = np.arange(len(classes))
    plt.xticks(ticks_mark, classes, rotation=45)
    plt.yticks(ticks_mark, classes)
    
    thresh = cm.max()/2

    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], 'd'),
                 horizontalalignment='center',
                  color='white' if cm[i,j]>thresh else 'black') 
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(path)


if __name__ == '__main__':
    args = get_args()
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor()
    ])
    
    download = False if os.path.isdir('./data') else True
    
    dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transform, download=download)
    dataset = CNN_Dataset(dataset)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    model = CNN()
    model.load_state_dict(torch.load(args.checkpoint)['model_state_dict'])
    avg_acc = 0
    pred = []
    lbl = []
    for i,(features, labels) in tqdm(enumerate(dataloader)):
        output = model(features)
        _, predicted = torch.max(output, 1)
        avg_acc += accuracy_score(predicted, labels)
        for idx in range(len(predicted)):
            pred.append(predicted[idx])
            lbl.append(labels[idx])



    cm = confusion_matrix(lbl, pred)

    path = os.path.join(os.path.split(args.checkpoint)[0],'test')
    if not os.path.isdir(path):
        os.makedirs(path)

    avg_acc /= (i+1)
    results = {'checkpoint': args.checkpoint, 'acc': avg_acc}
    with open(os.path.join(path, 'results.json'), 'w') as json_file:
        json.dump(results, json_file, indent=2)
    get_confusion_matrix(cm, range(10),os.path.join(path, 'confusion_matrix.jpg'))
    print('Model tested with success!')

    