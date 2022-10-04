import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score
from dataset import New_Dataset
from tqdm import tqdm
import json
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dataset', type=bool, default=False,
                        help='Set False in case you already downloaded the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Test batch size.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path of the checkpoint file')
    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_args()
    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
    model.classifier= nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280,10)
    )
    
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])

    for param in model.parameters():
        param.requires_grad_(False)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28,28))
    ])
    
    dataset = torchvision.datasets.MNIST('./data/test',train=False, transform=transform, download=args.download_dataset)
    dataset = New_Dataset(dataset=dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    acc = prec = rec = 0
    for i,(features,labels) in tqdm(enumerate(dataloader)):
        output = model(features)
        _,predicted = torch.max(output, 1)
        acc += accuracy_score(labels,predicted)
        prec += precision_score(labels, predicted, average='micro')
    
    acc /= (i+1)
    prec /= (i+1)
    
    results={
        'checkpoint':args.checkpoint,
        'accuracy':acc,
        'precision':prec
    }

    print(results)
    
    path='./log/test/test_1'
    
    if not os.path.isdir(path):
        os.makedirs(path)
    
    with open(os.path.join(path,'results.json'),'w') as json_file:
        json.dump(results, json_file, indent=2)
    json_file.close()
    
    print('test ended with success!')