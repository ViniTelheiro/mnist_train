from email.policy import default
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import New_Dataset
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dataset', type=bool, default=False,
                                    help='set False if the dataset has already been downloaded.')
    parser.add_argument('--checkpoint', type=str, default='', 
                        help='if the train will occour by a checkpoint put the path of the checkpoint. else the train will beggin in the epoch 0.')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs that will be trained.')
    return parser.parse_args()


def train_from_zero(model, train_dataloader, val_dataloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    val_losses = []
    acc = []
    min_valid_loss = np.inf
    cont = 0
    for epoch in tqdm(range(1,epochs+1)):
        avg_acc = avg_train_loss = avg_val_loss = 0
        
        model.train()
        for i ,(features, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(features)
            
            train_loss = criterion(output, labels)
            train_loss.backward()
            optimizer.step()

            avg_train_loss += train_loss.item()
        avg_train_loss /= (i+1)
        train_losses.append(avg_train_loss)

        model.eval()
        for i, (features, labels) in enumerate(val_dataloader):
            output = model(features)
            val_loss = criterion(output, labels)

            avg_val_loss += val_loss.item()

            _, predicted = torch.max(output, 1)            
            avg_acc += accuracy_score(labels, predicted)
        
        avg_val_loss /= (i+1)
        avg_acc /= (i+1)
        
        val_losses.append(avg_val_loss)
        acc.append(avg_acc)

        print(f'Epoch: {epoch}\tTrain Loss: {avg_train_loss:.5f}\tValidation Loss: {avg_val_loss:.5f}\tAccuracy: {avg_acc:.5f}')
        
        cont += 1
        if min_valid_loss > avg_val_loss:
            cont = 0
            print(f'Validation loss decresed ({min_valid_loss} ---> {avg_val_loss}) saving the model at epoch {epoch}')
            min_valid_loss = avg_val_loss
            

            path = f'./log/train/train_1/epoch{epoch}'
            os.makedirs(path)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'acc': acc,
            }, os.path.join(path,'chekpoint.pth'))
            
        if cont==10:
            print(f"The validation loss didn't decreased in 10 epochs.\tquittin train at epoch {epoch}")
            break
        
    plt.plot(train_losses, 'b', label='Train Loss')
    plt.plot(val_losses, 'r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()
    plt.savefig(os.path.join(path,'loss_graph.jpg'))

    plt.clf()

    plt.plot(acc, 'b')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(path,'acc_graph.jpg'))
            

def train_from_checkpoint(model, train_dataloader, val_dataloader, epochs, checkpoint):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    checkpoint = torch.load(checkpoint)

    model.load_state_dict(checkpoint['model_state_dict'])
    criterion.load_state_dict(checkpoint['criterion_state_dict'])

    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    acc = checkpoint['acc']
    min_valid_loss = val_loss[-1]
    cont = 0
    for epoch in tqdm(range(checkpoint['epoch']+1,epochs+1)):
        avg_acc = avg_train_loss = avg_val_loss = 0
        
        model.train()
        for i ,(features, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(features)
            
            train_loss = criterion(output, labels)
            train_loss.backward()
            optimizer.step()

            avg_train_loss += train_loss.item()
        avg_train_los /= (i+1)
        train_losses.append(avg_train_loss)

        model.eval()
        for i, (features, labels) in enumerate(val_dataloader):
            output = model(features)
            val_loss = criterion(output, labels)

            avg_val_loss += val_loss.item()
            
            _, predicted = torch.max(output, 1)
            avg_acc += accuracy_score(labels, predicted)
        
        avg_val_loss /= (i+1)
        avg_acc /= (i+1)
        
        val_losses.append(avg_val_loss)
        acc.append(avg_acc)

        print(f'Epoch: {epoch}\tTrain Loss: {avg_train_loss:.5f}\tValidation Loss: {avg_val_loss:.5f}\tAccuracy: {avg_acc:.5f}')
        
        cont += 1
        if min_valid_loss > avg_val_loss:
            cont = 0
            print(f'Validation loss decresed ({min_valid_loss} ---> {avg_val_loss}) saving the model at epoch {epoch}')
            min_valid_loss = avg_val_loss
            

            path = f'./log/train/train_1/epoch{epoch}'
            os.makedirs(path)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'acc': acc,
            }, os.path.join(path,'chekpoint.pth'))
            
        if cont==10:
            print(f"The validation loss didn't decreased in 10 epochs.\tquittin train at epoch {epoch}")
            break
        
        plt.plot(train_losses, 'b', label='Train Loss')
        plt.plot(val_losses, 'r', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.legend()
        plt.savefig(os.path.join(path,'loss_graph.jpg'))

        plt.clf()

        plt.plot(acc, 'b')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig(os.path.join(path,'acc_graph.jpg'))


if __name__ == '__main__':
    args = get_args()

    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 10)
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28,28))
    ])

    dataset = torchvision.datasets.MNIST('./data/train', train=True,transform=transform, download=args.download_dataset)
    dataset = New_Dataset(dataset=dataset)

    val_len = int(10*len(dataset)/100)
    train_dataset, val_dataset = random_split(dataset,((np.shape(dataset)[0]-val_len, val_len)))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if args.checkpoint:
        
        if not os.path.isfile(args.checkpoint):
            raise Exception(f'the file {args.checkpoint} does not exist.')
        if '.pt' not in args.checkpoint:
            raise Exception(f"the file {args.checkpoint} isn't a checkpoint file.")
        
        train_from_checkpoint(model=model, train_dataloader=train_dataloader,val_dataloader=val_dataloader, epochs=args.epochs, checkpoint=args.checkpoint)
    
    else:
        train_from_zero(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=args.epochs) 
    
    print('model trained with success!')
