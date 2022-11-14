import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from dataset import CNN_Dataset
from model import CNN
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='size of the trainning and validation batch.')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((28,28)),
    torchvision.transforms.ToTensor()
])

    download = False
    if not os.path.isdir('./data'):
        os.makedirs('./data')
        download = True

    dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms,download=download)

    train_dataset, val_dataset = train_test_split(dataset, test_size=.1, shuffle=True)

    train_dataset = CNN_Dataset(train_dataset)
    val_dataset = CNN_Dataset(val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = CNN()
    
    cont = 1
    path = f'./log/train/train{cont}'
    while os.path.isdir(path):
        cont+=1
        path = f'./log/train/train{cont}'
    os.makedirs(path)

    print(path)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=.9)
    train_losses = []
    val_losses = []
    acc = []
    min_valid_loss = np.inf
    cont = 0
    epochs=100

    for epoch in tqdm(range(1,epochs+1), total=epochs):
        avg_train_loss = avg_val_loss = avg_acc = 0
        model.train()
        for i,(features, labels) in enumerate(train_dataloader):
        

            output = model(features)
            
            train_loss = criterion(output, labels)
            avg_train_loss += train_loss.item()
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        avg_train_loss /= (i+1)
        train_losses.append(avg_train_loss)

        model.eval()
        for i,(features, labels) in enumerate(train_dataloader):

            output = model(features)

            val_loss = criterion(output, labels)
            avg_val_loss += val_loss.item()
            _, predicted = torch.max(output, 1)
            avg_acc += accuracy_score(labels.cpu(), predicted.cpu())

        avg_val_loss /= (i+1)
        avg_acc /=(i+1)
            
        val_losses.append(avg_val_loss)
        acc.append(avg_acc)
        cont+=1
        tqdm.write(f'epoch:{epoch}\t Train_loss:{avg_train_loss}\t Validation_Loss: {avg_val_loss}\t Acc: {avg_acc}')
        if avg_val_loss < min_valid_loss:
            tqdm.write(f'Validation loss decreased: {min_valid_loss} ---> {avg_val_loss}\t Saving model at epoch {epoch} ')
            min_valid_loss = avg_val_loss
            cont = 0
            torch.save({'model_state_dict':model.state_dict()}, os.path.join(path,'best_model.pth'))
        if cont == 10:
            print(f"The validation loss didn't decreased in 10 epochs. Quitting the model at epoch {epoch}")
            break
    

    

    plt.plot(train_losses, 'b', label='Train')
    plt.plot(val_losses, 'r', label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(path, 'loss_graph.jpg'))
    plt.clf()

    plt.plot(acc, 'b')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(path,'acc.jpg'))