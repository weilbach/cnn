import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import png
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from colormap.colors import Color, hex2rgb
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import torch.nn.functional as F

from dataset import FacadeDataset

N_CLASS=5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_class = N_CLASS
        self.layers = nn.Sequential(
            nn.Conv2d(3, self.n_class, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 10, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 5)
        )
        self.conv1 = nn.Conv2d(3, 64, 3, padding=0)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=0)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=0)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=0)
        self.conv5 = nn.Conv2d(128, 64, 3, padding=0)
        self.conv6 = nn.Conv2d(64, 5, 3, padding=0)

        self.fullback = nn.Conv2d(128, 5, 1)


        self.pool = nn.MaxPool2d(kernel_size=2, stride=(2,2), padding=0,return_indices=True)
        self.up2 = nn.MaxUnpool2d(2, stride=2)

        self.trans1 = nn.ConvTranspose2d(5, 5, 4, stride=2, bias=False)
        self.trans2 = nn.ConvTranspose2d(5, 5, 32, stride=16, bias=False)
        #this didn't work lol
        self.drop = nn.Dropout2d()

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x, indices = self.pool(x)
        x = self.up2(x, indices)
       
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x, indices = self.pool(x)
        x = self.up2(x, indices)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        return x


def save_label(label, path):
    '''
    Function for ploting labels.
    '''
    colormap = [
        '#000000',
        '#0080FF',
        '#80FF80',
        '#FF8000',
        '#FF0000',
    ]
    assert(np.max(label)<len(colormap))
    colors = [hex2rgb(color, normalise=False) for color in colormap]
    w = png.Writer(label.shape[1], label.shape[0], palette=colors, bitdepth=4)
    with open(path, 'wb') as f:
        w.write(f, label)

def train(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    net = net.train()
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        # output = output.float()
        print('inside train')
        print(output.shape)
        print(labels.shape)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
    end = time.time()
    print('[epoch %d] loss: %.3f elapsed time %.3f' %
          (epoch, running_loss, end-start))

def test(testloader, net, criterion, device):
    '''
    Function for testing.
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = criterion(output, labels)
            losses += loss.item()
            cnt += 1
    print(losses / cnt)
    return (losses/cnt)


def cal_AP(testloader, net, criterion, device):
    '''
    Calculate Average Precision
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        preds = [[] for _ in range(5)]
        heatmaps = [[] for _ in range(5)]
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images).cpu().numpy()
            for c in range(5):
                preds[c].append(output[:, c].reshape(-1))
                heatmaps[c].append(labels[:, c].cpu().numpy().reshape(-1))

        aps = []
        for c in range(5):
            preds[c] = np.concatenate(preds[c])
            heatmaps[c] = np.concatenate(heatmaps[c])
            if heatmaps[c].max() == 0:
                ap = float('nan')
            else:
                ap = ap_score(heatmaps[c], preds[c])
                aps.append(ap)
            print("AP = {}".format(ap))
        print('average precision')
        print(np.mean(aps))

    # print(losses / cnt)
    return None


def get_result(testloader, net, device, folder='output_train'):
    result = []
    cnt = 1
    with torch.no_grad():
        net = net.eval()
        cnt = 0
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)[0].cpu().numpy()
            c, h, w = output.shape
            print(output.shape)
            assert(c == N_CLASS)
            y = np.zeros((h,w)).astype('uint8')
            for i in range(N_CLASS):
                mask = output[i]>0.5
                y[mask] = i
            gt = labels.cpu().data.numpy().squeeze(0).astype('uint8')
            save_label(y, './{}/y{}.png'.format(folder, cnt))
            save_label(gt, './{}/gt{}.png'.format(folder, cnt))
            plt.imsave(
                './{}/x{}.png'.format(folder, cnt),
                ((images[0].cpu().data.numpy()+1)*128).astype(np.uint8).transpose(1,2,0))

            cnt += 1

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # TODO change data_range to include all train/evaluation/test data.
    # TODO adjust batch_size.
    data = FacadeDataset(flag='train')
    print(data.__len__)
    train_data = FacadeDataset(flag='train', data_range=(0,20), onehot=False)
    train_loader = DataLoader(train_data, batch_size=20)
    test_data = FacadeDataset(flag='test_dev', data_range=(0,114), onehot=False)
    test_loader = DataLoader(test_data, batch_size=19)
    ap_data = FacadeDataset(flag='test_dev', data_range=(0,114), onehot=True)
    ap_loader = DataLoader(ap_data, batch_size=19)

    name = 'starter_net'
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)

    print('\nStart training')
    for epoch in range(10): #TODO decide epochs
        print('-----------------Epoch = %d-----------------' % (epoch+1))
        train(train_loader, net, criterion, optimizer, device, epoch+1)
        # TODO create your evaluation set, load the evaluation set and test on evaluation set
        evaluation_loader = train_loader
        test(evaluation_loader, net, criterion, device)

    print('\nFinished Training, Testing on test set')
    test(test_loader, net, criterion, device)
    print('\nGenerating Unlabeled Result')
    result = get_result(test_loader, net, device, folder='output_test')

    torch.save(net.state_dict(), './models/model_{}.pth'.format(name))

    cal_AP(ap_loader, net, criterion, device)

if __name__ == "__main__":
    main()
