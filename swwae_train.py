from __future__ import print_function
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pickle
from swwae import *

model = DSWWAE()


train_labeled = pickle.load(open("train_labeled.p", "rb" ))
train_unlabeled = pickle.load(open("train_unlabeled.p", "rb" ))
train_unlabeled.train_labels = torch.ones([47000])
train_unlabeled.k = 47000
val_data = pickle.load(open("validation.p","rb"))
#test_data = pickle.load(open("test.p","rb"))


train_loader_unlabeled = torch.utils.data.DataLoader(train_unlabeled, batch_size=64, shuffle=True)
train_loader_labeled = torch.utils.data.DataLoader(train_labeled, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)



opt = optim.Adam(model.parameters(), lr=0.001)

nll = torch.nn.NLLLoss()
mse = torch.nn.MSELoss()


def train_unsup():
    avg_loss = 0
    count = 0
    avg_forward_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader_unlabeled):

        data, target = Variable(data), Variable(target)

        opt.zero_grad()

        X_hat, y_hat, h_d, h_e = model(data)

        loss = mse(X_hat, data)

        for _h_d, _h_e in zip(h_d, h_e):
            loss += ((_h_d - _h_e) ** 2).mean()

        loss.backward()

        opt.step()

        avg_loss += loss
        count += 1

    print("averge loss: ", (avg_loss / count).data[0])


def train_sup():
    avg_loss = 0
    count = 0
    avg_forward_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader_labeled):

        data, target = Variable(data), Variable(target)

        opt.zero_grad()

        X_hat, y_hat, h_d, h_e = model(data)

        class_loss = nll(y_hat, target)
        loss = mse(X_hat, data) + class_loss

        for _h_d, _h_e in zip(h_d, h_e):
            loss += ((_h_d - _h_e) ** 2).mean()

        loss.backward()

        opt.step()

        avg_forward_loss += class_loss
        avg_loss += loss
        count += 1

    print("averge loss: ", (avg_loss / count).data[0], " average classificition loss: ",
          (avg_forward_loss / count).data[0])


def test():
    test_loss = 0
    correct = 0

    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)

        X_hat, y_hat, _, _ = model(data)

        loss = nll(y_hat, target)

        test_loss += loss.data[0]

        pred = y_hat.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(val_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


for i in range(20):

    #train_unsup()    
    train_sup()
    test()
