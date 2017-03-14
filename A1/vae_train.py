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
import torchvision
import matplotlib.pyplot as plt

from vae import CVAE,CNN



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



vae = CVAE()

model = CNN()

#----------------------------------------
#Unsupevised VAE training
#----------------------------------------


opt = optim.Adam(vae.parameters(), lr=0.001)
mse = nn.MSELoss()
bce = nn.BCELoss()


def test_vae():
    vae.eval()
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = vae(data)

        recon_loss = mse(output, data)
        kl_loss = 0.5 * torch.mean(torch.exp(vae.log_sig) + vae.mu ** 2 - 1. - vae.log_sig)
        loss = .5 * recon_loss + .5 * kl_loss

        test_loss += loss.data[0]

    test_loss = test_loss
    test_loss /= len(val_loader)  # loss function already averages over batch size
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))# , avg_mse.data[0]/count ,avg_kl.data[0]/count)


def train_vae():


    vae.train()
    avg_loss = 0
    count = 0

    avg_mse = 0
    avg_kl = 0

    for batch_idx, (data, target) in enumerate(train_loader_unlabeled):
        data, target = Variable(data), Variable(target)
        opt.zero_grad()

        output = vae(data)

        recon_loss = mse(output, data)
        #recon_loss = bce(output, data)

        kl_loss = 0.5 * torch.mean(torch.exp(vae.log_sig) + vae.mu ** 2 - 1. - vae.log_sig)
        loss = .5 * recon_loss + .5 * kl_loss

        loss.backward()

        opt.step()

        avg_loss += loss
        avg_mse += recon_loss
        avg_kl += kl_loss
        count += 1

    print("averge loss: ", avg_loss.data[0] / count , avg_mse.data[0]/count ,avg_kl.data[0]/count)

for i in range(20):

    train_vae()
    test_vae()



#----------------------------------------
#Supevised CNNtraining
#----------------------------------------

model.conv1 = vae.conv1
model.conv2 = vae.conv2
#model.conv3 = vae.conv3

# CPU only training
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def train_cnn(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader_labeled):

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader_labeled.dataset),
                100. * batch_idx / len(train_loader_labeled), loss.data[0]))

def test_cnn(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in val_loader:

        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(val_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

for epoch in range(1, 40):
    train_cnn(epoch)
    test_cnn(epoch)
