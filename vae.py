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



class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5 ,stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5 ,stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)

        self.flat_dim = [32 ,4 ,4]
        self.flat_dim = [64 ,2 ,2]

        self.h = self.flat_dim[0 ] *self.flat_dim[1 ] *self.flat_dim[2]

        self.latent_vars = 200

        self.fc1 = nn.Linear(self.h, self. latent_vars *2)

        self.fc2 = nn.Linear(self.latent_vars, self.h)

        self.dconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3)
        self.dconv2 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2 ,output_padding=1)
        self.dconv3 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2 ,output_padding=1)


    def encode(self ,x):


        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, x.size()[1] *x.size()[2] * x.size()[3])
        z = self.fc1(x)

        self.mu = z[:, 0:self.latent_vars]
        self.log_sig = z[:, self.latent_vars:]

        # sample
        eps = Variable(torch.randn(self.log_sig.size()))

        return self.mu + torch.exp(self.log_sig / 2) * eps

    def decode(self, z):
        x = F.relu(self.fc2(z))

        x = x.view(-1, self.flat_dim[0], self.flat_dim[1], self.flat_dim[2])

        x = F.relu(self.dconv1(x))

        x = F.relu(self.dconv2(x))

        x = self.dconv3(x)

        x = F.sigmoid(x)
        return (x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)

        return (x_hat)

    def sample(self, n):
        z = Variable(torch.randn((n, self.latent_vars)))

        return (self.decode(z))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=5, stride=2)

        print('here')
        self.flat_dim = [32, 4, 4]

        self.conv2_drop = nn.Dropout2d()
        self.conv3_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(self.flat_dim[0] * self.flat_dim[1] * self.flat_dim[2], 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))
        x = self.conv2_drop(x)

        # x = F.relu(self.conv3(x))
        # x = self.conv3_drop(x)

        x = x.view(-1, self.flat_dim[0] * self.flat_dim[1] * self.flat_dim[2])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


