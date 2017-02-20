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


class SWWAE(nn.Module):
    def __init__(self):
        super(SWWAE, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        self.flat_dim = [32, 4, 4]
        self.flat_dim = [64, 2, 2]

        self.h = self.flat_dim[0] * self.flat_dim[1] * self.flat_dim[2]

        self.latent_vars = 10

        self.fc1 = nn.Linear(self.h, self.latent_vars)

        self.fc2 = nn.Linear(self.latent_vars, self.h)

        self.dconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)
        self.dconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, output_padding=1)
        self.dconv3 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, output_padding=1)

        self.fc_out = nn.Linear(self.latent_vars, 10)
        self.fc4 = nn.Linear(10, self.latent_vars)

        self.bnf1 = nn.BatchNorm2d(64)
        self.bnf2 = nn.BatchNorm2d(64)
        self.bnf3 = nn.BatchNorm2d(64)

        self.bnb1 = nn.BatchNorm2d(64)
        self.bnb2 = nn.BatchNorm2d(64)
        self.bnb3 = nn.BatchNorm2d(64)

    def encode(self, x):
        h = []

        h.append(F.relu(self.bnf1(self.conv1(x))))

        h.append(F.relu(self.bnf2(self.conv2(h[0]))))

        h.append(self.bnf3(F.relu(self.conv3(h[1]))))

        z = self.fc1(h[2].view(-1, h[-1].size()[1] * h[-1].size()[2] * h[-1].size()[3]))

        '''
        h = []

        h.append(F.relu(self.conv1(x)))

        h.append(F.relu(self.conv2(h[0])))

        h.append(F.relu(self.conv3(h[1])))

        z = self.fc1(h[2].view(-1, h[-1].size()[1] *h[-1].size()[2] *h[-1].size()[3]))
        '''

        return h, z

    def decode(self, z):
        h = []

        h.append(F.relu(self.bnb1(self.fc2(z).view(-1, self.flat_dim[0], self.flat_dim[1], self.flat_dim[2]))))

        h.append(F.relu(self.bnb2(self.dconv1(h[-1]))))

        h.append(F.relu(self.bnb3(self.dconv2(h[-1]))))

        x = self.dconv3(h[-1])

        '''
        h = []

        h.append(F.relu(self.fc2(z)).view(-1,self.flat_dim[0], self.flat_dim[1], self.flat_dim[2]))

        h.append( F.relu(self.dconv1(h[-1]))  )

        h.append(F.relu(self.dconv2(h[-1])))

        x = self.dconv3(h[-1])
        '''

        h.reverse()

        return (h, x)

    def forward(self, x):
        h, z = self.encode(x)
        h_d, x_hat = self.decode(z)
        y_hat = F.log_softmax(z)

        return (x_hat, y_hat, h_d, h)

    def test(self, x):
        x = self.conv1(x)
        x = self.conv2(x)


class DSWWAE(nn.Module):
    def __init__(self):
        super(DSWWAE, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        self.flat_dim = [32, 4, 4]
        self.flat_dim = [64, 2, 2]

        self.h = self.flat_dim[0] * self.flat_dim[1] * self.flat_dim[2]

        self.latent_vars = 10

        self.fc1 = nn.Linear(self.h, self.latent_vars)

        self.fc2 = nn.Linear(self.latent_vars, self.h)

        self.dconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2)
        self.dconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, output_padding=1)
        self.dconv3 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, output_padding=1)

        self.fc_out = nn.Linear(self.latent_vars, 10)
        self.fc4 = nn.Linear(10, self.latent_vars)

        self.bnf1 = nn.BatchNorm2d(64)
        self.bnf2 = nn.BatchNorm2d(64)
        self.bnf3 = nn.BatchNorm2d(64)

        self.bnb1 = nn.BatchNorm2d(64)
        self.bnb2 = nn.BatchNorm2d(64)
        self.bnb3 = nn.BatchNorm2d(64)

        self.noise_std = .2

    def encode(self, x, noise=False):

        h = []

        if noise:
            x_in = x + Variable(self.noise_std * torch.randn(x.size()))
        else:
            x_in = x

        h.append(self.step(self.conv1, self.bnf1, x_in, noise=False))
        h.append(self.step(self.conv2, self.bnf2, h[-1], noise=False))
        h.append(self.step(self.conv3, self.bnf3, h[-1], noise=False))

        z = self.fc1(h[2].view(-1, h[-1].size()[1] * h[-1].size()[2] * h[-1].size()[3]))

        return h, z

    def step(self, module, bn, inputs, noise=False):

        z_pre = module(inputs)

        if noise:
            z_pre = z_pre + Variable(self.noise_std * torch.randn(z_pre.size()))

        h = F.relu(bn(z_pre))
        return (h)

    def decode(self, z):

        h = []

        h.append(F.relu(self.bnb1(self.fc2(z).view(-1, self.flat_dim[0], self.flat_dim[1], self.flat_dim[2]))))

        h.append(F.relu(self.bnb2(self.dconv1(h[-1]))))

        h.append(F.relu(self.bnb3(self.dconv2(h[-1]))))

        x = self.dconv3(h[-1])

        h.reverse()

        return (h, x)

    def predict(self, x):

        h, z = self.encode(x, noise=False)
        y_hat = F.log_softmax(z)

        return (y_hat)

    def forward(self, x):

        h_noise, z_noise = self.encode(x, noise=True)

        h, z = self.encode(x, noise=False)

        h_d, x_hat = self.decode(z_noise)

        y_hat = F.log_softmax(z_noise)

        return (x_hat, y_hat, h_d, h)


