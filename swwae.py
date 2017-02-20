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


model = SWWAE()

a = model(Variable(torch.randn(64, 1, 28, 28)))[0]


model = SWWAE()


train_labeled = pickle.load(open("train_labeled.p", "rb" ))
train_unlabeled = pickle.load(open("train_unlabeled.p", "rb" ))
train_unlabeled.train_labels = torch.ones([57000])

train_unlabeled.k = 20000

train_loader_unlabeled = torch.utils.data.DataLoader(train_unlabeled, batch_size=32, shuffle=True)

train_loader_labeled = torch.utils.data.DataLoader(train_labeled, batch_size=32, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=32, shuffle=False)

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

        loss.backward(retain_variables=True)

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

        loss.backward(retain_variables=True)

        opt.step()

        avg_forward_loss += class_loss
        avg_loss += loss
        count += 1

    print("averge loss: ", (avg_loss / count).data[0], " average classificition loss: ",
          (avg_forward_loss / count).data[0])


def test():
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)

        X_hat, y_hat, _, _ = model(data)

        loss = nll(y_hat, target)

        test_loss += loss.data[0]

        pred = y_hat.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


eps = .0001
cont = True
prev_error = 10e10

while cont:

    train_unsup()    
    train_sup()
    test()

    #train_sup()

    test()
