"""
Different model structure to be shared in tests
"""


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



class NetSimple(nn.Module):
    def __init__(self, model_paras):
        super(NetSimple, self).__init__()
        self.p = model_paras.get('p',1.0)
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p = self.p)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
                
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p = self.p, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

class trainModel(object):
    def __init__(self, train_loader, test_loader, model, optimizer, n_iter, log_interval, flg_cuda = False):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.n_iter = n_iter
        self.log_interval = log_interval # list of two numbers: [log_per_n_epoch, log_per_n_batch]
        self.flg_cuda = flg_cuda
        
    def run(self):
        lsTrainAccuracy = []
        lsTestAccuracy = []
        for epoch in range(self.n_iter):
            self._train(epoch, lsTrainAccuracy)
            self._test(epoch, lsTestAccuracy)
        return self.model, lsTrainAccuracy, lsTestAccuracy

    def _train(self, epoch, lsTrainAccuracy):
        correct = 0
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.flg_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            correct += self._getAccuracy(output, target)
            
            """
            if (epoch == 0) | (epoch % self.log_interval[0] == 0):
                if batch_idx % self.log_interval[1] == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.data[0]))
            """
            
        trainAccuracy = 100. * correct / len(self.train_loader.dataset)
        lsTrainAccuracy.append(trainAccuracy)
        
        if (epoch == 0) | (epoch % self.log_interval[0] == 0):
            print('\nTrain Epoch: {} Last loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                epoch, loss.data[0], correct, len(self.train_loader.dataset),
                trainAccuracy))
            
    
    def _test(self, epoch, lsTestAccuracy):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            if self.flg_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target).data[0]
            correct += self._getAccuracy(output, target)
    
        testAccuracy = 100. * correct / len(self.test_loader.dataset)
        test_loss /= len(self.test_loader) # loss function already averages over batch size

        if (epoch == 0) | (epoch % self.log_interval[0] == 0):
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(self.test_loader.dataset),
                testAccuracy))
            lsTestAccuracy.append(testAccuracy)
    
    def _getAccuracy(self, output, target):
        pred = output.data.max(1)[1] # get the index of the max log-probability
        accuracy = pred.eq(target.data).cpu().sum()
        return accuracy


