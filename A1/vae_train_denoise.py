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

from vae import *


print("VAE")


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


model = DCVAE2_Pool()

opt = optim.Adam(model.parameters(), lr=0.001)

nll = torch.nn.NLLLoss()
mse = torch.nn.MSELoss()

C = 1

def train_unsup():
    avg_loss = 0
    count = 0

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader_unlabeled):

        data, target = Variable(data), Variable(target)

        opt.zero_grad()

        X_hat, mu,log_sig  = model(data)

        recon_loss = mse(X_hat, data)
        kl_loss = 0.5 * torch.mean(torch.exp(log_sig) + mu ** 2 - 1. - log_sig)

        loss = (recon_loss + kl_loss) * C

        loss.backward()

        opt.step()

        avg_loss += loss
        count += 1

    print("averge loss: ", (avg_loss / count).data[0])


def train_sup():


    avg_loss = 0
    count = 0
    avg_forward_loss = 0

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader_labeled):

        data, target = Variable(data), Variable(target)

        opt.zero_grad()

        X_hat, mu, log_sig  = model(data)

        recon_loss = mse(X_hat, data)
        kl_loss = 0.5 * torch.mean(torch.exp(log_sig) + mu ** 2 - 1. - log_sig)

        y_hat = F.log_softmax(mu)

        class_loss = nll(y_hat,target)

        loss = (recon_loss + kl_loss) * C +  class_loss

        loss.backward()

        opt.step()

        avg_forward_loss += class_loss
        avg_loss += loss
        count += 1

    print("averge loss: ", (avg_loss / count).data[0], " average classificition loss: ",
          (avg_forward_loss / count).data[0])


def test(record = False):

    test_loss = 0
    correct = 0

    right = []
    predicted = []
    image = []
    recon_image =[]

    model.eval()

    z = 0

    for data, target in val_loader:


        data, target = Variable(data, volatile=True), Variable(target)

        X_hat, mu, log_sig = model(data)

        y_hat = F.log_softmax(mu)

        loss = nll(y_hat, target)

        test_loss += loss.data[0]

        pred = y_hat.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        if record == True:

            #predicted = predicted + list(pred)
            #right = right + list(target.data)

            for i in range(len(list(pred))):

                if pred[i] != target.data[i]:
                    
                    right.append(target.data[i])
                    predicted.append(pred[i])
                    image.append(data.data[i])
                    recon_image.append(X_hat.data[i])

        z += 1

    test_loss /= len(val_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    
    return predicted,right,image,recon_image

for i in range(20):

    train_sup()


for i in range(20):
    
    train_sup()

    test()
   
'''
predicted,right,image,recon_image = test(True)


pickle.dump(predicted,open('predicted.p','wb'))
pickle.dump(right,open('right.p','wb'))
pickle.dump(image,open('X_test.p','wb'))
pickle.dump(recon_image,open('X_hat.p','wb'))
'''



