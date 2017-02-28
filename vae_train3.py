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
import sys
from vae import *
import pandas as pd

print("VAE")

train_labeled = pickle.load(open("train_labeled.p", "rb"))
train_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))
train_unlabeled.train_labels = torch.ones([47000])
train_unlabeled.k = 47000
val_data = pickle.load(open("validation.p", "rb"))
test_data = pickle.load(open("test.p","rb"))


train_loader_unlabeled = torch.utils.data.DataLoader(train_unlabeled, batch_size=64, shuffle=True)
train_loader_labeled = torch.utils.data.DataLoader(train_labeled, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)


if len(sys.argv ) > 1:

    unsup_cost = float(sys.argv[1])

    print(unsup_cost)

model = DCVAE2_Pool_Deeper(cost_rec = unsup_cost)


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

        loss = model.unsup_cost(data)

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

        loss = model.cost(data, target)

        loss.backward()

        opt.step()

        #avg_forward_loss += class_loss
        avg_loss += loss
        count += 1

    #print("averge loss: ", (avg_loss / count).data[0], " average classificition loss: ",
    #      (avg_forward_loss / count).data[0])
    print("averge loss: ", (avg_loss / count).data[0])

def test(record=False):
    test_loss = 0
    correct = 0

    right = []
    predicted = []
    image = []
    recon_image = []

    model.eval()

    z = 0

    for data, target in val_loader:

        data, target = Variable(data, volatile=True), Variable(target)

        y_hat = model.predict(data)

        loss = model.cost(data,target)

        test_loss += loss.data[0]

        pred = y_hat.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        if record == True:

            eq = pred.eq(target.data).cpu()

            for i in range(len(list(pred))):

                if eq[i] == 0:
                    right.append(target.data[i])
                    predicted.append(pred[i])
                    image.append(data.data[i])
                    #recon_image.append(X_hat.data[i])

        z += 1

    test_loss /= len(val_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    #return predicted, right, image, recon_image
    return 100. * correct / len(val_loader.dataset)


def predict_test_data():

    label_predict = np.array([])
    
    model.eval()

    for data, target in test_loader:
    
        data, target = Variable(data, volatile=True), Variable(target)
        
        output = model.predict(data)

        temp = output.data.max(1)[1].numpy().reshape(-1)
        
        label_predict = np.concatenate((label_predict, temp))


    predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
    predict_label.reset_index(inplace=True)
    predict_label.rename(columns={'index': 'ID'}, inplace=True)

    predict_label.to_csv('submission.csv', index=False)


num_right = []

for i in range(30):
    
    #train_unsup()

    train_sup()

    if i > 20:

        num_right.append(test())


print("AVERAGE TESTING ACCURACY:")
print(np.array(num_right).mean() )

predict_test_data()



