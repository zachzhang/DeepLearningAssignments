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
test_data = pickle.load(open("test.p", "rb"))

train_loader_unlabeled = torch.utils.data.DataLoader(train_unlabeled, batch_size=64, shuffle=True)
train_loader_labeled = torch.utils.data.DataLoader(train_labeled, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

if len(sys.argv) > 1:
    unsup_cost = float(sys.argv[1])
    m_cost = float(sys.argv[1])

# model = DCVAE2_Pool_Deeper(cost_rec = unsup_cost)
#model = DCVAE2_Pool_Deeper_Ladder(1, .1)
model = DCVAE2_Pool_Deeper_Ladder(0., 0.)

opt = optim.Adam(model.parameters(), lr=0.001)
#opt = optim.SGD(model.parameters(), lr=0.001)


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

        # avg_forward_loss += class_loss
        avg_loss += loss
        count += 1

    # print("averge loss: ", (avg_loss / count).data[0], " average classificition loss: ",
    #      (avg_forward_loss / count).data[0])
    print("averge loss: ", (avg_loss / count).data[0])


def train_semi_sup():

    test_results = []
    train_results = []

    avg_sup_cost = 0
    avg_unsup_cost = 0

    count = 0

    sup_iter = iter(train_loader_labeled)

    for batch_idx, (data, target) in enumerate(train_loader_unlabeled):


        data_sup,target_sup = next(sup_iter,[None,None])

        if data_sup is not  torch.FloatTensor:

            sup_iter = iter(train_loader_labeled)

            data_sup, target_sup = next(sup_iter, [None, None])

            #train_results.append(train_acc())
            #test_results.append(test())

            count+=1
            
            if count % 5 == 0:
                train_results.append(train_acc())
                test_results.append(test())

            #if count ==150:


                #return train_results,test_results

        opt.zero_grad()

        sup_loss = model.cost(Variable(data_sup), Variable(target_sup))
        sup_loss.backward()

        opt.step()

        opt.zero_grad()

        unsup_loss = model.unsup_cost(Variable(data))
        unsup_loss.backward()

        opt.step()

        avg_sup_cost += sup_loss.data[0]
        avg_unsup_cost += unsup_loss.data[0]

    print("averge supervised loss: ", avg_sup_cost / len(train_loader_unlabeled) , " averge unsupervised loss: ", avg_unsup_cost / len(train_loader_unlabeled))

    return train_results,test_results

def train_acc():

    correct = 0
    test_loss = 0
    model.eval()

    for data, target in train_loader_labeled:

        data, target = Variable(data, volatile=True), Variable(target)

        y_hat = model.predict(data)

        loss = model.cost(data, target)

        test_loss += loss.data[0]

        pred = y_hat.data.max(1)[1]
                                   
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(val_loader)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(train_loader_labeled.dataset),
                            100. * correct / len(train_loader_labeled.dataset)))


    return 100. * correct / len(train_loader_labeled.dataset)

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

        loss = model.cost(data, target)

        test_loss += loss.data[0]

        pred = y_hat.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

        z += 1

    test_loss /= len(val_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    # return predicted, right, image, recon_image
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



#train_semi_sup()


test_results = []
train_results = []

for i in range(3):

    tr,ts = train_semi_sup()

    test_results += ts
    train_results += tr

    #train_unsup()
    #train_sup()

    #train_results.append(train_acc())
    #test_results.append(test())


pickle.dump(test_results, open('test_results.p','wb'))
pickle.dump(train_results, open('train_results.p','wb'))

predict_test_data()


