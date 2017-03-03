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
from sub import subMNIST
from ladder import *
import pandas as pd

train_labeled = pickle.load(open("train_labeled.p", "rb"))
train_unlabeled = pickle.load(open("train_unlabeled.p", "rb"))
train_unlabeled.train_labels = torch.ones([47000]).long()
train_unlabeled.k = 47000
val_data = pickle.load(open("validation.p", "rb"))
test_data = pickle.load(open("test.p","rb"))


train_loader_unlabeled = torch.utils.data.DataLoader(train_unlabeled, batch_size=64, shuffle=True)
train_loader_labeled = torch.utils.data.DataLoader(train_labeled, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

model = LadderNet()

params = []

for i in range(len(model.bn_layers_forward)):
    params += list(model.bn_layers_forward[i].parameters())
    params += list(model.bn_layers_backward[i].parameters())

for i in range(len(model.modules)):
    params += list(model.modules[i].parameters())
    params += list(model.decoder_modules[i].parameters())

denoising_params = [item for sublist in model.denoising_params for item in sublist]

# params = list(model.parameters())  + model.beta + model.gamma + denoise_params

params += denoising_params

opt = optim.Adam(params, lr=0.004)

nll = torch.nn.NLLLoss()


def train_semi_sup():

    avg_sup_cost = 0
    avg_unsup_cost = 0

    sup_iter = iter(train_loader_labeled)

    for batch_idx, (data, target) in enumerate(train_loader_unlabeled):


        data_sup, target_sup = next(sup_iter, [None, None])

        if data_sup is not torch.FloatTensor:
            sup_iter = iter(train_loader_labeled)

            data_sup, target_sup = next(sup_iter, [None, None])

        data = data.view(data.size()[0], 28 * 28)
        data_sup = data_sup.view(data_sup.size()[0], 28 * 28)

        data,target = Variable(data) , Variable(target)
        data_sup , target_sup = Variable(data_sup) , Variable(target_sup)

        opt.zero_grad()

        z_hat_bn, z, y_hat = model.forward2(data_sup)

        C_denoise, C_forward = model.cost(z_hat_bn, z, y_hat, target_sup)

        sup_loss = C_denoise + C_forward

        sup_loss.backward()

        opt.step()

        opt.zero_grad()

        z_hat_bn, z, y_hat = model.forward2(data)
        C_denoise, _ = model.cost(z_hat_bn, z, y_hat, target)

        unsup_loss = C_denoise

        unsup_loss.backward()

        opt.step()

        avg_sup_cost += sup_loss.data[0]
        avg_unsup_cost += unsup_loss.data[0]

    print("averge supervised loss: ", avg_sup_cost / len(train_loader_unlabeled), " averge unsupervised loss: ",
          avg_unsup_cost / len(train_loader_unlabeled))


def train():

    avg_loss = 0
    avg_unsup_loss = 0
    count = 0
    avg_forward_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader_labeled):
    #for data, target in train_loader_labeled[0:10]:

        data = data.view(data.size()[0], 28 * 28)
        mask = Variable(target != -1)
        mask = mask.unsqueeze(1)

        data, target = Variable(data), Variable(target)

        opt.zero_grad()

        z_hat_bn, z, y_hat = model.forward2(data)
        C_denoise, C_forward = model.cost(z_hat_bn, z, y_hat, target, mask)

        loss = C_denoise + C_forward

        loss.backward(retain_variables=True)

        opt.step()

        avg_loss += loss
        avg_forward_loss += C_forward
        count += 1

    print("averge loss: ", (avg_loss / count).data[0], " average classificition loss: ",
          (avg_forward_loss / count).data[0])


def test():
    test_loss = 0
    correct = 0

    for data, target in val_loader:
        data = data.view(data.size()[0], 28 * 28)

        mask = Variable(target != -1)
        mask = mask.unsqueeze(1)

        data, target = Variable(data, volatile=True), Variable(target)

        y_hat = model.predict(data)

        loss = nll(y_hat, target)

        test_loss += loss.data[0]

        pred = y_hat.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def predict_test_data():

    label_predict = np.array([])

    model.eval()

    for data, target in test_loader:

        data = data.view(data.size()[0], 28 * 28)
        data, target = Variable(data, volatile=True), Variable(target)

        output = model.predict(data)

        temp = output.data.max(1)[1].numpy().reshape(-1)

        label_predict = np.concatenate((label_predict, temp))

    predict_label = pd.DataFrame(label_predict, columns=['label'], dtype=int)
    predict_label.reset_index(inplace=True)
    predict_label.rename(columns={'index': 'ID'}, inplace=True)

    predict_label.to_csv('submission_ladder.csv', index=False)


for i in range(15):

    train_semi_sup()
    test()


predict_test_data()
