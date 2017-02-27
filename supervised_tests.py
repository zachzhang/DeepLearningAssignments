"""
Performance comparison with supervised model
1. Different dropout probability
2. Different initialization: naive, He, Xavier
3. Data augmentation

"""

import pickle 
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from models import *


def main(train_loader, valid_loader, p, n_iter):

    print('Build model!')
    model_paras = {'p': p}
    model = NetSimple(model_paras)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    m = trainModel(train_loader, valid_loader, model, optimizer, n_iter, log_interval = [1,20])
    _, lsTrainError, lsTestError = m.run()
    
    print lsTrainError
    print lsTestError


if __name__ == '__main__':
    print('loading data!')

    trainset_labeled = pickle.load(open("train_labeled.p", "rb"))
    validset = pickle.load(open("validation.p", "rb"))
    
    train_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)


    main(train_loader, valid_loader, 0.5, 100)