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


        arc = [32,32,32]

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5 ,stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3 ,stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        self.flat_dim = [32 ,4 ,4]
        self.flat_dim = [64 ,2 ,2]

        self.h = self.flat_dim[0 ] *self.flat_dim[1 ] *self.flat_dim[2]



        self.latent_vars = 200

        self.fc1 = nn.Linear(self.h, self. latent_vars *2)

        self.fc2 = nn.Linear(self.latent_vars, self.h)

        self.dconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.dconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2 ,output_padding=1)
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


class CVAE2(nn.Module):

    def __init__(self):

        super(CVAE2, self).__init__()

        arc = [32,32,32]

        self.bnf1 = nn.BatchNorm2d(arc[0])
        self.bnf2 = nn.BatchNorm2d(arc[1])
        self.bnf3 = nn.BatchNorm2d(arc[2])

        self.bnb1 = nn.BatchNorm2d(arc[2])
        self.bnb2 = nn.BatchNorm2d(arc[1])
        self.bnb3 = nn.BatchNorm2d(arc[0])


        self.conv1 = nn.Conv2d(1, arc[0], kernel_size=5 ,stride=2)
        self.conv2 = nn.Conv2d(arc[0], arc[1], kernel_size=5 ,stride=2)
        self.conv3 = nn.Conv2d(arc[1], arc[2], kernel_size=3)

        self.flat_dim = self.get_flat_dim()

        self.h = self.flat_dim[0 ] *self.flat_dim[1 ] *self.flat_dim[2]

        self.latent_vars = 10

        self.fc1 = nn.Linear(self.h, self. latent_vars *2)

        self.fc2 = nn.Linear(self.latent_vars, self.h)

        self.dconv1 = nn.ConvTranspose2d(arc[2], arc[1], kernel_size=3)
        self.dconv2 = nn.ConvTranspose2d(arc[1], arc[0], kernel_size=5, stride=2 ,output_padding=1)
        self.dconv3 = nn.ConvTranspose2d(arc[0], 1, kernel_size=5, stride=2 ,output_padding=1)
        

    def get_flat_dim(self):

        x = Variable(torch.randn(64,1,28,28))
        x = F.relu(self.bnf1(self.conv1(x)))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = F.relu(self.bnf3(self.conv3(x)))

        return(x.size()[1:])


    def encode(self ,x):


        x = F.relu(self.bnf1(self.conv1(x)))
        x = F.relu(self.bnf2(self.conv2(x)))
        x = F.relu(self.bnf3(self.conv3(x)))

        x = x.view(-1, x.size()[1] *x.size()[2] * x.size()[3])
        z = self.fc1(x)

        mu = z[:, 0:self.latent_vars]
        log_sig = z[:, self.latent_vars:]

        return mu, log_sig


    def decode(self, z):
        x = self.fc2(z)
   
        x = x.view(-1, self.flat_dim[0], self.flat_dim[1], self.flat_dim[2])
   
        x = F.relu(self.bnb1(x))

        x = F.relu(self.bnb2(self.dconv1(x)))

        x = F.relu(self.bnb3(self.dconv2(x)))

        x = self.dconv3(x)

        #x = F.sigmoid(x)
        return (x)

    def forward(self, x):
        mu,log_sig = self.encode(x)

        eps = Variable(torch.randn(log_sig.size()))

        z = mu + torch.exp(log_sig / 2) * eps

        x_hat = self.decode(z)

        return(x_hat,mu,log_sig)

    def sample(self, n):

        z = Variable(torch.randn((n, self.latent_vars)))
        return (self.decode(z))


class CVAE2_Pool(nn.Module):
    def __init__(self):
        super(CVAE2_Pool, self).__init__()

        arc = [64, 128, 128]
        self.arc = arc

        self.bnf1 = nn.BatchNorm2d(arc[0])
        self.bnf2 = nn.BatchNorm2d(arc[1])
        self.bnf3 = nn.BatchNorm2d(arc[2])

        self.bnb1 = nn.BatchNorm2d(arc[2])
        self.bnb2 = nn.BatchNorm2d(arc[1])
        self.bnb3 = nn.BatchNorm2d(arc[0])

        self.poolf1 = nn.MaxPool2d(2,return_indices=True)
        self.poolf2 = nn.MaxPool2d(2,return_indices=True)
        self.poolf3 = nn.MaxPool2d(2,return_indices=True)

        self.unpool1 = nn.MaxUnpool2d(2)
        self.unpool2 = nn.MaxUnpool2d(2)
        self.unpool3 = nn.MaxUnpool2d(2)


        self.conv1 = nn.Conv2d(1, arc[0], kernel_size=5)
        self.conv2 = nn.Conv2d(arc[0], arc[1], kernel_size=3)
        self.conv3 = nn.Conv2d(arc[1], arc[2], kernel_size=3)

        self.flat_dim = self.get_flat_dim()

        self.h = self.flat_dim[0] * self.flat_dim[1] * self.flat_dim[2]

        self.latent_vars = 10

        self.fc1 = nn.Linear(self.h, self.latent_vars * 2)

        self.fc2 = nn.Linear(self.latent_vars, self.h)

        self.dconv1 = nn.ConvTranspose2d(arc[2], arc[1], kernel_size=3)
        self.dconv2 = nn.ConvTranspose2d(arc[1], arc[0], kernel_size=3)
        self.dconv3 = nn.ConvTranspose2d(arc[0], 1, kernel_size=5)

    def get_flat_dim(self):
        x = Variable(torch.randn(64, 1, 28, 28))
        x = F.relu(F.max_pool2d( self.bnf1(self.conv1(x)) , 2))
        x = F.relu(F.max_pool2d(self.bnf2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.bnf3(self.conv3(x)), 2))

        return (x.size()[1:])

    def encode(self, x):

        x,id1 = self.poolf1(F.relu(self.bnf1(self.conv1(x))))
        x, id2 = self.poolf2(F.relu(self.bnf2(self.conv2(x))))
        x, id3 = self.poolf3(F.relu(self.bnf3(self.conv3(x))))


        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        z = self.fc1(x)

        mu = z[:, 0:self.latent_vars]
        log_sig = z[:, self.latent_vars:]

        return mu, log_sig , [id1,id2,id3]

    def decode(self, z , ids):

        x = self.fc2(z)

        x = x.view(-1, self.flat_dim[0], self.flat_dim[1], self.flat_dim[2])

        x = F.relu(self.bnb1(x))

        x = self.unpool1(x,ids[2] , output_size=torch.Size([64, self.arc[2], 3, 3]))

        x = F.relu(self.bnb2(self.dconv1(x)))

        x = self.unpool2(x,ids[1] , output_size=torch.Size([64, self.arc[1], 10, 10]))

        x = F.relu(self.bnb3(self.dconv2(x)))

        x = self.unpool3(x,ids[0] , output_size=torch.Size([64, self.arc[0], 24, 24]))

        x = self.dconv3(x)

        # x = F.sigmoid(x)
        return (x)

    def forward(self, x):
        mu, log_sig , ids = self.encode(x)

        eps = Variable(torch.randn(log_sig.size()))

        z = mu + torch.exp(log_sig / 2) * eps

        x_hat = self.decode(z,ids)

        return (x_hat, mu, log_sig)

    def sample(self, n):
        z = Variable(torch.randn((n, self.latent_vars)))
        return (self.decode(z))


class DCVAE2_Pool(nn.Module):
    def __init__(self):
        super(DCVAE2_Pool, self).__init__()

        arc = [64, 128, 128]

        self.bnf1 = nn.BatchNorm2d(arc[0])
        self.bnf2 = nn.BatchNorm2d(arc[1])
        self.bnf3 = nn.BatchNorm2d(arc[2])

        self.bnb1 = nn.BatchNorm2d(arc[2])
        self.bnb2 = nn.BatchNorm2d(arc[1])
        self.bnb3 = nn.BatchNorm2d(arc[0])

        self.poolf1 = nn.MaxPool2d(2,return_indices=True)
        self.poolf2 = nn.MaxPool2d(2,return_indices=True)
        self.poolf3 = nn.MaxPool2d(2,return_indices=True)

        self.unpool1 = nn.MaxUnpool2d(2)
        self.unpool2 = nn.MaxUnpool2d(2)
        self.unpool3 = nn.MaxUnpool2d(2)


        self.conv1 = nn.Conv2d(1, arc[0], kernel_size=5)
        self.conv2 = nn.Conv2d(arc[0], arc[1], kernel_size=3)
        self.conv3 = nn.Conv2d(arc[1], arc[2], kernel_size=3)

        self.flat_dim = self.get_flat_dim()

        self.h = self.flat_dim[0] * self.flat_dim[1] * self.flat_dim[2]

        self.latent_vars = 10

        self.fc1 = nn.Linear(self.h, self.latent_vars * 2)

        self.fc2 = nn.Linear(self.latent_vars, self.h)

        self.dconv1 = nn.ConvTranspose2d(arc[2], arc[1], kernel_size=3)
        self.dconv2 = nn.ConvTranspose2d(arc[1], arc[0], kernel_size=3)
        self.dconv3 = nn.ConvTranspose2d(arc[0], 1, kernel_size=5)

        self.noise_std = 0


    def get_flat_dim(self):
        x = Variable(torch.randn(64, 1, 28, 28))
        x = F.relu(F.max_pool2d( self.bnf1(self.conv1(x)) , 2))
        x = F.relu(F.max_pool2d(self.bnf2(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.bnf3(self.conv3(x)), 2))

        return (x.size()[1:])

    def encode(self, x):

        if self.training:

            x, id1 = self.poolf1(F.relu(self.bnf1(self.conv1(x+ Variable(self.noise_std * torch.randn(x.size()))))))
            x, id2 = self.poolf2(F.relu(self.bnf2(self.conv2(x+ Variable(self.noise_std * torch.randn(x.size()))))))
            x, id3 = self.poolf3(F.relu(self.bnf3(self.conv3(x+ Variable(self.noise_std * torch.randn(x.size()))))))

        else:

            x, id1 = self.poolf1(F.relu(self.bnf1(self.conv1(x))))
            x, id2 = self.poolf2(F.relu(self.bnf2(self.conv2(x))))
            x, id3 = self.poolf3(F.relu(self.bnf3(self.conv3(x))))

        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        z = self.fc1(x)

        mu = z[:, 0:self.latent_vars]
        log_sig = z[:, self.latent_vars:]

        return mu, log_sig , [id1,id2,id3]

    def decode(self, z , ids):

        x = self.fc2(z)

        x = x.view(-1, self.flat_dim[0], self.flat_dim[1], self.flat_dim[2])

        x = F.relu(self.bnb1(x))

        x = self.unpool1(x,ids[2] , output_size=torch.Size([64, 32, 3, 3]))

        x = F.relu(self.bnb2(self.dconv1(x)))

        x = self.unpool2(x,ids[1] , output_size=torch.Size([64, 32, 10, 10]))

        x = F.relu(self.bnb3(self.dconv2(x)))

        x = self.unpool3(x,ids[0] , output_size=torch.Size([64, 32, 24, 24]))

        x = self.dconv3(x)

        # x = F.sigmoid(x)
        return (x)

    def forward(self, x):
        mu, log_sig , ids = self.encode(x)

        eps = Variable(torch.randn(log_sig.size()))

        z = mu + torch.exp(log_sig / 2) * eps

        x_hat = self.decode(z,ids)

        return (x_hat, mu, log_sig)

    def sample(self, n):
        z = Variable(torch.randn((n, self.latent_vars)))
        return (self.decode(z))



class DCVAE2_Pool_Deeper(nn.Module):
    def __init__(self,cost_rec = 1.):

        super(DCVAE2_Pool_Deeper, self).__init__()

        self.cost_rec = cost_rec
        self.arc =[]
        arc = [32 , 64, 64, 128]

        self.bnf1 = nn.BatchNorm2d(arc[0])
        self.bnf2 = nn.BatchNorm2d(arc[1])
        self.bnf3 = nn.BatchNorm2d(arc[2])
        self.bnf4 = nn.BatchNorm2d(arc[3])

        self.bnb1 = nn.BatchNorm2d(arc[3])
        self.bnb2 = nn.BatchNorm2d(arc[2])
        self.bnb3 = nn.BatchNorm2d(arc[1])
        self.bnb4 = nn.BatchNorm2d(arc[0])

        self.poolf1 = nn.MaxPool2d(2,return_indices=True)
        self.poolf2 = nn.MaxPool2d(2,return_indices=True)

        self.unpool1 = nn.MaxUnpool2d(2)
        self.unpool2 = nn.MaxUnpool2d(2)

        self.unpool3 = nn.MaxUnpool2d(2)

        self.conv1 = nn.Conv2d(1, arc[0], kernel_size=5)
        self.conv2 = nn.Conv2d(arc[0], arc[1], kernel_size=3)
        self.conv3 = nn.Conv2d(arc[1], arc[2], kernel_size=3)
        self.conv4 = nn.Conv2d(arc[2], arc[3], kernel_size=3)

        self.flat_dim = self.get_flat_dim()

        self.h = self.flat_dim[0] * self.flat_dim[1] * self.flat_dim[2]

        self.latent_vars = 10

        self.fc1 = nn.Linear(self.h, self.latent_vars * 2)

        self.fc2 = nn.Linear(self.latent_vars, self.h)

        self.dconv1 = nn.ConvTranspose2d(arc[3], arc[2], kernel_size=3)
        self.dconv2 = nn.ConvTranspose2d(arc[2], arc[1], kernel_size=3)
        self.dconv3 = nn.ConvTranspose2d(arc[1], arc[0], kernel_size=3)
        self.dconv4 = nn.ConvTranspose2d(arc[0], 1, kernel_size=5)

        self.noise_std = .2


    def get_flat_dim(self):
        x = Variable(torch.randn(64, 1, 28, 28))

        x = F.relu(self.bnf1(self.conv1(x)))
        x, id1 = self.poolf1(x)
        x = F.relu(self.bnf2(self.conv2(x)))
        x = F.relu(self.bnf3(self.conv3(x)))
        x, id2 = self.poolf2(x)
        x = F.relu(self.bnf4(self.conv4(x)))

        x,i = F.max_pool2d(x,2,return_indices=True)

        return (x.size()[1:])

    def encode(self, x):

        if self.training:
            noise = self.noise_std
        else:
            noise = 0

        x = F.relu(self.bnf1(self.conv1(x+ Variable(noise * torch.randn(x.size())))))
        x, id1 = self.poolf1(x)
        x = F.relu(self.bnf2(self.conv2(x+ Variable(noise * torch.randn(x.size())))))

        x = F.relu(self.bnf3(self.conv3(x+ Variable(noise * torch.randn(x.size())))))

        x ,id2 = self.poolf2(x)

        x = F.relu(self.bnf4(self.conv4(x+ Variable(noise * torch.randn(x.size())))))

        x = F.avg_pool2d(x,2)

        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])

        z = self.fc1(x)

        mu = z[:, 0:self.latent_vars]
        log_sig = z[:, self.latent_vars:]

        return mu, log_sig , [id1,id2]

    def avg_unpool(self,x):

        x = x.expand(x.size()[0],x.size()[1],2,2)

        return x

    def decode(self, z , ids):

        x = self.fc2(z)

        x = x.view(-1, self.flat_dim[0], self.flat_dim[1], self.flat_dim[2])

        #x = self.unpool3(x,ids[2],output_size=torch.Size([64,128,2,2]))
        x = self.avg_unpool(x)

        x = F.relu(self.bnb1(x))

        x = F.relu(self.bnb2(self.dconv1(x)))

        x = self.unpool1(x,ids[1] , output_size=torch.Size([64, 64, 8, 8]))

        x = F.relu(self.bnb3(self.dconv2(x)))

        x = F.relu(self.bnb4(self.dconv3(x)))

        x = self.unpool2(x,ids[0] , output_size=torch.Size([64, 32, 24, 24]))

        x = self.dconv4(x)

        return (x)

    def forward(self, x):
        mu, log_sig , ids = self.encode(x)

        eps = Variable(torch.randn(log_sig.size()))

        z = mu + torch.exp(log_sig / 2) * eps

        x_hat = self.decode(z,ids)

        return (x_hat, mu, log_sig)


    def cost(self, x, y):

        x_hat, mu, log_sig = self.forward(x)

        y_hat = F.log_softmax(mu)

        class_loss = F.nll_loss(y_hat,y)

        recon_loss =  ((x_hat - x)**2).mean()

        loss = class_loss + recon_loss * self.cost_rec

        return(loss)


    def unsup_cost(self, x):

        x_hat, mu, log_sig = self.forward(x)

        recon_loss =  ((x_hat - x)**2).mean() * self.cost_rec

        return(recon_loss)


    def predict(self,x):

        x_hat, mu, log_sig = self.forward(x)

        return F.log_softmax(mu)

    def sample(self, n):
        z = Variable(torch.randn((n, self.latent_vars)))
        return (self.decode(z))


class DCVAE2_Pool_Deeper_Ladder(nn.Module):
    def __init__(self, cost_rec = 1., cost_m = .1):

        super(DCVAE2_Pool_Deeper_Ladder, self).__init__()

        self.cost_rec = cost_rec
        self.cost_m = cost_m

        self.arc =[]
        arc = [32 , 64, 64, 128]

        self.bnf1 = nn.BatchNorm2d(arc[0])
        self.bnf2 = nn.BatchNorm2d(arc[1])
        self.bnf3 = nn.BatchNorm2d(arc[2])
        self.bnf4 = nn.BatchNorm2d(arc[3])

        self.bnb1 = nn.BatchNorm2d(arc[3])
        self.bnb2 = nn.BatchNorm2d(arc[2])
        self.bnb3 = nn.BatchNorm2d(arc[1])
        self.bnb4 = nn.BatchNorm2d(arc[0])

        self.poolf1 = nn.MaxPool2d(2,return_indices=True)
        self.poolf2 = nn.MaxPool2d(2,return_indices=True)

        self.unpool1 = nn.MaxUnpool2d(2)
        self.unpool2 = nn.MaxUnpool2d(2)

        self.unpool3 = nn.MaxUnpool2d(2)

        self.conv1 = nn.Conv2d(1, arc[0], kernel_size=5)
        self.conv2 = nn.Conv2d(arc[0], arc[1], kernel_size=3)
        self.conv3 = nn.Conv2d(arc[1], arc[2], kernel_size=3)
        self.conv4 = nn.Conv2d(arc[2], arc[3], kernel_size=3)

        self.flat_dim = self.get_flat_dim()

        self.h = self.flat_dim[0] * self.flat_dim[1] * self.flat_dim[2]

        self.latent_vars = 10

        self.fc1 = nn.Linear(self.h, self.latent_vars * 2)

        self.fc2 = nn.Linear(self.latent_vars, self.h)

        self.dconv1 = nn.ConvTranspose2d(arc[3], arc[2], kernel_size=3)
        self.dconv2 = nn.ConvTranspose2d(arc[2], arc[1], kernel_size=3)
        self.dconv3 = nn.ConvTranspose2d(arc[1], arc[0], kernel_size=3)
        self.dconv4 = nn.ConvTranspose2d(arc[0], 1, kernel_size=5)

        self.noise_std = .2


    def get_flat_dim(self):
        x = Variable(torch.randn(64, 1, 28, 28))

        x = F.relu(self.bnf1(self.conv1(x)))
        x, id1 = self.poolf1(x)
        x = F.relu(self.bnf2(self.conv2(x)))
        x = F.relu(self.bnf3(self.conv3(x)))
        x, id2 = self.poolf2(x)
        x = F.relu(self.bnf4(self.conv4(x)))

        x,i = F.max_pool2d(x,2,return_indices=True)

        return (x.size()[1:])

    def encode(self, x):

        if self.training:
            noise = self.noise_std
        else:
            noise = 0

        z1 = F.relu(self.bnf1(self.conv1(x+ Variable(noise * torch.randn(x.size())))))

        z1, id1 = self.poolf1(z1)

        z2 = F.relu(self.bnf2(self.conv2(z1+ Variable(noise * torch.randn(z1.size())))))

        z3 = F.relu(self.bnf3(self.conv3(z2+ Variable(noise * torch.randn(z2.size())))))

        z3 ,id2 = self.poolf2(z3)

        z4 = F.relu(self.bnf4(self.conv4(z3+ Variable(noise * torch.randn(z3.size())))))

        x = F.avg_pool2d(z4,2)

        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])

        z = self.fc1(x)

        mu = z[:, 0:self.latent_vars]
        log_sig = z[:, self.latent_vars:]

        return mu, log_sig , [z1,z2,z3,z4] ,[id1,id2]

    def avg_unpool(self,x):

        x = x.expand(x.size()[0],x.size()[1],2,2)

        return x

    def decode(self, z , ids):

        x = self.fc2(z)

        x = x.view(-1, self.flat_dim[0], self.flat_dim[1], self.flat_dim[2])

        x = self.avg_unpool(x)

        h1 = self.bnb1(x)
        x = F.relu(h1)

        h2 = self.bnb2(self.dconv1(x))
        x = F.relu(h2)

        x = self.unpool1(x,ids[1] , output_size=torch.Size([64, 64, 8, 8]))

        h3 =self.bnb3(self.dconv2(x))
        x = F.relu(h3)

        h4 = self.bnb4(self.dconv3(x))
        x = F.relu(h4)

        x = self.unpool2(x,ids[0] , output_size=torch.Size([64, 32, 24, 24]))

        x = self.dconv4(x)

        return (x,[h1,h2,h3,h4])

    def forward(self, x):
        mu, log_sig , h_encode ,  ids = self.encode(x)

        eps = Variable(torch.randn(log_sig.size()))

        z = mu + torch.exp(log_sig / 2) * eps

        x_hat , h_decode = self.decode(z,ids)

        h_decode.reverse()

        return (x_hat, mu, log_sig , h_decode , h_encode)

    def cost(self, x, y):

        x_hat, mu, log_sig, h_decode, h_encode = self.forward(x)

        y_hat = F.log_softmax(mu)

        class_loss = F.nll_loss(y_hat,y)

        recon_loss =  ((x_hat - x)**2).mean()

        h_loss = Variable(torch.zeros(1))

        for h_d, h_e in zip(h_decode, h_encode):
            h_loss += ((h_d - h_e) ** 2).mean()

        loss = class_loss + self.cost_rec * recon_loss + self.cost_m *h_loss

        return(loss)

    def unsup_cost(self, x):

        x_hat, mu, log_sig, h_decode, h_encode = self.forward(x)

        y_hat = F.log_softmax(mu)

        recon_loss =  ((x_hat - x)**2).mean()

        h_loss = Variable(torch.zeros(1))

        for h_d, h_e in zip(h_decode, h_encode):
            h_loss += ((h_d - h_e) ** 2).mean()

        loss = self.cost_rec * recon_loss + self.cost_m *h_loss

        return(loss)


    def predict(self,x):

        x_hat, mu, log_sig, h_decode, h_encode = self.forward(x)

        return F.log_softmax(mu)

    def sample(self, n):
        z = Variable(torch.randn((n, self.latent_vars)))
        return (self.decode(z))


model = DCVAE2_Pool_Deeper_Ladder()


#print(model.encode(Variable(torch.randn(64,1,28,28)))[0].size()  )
#print(model.forward(Variable(torch.randn(64,1,28,28)))[0].size()  )
print(model.cost(Variable(torch.randn(64,1,28,28)) ,  Variable(torch.ones(64)).long())  )

#model.forward(Variable(torch.randn(64,1,28,28)))
#model.encode(Variable(torch.randn(64,1,28,28)))

#print(model.get_flat_dim())

#a = model(Variable(torch.randn(64,1,28,28)))[0]

#print(a.size())

