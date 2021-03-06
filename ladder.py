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


class LadderNet(nn.Module):

    def __init__(self):

        super(LadderNet, self).__init__()

        self.noise_std = 0.3

        self.layer_sizes = [784, 1000, 500, 250, 250, 250, 10]

        # self.layer_sizes =[28*28,256,256, 128,10]
        self.denoise_cost = [1000.0, 10.0 ,0.1 ,0.1, 0.10, 0.10, 0.10]

        self.shapes = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))

        self.modules = [ nn.Linear(shape[0] ,shape[1]) for shape in self.shapes]
        self.decoder_modules = [ nn.Linear(shape[1] ,shape[0]) for shape in self.shapes]

        self.bn_layers_forward = [nn.BatchNorm1d(size) for size in self.layer_sizes ]
        self.bn_layers_backward = [nn.BatchNorm1d(size) for size in self.layer_sizes]

        self.init_denoising_params()
        self.init_bn_params()


    def init_denoising_params(self):

        self.denoising_params = []
        init = [0 ,1 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0]

        for i in range(len(self.layer_sizes)):

            a = []

            for j in init:
                # a.append( nn.Parameter(.01 * torch.randn([1,self.layer_sizes[i]])) )
                # a.append( nn.Parameter(j * torch.ones([1,self.layer_sizes[i]])) +.01 * torch.randn([1,self.layer_sizes[i]])) )
                a.append( nn.Parameter(j * torch.ones([1 ,self.layer_sizes[i]])) )

            self.denoising_params.append(a)

    def init_bn_params(self):

        self. beta= []
        self.gamma = []
        for i in range(1 ,len(self.layer_sizes)):

            self.beta.append( nn.Parameter( torch.zeros([1 ,self.layer_sizes[i]])) )
            self.gamma.append( nn.Parameter( torch.ones([1 ,self.layer_sizes[i]])) )

    def batch_norm(self ,x):

        mu = x.mean(0)
        z = x- mu.expand(x.size())
        std = ((z) ** 2).mean(0).sqrt()

        return z / (std + Variable(10e-10 * torch.ones(x.size()[1]))).expand(z.size()), mu, std

    def index_mat(self, x, mask):

        return (x[mask.expand(x.size())].resize(mask.sum().data[0], x.size()[1]))

    def predict(self, x):

        z = [x]
        h = [x]

        for i in range(len(self.layer_sizes) - 1):

            z_pre = self.modules[i](h[i])

            z_bn = self.bn_layers_forward[i + 1](z_pre)

            z.append(z_bn)

            if i == (len(self.layer_sizes) - 2):
                h.append(F.log_softmax(z[i + 1]))
            else:
                h.append(F.relu(z[i + 1]))

        return h[-1]

    def forward2(self, x):

        h_noise = [x + self.noise_std *Variable(torch.randn(x.size()))]
        z_noise = [h_noise[0]]
        z = [x]
        h = [x]

        # forward pass
        for i in range(len(self.layer_sizes) - 1):

            z_pre = self.modules[i](h[i])
            z_pre_noise = self.modules[i](h_noise[i])

            z_bn = self.bn_layers_forward[i + 1](z_pre)
            z_bn_noise = self.bn_layers_forward[i + 1](z_pre_noise)

            z.append(z_bn)
            z_noise.append(z_bn_noise + self.noise_std * Variable(torch.randn(z_pre_noise.size())))

            if i == (len(self.layer_sizes) - 2):
                h_noise.append(F.log_softmax(z_noise[i + 1]))
                h.append(F.log_softmax(z[i + 1]))
            else:
                h_noise.append(F.relu(z_noise[i + 1]))
                h.append(F.relu(z[i + 1]))

        # decoder
        j = 0
        u = []
        z_hat_bn = []

        for i in range(len(self.layer_sizes) - 1, -1, -1):

            if i == (len(self.layer_sizes) - 1):
                bn,_,_ = self.batch_norm( h_noise[-1])
                #bn = self.bn_layers_backward[i](h_noise[-1])
            else:
                bn, _, _ = self.batch_norm(self.decoder_modules[i](z_hat))
                #bn = self.bn_layers_backward[i](self.decoder_modules[i](z_hat))

            u.append(bn)

            z_hat = self.g_gauss(z_noise[i], u[j], i)

            z_hat_bn.append(z_hat)

            j += 1

        z_hat_bn.reverse()

        return (z_hat_bn, z, h_noise[-1])


    def forward(self, x):

        h_noise = [x + self.noise_std *Variable(torch.randn(x.size()))]
        z_noise = [h_noise[0]]
        z = [x]
        h = [x]

        # forward pass
        for i in range(len(self.layer_sizes) - 1):

            z_pre = self.modules[i](h[i])
            z_pre_noise = self.modules[i](h_noise[i])

            z_bn = self.bn_layers_forward[i + 1](z_pre)
            z_bn_noise = self.bn_layers_forward[i + 1](z_pre_noise)

            z.append(z_bn)
            z_noise.append(z_bn_noise + self.noise_std * Variable(torch.randn(z_pre_noise.size())))

            if i == (len(self.layer_sizes) - 2):
                h_noise.append(F.log_softmax(z_noise[i + 1]))
                h.append(F.log_softmax(z[i + 1]))
            else:
                h_noise.append(F.relu(z_noise[i + 1]))
                h.append(F.relu(z[i + 1]))

        # decoder
        j = 0
        u = []
        z_hat_bn = []

        for i in range(len(self.layer_sizes) - 1, -1, -1):

            if i == (len(self.layer_sizes) - 1):
                bn = self.bn_layers_backward[i](h_noise[-1])
            else:
                bn = self.bn_layers_backward[i](self.decoder_modules[i](z_hat))

            u.append(bn)

            z_hat = self.g_gauss(z_noise[i], u[j], i)

            z_hat_bn.append(z_hat)

            j += 1

        z_hat_bn.reverse()

        return (z_hat_bn, z, h_noise[-1])

    def cost(self, z_hat, z_clean, y_hat, y):

        C_denoise = Variable(torch.zeros(1))

        for i in range(len(z_hat)):
            C_denoise += ((z_hat[i] - z_clean[i]) ** 2).mean() * Variable(torch.ones([1]) * self.denoise_cost[i])

        C_forward = F.nll_loss(y_hat, y)

        return (C_denoise, C_forward)

    def g_gauss(self, z, u, i):

        a = [_a.expand(z.size()) for _a in self.denoising_params[i]]

        mu = a[0] * F.sigmoid(a[1] * u + a[2]) + a[3] * u + a[4]
        v = a[5] * F.sigmoid(a[6] * u + a[7]) + a[8] * u + a[9]

        return (z - mu) * v + mu




class LadderNet2(nn.Module):

    def __init__(self):

        super(LadderNet2, self).__init__()

        self.noise_std = 0.3

        self.layer_sizes = [784, 1000, 500, 250, 250, 250, 10]

        # self.layer_sizes =[28*28,500,256,256, 128,10]

        self.denoise_cost = [1000.0, 10.0, 0.1, 0.1, 0.10, 0.10, 0.10]

        self.shapes = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))

        self.modules = [nn.Linear(shape[0], shape[1]) for shape in self.shapes]
        self.decoder_modules = [nn.Linear(shape[1], shape[0]) for shape in self.shapes]


        self.init_denoising_params()
        self.init_bn_params()

    def init_denoising_params(self):

        self.denoising_params = []
        init = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]

        for i in range(len(self.layer_sizes)):

            a = []

            for j in init:
                # a.append( nn.Parameter(.01 * torch.randn([1,self.layer_sizes[i]])) )
                # a.append( nn.Parameter(j * torch.ones([1,self.layer_sizes[i]])) +.01 * torch.randn([1,self.layer_sizes[i]])) )
                a.append(nn.Parameter(j * torch.ones([1, self.layer_sizes[i]])))

            self.denoising_params.append(a)

    def init_bn_params(self):

        self.beta = []
        self.gamma = []
        for i in range(1, len(self.layer_sizes)):
            self.beta.append(nn.Parameter(torch.zeros([1, self.layer_sizes[i]])))
            self.gamma.append(nn.Parameter(torch.ones([1, self.layer_sizes[i]])))

    def batch_norm(self, x):

        mu = x.mean(1)
        z = x - mu.expand(x.size())
        std = ((z) ** 2).mean(1).sqrt()

        return z / (std).expand(z.size()), mu, std

    def index_mat(self, x, mask):

        return (x[mask.expand(x.size())].resize(mask.sum().data[0], x.size()[1]))

    def predict(self, x):

        z = [x]
        h = [x]

        for i in range(len(self.layer_sizes) - 1):

            z_pre = self.modules[i](h[i])

            # z_bn = self.bn_layers_forward[i + 1](z_pre)
            z_bn, _mu, _sig = self.batch_norm(z_pre)

            z.append(z_bn)

            beta = self.beta[i].expand(z[i + 1].size())
            gamma = self.gamma[i].expand(z[i + 1].size())

            if i == (len(self.layer_sizes) - 2):
                h.append(F.log_softmax(gamma * (z[i + 1] + beta)))
            else:
                h.append(F.relu(gamma * (z[i + 1] + beta)))
                # h.append(F.relu(z[i+1]))

        return h[-1]

    def forward(self, x):

        h_noise = [x + self.noise_std * Variable(torch.randn(x.size()))]
        z_noise = [h_noise[0]]
        z = [x]
        h = [x]
        mu = []
        sig = []

        # forward pass
        for i in range(len(self.layer_sizes) - 1):

            z_pre = self.modules[i](h[i])
            z_pre_noise = self.modules[i](h_noise[i])

            z_bn, _mu, _sig = self.batch_norm(z_pre)

            z_bn_noise, _, _ = self.batch_norm(z_pre_noise)

            z.append(z_bn)
            z_noise.append(z_bn_noise + self.noise_std * Variable(torch.randn(z_pre_noise.size())))
            mu.append(_mu)
            sig.append(_sig)

            beta = self.beta[i].expand(z_noise[i + 1].size())
            gamma = self.gamma[i].expand(z_noise[i + 1].size())

            if i == (len(self.layer_sizes) - 2):

                h_noise.append(F.log_softmax(gamma * (z_noise[i + 1] + beta)))
                h.append(F.log_softmax(gamma * (z[i + 1] + beta)))
                # h_noise.append(F.log_softmax(z_noise[i+1])
                # h.append(F.log_softmax(z[i+1]))

            else:
                h_noise.append(F.relu(gamma * (z[i + 1] + beta)))
                h.append(F.relu(gamma * (z[i + 1] + beta)))
                # h_noise.append(F.relu(z_noise[i+1]))
                # h.append(F.relu(z[i+1]))

        # decoder
        j = 0
        u = []
        z_hat_bn = []

        for i in range(len(self.layer_sizes) - 1, -1, -1):

            if i == (len(self.layer_sizes) - 1):
                bn, _, _ = self.batch_norm(h_noise[-1])
            else:
                bn, _, _ = self.batch_norm(self.decoder_modules[i](z_hat))

            u.append(bn)

            z_hat = self.g_gauss(z_noise[i], u[j], i)

            if i != 0:
                _mu = mu[i - 1].expand(z_hat.size())
                _std = sig[i - 1].expand(z_hat.size())

                z_hat_bn.append((z_hat - _mu) / (_std))

            else:

                z_hat_bn.append(z_hat)

            j += 1

        z_hat_bn.reverse()

        return (z_hat_bn, z, h_noise[-1])

    def cost(self, z_hat, z_clean, y_hat, y):

        C_denoise = Variable(torch.zeros(1))

        for i in range(len(z_hat)):
            C_denoise += ((z_hat[i] - z_clean[i]) ** 2).mean() * Variable(torch.ones([1]) * self.denoise_cost[i])

        C_forward = F.nll_loss(y_hat, y)

        return (C_denoise, C_forward)

    def g_gauss(self, z, u, i):

        a = [_a.expand(z.size()) for _a in self.denoising_params[i]]

        mu = a[0] * F.sigmoid(a[1] * u + a[2]) + a[3] * u + a[4]
        v = a[5] * F.sigmoid(a[6] * u + a[7]) + a[8] * u + a[9]

        return (z - mu) * v + mu

