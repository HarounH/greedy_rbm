'''
    Implementation of SBN such that
    we sparsify latent variables

    Note: single layer
'''
__author__ = 'haroun habeeb'
__mail__ = 'haroun7@gmail.com'

import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Bernoulli
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from utils import display, make_dot, \
    smooth_distribution, EPS, binary_row_reduce, \
    weighted_average_logsumexp


class SBN_Greedy(nn.Module):
    eps = EPS

    def __init__(self, nx, nz):
        super(SBN_Greedy, self).__init__()
        self.nx = nx
        self.nz = nz

        def sample_range(denom):
            return np.sqrt(6 / denom)
        wr = sample_range(nx + nz)
        self.U = nn.Parameter(torch.rand(nz, nx) * 2 * wr - wr)
        # self.V = nn.Parameter(torch.rand(nx, nz) * 2 * wr - wr)
        xr = sample_range(nx)
        self.x_bias = nn.Parameter(torch.rand(nx) * 2 * xr - xr)
        zr = sample_range(nz)
        self.z_bias = nn.Parameter(torch.rand(nz) * 2 * zr - zr)

    def sample_from(self, d, S):
        '''
            ARGS
            ----
            d   FloatTensor
                tensor containing of Bernoulli paramater of
                independent variables
            RETURNS
            ----
            samples: (S, *list(d.size())) sized tensor
            sampler: a torch.distributions.Bernoulli object
        '''
        sampler = Bernoulli(d)
        samples = sampler.sample_n(S)
        return samples, sampler

    def x2z(self, x):
        '''
            Inference (q(z | x))
            ARGS
            ----
                x: (nx) sized FloatTensor
            RETURNS
            ----
                q: (nz) sized FloatTensor which represents
                    Pr[Z_i=1 | x]
        '''
        return F.sigmoid(F.linear(x, self.U, self.z_bias))

    def z2x(self, z):
        '''
            Generative (p(x  | z))
            ARGS
            ----
                z: (nz) sized FloatTensor
            RETURNS
            ----
                p: (nx) sized FloatTensor which represents
                    Pr[X_i=1 | z]
        '''
        return F.sigmoid(F.linear(z, self.U.t(), self.x_bias))

    def forward(self, xin_distribution, S=1,
                compute_loss=True,
                k=25):
        '''
            ARGS
            ----
                xin_distribution: (batch_size, nx) tensor of bernoulli params
                S: # samples
                compute_loss:
                k: sparsity hyperparameter for greedy
            RETURNS
            ----
        '''
        xin = xin_distribution.bernoulli()
        batch_size = xin.size()[0]
        q = smooth_distribution(self.x2z(xin))  # batch size, nz
        samples_z, sampler_z = self.sample_from(q, S)  # S, batch_size, nz
        ps = torch.stack([
            smooth_distribution(self.z2x(sample_z)) for sample_z in samples_z
            ])
        xouts = torch.stack([self.sample_from(psi, 1)[0][0] for psi in ps])
        if compute_loss:
            # Need to compute which variables are free

            # Get which variables are free
            # binary vector of length nz
            Z_f_indices, Z_f_c_indices = \
                self.get_k_free_latent_variables(k, xin)  # batch_size, nz
            # Compute elbo
            self.set_latent_variables_constants(samples_z,
                                                Z_f_c_indices,
                                                q.expand(S,
                                                         batch_size,
                                                         self.nz)
                                                )

            logq_zi = torch.stack([
                sampler_z.log_prob(sample_z) for sample_z in samples_z
                ])  # S, batch_size, nz
            pz = smooth_distribution(
                F.sigmoid(self.z_bias).expand(S, batch_size, -1)
                )
            z_prior = Bernoulli(pz)
            logp_zi = z_prior.log_prob(samples_z)
            ps_new = torch.stack([
                smooth_distribution(self.z2x(sample_z))
                for sample_z in samples_z
                ])
            px_given_z = Bernoulli(ps_new)
            logp_xin_given_samples_z = px_given_z.log_prob(
                xin.expand(S, batch_size, self.nx)
                )  # S, batch_size, nx

            # Now to put it all together
            logq = logq_zi.sum(dim=2)
            logp = logp_zi.sum(dim=2) + logp_xin_given_samples_z.sum(dim=2)
            # Eh, the names can be fixed later
            term2 = weighted_average_logsumexp(logq, logq)
            term1 = weighted_average_logsumexp(logq, logp)
            loss = -(term1 - term2)
        else:
            loss = None
        return q, samples_z, ps, xouts, loss

    def set_latent_variables_constants(self,
                                       inp,
                                       indices,
                                       bernoulli_parameters):
        '''
            inp: S, batch_size, n sized
            sets indices along dim=2 to constants? function of bernoulli?
        '''
        inp[:, :, indices] = (bernoulli_parameters[:, :, indices] > 0.5).float()

    def set_visible_variables_constants(self,
                                        inp,
                                        indices,
                                        bernoulli_parameters):
        '''
            inp: S, batch_size, n sized
            sets indices along dim=2 to constants? function of bernoulli?
        '''
        inp[:, :, indices] = (bernoulli_parameters[:, :, indices] > 0.5).float()

    def get_k_free_latent_variables(self,
                                    k,
                                    xin_distribution,
                                    q=None,
                                    p=None,
                                    noisy=False,
                                    set_constants=None):
        '''
            ARGS
            ----
                k: # of free variables
                xin: the input x that we can use to compute other stuff
            RETURNS
            ----
                Z_f_indices: indices of free variables, len=k
                Z_f_c_indices: indices of non-free variables. len=self.nz - k
        '''
        if q is None:
            xin = xin_distribution.bernoulli()
            q = smooth_distribution(self.x2z(xin))  # nz
        if p is None:
            p = smooth_distribution(F.sigmoid(self.z_bias))  # nz
        if set_constants is None:
            set_constants = self.set_latent_variables_constants
        if noisy is True:
            raise NotImplementedError
        else:
            # Simply pick the top-k indices of p.
            constants = Variable(torch.rand(1, 1, self.nz))
            set_constants(constants, list(range(self.nz)), p.expand(1, 1, -1))
            z_prior = Bernoulli(p.squeeze())
            logps = z_prior.log_prob(constants.squeeze())
            Z_f_indices = logps.topk(k)[1].data.tolist()
            # Z_f_c_indices is common to if else thingy
        Z_f_c_indices = list(set(range(self.nz)) - set(Z_f_indices))
        return Z_f_indices, Z_f_c_indices

    def get_k_free_visible_variables(self,
                                     k,
                                     xin_distribution,
                                     q=None,
                                     p=None,
                                     noisy=False,
                                     set_constants=None,
                                     Z_f_indices=[]):
        '''
            ARGS
            ----
            RETURNS
            ----
                X_f_indices
                X_f_c_indices
        '''
        raise NotImplementedError


def sanity_test():
    batch_size = 20
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('MNIST_data/', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('MNIST_data/',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size)

    sbn = SBN_Greedy(784, 200)
    optimizer = optim.SGD(sbn.parameters(), lr=0.001)

    for epoch in range(10):
        losses = []
        for _, (data, target) in enumerate(train_loader):
            data = Variable(data.view(-1, 784))  # visible
            # pdb.set_trace()
            data_sample = data.bernoulli()  # xin
            z, z_sample, xp, xp_sample, loss = sbn(data_sample,
                                                   compute_loss=True,
                                                   k=sbn.nz - 25)
            losses.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch', epoch, 'loss=', np.mean(losses))

    display("real", make_grid(data_sample.view(-1, 1, 28, 28).data))
    display("generate", make_grid(xp_sample[0].view(-1, 1, 28, 28).data))
    plt.show()


if __name__ == '__main__':
    sanity_test()
