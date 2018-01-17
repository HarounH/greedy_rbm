'''
  File to make a SBN and a DBN, trained using VMCS Steroids
  http://cs.stanford.edu/~ermon/papers/variational_nips2016.pdf
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

from utils import display, make_dot, smooth_distribution, EPS

# TODO: set torch seed
torch.manual_seed(1337)
EPS = 10**-6

class SBN_Steroids(nn.Module):
    eps = EPS
    def __init__(self, nx, nz):
        super(SBN_Steroids, self).__init__()
        self.nx = nx
        self.nz = nz
        def sample_range(denom):
            return np.sqrt(6 / denom)
        wr = sample_range(nx + nz)
        self.U = nn.Parameter(torch.rand(nz, nx) * 2 * wr - wr)
        self.V = nn.Parameter(torch.rand(nx, nz) * 2 * wr - wr)
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
        return F.sigmoid(F.linear(z, self.V, self.x_bias))

    def forward(self, xin, S=1,
                compute_loss=True, aggregate_fn=None, k=25, T=30):
        '''
            ARGS
            ----
                xin
                S
                compute_elbo
                aggregate_fn
            RETURNS
                q
                samples_z
                ps
                xouts
        '''
        q = smooth_distribution(self.x2z(xin))
        samples_z, sampler_z = self.sample_from(q, S)
        ps_list = [smooth_distribution(self.z2x(z)) for z in samples_z]
        ps = torch.stack(ps_list)
        xouts = torch.stack([self.sample_from(psi, 1)[0][0] for psi in ps])
        if compute_loss:
            if aggregate_fn is None:
                aggregate_fn = lambda vars: sum(vars) / len(vars)  # Mean by default
            projected_elbos = []
            for t in range(T):
                sampler_A = Bernoulli(0.5 * torch.ones(k, self.nz))
                A = sampler_A.sample()
                sampler_b = Bernoulli(0.5 * torch.ones(k))
                b = sampler_b.sample()
                # Get C, bp
                pdb.set_trace()
                C, bp = binary_row_reduce(A, b)
                # Implement computeProjectedELBO - pass in whatever it needs.
                pdb.set_trace()
                projected_elbos.append(self.computeProjectedELBO(xin,
                                                                 samples_z.clone(),
                                                                 sampler_z,
                                                                 C,
                                                                 bp))
                pdb.set_trace()
            loss = -aggregate_fn(projected_elbos)
        else:
            loss = None
        return q, samples_z, ps, xouts, loss

    def computeProjectedELBO(self, xin, samples_z, sampler_z, C, bp):
        '''
            Computes projected ELBO
            xin: batch_size, nx
            samples_z: ns, batch_size, nz
            sampler_z: Bernoulli(batch_size, nz) object
            C: k*n
            bp: k
        '''
        S = samples_z.size()[0]
        batch_size = xin.size()[0]
        k, n = C.size()[0], C.size()[1]

        samples_z[:, :, :k] = (samples_z[:, :, k:].matmul(C[:, k:].t()) + bp) % 2

        logq_zi = torch.stack([sampler_z.log_prob(sample_z) for sample_z in samples_z])  # S, batch_size, nz
        pz = smooth_distribution(F.sigmoid(self.z_bias).expand(S, batch_size, -1))

        z_prior = Bernoulli(pz)

        logp_z = z_prior.log_prob(samples_z)  # S, batch_size, nz

        # We need a new px_given_z because the zs have changed.
        ps = torch.stack([smooth_distribution(self.z2x(sample_z)) for sample_z in samples_z])
        px_given_z = Bernoulli(ps)
        logp_xin_given_samples_z = px_given_z.log_prob(xin.expand(S, batch_size, self.nx)) # S, batch_size, nx

        logq = logq_zi.sum(dim=2)
        logp = logp_z.sum(dim=2) + logp_xin_given_samples_z.sum(dim=2)
        term2 = self.compute_term1(logq, logq)
        term1 = self.compute_term1(logq, logp)

        return term1 - term2
        # Note that nothing has been summed up
    def compute_term1(self, logq, logp, dim=0):
        '''
            Numerically stable way of computing
                [ \Sigma q * log(p) ]/ [ \Sigma q ]
                (along dimension 0 for now)

            given logq, logp
                everything is negative

            operations have to be independent of dim=1, dim=2

            everything will have to be done in logspace
            num = \Sigma q * log(p)
            den = \Sigma q

            need to compute num / den => torch.log(num / den).exp() would do
            torch.log(-num) is doable
            torch.log(den) is doable
            torch.log(num) is NOT doable
            -1 * (torch.log(-num) - torch.log(den)).exp() provides hope

            ARGS
            ----
            logq : S, batch_size
            logp : S, batch_size

            RETURNS
            ----
        '''
        S, batch_size = logq.size()[0], logq.size()[1]
        # pdb.set_trace()
        logq_max = logq.max(dim=0)[0]  # 1, batch_size, 1
        logq_max_expanded = logq_max.expand(S, batch_size)
        # numerator part
        neg_num = logq_max + \
            torch.log(((logq - logq_max_expanded).exp() * (-logp)).sum(dim=0))
        # pdb.set_trace()
        # denominator part
        pos_den = logq_max + \
            torch.log((logq - logq_max_expanded).exp().sum(dim=0))
        # pdb.set_trace()
        return -1 * (neg_num - pos_den).exp().sum()

def sanity_test():
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('MNIST_data/', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('MNIST_data/', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size)

    sbn = SBN_Steroids(784, 200)
    optimizer = optim.SGD(sbn.parameters(), lr=0.1)
    # Quick aggregation function - mean
    def mean_fn(ls):
        return sum(ls) / len(ls)

    for epoch in range(4):
        losses = []
        for _, (data, target) in enumerate(train_loader):
            data = Variable(data.view(-1, 784))  # visible
            # pdb.set_trace()
            data_sample = data.bernoulli()  # xin
            z, z_sample, xp, xp_sample, loss = sbn(data_sample,
                                                   compute_loss=True,
                                                   aggregate_fn=mean_fn)
            # pdb.set_trace()
            losses.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch', epoch, 'loss=', np.mean(losses))

    display("real", make_grid(data_sample.view(32, 1, 28, 28).data))
    display("generate", make_grid(xp_sample[0].view(32, 1, 28, 28).data))
    plt.show()


if __name__ == '__main__':
    sanity_test()
