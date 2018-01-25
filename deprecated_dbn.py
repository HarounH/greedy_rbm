'''
    SBN with multiple layers
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
from utils import make_dot, display, smooth_distribution, EPS, sample_range, \
    compute_elbo


class _deprecated_DBN(nn.Module):
    '''
        Inference (encode):
            x -> z0 -> z1 ... -> z(L - 1)

            q(z(0) | x) = sigmoid(W(0) * x + bq(0))

            for l=1, 2, ... L-1
            q(z(l) | z(l-1)) = sigmoid(W(l) * z(l-1) + bq(l))


        Hence, W(l) is of size (n(l), n(l-1)), such that n(-1) = nx
        Generation (decode):
            z(L-1) -> z(L-2) -> ... z(0) -> x

            p(z(L-1)) = sigmoid(bp(L)) // L, not L - 1
            for l=1, 2, 3... L-1
            p(z(l-1) | z(l)) = sigmoid(W(l).t() * z(l) + bp(l))

            p(x | z(0)) = sigmoid(W(0).t() * z(0) + bp(0))
    '''
    eps = EPS

    def __init__(self, nx, nzs):
        raise DeprecationWarning
        raise NotImplementedError
        super(DBN, self).__init__()
        self.nx = nx
        self.nzs = nzs
        self.L = len(nzs)
        self.Ws = []
        self.bq = []  # Inference
        self.bp = []  # Generation

        # (0)
        wr = sample_range(nzs[0] * nx)
        self.Ws.append(
            nn.Parameter(
                torch.randn(nzs[0], nx) * 2 * wr - wr
            )
        )
        bqr = sample_range(nzs[0])
        self.bq.append(
            nn.Parameter(
                torch.randn(nzs[0]) * 2 * bqr - bqr
            )
        )
        bpq = sample_range(nx)
        self.bp.append(
            nn.Parameter(
                torch.randn(nx) * 2 * bpq - bpq
            )
        )

        for l in range(1, self.L):
            raise NotImplementedError
            wr = sample_range(nzs[l] * nzs[l - 1])
            bqr = sample_range(nzs[l])
            bpr = sample_range(nzr[l - 1])
            self.Ws.append(
                nn.Parameter(
                    torch.randn(nzs[l], nzs[l - 1]) * 2 * wr - wr
                )
            )
            self.bq.append(
                nn.Parameter(
                    torch.randn(nzs[l]) * 2 * bqr - bqr
                )
            )
            self.bp.append(
                nn.Parameter(
                    torch.randn(nzs[l - 1]) * 2 * bpq - bpq
                )
            )

        # bpr = sample_range(nzs[-1])
        # self.bp.append(
        #     self.bq[0]
        # )  # Ignored, really

        self.Ws = nn.ParameterList(self.Ws)
        self.bq = nn.ParameterList(self.bq)
        self.bp = nn.ParameterList(self.bp)

    def generate(self, z0=None, bs=None, batch_dim=0, smooth_eps=eps, S=1):
        '''
            z(L-1) -> z(L-2) -> ... z(0) -> x

            p(z(L-1)) = sigmoid(bq(0)) // L, not L - 1
            bp(L) = bq(0) (<--- that)


            for l=1, 2, 3... L-1
            p(z(l-1) | z(l)) = sigmoid(W(l).t() * z(l) + bp(l))

            p(x | z(0)) = sigmoid(W(0).t() * z(0) + bp(0))

            ARGS
            ----
                z0: FloatTensor list
            RETURNS
            ----
                samples: FloatTensor list list, each FloatTensor is batch_size, nzs[i]
                samplers: Bernoulli list list, each one has batch_size, nzs[i]
        '''
        if bs is None:
            bs = z0.size()[batch_dim]

        samples = [[] for i in range(S)]
        samplers = [[] for i in range(S)]
        for i in range(S):
            sample = samples[i]
            sampler = samplers[i]
            sampler = [Bernoulli(smooth_distribution(F.sigmoid(self.bq[0]),
                                                     eps=smooth_eps).expand(bs, -1))]
            if z0 is None:
                sample = sampler[0].sample()
            sample = [z0[i]]
            for l in range(self.L - 1, 0, -1):
                dist = smooth_distribution(
                    F.sigmoid(F.linear(sample[-1], self.Ws[l].t(), self.bp[l])),
                    eps=smooth_eps
                    )
                sampler.append(Bernoulli(dist))
                sample.append(sampler[-1].sample())

            dist = smooth_distribution(
                F.sigmoid(F.linear(sample[-1], self.Ws[0].t(), self.bp[0])),
                eps=smooth_eps
                )
            sampler.append(Bernoulli(dist))
            sample.append(sampler[-1].sample())

        return samples, samplers

    def infer(self,  xin_dist, smooth_eps=eps, S=1):
        '''
        Inference (encode):
            x -> z0 -> z1 ... -> z(L - 1)

            q(z(0) | x) = sigmoid(W(0) * x + bq(0))

            for l=1, 2, ... L-1
            q(z(l) | z(l-1)) = sigmoid(W(l) * z(l-1) + bq(l))
        '''

        samples = [[] for i in range(S)]
        samplers = [[] for i in range(S)]

        for i in range(S):
            sampler = samplers[i]
            sample = sample[i]
            sampler = [Bernoulli(xin_dist)]
            sample = [sampler[-1].sample()]
            for l in range(0, self.L):
                dist = smooth_distribution(
                    F.sigmoid(F.linear(sample[-1], self.Ws[l], self.bq[l])),
                    eps=smooth_eps
                    )
                sampler.append(Bernoulli(dist))
                sample.append(sampler[-1].sample())
        return sample, sampler

    def forward(self, xin_dist, compute_loss=False, S=1):
        '''
            S only matters in computing the loss
        '''
        input_sampler = Bernoulli(xin_dist)
        xin = input_sampler.sample()  # batch_size, nx
        if compute_loss:
            # Stochastic noise... parallelize this.
            q_samples, q_samplers = self.infer(xin, S=S)

        else:
            q_sample_out, q_sampler_out = self.infer(xin)
            p_sample_out, p_sampler_out = self.generate(q_sample_out[-1])
            return xin,\
                q_sampler_out[0],\
                q_sample_out[0],\
                p_sampler_out[0],\
                p_sample_out[0],\
                None


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

    dbn = DBN(784, [200])
    # pdb.set_trace()
    optimizer = optim.SGD(dbn.parameters(), lr=0.1)

    for epoch in range(3):
        losses = []
        for _, (data, target) in enumerate(train_loader):
            data = Variable(data.view(-1, 784))  # visible
            # pdb.set_trace()
            data_sample, z, z_sample, xp, xp_sample, loss = dbn(
                data,
                compute_loss=True,
                S=5
                )

            # loss = -elbo
            # pdb.set_trace()
            losses.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch', epoch, 'loss=', np.mean(losses))

    display("real", make_grid(data_sample.view(-1, 1, 28, 28).data))
    display("generate", make_grid(xp_sample[-1].view(-1, 1, 28, 28).data))
    plt.show()

if __name__ == '__main__':
    sanity_test()
