'''
  File to make a SBN and a DBN
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

from utils import make_dot

# TODO: set torch seed
torch.manual_seed(1337)
EPS = 10**-6

class SBN(nn.Module):
    '''
        Rough and shoddy implementation of a sigmoid belief network.
        It is a single layer SBN
    '''
    eps = EPS
    def __init__(self, nx, nz):
        super(SBN, self).__init__()  # Dont remember how exactly this works
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

    def sample_from(self, d, ns):
        '''
            d: tensor whose every location represents probability of
                variable at that location being 1
                intended to be a batch_size, n_variables sized FloatTensor
            returns:
                ns, *(list(d.size())) sized vector - note increase in dimensionality
                    intended to be a (ns, batch_size, n_variables) sized tensor
        '''
        sampler = Bernoulli(d)
        samples = sampler.sample_n(ns)
        return samples, sampler
        # return (1 + (torch.sign(sampling_d - e))) / 2
        # return sampling_d.bernoulli()
        # return F.relu(torch.sign(sampling_d - torch.rand(sampling_d.size())))
        # return F.relu(torch.sign(sampling_d - Variable(torch.rand(sampling_d.size()))))

    def x2z(self, x):
        '''
            Inference (q)
            returns Pr(z=1 | x)
        '''
        return F.sigmoid(F.linear(x, self.U, self.z_bias))

    def z2x(self, z):
        '''
            Generator (p)
            return Pr(x=1 | z)
        '''
        return F.sigmoid(F.linear(z, self.V, self.x_bias))

    def forward(self, xin, S=1, compute_loss=True):
        '''
            ARGS
            ----
                xin: (batch_size, nx) sized tensor
                S: int, default=1
                compute_loss: bool, default=True
            RETURNS
            ----
                q,
                samples_z,
                ps,
                xouts,
                full_elbo
        '''
        batch_size = xin.size()[0]

        # Forward passes
        q = self.smooth_distribution(self.x2z(xin))  # batch_size, nz
        samples_z, sampler_z = self.sample_from(q, S)  # S, batch_size, nz
        ps_list = [self.smooth_distribution(self.z2x(sample_z))
                   for sample_z in samples_z]
        ps = torch.stack(ps_list)  # S, batch_size, nx
        xouts = torch.stack([self.sample_from(p, 1)[0][0]
                             for p in ps_list])  # S, batch_size, nx
        if compute_loss:
            # Compute full_elbo to use as (negative) loss
            '''
                ELBO = \Sigma_z q(z)log(p(xin(?), z)) - \Sigma q(z)log(q(z))
            '''
            term2 = (q * torch.log(q) + (1 - q) * torch.log(1 - q)).sum()
            # we need probability of samples to compute term1
            # logq_samples_z = sampler_z.log_prob(samples_z).sum(dim=2)  # S, batch_size
            logq_samples_z = torch.stack([sampler_z.log_prob(sample_z) for sample_z in samples_z]).sum(dim=2)
            #   self.log_bernoulli_probability_of_samples(samples_z, q)
            pz = F.sigmoid(self.z_bias).expand(S, batch_size, -1)  # S, batch_size, nz
            z_prior = Bernoulli(pz)
            # logp_samples_z =
            #   self.log_bernoulli_probability_of_samples(samples_z, pz)
            logp_samples_z = z_prior.log_prob(samples_z).sum(dim=2)  # S, batch_size
            # logp_xin_given_samples_z =
            #   self.log_bernoulli_probability_given_distributions(xin, ps)
            px_given_z = Bernoulli(ps)
            logp_xin_given_samples_z = px_given_z.log_prob(xin.expand(S,
                                                                      batch_size,
                                                                      self.nx)).sum(dim=2)
            logp_xzs = logp_samples_z + logp_xin_given_samples_z  # S, batch_size, 1
            term1 = self.compute_term1(logq_samples_z, logp_xzs)
            # pdb.set_trace()
            full_elbo = term1 - term2
        else:
            full_elbo = None
        return q, samples_z, ps, xouts, full_elbo

    def _deprecated_forward(self, xin):
        '''
            Does one pass through inference network
            and then one pass through generator

            does so stochastically, i.e., samples from intermediate distribution

            distributions are returned for convenience.
        '''
        raise DeprecationWarning
        q = self.x2z(xin)
        sample_z = self.sample_from(q, 1)[0]
        p = self.z2x(sample_z)
        sample_x = self.sample_from(p, 1)[0]

        # Also need to compute full_elbo and return that too
        return q, sample_z, p, sample_x

    def _deprecated_log_bernoulli_probability_of_samples(self, samples, distribution):
        '''
            Returns Pr[samples | distribution]
            ARGS
            ----
                samples: (n_samples, batch_size, n_variables) tensor
                distribution: (batch_size, n_variables) tensor
            RETURNS
            ----
                Pr[samples | distribution] : n_samples, batch_size tensor
        '''
        raise DeprecationWarning
        ns = samples.size()[0]
        per_sample_distribution = distribution.repeat(ns, *([1]*len(distribution.size())))
        # v ~ bernoulli(\Theta) \Rightarrow
        # Pr[v | \Theta] = \Theta^v * (1 - \Theta)^(1 - v)
        # pdb.set_trace()
        return torch.log(per_sample_distribution.pow(samples) *
                         (1 - per_sample_distribution).pow(1 - samples)).sum(dim=2)

    def _deprecated_log_bernoulli_probability_given_distributions(self, sample, distributions):
        '''
            Essentially returns
                [Pr[sample | distribution] for distribution in distributions]
            ARGS
            ----
            sample: (batch_size, nv)
            distributions: (nd, batch_size, nv)

            Returns
            ----
            (nd, batch_size) tensor
        '''
        raise DeprecationWarning
        nd = distributions.size()[0]  # number of distributions
        sample_per_distribution = sample.repeat(nd,
                                                *([1]*len(sample.size())))
        # pdb.set_trace()
        return torch.log(distributions.pow(sample_per_distribution) *
                         (1 - distributions).pow(1 - sample_per_distribution)).sum(dim=2)

    def smooth_distribution(self, d, eps=EPS):
        '''
            Used to convert probability distributions from range 0,1
                to eps, 1 - eps
            Takes a vector in range [lo, hi]
            and converts it into range [eps + lo*(1 - 2*eps), eps + hi*(1 - 2*eps)]
        '''
        return d * (1 - 2*eps) + eps

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

    def _deprecated_full_elbo(self, xin, S=1):
        '''
            For a single data point
            ELBO(v) = \Sigma_z q(z|x)[log(p(z, x)) - log(q(z|x))]
            Second term decomposes nicely because of mean field assumption
                \Sigma_z q(z|x) log(q(z|x)) =
                    \Sigma_i [q(z_i=1|x)log(q(z_i=1|x))
                        + q(z_i=0|x)log(q(z_i=0|x))]

            First term,
                \Sigma_z q(z|x)[log(p(z, x))]
                is not tractable really. Hence we must approximate by sampling.
        '''
        raise DeprecationWarning
        batch_size = xin.size()[0]

        # qzx and second term computation
        qzx = self.x2z(xin)  # batch_size, nz
        qzx_smooth = self.smooth_distribution(qzx)  # batch_size, nz
        # term2 is scalar
        term2 = (qzx_smooth * torch.log(qzx_smooth) +
                 (1 - qzx_smooth) * torch.log(1 - qzx_smooth)).repeat(batch_size, 1).sum()

        zs = self.sample_from(qzx_smooth, S)  # S, batch_size, nz sized
        # ugly line follows :D
        log_qzx_of_zs = self.log_bernoulli_probability_of_samples(zs, qzx_smooth)  # S, batch_size
        pzs = F.sigmoid(self.z_bias).repeat(batch_size, 1)  # batch_size, nz
        pzs_smooth = self.smooth_distribution(pzs)  # batch_size, nz
        log_pz_of_zs = self.log_bernoulli_probability_of_samples(zs, pzs_smooth)  # S, batch_size

        # pytorch doesnt have function application along a dimension, so...
        px_given_zs = torch.stack([
            self.z2x(zsi) for _, zsi in enumerate(torch.unbind(zs, dim=0))
        ], dim=0)  # S, batch_size, nx ... distribution
        pdb.set_trace()
        px_given_zs_smooth = self.smooth_distribution(px_given_zs)  # S, batch_size, nx
        # Use samples, x to compute p(x, z) and
        log_px_given_zs_of_xin = self.log_bernoulli_probability_given_distributions(
            xin,
            px_given_zs_smooth)  # S, batch_size
        log_pxzs = log_px_given_zs_of_xin + log_pz_of_zs  # S, batch_size
        # weighted average them using qzx(z = zs)
        term1 = self.compute_term1(log_qzx_of_zs, log_pxzs)
        # pdb.set_trace()
        return term1 - term2


def display(title, img):
    plt.figure()
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    plt.imshow(npimg)
    plt.title(title)

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

    sbn = SBN(784, 200)
    optimizer = optim.SGD(sbn.parameters(), lr=0.1)

    for epoch in range(4):
        losses = []
        for _, (data, target) in enumerate(train_loader):
            data = Variable(data.view(-1, 784))  # visible
            # pdb.set_trace()
            data_sample = data.bernoulli()  # xin
            z, z_sample, xp, xp_sample, elbo = sbn(data_sample,
                                                   compute_loss=True)

            loss = -elbo
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
