
__author__ = 'haroun habeeb'
__mail__ = 'haroun7@gmail.com'

import pdb
import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Bernoulli

from utils import smooth_distribution, EPS, sample_range, \
    compute_elbo_sampled_batched, glorot_init, binary_row_reduce


torch.manual_seed(1337)


class DBN(nn.Module):
    '''
        RBM learnt variationally
    '''
    eps = EPS

    def __init__(self, nx, nz, mode='vanilla', S=30, ncs=25, T=3):
        super(DBN, self).__init__()
        self.mode = mode
        self.nx = nx
        self.nz = nz
        self.W = nn.Parameter(glorot_init(torch.rand(nx, nz)))
        self.V = nn.Parameter(glorot_init(torch.rand(nx, nz)))
        self.bp = nn.Parameter(glorot_init(torch.rand(nx)))
        self.bq = nn.Parameter(glorot_init(torch.rand(nz)))
        self.S = S
        self.k = ncs
        self.T = T
    @property
    def p_parameters(self):
        return [self.W, self.bp]
    @property
    def q_parameters(self):
        return [self.V, self.bq]

    def generate(self, z, eps=eps):
        '''
            z: list of (S * bs * nz) tensors
        '''
        samples = [z, []]
        pz = F.sigmoid(self.bq)
        samplers = [[Bernoulli(smooth_distribution(pz.expand(*z[t].size()))) for t in range(len(z))]]
        samplers.append([])

        sample = samples[-1]
        sampler = samplers[-1]
        for zt in samples[0]:
            S, bs, nz = zt.size()
            p = F.sigmoid(F.linear(zt, self.W, self.bp))
            sampler.append(Bernoulli(smooth_distribution(p.expand(S, bs, self.nx))))
            sample.append(sampler[-1].sample())
        return samples, samplers

    def inference(self, x_in=None, x_dist=None, S=None, k=None, T=None, eps=eps):
        if S is None:
            S = self.S
        if k is None:
            k = self.k
        if T is None:
            T = self.T
        if x_dist is None and x_in is None:
            raise NotImplementedError("Cannot magically imagine a distribution over x yet")
        if x_in is None:
            x_in = x_dist.bernoulli()  # bs, nx
        # x = x_in.expand(S, *x_in.size())
        samples = [[x_in.expand(S, *x_in.size()) for t in range(T)], []]
        samplers = [[None for t in range(T)], []]
        sample = samples[-1]
        sampler = samplers[-1]
        for t in range(T):
            q = F.sigmoid(F.linear(x_in, self.V.t(), self.bq))  # bs, nx
            q_expanded = q.expand(S, *q.size())

            sampler.append(Bernoulli(smooth_distribution(q_expanded)))
            new_sample = sampler[-1].sample()
            new_sample = self.constrain_latent_variables(new_sample, q, k)  # Greedy/random/vanilla is hidden

            sample.append(new_sample)
        return samples, samplers

    def constrain_latent_variables(self, new_sample, q, k):
        '''
            new_sample: S, bs, nz FloatTensor
            q: bs, nz FloatTensor
            k: int, # constraints
        '''
        if self.mode == 'vanilla':
            pass
        elif self.mode == 'greedy':
            S, batch_size, _ = new_sample.size()
            constants = (q > 0.5).float()
            # pdb.set_trace()
            Z_f_c_indices = Bernoulli(q).log_prob(constants).topk(k)[1].data.tolist()
            dim1 = [ ind // k for ind in range(k * batch_size)]
            dim2 = sum(Z_f_c_indices, [])
            new_values = constants[dim1, dim2]
            new_values = new_values.expand(S, *new_values.size())
            new_sample[:, dim1, dim2] = new_values
            # Commented code is inefficient.
            # pdb.set_trace()
            # unbound = [temp.clone() for temp in torch.unbind(new_sample, dim=1)]
            # new_samples = []
            # unbound_constants = [temp.clone() for temp in torch.unbind(constants, dim=0)]
            # try:
            #     for point_idx in range(len(unbound)):
            #         new_sample_batch = unbound[point_idx]  # nx
            #         point_zfc = Z_f_c_indices[point_idx]
            #         # pdb.set_trace()
            #         new_sample_batch[:, point_zfc] = unbound_constants[point_idx][point_zfc].expand(*new_sample_batch[:, point_zfc].size())
            #         new_samples.append(new_sample_batch)
            #     new_sample = torch.stack(new_samples, dim=1)
            # except:
            #     pdb.set_trace()
        elif self.mode == 'random':
            sampler_A = Bernoulli(0.5 * torch.ones(k, self.nz))
            A = sampler_A.sample()
            sampler_b = Bernoulli(0.5 * torch.ones(k))
            b = sampler_b.sample()
            C, bp = binary_row_reduce(A, b)
            C = Variable(C)
            bp = Variable(bp)
            new_sample[:, :, :k] = \
                (new_sample[:, :, k:].matmul(C[:, k:].t()) + bp) % 2
        return new_sample

    def forward(self, x_in, S=None, k=None, T=None, eps=eps, compute_loss=True):
        if S is None:
            S = self.S
        if k is None:
            k = self.k
        if T is None:
            T = self.T
        q_samples_T, q_samplers_T = self.inference(x_in=x_in, S=S, k=k, T=T, eps=eps)
        p_samples_T, p_samplers_T = self.generate([q_samples_T[1][t] for t in range(T)])
        if compute_loss:
            loss_T = [-self.compute_elbo([q_samples_T[i][t] for i in range(len(q_samples_T))],
                                         [q_samplers_T[i][t] for i in range(len(q_samplers_T))],
                                         [p_samples_T[i][t] for i in range(len(p_samples_T))],
                                         [p_samplers_T[i][t] for i in range(len(p_samplers_T))])
                      for t in range(T)]
            loss = torch.stack(loss_T)
        else:
            loss = None
        return q_samples_T, q_samplers_T, p_samples_T, p_samplers_T, loss

    def compute_elbo(self, q_sample, q_sampler, p_sample, p_sampler):
        '''
            q_sample: [x, z]
            q_sampler: [None, q(z|x)]
            p_sample: [z, x]
            p_sampler: [p(z), p(x|z)]
                All things are S, bs, nx/nz appropriately.

            Use samples to evaluate expectation in elbo
        '''
        logq = q_sampler[1].log_prob(q_sample[1]).sum(dim=2)
        logpz = p_sampler[0].log_prob(p_sample[0]).sum(dim=2)
        logpx = p_sampler[1].log_prob(p_sample[1]).sum(dim=2)
        logpxz = logpz + logpx
        return compute_elbo_sampled_batched(logpxz, logq)


'''
    Metric functions: ELBO, NLL, "perplexity"
'''

def evaluate_elbo(dbn, q_samples_T, p_samples_T):
    T = len(q_samples_T)
    S, bs, _ = q_samples_T[0][0].size()
    nx = dbn.nx
    nz = dbn.nz
    elbos = []
    for t in range(T):
        q_sample = q_samples_T[t]  # x, z
        p_sample = p_samples_T[t]  # z, x
        pz_sampler = Bernoulli(smooth_distribution(F.sigmoid(dbn.bq)).expand(S, bs, nz))
        px_sampler = Bernoulli(smooth_distribution(F.sigmoid(F.linear(p_sample[0], dbn.W, dbn.bp))))
        qz_sampler = Bernoulli(smooth_distribution(F.sigmoid(F.linear(q_sample[0], dbn.V.t(), dbn.bq))))

        logq = qz_sampler.log_prob(q_sample[1]).sum(dim=2)
        logpz = pz_sampler.log_prob(p_sample[0]).sum(dim=2)
        logpx = px_sampler.log_prob(p_sample[1]).sum(dim=2)
        logp = logpx + logpx
        elbos.append(compute_elbo_sampled_batched(logp, logq))
    return torch.stack(elbos).median()



def evaluate_nll(dbn, q_samples_T, p_samples_T):
    T = len(q_samples_T)
    S, bs, _ = q_samples_T[0][0].size()
    nx = dbn.nx
    nz = dbn.nz
    nlls = []
    for t in range(T):
        q_sample = q_samples_T[t]  # x, z
        p_sample = p_samples_T[t]  # z, x
        pz_sampler = Bernoulli(smooth_distribution(F.sigmoid(dbn.bq)).expand(S, bs, nz))
        px_sampler = Bernoulli(smooth_distribution(F.sigmoid(F.linear(p_sample[0], dbn.W, dbn.bp))))
        qz_sampler = Bernoulli(smooth_distribution(F.sigmoid(F.linear(q_sample[0], dbn.V.t(), dbn.bq))))

        logq = qz_sampler.log_prob(q_sample[1]).sum(dim=2)
        logpz = pz_sampler.log_prob(p_sample[0]).sum(dim=2)
        logpx = px_sampler.log_prob(p_sample[1]).sum(dim=2)
        logp = logpx + logpx

        # log(\Sigma_S p(x, z)) - log(\Sigma_S p(z))
        logp_max = logp.max(dim=0)[0]
        logp_max_expanded = logp_max.expand(*logp.size())
        term1 = logp_max +\
            torch.log((logp - logp_max_expanded).exp().sum(dim=0))
        logpz_max = logpz.max(dim=0)[0]
        log_pz_max_expanded = logpz_max.expand(*logpz.size())
        term2 = logpz_max +\
            torch.log((logpz - log_pz_max_expanded).exp().sum(dim=0))
        nlls.append(- term1 + term2)
    return torch.stack(nlls).median()


def evaluate_perplexity(dbn, q_samples_T, p_samples_T):
    T = len(q_samples_T)
    S, bs, _ = q_samples_T[0][0].size()
    nx = dbn.nx
    nz = dbn.nz
    perps = []
    for t in range(T):
        q_sample = q_samples_T[t]  # x, z
        p_sample = p_samples_T[t]  # z, x
        pz_sampler = Bernoulli(smooth_distribution(F.sigmoid(dbn.bq)).expand(S, bs, nz))
        px_sampler = Bernoulli(smooth_distribution(F.sigmoid(F.linear(p_sample[0], dbn.W, dbn.bp))))
        qz_sampler = Bernoulli(smooth_distribution(F.sigmoid(F.linear(q_sample[0], dbn.V.t(), dbn.bq))))
        Li = q_sample[0].sum(dim=1)
        logq = qz_sampler.log_prob(q_sample[1]).sum(dim=2)
        logpz = pz_sampler.log_prob(p_sample[0]).sum(dim=2)
        # IMPORTANT NOTE difference in logpx definition
        logpx = torch.log(px_sampler.probs.pow(p_sample[1])).sum(dim=2)
        logp = logpx + logpx
        logp_max = logp.max(dim=0)[0]
        logp_max_expanded = logp_max.expand(*logp.size())
        term1 = logp_max +\
            torch.log((logp - logp_max_expanded).exp().sum(dim=0))
        logpz_max = logpz.max(dim=0)[0]
        log_pz_max_expanded = logpz_max.expand(*logpz.size())
        term2 = logpz_max +\
            torch.log((logpz - log_pz_max_expanded).exp().sum(dim=0))
        logp_x_no_z = (term1 - term2) / Li
        perps.append(-logp_x_no_z)
    return perps

if __name__ == '__main__':
    dbn = DBN(784, 200)
    pdb.set_trace()
