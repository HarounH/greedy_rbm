'''
    Implementation of SBN such that
    we sparsify latent variables
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

from utils import display, make_dot, smooth_distribution, EPS, binary_row_reduce


class SBN_Greedy(nn.Module):
    eps = EPS
        def __init__(self, nx, nz):
            super(SBN_Steroids, self).__init__()
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

        def forward(self, xin, S=1,
                    compute_loss=True,
                    k=25):
            '''
                ARGS
                ----
                    xin: (batch_size, nx) tensor
                    S: # samples
                    compute_loss:
                    k: sparsity hyperparameter for greedy
                RETURNS
                ----
            '''
            batch_size = xin.size()[0]
            q = smooth_distribution(self.x2z(xin))  # batch size, nz
            samples_z, sampler_z = self.sample_from(q, S)  # S, batch_size, nz
            ps = torch.stack([smooth_distribution(self.z2x(sample_z)) for samples_z])
            xouts = torch.stack([self.sample_from(psi, 1)[0][0] for psi in ps])
            if compute_loss:
                # Need to compute which variables are free

                # Get which variables are free
                # binary vector of length nz
                Z_f_indices, Z_f_c_indices = self.get_k_free_latent_variables(k, xin)  # batch_size, nz
                # Compute elbo
                self.set_latent_variables_constants(samples_z, Z_f_c_indices, q.expand(S, batch_size, self.nz))


                logq_zi = torch.stack([sampler_z.log_prob(sample_z) for sample_z in samples_z])  # S, batch_size, nz
                pz = smooth_distribution(F.sigmoid(self.z_bias).expand(S, batch_size, -1))
                z_prior = Bernoulli(pz)
                logp_zi = z_prior.log_prob(samples_z)
                ps_new = torch.stack([smooth_distribution(self.z2x(sample_z)) for sample_z in samples_z])
                px_given_z = Bernoulli(ps_new)
                logp_xin_given_samples_z = px_given_z.log_prob(xin.expand(S, batch_size, self.nx))  # S, batch_size, nx

                # Now to put it all together
                logq = logq_zi.sum(dim=2)
                logp = (logp_zi + logp_xin_given_samples_z).sum(dim=2)
                term2 = self.compute_term1(logq, logq)  # Eh, the names can be fixed later
                term1 = self.compute_term1(logq, logp)
                loss = -(term1 - term2)
            else:
                loss = None
            return q, samples_z, ps, xouts, loss

    def set_latent_variables_constants(self, inp, indices, bernoulli_parameters):
        '''
            inp: S, batch_size, n sized
            sets indices along dim=2 to constants that we need to figure out still
        '''
        inp[:, :, indices] = bernoulli_parameters[:, :, indices] > 0.5
