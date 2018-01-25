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
    compute_elbo_sampled_batched, glorot_init, binary_row_reduce


torch.manual_seed(1337)


class DBN(nn.Module):
    '''
        Notation:
        q (inference) : x -> z^0 -> z^1 ... z^{L-1}

        p (generate): z^{L-1} -> z^{L-2} -> z^{L-3} ... z^0 -> x

        Layer sizes are provided in z^0, z^1 ... z^{L-1} order.
            nzs[l] is size of layer l

        Parameters:

            W^l for l in range 0, 1, 2, ... L-1.
            bq^l for l in range 0, 1, ... L-1
            bp^l for l in range 0, 1, 2... L-1 and kept in that order.

            Inference will use a normal loop,
            Generate will use a reverse loop

        Equations and sizes:
            q(z^0 | x) = sigmoid(x * W^0 + bq^0)
                = Sigmoid(Linear(x, W^0.t(), bq^0))
                Hence, W^0 has size: nx, nzs[0]
                    bq^0 has size: nzs[0]

            for l \in [1 ... L)
            q(z^l | z^{l-1}) = Sigmoid(z^{l-1} * W^l + bq^l)
                W^l has size nzs[l - 1], nzs[l]
                bq^l has size nzs[l]

            p(z^{L-1}) = sigmoid(bq^{L-1})
            p(x | z^0) = Sigmoid(z^0 * W^0.t() + bp^0)
                =Sigmoid(Linear(z^0, W^0, bp^0))
                Hence, bp^0 has size nx
            p(z^{l-1} | z^l) = Sigmoid(z^l * W^{l}.t() + bp^l)
                bp^l has size nzs[l - 1]
    '''
    eps = EPS

    def __init__(self, nx, nzs, mode='vanilla'):
        super(DBN, self).__init__()
        self.mode = mode
        self.L = len(nzs)
        self.nx = nx
        self.nzs = nzs
        self.W = nn.ParameterList([nn.Parameter(glorot_init(torch.rand(nx, nzs[0])))])
        self.bp = nn.ParameterList([nn.Parameter(glorot_init(torch.rand(nx)))])
        self.bq = nn.ParameterList([nn.Parameter(glorot_init(torch.rand(nzs[0])))])
        for l in range(1, self.L):
            self.W.append(nn.Parameter(glorot_init(torch.rand(nzs[l-1], nzs[l]))))
            self.bp.append(nn.Parameter(glorot_init(torch.rand(nzs[l-1]))))
            self.bq.append(nn.Parameter(glorot_init(torch.rand(nzs[l]))))

    def generate(self, z_Lm1, S=1, smooth_eps=eps, n_constraints=25, T=3):
        if self.mode == 'greedy':
            return self.greedy_generate(z_Lm1,
                                        S=S,
                                        smooth_eps=smooth_eps,
                                        k=self.nzs[0] - n_constraints)
        elif self.mode == 'random':
            return self.random_generate(z_Lm1,
                                        S=S,
                                        smooth_eps=smooth_eps,
                                        k=n_constraints,
                                        T=T)
        elif self.mode == 'vanilla':
            return self.vanilla_generate(z_Lm1, S=S, smooth_eps=smooth_eps)

    def vanilla_generate(self, z_Lm1, S=1, smooth_eps=eps):
        '''
            ARGS
            ----
                z_Lm1: S * bs * nzs[L - 1]
            RETURNS
            ----
                samples: FloatTensor list. Each element is S * bs * nzs[i]
                    length is self.L + 1  (+1 for x)
                samplers: Bernoulli list. Each element is S * bs * nzs[i]
                    length is self.L + 1  (+1 for x)
                The order is z_{L-1}... z_0... x
        '''
        bs = z_Lm1.size()[1]
        if z_Lm1.size()[0] != S:
            print('Invalid size for z_Lm1 along dim=0')
            exit()
        samples = [z_Lm1]
        samplers = [Bernoulli(
            smooth_distribution(F.sigmoid(self.bq[-1]).expand(S, bs, -1),
                                eps=smooth_eps))
                    ]
        for l in range(self.L-1, -1, -1):  # z^l is GIVEN... includes z^0 given
            # Need to generate z_{l-1} given z_l
            p = smooth_distribution(
                F.sigmoid(F.linear(samples[-1], self.W[l], self.bp[l])),
                eps=smooth_eps
            )
            samplers.append(Bernoulli(p))
            samples.append(samplers[-1].sample())
        return samples, samplers

    def random_generate(self, z_Lm1, S=1, smooth_eps=eps, k=5, T=3):
        '''
            Generates z^{L-2}...z^0, x using given paramaters and random projections.
            Only implemented for 1 layer networks...
                probabilities are a little messed up otherwise.
            ARGS
            ----
                z_Lm1 : Muxed - either a single FloatTensor of S, bs, nz
                    OR a list of length T, of FloatTensor
            RETURNS
            ----
                samples_T: FloatTensor list list. 1st dimension is along t. 2nd is along layer.
                samplers_T: Bernoulli list list. 1st dim is along t, 2nd along layer.
                Cs: FloatTensor list: Random projections used, indexed by t
                bps: FloatTensor list: Random projections used, indexed by t
        '''

        if self.L != 1:
            raise NotImplementedError('Random projections only implemented for single layer')
        Cs = []  # Will be length T
        bps = []  # Will be length T
        if isinstance(z_Lm1, list):
            bs = z_Lm1[0][0].size()[1]
            if z_Lm1[0][0].size()[0] != S:
                print('Invalid size for z_Lm1 along dim=0')
            samples_T = z_Lm1
        else:
            bs = z_Lm1[0].size()[1]
            if z_Lm1[0].size()[0] != S:
                print('Invalid size for z_Lm1 along dim=0')
            samples_T = [[z_Lm1] for t in range(T)]
        samplers_T = [
            [
                Bernoulli(
                    smooth_distribution(
                        F.sigmoid(self.bq[-1]).expand(S, bs, -1),
                        eps=smooth_eps
                        )
                    )
                ] for t in range(T)
            ]  # Annoying PEP8 standards.

        for t in range(T):
            sampler = samplers_T[t]
            sample = samples_T[t]
            sampler_A = Bernoulli(0.5 * torch.ones(k, self.nzs[0]))
            A = sampler_A.sample()
            sampler_b = Bernoulli(0.5 * torch.ones(k))
            b = sampler_b.sample()
            C, bp = binary_row_reduce(A, b)
            C = Variable(C)
            bp = Variable(bp)
            # Imposing constraints
            sample[0][:, :, :k] = \
                (sample[0][:, :, k:].matmul(C[:, k:].t()) + bp) % 2
            # Building sampler (for x)
            p = smooth_distribution(
                F.sigmoid(F.linear(sample[0], self.W[0], self.bp[0])),
                eps=smooth_eps
            )
            sampler.append(Bernoulli(p))
            sample.append(sampler[-1].sample())
            Cs.append(C)
            bps.append(bp)

            # Enforce on z_Lm1
        return samples_T, samplers_T, Cs, bps

    def greedy_generate(self, z_Lm1, S=1, smooth_eps=eps, k=195):
        '''
            ARGS
            ----
                z_Lm1: S * bs *nzs[-1]
                S
                smooth_eps
                k: # of free variables

            RETURNS
            ----
                samples: FloatTensor list. list along layer.
                samplers: Bernoulli list. list along layer.
                Z_f_indices:
                Z_f_c_indices:
                constants: nz sized tensor telling us what constant was considered for each position
        '''
        if self.L != 1:
            raise NotImplementedError('Random projections only implemented for single layer')
        bs = z_Lm1.size()[1]
        assert(S == z_Lm1.size()[0])  # Remove this ?
        samples = [z_Lm1]
        samplers = [Bernoulli(
            smooth_distribution(F.sigmoid(self.bq[-1]).expand(S, bs, -1),
                                eps=smooth_eps))
                    ]

        # Constrain z_Lm1
        constants = self.assigned_constants(samplers[0].probs[0, 0, :])
        Z_f_indices, Z_f_c_indices = self.get_k_free_latent_variables(samplers[0], k=k, constants=constants)
        # pdb.set_trace()  # : See if this works
        samples[0][:, :, Z_f_c_indices] = constants[Z_f_c_indices].expand(samples[0][:, :, Z_f_c_indices].size())
        p = smooth_distribution(
            F.sigmoid(F.linear(samples[0], self.W[0], self.bp[0])),
            eps=smooth_eps
        )
        samplers.append(Bernoulli(p))
        samples.append(samplers[-1].sample())
        return samples, samplers, Z_f_indices, Z_f_c_indices, constants

    def assigned_constants(self, paramaters):
        return (paramaters > 0.5).float()

    def get_k_free_latent_variables(self, sampler, k=195, constants=None):
        '''
            ARGS
            ----
                sampler: Bernoulli(t) where t is S, bs, nz tensor
                k: # of free variables
                constants: nz
            Returns
                Z_f_indices
                Z_f_c_indices
        '''
        nz = sampler.probs.size()[2]
        assert(nz == constants.size()[0])
        log_probability = sampler.log_prob(constants.expand(sampler.probs.size()))  # S, bs, -1
        Z_f_c_indices = log_probability[0, 0, :].topk(nz - k)[1].data.tolist()
        Z_f_indices = list(set(range(nz)) - set(Z_f_c_indices))
        return Z_f_indices, Z_f_c_indices

    def inference(self, xin, S=1, smooth_eps=eps, n_constraints=25, T=3):
        if self.mode == 'greedy':
            return self.greedy_inference(xin,
                                         S=S,
                                         smooth_eps=smooth_eps,
                                         k=self.nzs[0]-n_constraints)
        elif self.mode == 'random':
            return self.random_inference(xin,
                                         S=S,
                                         smooth_eps=smooth_eps,
                                         k=n_constraints,
                                         T=T)
        elif self.mode == 'vanilla':
            return self.vanilla_inference(xin, S=S, smooth_eps=smooth_eps)

    def vanilla_inference(self, xin, S=1, smooth_eps=eps):
        '''
            ARGS
            ----
            xin: bs, nx sized FloatTensor

            RETURNS
            ----
                samples: FloatTensor list. Each element is S * bs * nzs[i]
                    length is self.L + 1 (+1 is for x)
                samplers: Bernoulli list. Each element is S * bs * nzs[i]
                    length is self.L + 1  (+1 for x, but is None)
                The order is x, z^0, z^1 ... z^{L-1}
        '''
        samples = [xin.expand(S, *(xin.size()))]
        samplers = [None]  # Welp
        for l in range(self.L):  # l is being inferred
            q = smooth_distribution(
                F.sigmoid(F.linear(samples[-1], self.W[l].t(), self.bq[l])),
                eps=smooth_eps
            )
            samplers.append(Bernoulli(q))
            samples.append(samplers[-1].sample())
        return samples, samplers

    def random_inference(self, xin, S=5, smooth_eps=eps, k=5, T=3):
        '''
        Infer z^0 given xin_dist, from which samples are drawn.
        Random projections are used.
        Only works for single layer networks because
        ARGS
        ----
        xin: bs * nx sized FloatTensor. It should ideally be binary.
        S=5: Number of samples
        k: number of constraints
        T: Number of projections to be done
        RETURNS
        ----
            samples_T: FloatTensor list list.
                1st dimension is along t. 2nd is along layer.
            samplers_T: Bernoulli list list.
                1st dim is along t, 2nd along layer.
            Cs: FloatTensor list: Random projections used, indexed by t
            bps: FloatTensor list: Random projections used, indexed by t

            The order is x, z^0, z^1 ... z^{L-1}
        '''
        if self.L > 1:
            raise NotImplementedError('projections havent been \
                                      implemented for >1 hidden layer')
        samples_T = [[xin.expand(S, *xin.size())] for t in range(T)]
        samplers_T = [[None] for t in range(T)]
        Cs = []
        bps = []
        for t in range(T):
            sample = samples_T[t]
            sampler = samplers_T[t]
            sampler_A = Bernoulli(0.5 * torch.ones(k, self.nzs[0]))
            A = sampler_A.sample()
            sampler_b = Bernoulli(0.5 * torch.ones(k))
            b = sampler_b.sample()
            C, bp = binary_row_reduce(A, b)
            C = Variable(C)
            bp = Variable(bp)
            # Sample z from q and then impose constraints
            q = smooth_distribution(
                F.sigmoid(F.linear(sample[-1], self.W[0].t(), self.bq[0])),
                eps=smooth_eps
            )
            sampler.append(Bernoulli(q))
            new_sample = sampler[-1].sample()  # S, bs, nz
            new_sample[:, :, :k] = \
                (new_sample[:, :, k:].matmul(C[:, k:].t()) + bp) % 2
            sample.append(new_sample)
            Cs.append(C)
            bps.append(bp)
        return samples_T, samplers_T, Cs, bps

    def greedy_inference(self, xin, S=5, smooth_eps=eps, k=195):
        '''
            RETURNs
            ----
                samples: FloatTensor list, per layer
                samplers: Bernoulli list, per layer
                Z_f_indices: free latent variables
                Z_f_c_indices: not free latent variables
        '''
        if self.L > 1:
            raise NotImplementedError('projections not implemented for L>1')
        samples = [xin.expand(S, *xin.size())]
        samplers = [None]
        q = smooth_distribution(
            F.sigmoid(F.linear(samples[0], self.W[0].t(), self.bq[0])),
            eps=smooth_eps
        )
        samplers.append(Bernoulli(q))
        new_sample = samplers[-1].sample()
        constants = self.assigned_constants(q[0, :, :].mean(dim=0))
        Z_f_indices, Z_f_c_indices = self.get_k_free_latent_variables(samplers[-1], k=k, constants=constants)
        new_sample[:, :, Z_f_c_indices] = constants[Z_f_c_indices].expand(new_sample[:, :, Z_f_c_indices].size())
        samples.append(new_sample)
        return samples, samplers, Z_f_indices, Z_f_c_indices, constants

    def forward(self, xin_dist, compute_loss=True, S=5, n_constraints=5, T=3, aggregate_fn=None):
        if self.mode == 'vanilla':
            return self.vanilla_forward(xin_dist,
                                        compute_loss=compute_loss,
                                        S=S)
        elif self.mode == 'random':
            if aggregate_fn is None:
                def aggregate_fn(ls):
                    return sum(ls) / len(ls)
            return self.random_forward(xin_dist,
                                       compute_loss=compute_loss,
                                       S=S,
                                       k=n_constraints,
                                       T=T)
        elif self.mode == 'greedy':
            return self.greedy_forward(xin_dist,
                                       compute_loss=compute_loss,
                                       S=S,
                                       k=self.nzs[0] - n_constraints)
        else:
            raise NotImplementedError

    def vanilla_forward(self, xin_dist, compute_loss=True, S=5):
        # We'll have samples anyway
        xin = xin_dist.bernoulli()  # Fix xin: bs, nx
        q_samples, q_samplers = self.inference(xin, S)
        p_samples, p_samplers = self.generate(q_samples[-1], S)
        if compute_loss:
            # Construct logq first
            # (S sized FloatTensor) list
            logq = [q_samplers[l].log_prob(q_samples[l]).sum(dim=2)
                    for l in range(1, 1 + self.L)]
            logq = sum(logq)
            # construct logp
            # (S sized FloatTensor) list
            logp = [p_samplers[l].log_prob(p_samples[l]).sum(dim=2)
                    for l in range(self.L)]
            logp.append(p_samplers[-1].log_prob(
                xin.expand(
                    S,
                    *xin.size()
                    )
                ).sum(dim=2))
            logp = sum(logp)
            # Now, use logp, logq to compute elbo
            loss = -compute_elbo_sampled_batched(logp, logq)
        else:
            loss = None
        return xin, q_samplers, q_samples, p_samplers, p_samples, loss

    def random_forward(self, xin_dist, compute_loss=True,
                       aggregate_fn=None, S=5, k=5, T=3):
        '''
            Performs random projections and provides a loss
            computed using random projections
        '''
        if self.L > 1:
            raise NotImplementedError('Projections not implemented for > 1 layer')
        xin = xin_dist.bernoulli()  # no gradient through this
        q_samples_T, q_samplers_T, _, _ = self.inference(xin,
                                                         S,
                                                         n_constraints=k,
                                                         T=T)
        p_samples_T, p_samplers_T, _, _ = self.generate([[temp[-1]] for temp in q_samples_T],
                                                        S,
                                                        n_constraints=k,
                                                        T=T)
        if compute_loss:
            if aggregate_fn is None:
                def aggregate_fn(ls):
                    return sum(ls) / len(ls)
            projected_elbos = []
            for t in range(T):
                # Compute projected loss basically
                q_samples = q_samples_T[t]
                q_samplers = q_samplers_T[t]
                p_samples = p_samples_T[t]
                p_samplers = p_samplers_T[t]
                # : Compute loss.
                logq = sum([q_samplers[l].log_prob(q_samples[l]).sum(dim=2)
                            for l in range(1, 1 + self.L)])
                logp = [p_samplers[l].log_prob(p_samples[l]).sum(dim=2)
                        for l in range(self.L)]
                logp.append(
                    p_samplers[-1].log_prob(
                        xin.expand(S, *xin.size())
                        ).sum(dim=2)
                    )
                logp = sum(logp)
                projected_elbos.append(compute_elbo_sampled_batched(logp, logq))

            # : Aggregate loss
            elbo = aggregate_fn(projected_elbos)
            loss = -elbo
        else:
            loss = None
        #  Return correct stuff
        return xin, q_samplers_T, q_samples_T, p_samplers_T, p_samples_T, loss

    def greedy_forward(self, xin_dist, compute_loss=True, S=5, k=195):
        '''
            k: # of free variables
        '''
        xin = xin_dist.bernoulli()
        q_samples, q_samplers, q_z_f, q_z_f_c, q_constants = \
            self.inference(xin, S=S, n_constraints=self.nzs[0] - k)
        p_samples, p_samplers, p_z_f, p_z_f_c, p_constants = \
            self.generate(q_samples[-1],
                          S=S,
                          n_constraints=self.nzs[0] - k)
        if compute_loss:
            # Construct logq first
            # (S sized FloatTensor) list
            logq = [q_samplers[l].log_prob(q_samples[l]).sum(dim=2)
                    for l in range(1, 1 + self.L)]
            logq = sum(logq)
            # construct logp
            # (S sized FloatTensor) list
            logp = [p_samplers[l].log_prob(p_samples[l]).sum(dim=2)
                    for l in range(self.L)]
            logp.append(p_samplers[-1].log_prob(
                xin.expand(
                    S,
                    *xin.size()
                    )
                ).sum(dim=2))
            logp = sum(logp)
            # Now, use logp, logq to compute elbo
            elbo = compute_elbo_sampled_batched(logp, logq)
            loss = -elbo
        else:
            loss = None
        return xin, q_samplers, q_samples, p_samplers, p_samples, loss

    def evaluate_sample(self, q_samples_T, p_samples_T, smooth_eps=eps):
        '''
            ARGS
            ----
            q_samples_T: FloatTensor list list
                each float tensor is binary.
                represents  x, z^0, z^2  .... z^Lm1
            p_samples_T: FloatTensor list list
                each float tensor is binary.
                represents z^Lm1, z^Lm2 ... z^0, x

            NOTE: dimension 0 of the lists is intended to be used for time (random projs)
                but the hack for the other two is to simply have a single timestep.

            RETURNS
            ----
            elbos: Variable list
                elbo corresponding to each element pair in q_samples_T, p_samples_T
        '''
        (S, bs, _) = q_samples_T[0][0].size()
        T = len(q_samples_T)
        elbos = []
        for t in range(T):
            q_samples = q_samples_T[t]
            p_samples = p_samples_T[t]
            # Generate samplers, and then the probabilities
            # Evaluate logq
            logq = []
            for l in range(self.L):  # 0, 1, 2, ...Lm1
                q = smooth_distribution(
                    F.sigmoid(F.linear(q_samples[l], self.W[l].t(), self.bq[l])),
                    eps=smooth_eps
                )  # S, bs, nzs[l + 1]
                logq.append(Bernoulli(q).log_prob(q_samples[l + 1]).sum(dim=2))
            logq = sum(logq)

            # Evaluate logp
            logp = []
            p = smooth_distribution(F.sigmoid(self.bq[-1]).expand(S, bs, -1),
                                    eps=smooth_eps)
            logp.append(Bernoulli(p).log_prob(p_samples[0]).sum(dim=2))
            for l in range(1, self.L + 1):
                # Use p_samples[l-1] to generate p
                p = smooth_distribution(
                    F.sigmoid(F.linear(p_samples[l - 1], self.W[self.L - l], self.bp[self.L - l])),
                    eps=smooth_eps
                )
                logp.append(Bernoulli(p).log_prob(p_samples[l]).sum(dim=2))
                # Use p to evaluate log_prob(p_samples[l])
            logp = sum(logp)
            # logq, logp should be S, bs sized FloatTensor
            elbos.append(compute_elbo_sampled_batched(logp, logq))
        return elbos

if __name__ == '__main__':
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

    dbn = DBN(784, [200], mode='greedy')
    optimizer = optim.SGD(dbn.parameters(), lr=0.1)

    for epoch in range(5):
        losses = []
        for _, (data, target) in enumerate(train_loader):
            data = Variable(data.view(-1, 784))  # visible
            data_sample, z, z_sample, xp, xp_sample, loss = dbn(
                data,
                compute_loss=True,
                S=5
                )

            # loss = -elbo
            losses.append(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch', epoch, 'loss=', np.mean(losses))
    display("real", make_grid(data_sample.view(-1, 1, 28, 28).data))
    display("generate", make_grid(xp_sample[-1][0].view(-1, 1, 28, 28).data))
    plt.show()
