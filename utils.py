# from graphviz import Digraph
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
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
EPS = 10**-6


def sample_range(denom):
    return np.sqrt(6 / denom)




def smooth_distribution(d, eps=EPS):
    '''
        Used to convert probability distributions from range 0,1
            to eps, 1 - eps
        Takes a vector in range [lo, hi]
        and converts it into range [eps + lo*(1 - 2*eps), eps + hi*(1 - 2*eps)]
    '''
    return d * (1 - 2*eps) + eps


def binary_row_reduce(A, b):
    '''
        Performs gaussian-jordan to get row reduced form
        ARGS
            A: (k, n) binary FloatTensor
            b: (k) binary FloatTensor
        RETURNS
            C = [I_k | A']: (k, n) binary FloatTensor
            bp: (k) binary FloatTensor

            Courtesy: http://www.csun.edu/~panferov/math262/262_rref.pdf
    '''
    k, n = A.size()[0], A.size()[1]
    i = 0
    j = 0
    C = A.clone()
    bp = b.clone()
    # pdb.set_trace()
    while (i < k) and (j < n):
        # Pick pivot
        if C[i, j] == 0:
            # Find row with non zero element
            swap_i = None
            for ip in range(i + 1, k):
                if C[ip, j] == 0:
                    continue
                else:
                    swap_i = ip
                    break
            if swap_i is None:
                j += 1
                continue  # Outer loop. Entire column is zero.
            else:
                # Swap i, swap_i
                C[[i, swap_i]] = C[[swap_i, i]]
                bp[[i, swap_i]] = bp[[swap_i, i]]
                # Dafuq pytorch.
                # indices = list(range(k))
                # indices[i], indices[swap_i] = swap_i, i
                # C[indices] = C  # Swapped!
                # bp[indices] = bp  # Swapped!
        # Normalize pivot - unnecessary because binary :)
        # assert(C[i, j] == 1)
        # Elimination - preserve binary nature
        for ip in range(k):
            if ip == i:
                continue
            if C[ip, j] == 0:
                continue
            else:
                C[ip] = (C[ip] - C[i]) % 2
                b[ip] = (b[ip] - b[i]) % 2
        # Onwards!
        i += 1
        j += 1
    return C, bp


def weighted_average_logsumexp(logq, logp):
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
        [ \Sigma q * log(p) ]/ [ \Sigma q ]
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


def compute_elbo_sampled_batched(logp, logq):
    '''
        Numerically stable computation of
        \Sigma_batches ((\Sigma_i q[i] log(p[i]) - log(q[i])) / (\Sigma_i q[i]))
    '''
    (S, bs) = list(logq.size())
    logq_max = logq.max(dim=0)[0]  # 1, bs
    logq_max_expanded = logq_max.expand(S, bs)  # S, bs
    inner_weight = (logq - logq_max_expanded).exp()

    # Thr following are 1, bs
    term1_num = logq_max + torch.log((inner_weight * (-logp)).sum(dim=0))
    term2_num = logq_max + torch.log((inner_weight * (-logq)).sum(dim=0))
    # num = (term1_num - term2_num)  # UNSTABLE
    den = logq_max + torch.log(inner_weight.sum(dim=0))
    return -1 * (term1_num - den).exp().sum() + (term2_num - den).exp().sum()



def glorot_init(param):
    '''
        ARGS
        ----
            param: receives a paramater of some size, initialized unformly in [0, 1]
    '''
    denom = sum(list(param.size()))
    r = np.sqrt(6 / denom)
    return param * 2 * r - r


'''
    For the next 2 functions, checkpoints are states of model, optimizer essentially
'''

def save_checkpoint(state, filename, is_best=False, best_filename='models/best.pytorch.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

def load_checkpoint(filename):
    return torch.load(filename)


def every(epoch, stride=5, start=-1):
    '''
        Function is true for every stideth epoch
    '''
    if start == -1:
        start = stride - 1
    return epoch % stride == start

'''
    The following code has problems.
'''


def _incorrect_compute_elbo(logp, logq):
    '''
        Compute \Sigma q * [logp - logq] / \Sigma q
    '''
    logq_max = logq.max().expand(logq.size())
    num_te1 = (logq_max +
               torch.log((logq - logq_max).exp() * (-logp))).sum(dim=0)
    num_te2 = (logq_max +
               torch.log((logq - logq_max).exp() * (-logq))).sum(dim=0)
    neg_num = num_te1 - num_te2
    pos_den = (logq_max +
               torch.log((logq - logq_max).exp())).sum(dim=0)
    return -1 * (neg_num - pos_den).exp().sum()
