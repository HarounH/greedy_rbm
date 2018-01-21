from graphviz import Digraph
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
EPS = 10**-6


def sample_range(denom):
    return np.sqrt(6 / denom)


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)),
                         size_to_str(var.size()),
                         fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


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
def display(title, img):
    plt.figure()
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    plt.imshow(npimg)
    plt.title(title)
