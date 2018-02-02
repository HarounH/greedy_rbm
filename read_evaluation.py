
__author__ = 'haroun habeeb'
__mail__ = 'haroun7@gmail.com'

import argparse
import pickle
import os
import sys
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
from utils import smooth_distribution, EPS, sample_range, \
    compute_elbo_sampled_batched, glorot_init

from utils import save_checkpoint, load_checkpoint, every
from statistics import median
from visualize import make_dot, display
from dbn import DBN


torch.manual_seed(1337)

MODELS_DIR = 'models/'

filename = 'models/1.28.2018.data_losses.pickle'


def avg(ls):
    return sum(ls) / len(ls)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    with open(filename, 'rb') as f:
        losses = pickle.load(f)
    vll = losses['vanilla']
    gll = losses['greedy']
    rll = losses['random']
    vl = [t[0].data[0] for t in vll]
    gl = [t[0].data[0] for t in gll]
    rl = [median([ti.data[0] for ti in t]) for t in rll]
    v = avg(vl)
    g = avg(gl)
    r = avg(rl)
    print('vanilla:', v)
    print('greedy:', g)
    print('random:', r)
