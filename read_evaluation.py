
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
from utils import make_dot, display, smooth_distribution, EPS, sample_range, \
    compute_elbo_sampled_batched, glorot_init

from utils import save_checkpoint, load_checkpoint, every

from dbn import DBN


torch.manual_seed(1337)

MODELS_DIR = 'models/'

filename = 'models/1.28.2018.data_losses.pickle'
if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    with open(filename, 'rb') as f:
        losses = pickle.load(f)
    pdb.set_trace()
    vll = losses['vanilla']
    gll = losses['greedy']
    rll = losses['random']
    vl = [t[0] for t in vll]
    gl = [t[0] for t in gll]
    rl = [sum(t) / len(t) for t in rll]
    v = sum(vl) / len(vl)
    g = sum(gl) / len(gl)
    r = sum(rl) / len(rl)
    print('vanilla:', v.data)
    print('greedy:', g.data)
    print('random:', r.data)
