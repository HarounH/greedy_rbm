
__author__ = 'haroun habeeb'
__mail__ = 'haroun7@gmail.com'

import argparse
import pickle
import os
import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
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
from dbn2 import DBN


torch.manual_seed(1337)


latex_output_file = 'latex_output_file.tex'

def avg(ls):
    return sum(ls) / len(ls)


base_dirs = ['eval_models/models/']
datasets = ['mnist', 'caltech', 'nips_data']
fileextension_dict = {'mnist':'lkl.pickle', 'nips_data':'perp.pickle', 'caltech':'lkl.pickle'}
modes = ['vanilla', 'greedy', 'random']
date_folders = ['jan30']
ks = ['5', '10', '20']
Ts = ['3']
epochs = ['49']
datestrs = ['1.28.2018']
results = {}
for dataset in datasets:
    results[dataset] = {}
    result = results[dataset]
    for mode in modes:  # Training mode.
        result[mode] = {}
        for date_folder in date_folders:
            for datestr in datestrs:
                for k in ks:
                    result[mode][k] = {}
                    for T in Ts:
                        result[mode][k][T] = {}
                        for epoch in epochs:
                            result[mode][k][T][epoch] = {}
                            filename = '.'.join([datestr, mode, k, T, epoch, fileextension_dict[dataset]])
                            filepath = os.path.join(base_dirs[0], dataset, mode, date_folder, filename)
                            with open(filepath, 'rb') as f:
                                losses = pickle.load(f)
                            for key in losses.keys():
                                ll = losses[key]
                                lldata = avg([median([ti.data[0] for ti in t]) for t in ll])
                                result[mode][k][T][epoch][key] = lldata

with open(latex_output_file, 'w') as f:
    headers = ['training method'] + ['test:' + mode for mode in modes]
    for date_folder in date_folders:
        for datestr in datestrs:
            for dataset in datasets:
                for k in ks:
                    for T in Ts:
                        for epoch in epochs:
                            table = []
                            for mode in modes:
                                data = [-results[dataset][mode][k][T][epoch][test_mode] for test_mode in modes]
                                table.append([mode] + data)
                            table_str = tabulate(table, headers, tablefmt='latex')
                            intro = 'Dataset:' + dataset + '\\\\\n\tk:' + k + '\\\\\n\tT:' + T + '\\\\\n\n'
                            f.write(intro + table_str + '\n\n--------------------\\\\\n\n')
pdb.set_trace()
