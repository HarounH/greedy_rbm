'''
    experiment for SBN with multiple layers
'''

__author__ = 'haroun habeeb'
__mail__ = 'haroun7@gmail.com'

import argparse
import pickle
import os
import pdb
import numpy as np
from scipy.io import loadmat
# import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Bernoulli


from utils import display, smooth_distribution, EPS, sample_range, \
    compute_elbo_sampled_batched, glorot_init

from utils import save_checkpoint, load_checkpoint, every

from dbn2 import DBN


torch.manual_seed(1337)

MODELS_DIR = 'models/'
date_str = '1.28.2018'


class CaltechDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        '''
            images: Tensor of size N, nx
        '''
        self.images = images.float()
        self.n, self.nx = images.size()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # pdb.set_trace()
        return self.images[idx], 0

def get_mnist_data(batch_size, loc):
    '''
        RETURN
        ----
            three dataloaders:
            Training
            validation
            testing
    '''
    print('Loading MNIST data')
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(loc, train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(loc,
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size)
    return train_loader, None, test_loader


def get_caltech101_data(batch_size, loc='caltech101/'):
    '''
        RETURN
        ----
            three dataloaders:
            Training
            validation
            testing
    '''
    print('Loading caltech data')
    mat = loadmat(loc + 'caltech101_silhouettes_28_split1.mat')
    train = torch.from_numpy(mat['train_data'])  # numpy array
    val = torch.from_numpy(mat['val_data'])  # numpy array
    test = torch.from_numpy(mat['test_data'])  # numpy array
    train_set = CaltechDataset(train)
    val_set = CaltechDataset(val)
    test_set = CaltechDataset(test)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader

def get_nips_data(batch_size, loc='nips_data/'):
    '''
        RETURN
        ----
            three dataloaders:
            Training
            validation
            testing
    '''
    print('Loading NIPS data')
    mat = loadmat(loc + 'nips_1-17.mat')
    pdb.set_trace()
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid, save_image
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',
                        type=bool,
                        default=False)
    parser.add_argument('-r', '--resume',
                        type=bool,
                        default=False)
    parser.add_argument('-f', '--checkpoint_file',
                        type=str,
                        default='')
    parser.add_argument('--test',
                        type=bool,
                        default=False)
    parser.add_argument('--n_epochs',
                        type=int,
                        default=50)
    parser.add_argument('--mode',
                        type=str,
                        default='vanilla')
    parser.add_argument('--dataset',
                        type=str,
                        choices=['mnist', 'caltech', 'nips_data'],
                        default='mnist')
    parser.add_argument('--model_folder',
                        type=str,
                        default='models/')
    args = parser.parse_args()
    MODELS_DIR = args.model_folder
    # Make the directory if it doesn't exist.
    if not os.path.exists(os.path.dirname(MODELS_DIR)):
        # pdb.set_trace()
        os.mkdir(os.path.dirname(MODELS_DIR))

    # pdb.set_trace()
    # Data loading
    batch_size = 20
    if args.dataset == 'mnist':
        train_loader, val_loader, test_loader = get_mnist_data(batch_size, 'MNIST_data/')
        nx = 784
        nz = 200
    elif args.dataset == 'caltech':
        train_loader, val_loader, test_loader = get_caltech101_data(batch_size, 'caltech101/')
        nx = 28 * 28
        nz = 200
    elif args.dataset == 'nips_data':
        train_loader, val_loader, test_loader = get_nips_data(batch_size, 'caltech101/')
        nx = -1
        nz = -1
        raise NotImplementedError
    else:
        raise NotImplementedError

    dbn = DBN(nx, [nz])
    dbn.mode = args.mode
    param_groups = [{'params': dbn.q_parameters(), 'lr': 0.6e-4},
                    {'params': dbn.p_parameters(), 'lr': 3e-4}]
    optimizer = optim.Adam(param_groups)
    if args.resume:
        # Load model etc
        checkpoint = load_checkpoint(args.checkpoint_file)
        start_epoch = checkpoint['epoch']
        dbn.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Model loaded')
    else:
        start_epoch = 0

    if args.train is True:
        print('Training')
        # Training loop
        for epoch in range(start_epoch, start_epoch + args.n_epochs):
            losses = []
            for _, (data, target) in enumerate(train_loader):
                data = Variable(data.view(-1, 784))  # visible
                data_sample, q, q_sample, p, p_sample, loss = dbn(
                    data,
                    compute_loss=True
                    )

                # loss = -elbo
                losses.append(loss.data[0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('epoch', epoch, 'loss=', np.mean(losses))
            if every(epoch, stride=5):
                save_checkpoint({'epoch': epoch,
                                 'model': dbn.state_dict(),
                                 'optimizer': optimizer.state_dict()},
                                os.path.join(MODELS_DIR,
                                             date_str + '.' + dbn.mode + '.' + str(epoch) + '.pytorch.tar'))
                print('Saved model on epoch', epoch)

    if args.test is True:
        print('Started testing')
        vanilla_elbos = []
        random_elbos = []
        greedy_elbos = []
        for mbi, (data, target) in enumerate(test_loader):
            print('mb ', mbi)
            data = Variable(data.view(-1, 784))  # visible
            # Get samples from all 3 modes
            dbn.mode = 'vanilla'
            S=20
            data_sample, _, q_sample_vanilla, _, p_sample_vanilla, _ = dbn(
                data,
                compute_loss=False,
                S=20
                )
            # print('v', end='')
            dbn.mode = 'greedy'
            data_sample, _, q_sample_greedy, _, p_sample_greedy, _ = dbn(
                data,
                compute_loss=False,
                S=20
                )
            # print('g', end='')
            dbn.mode = 'random'
            data_sample, _, q_sample_T_random, _, p_sample_T_random, _ = dbn(
                data,
                compute_loss=False,
                S=20
                )
            # NOTE Set the end of the sample to data input.
            p_sample_vanilla[-1] = data_sample.expand(S, *data_sample.size())
            p_sample_greedy[-1] = data_sample.expand(S, *data_sample.size())
            for p_sample_random in p_sample_T_random:
                p_sample_random[-1] = data_sample.expand(S, *data_sample.size())
            # print('r', end='')
            dbn.mode = 'vanilla'
            # Get the ELBO for those samples using vanilla paramaters
            vanilla_elbos.append(dbn.evaluate_sample([q_sample_vanilla], [p_sample_vanilla]))
            # print(' vl', end='')
            greedy_elbos.append(dbn.evaluate_sample([q_sample_greedy], [p_sample_greedy]))
            # print(' gl', end='')
            random_elbos.append(dbn.evaluate_sample(q_sample_T_random, p_sample_T_random))
            # print(' rl', end='')
            # print(vanilla_elbos[-1], greedy_elbos[-1], random_elbos[-1])
        # pdb.set_trace()
        all_elbos = {'vanilla': vanilla_elbos, 'greedy': greedy_elbos, 'random': random_elbos}
        with open(MODELS_DIR + date_str + '.elbos.pickle', 'wb') as f:
            pickle.dump(all_elbos, f)
