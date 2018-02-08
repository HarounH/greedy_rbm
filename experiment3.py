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


from utils import smooth_distribution, EPS, sample_range, \
    compute_elbo_sampled_batched, glorot_init

from utils import save_checkpoint, load_checkpoint, every

from dbn3 import DBN, evaluate_perplexity, evaluate_nll, evaluate_elbo


torch.manual_seed(1337)


class CaltechDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        '''
            images: Tensor of size N, nx
        '''
        super(CaltechDataset, self).__init__()
        self.images = images.float()
        self.n, self.nx = images.size()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # pdb.set_trace()
        return self.images[idx], 0


class NIPSDataset(torch.utils.data.Dataset):
    def __init__(self, counts, words=None):
        super(NIPSDataset, self).__init__()
        self.counts = counts# / counts.sum(dim=1, keepdim=True)
        self.n, self.nx = counts.size()
        self.words = words

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.counts[idx], 0


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
    # pdb.set_trace()
    mat = loadmat(loc + 'nips_1-17.mat')
    # pdb.set_trace()
    nd, nw = mat['counts'].shape
    train_start = 0
    train_end = int(0.8 * nd)
    val_end = int(0.9 * nd)
    train = NIPSDataset(torch.from_numpy(mat['counts'][train_start:train_end].todense()).float(),
                        mat['words'])
    val = NIPSDataset(torch.from_numpy(mat['counts'][train_end:val_end].todense()).float(),
                      mat['words'])
    test = NIPSDataset(torch.from_numpy(mat['counts'][val_end:].todense()).float(),
                       mat['words'])
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    return train_loader, val_loader, test_loader, nw


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
    parser.add_argument('-k', '--ncs',
                        type=int,
                        default=25,
                        help='number of constraints')
    parser.add_argument('-T', '--timesteps',
                        type=int,
                        default=3,
                        help='number of time steps for random projections')
    parser.add_argument('--perplexity',
                        type=bool,
                        default=False,
                        help='Should testing be using perplexity instead of ELBO')
    parser.add_argument('--likelihood',
                        type=bool,
                        default=False,
                        help='Should testing be using likelihood instead of ELBO')
    parser.add_argument('-S', '--nS',
                        type=int,
                        default=20,
                        help='number of samples to use per random projection.')
    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.model_folder)):
        # pdb.set_trace()
        os.makedirs(os.path.dirname(args.model_folder))
    inner_fix = 'perp' if args.perplexity else 'elbos'
    inner_fix = 'lkl' if args.likelihood else inner_fix

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
        train_loader, val_loader, test_loader, nx = get_nips_data(batch_size, 'nips_data/')
        # nx is returned.
        nz = 200
    else:
        raise NotImplementedError

    dbn = DBN(nx, nz, mode=args.mode, ncs=args.ncs, T=args.timesteps)
    dbn.mode = args.mode
    param_groups = [{'params': dbn.q_parameters, 'lr': 0.6e-4},
                    {'params': dbn.p_parameters, 'lr': 3e-4}]
    optimizer = optim.Adam(param_groups)
    if args.resume:
        # Load model etc
        checkpoint = load_checkpoint(os.path.join(args.model_folder, args.checkpoint_file))
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
            for lol1, (data, target) in enumerate(train_loader):
                data = Variable(data.view(-1, nx))  # visible
                if args.dataset == 'nips_data':
                    data_sample = data
                else:
                    data_sample = data.bernoulli()
                q_sample, q, p_sample, p, loss_T = \
                    dbn(data_sample,
                        S=args.nS,
                        k=args.ncs,
                        T=args.timesteps,
                        compute_loss=True)
                loss = loss_T.median()
                losses.append(loss.data[0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # pdb.set_trace()
                # print(lol1, 'of', len(train_loader))
            print('epoch', epoch, 'loss=', np.mean(losses))
            if every(epoch, stride=5):
                save_checkpoint({'epoch': epoch,
                                 'model': dbn.state_dict(),
                                 'optimizer': optimizer.state_dict()},
                                os.path.join(args.model_folder,
                                             dbn.mode +
                                             '.' + str(dbn.k) +
                                             '.' + str(dbn.T) +
                                             '.' + str(epoch) +
                                             '.pytorch.tar'))
                print('Saved model on epoch', epoch)

    if args.test is True:
        print('Started testing')
        vanilla_metrics = []
        random_metrics = []
        greedy_metrics = []
        for mbi, (data, target) in enumerate(test_loader):
            print('mb ', mbi)
            data = Variable(data.view(-1, nx))  # visible
            if args.dataset == 'nips_data':
                data_sample = data
            else:
                data_sample = data.bernoulli()
            dbn.mode = 'vanilla'
            q_samples_T_vanilla, _, p_samples_T_vanilla, _, _ = \
                dbn(data_sample,
                    S=args.nS,
                    k=args.ncs,
                    T=args.timesteps,
                    compute_loss=False)
            dbn.mode = 'random'
            q_samples_T_random, _, p_samples_T_random, _, _ = \
                dbn(data_sample,
                    S=args.nS,
                    k=args.ncs,
                    T=args.timesteps,
                    compute_loss=False)
            dbn.mode = 'greedy'
            q_samples_T_greedy, _, p_samples_T_greedy, _, _ = \
                dbn(data_sample,
                    S=args.nS,
                    k=args.ncs,
                    T=args.timesteps,
                    compute_loss=False)

            if args.perplexity:
                metric_fn = evaluate_perplexity
            elif args.likelihood:
                metric_fn = evaluate_nll
            else:
                metric_fn = evaluate_elbo
            if not args.perplexity:
                for t in range(args.timesteps):
                    p_samples_T_greedy[t][-1] = data_sample.expand(args.nS, *data.size())
                    p_samples_T_vanilla[t][-1] = data_sample.expand(args.nS, *data.size())
                    p_samples_T_random[t][-1] = data_sample.expand(args.nS, *data.size())

            dbn.mode = 'vanilla'
            vanilla_metrics.append(metric_fn(dbn,
                                             q_samples_T_vanilla,
                                             p_samples_T_vanilla))
            random_metrics.append(metric_fn(dbn,
                                            q_samples_T_random,
                                            p_samples_T_random))
            greedy_metrics.append(metric_fn(dbn,
                                            q_samples_T_greedy,
                                            p_samples_T_greedy))

        all_metrics = {'vanilla': vanilla_metrics, 'greedy': greedy_metrics, 'random': random_metrics}
        with open(args.model_folder +
                  args.mode +
                  '.' + str(dbn.k) + '.' + str(dbn.T) +
                  '.' + str(start_epoch) + '.' + (inner_fix) + '.pickle', 'wb') as f:
            pickle.dump(all_metrics, f)
