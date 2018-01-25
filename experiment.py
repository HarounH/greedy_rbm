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


if __name__ == '__main__':
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
    args = parser.parse_args()
    # pdb.set_trace()
    # Data loading
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

    dbn = DBN(784, [200])
    optimizer = optim.SGD(dbn.parameters(), lr=0.1)
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
        print('Training again')
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
                                             '1.25.2018.vanilla.' + str(epoch) + '.pytorch.tar'))
                print('Saved model on epoch', epoch)

    if args.test is True:
        print('Started testing')
        vanilla_losses = []
        random_losses = []
        greedy_losses = []
        for mbi, (data, target) in enumerate(test_loader):
            print('mb ', mbi)
            data = Variable(data.view(-1, 784))  # visible
            # Get samples from all 3 modes
            dbn.mode = 'vanilla'
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
            # print('r', end='')
            dbn.mode = 'vanilla'
            # Get the ELBO for those samples using vanilla paramaters
            vanilla_losses.append(dbn.evaluate_sample([q_sample_vanilla], [p_sample_vanilla]))
            # print(' vl', end='')
            greedy_losses.append(dbn.evaluate_sample([q_sample_greedy], [p_sample_greedy]))
            # print(' gl', end='')
            random_losses.append(dbn.evaluate_sample(q_sample_T_random, p_sample_T_random))
            # print(' rl', end='')
            # print(vanilla_losses[-1], greedy_losses[-1], random_losses[-1])
        pdb.set_trace()
    all_losses = {'vanilla': vanilla_losses, 'greedy': greedy_losses, 'random': random_losses}
    with open('models/1.25.2018.losses.pickle', 'wb') as f:
        pickle.dump(all_losses, f)
