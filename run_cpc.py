"""
main for LibriSpeech
"""
## Utilities
from __future__ import print_function
import argparse
import random
import time
import os
import logging
from timeit import default_timer as timer
from tqdm import tqdm

## Libraries
import numpy as np

## Torch
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim

## Custrom Imports
from network.cpc import CDCK2
from src.logger_v1 import setup_logs
from src.training_v1 import train, trainXXreverse, snapshot
from src.validation_v1 import validation, validationXXreverse

############ Control Center and Hyperparameter ###############
from stream_generators.comma_loader import CommaLoader
from utils.MatplotlibUtils import reduce_dims_and_plot

run_name = "cdc" + time.strftime("-%Y-%m-%d_%H_%M_%S")
print(run_name)


class ScheduledOptim(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = 128
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.delta = 1

    def state_dict(self):
        self.optimizer.state_dict()

    def step(self):
        """Step by the inner optimizer"""
        self.optimizer.step()

    def zero_grad(self):
        """Zero out the gradients by the inner optimizer"""
        self.optimizer.zero_grad()

    def increase_delta(self):
        self.delta *= 2

    def update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += self.delta
        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr


def get_args():
    ## Settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--data_path', type=str, default=r'C:\comma\comma2k19\comma_ai_all_data')
    parser.add_argument('--dataset', type=str, choices=['comma', 'udacity'], default='comma')
    # parser.add_argument('--validation-raw', required=True)
    # parser.add_argument('--eval-raw')
    # parser.add_argument('--train-list', required=True)
    # parser.add_argument('--validation-list', required=True)
    # parser.add_argument('--eval-list')
    # parser.add_argument('--logging-dir', required=True,
    #                     help='model save directory')
    parser.add_argument('--epochs', type=int, default=60, metavar='N', help='number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the loader')
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--device', type=int, default=7, help='which gpu to use')
    parser.add_argument('--audio_window', type=int, default=20480, help='window length to sample from each utterance')
    parser.add_argument('--timestep', type=int, default=12)
    parser.add_argument('--masked_frames', type=int, default=20)
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--window_length', type=int, default=4000,
                        help='window length to sample from each video')
    return parser.parse_args()


def get_dataset(args, data_path, window_length):
    if args.dataset == 'comma':
        dataset = CommaLoader(signals_dataset_path=data_path,
                              samples_interval=0.01,
                              signals_input='all',
                              window_length=window_length)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    return dataset


def main():
    args = get_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)
    global_timer = timer()  # global timer
    # logger = setup_logs(args.logging_dir, run_name)  # setup logs
    device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")

    ## Loading the dataset
    params = {'num_workers': args.num_workers,
              'pin_memory': True} if use_cuda else {}

    print('===> loading train, validation and eval dataset')
    training_set = get_dataset(args=args, data_path=args.data_path, window_length=args.window_length)

    train_loader = data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True,
                                   **params)  # set shuffle to True

    model = CDCK2(args.timestep, args.batch_size, args.window_length, in_features=training_set.n_features).to(device)
    model = nn.DataParallel(model).to(device=device)
    # nanxin optimizer
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('### Model summary below###\n {}\n'.format(str(model)))
    print('===> Model total parameter: {}\n'.format(model_params))
    ## Start training
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()

        # Train and validate
        # trainXXreverse(args, model, device, train_loader, optimizer, epoch, args.batch_size)
        # val_acc, val_loss = validationXXreverse(args, model, device, validation_loader, args.batch_size)
        train(args, model, device, train_loader, optimizer, epoch, args.batch_size)
        # val_acc, val_loss = validation(args, model, device, validation_loader, args.batch_size)

        # Save
        # if val_acc > best_acc:
        #     best_acc = max(val_acc, best_acc)
        #     snapshot(args.logging_dir, run_name, {
        #         'epoch': epoch + 1,
        #         'validation_acc': val_acc,
        #         'state_dict': model.state_dict(),
        #         'validation_loss': val_loss,
        #         'optimizer': optimizer.state_dict(),
        #     })
        #     best_epoch = epoch + 1
        # elif epoch - best_epoch > 2:
        if epoch - best_epoch > 2:
            optimizer.increase_delta()
            best_epoch = epoch + 1

        end_epoch_timer = timer()
        print("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))

    ## end
    end_global_timer = timer()
    print("################## Success #########################")
    print("Total elapsed time: %s" % (end_global_timer - global_timer))

    # Do some TSNE
    dataset = get_dataset(args=args, data_path=args.data_path, window_length=160)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                   **params)  # set shuffle to True

    projects = torch.tensor([])
    total = 10000
    count = 0
    with torch.no_grad():
        bar = tqdm(total=total)
        for batch in loader:
            if count >= total:
                break

            y = model.encode(batch.to(device)).detach().cpu()
            projects = torch.cat([projects, y])
            bar.update(y.shape[0])
            count += y.shape[0]

    reduce_dims_and_plot(projects,
                         y=None,
                         title=None,
                         file_name='all_cpc_tsne.png',
                         perplexity=50,
                         library='Multicore-TSNE',
                         perform_PCA=False,
                         projected=None,
                         figure_type='2d',
                         show_figure=True,
                         close_figure=False,
                         text=None)


if __name__ == '__main__':
    main()
