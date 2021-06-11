"""
main for LibriSpeech
"""
## Utilities
from __future__ import print_function

import argparse
import os
import pickle
import time
from os import path
from timeit import default_timer as timer

import matplotlib.pyplot as plt
## Libraries
import numpy as np
## Torch
import torch
import torch.optim as optim
from torch.utils import data
from tqdm import tqdm

## Custrom Imports
from network.cpc import CDCK2
from src.training_v1 import train
############ Control Center and Hyperparameter ###############
from stream_generators.comma_loader import CommaLoader
from utils.DateUtils import get_time_str
from utils.MatplotlibUtils import reduce_dims_and_plot
from utils.email_utils import GmailNotifier
from utils.LoggingUtils import get_clearml_logger
from utils.utils import register_exps_dir


"""
Command:
python run_cpc.py --data_path ../comma_ai_all_data/ --epochs 100 --num_workers 1 --batch_size 512 --device 1 --window_stride 200 --timestep 30 --window_length 2000 --exps_dir knn_loss_batch_512_k_16 --k 16 --send_email
"""


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
    parser.add_argument('--test_split_ratio', type=float, default=0.2, help='percentage for using test data')
    parser.add_argument('--features', nargs="+", default='all', help='names of featurse to use for cpc')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the loader')
    parser.add_argument('--n-warmup-steps', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--device', type=int, default=7, help='which gpu to use')
    parser.add_argument('--audio_window', type=int, default=20480, help='window length to sample from each utterance')
    parser.add_argument('--timestep', type=int, default=12)
    parser.add_argument('--masked_frames', type=int, default=20)
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--window_length', type=int, default=1500, help='window length to sample from each video')
    parser.add_argument('--tsne_window_length', type=int, default=1000, help='window length to sample from each video')
    parser.add_argument('--sample_stride', type=int, default=1, help='interval between two consecutive samples')
    parser.add_argument('--window_stride', type=int, default=50, help='interval between two consecutive windows')
    parser.add_argument('--num_tsne_samples', type=int, default=25000, help='how many samples to use for TSNE')
    parser.add_argument('--exps_dir', type=str, default="exps", help='name to use for the output folder')
    parser.add_argument('--k', type=int, default=4, help='Number of neighbors to consider for the knn loss')
    parser.add_argument('--send_email', default=False, action='store_true', help='Send an email when finished')
    parser.add_argument('--email_user', default=r'', help='Username for sending the email')
    parser.add_argument('--email_password', default='', help='Password for sending the email')
    parser.add_argument('--email_to', default=r'eitan.kosman@gmail.com',
                        help='Email of the receiver of the results email')
    parser.add_argument('--enable_clearml_logger',
                        default=False,
                        action='store_true',
                        help="Enable logging to ClearML server")
    return parser.parse_args()


def get_dataset(args, data_path, window_length):
    if args.features[0] == 'all':
        args.features = 'all'

    if args.dataset == 'comma':
        file_name = "comma.dataset"
        if path.exists(file_name):
            print(f"Loading dataset {args.dataset} from {file_name}")
            with open(file_name, 'rb') as fp:
                dataset = pickle.load(fp)
                dataset.set_params(window_length, args.window_stride, args.sample_stride)
        else:
            print(f"Creating dataset {args.dataset}")
            dataset = CommaLoader(signals_dataset_path=data_path,
                                  samples_interval=0.005,
                                  signals_input=args.features,
                                  window_length=window_length,
                                  window_stride=args.window_stride,
                                  sample_stride=args.sample_stride)
            with open(file_name, 'wb') as fp:
                print(f"Writing dataset {args.dataset} to {file_name}")
                pickle.dump(dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    return dataset


def main():
    args = get_args()

    if args.enable_clearml_logger:
        # tags = [
        #     f'mode: {args.mode}',
        #     f'model: {args.model_type}',
        #     f'steps: {args.conv_steps}',
        #     f'kernel: {args.conv_kernel}',
        #     f'stride: {args.conv_stride}',
        #     f'context: {args.history_window_length}',
        #     f'delta: {args.future_stride}',
        #     f'horizon: {args.forecast_window_length}',
        #     f'alpha: {args.alpha}',
        # ]
        clearml_logger = get_clearml_logger(project_name="EntangledExplainableClustering",
                                            task_name='Train_' + get_time_str(),
                                            # tags=tagsת
                                            )

    register_exps_dir(args.exps_dir)
    model_dir = path.join(args.exps_dir, 'models')
    data_dir = path.join(args.exps_dir, 'data')
    register_exps_dir(model_dir)
    register_exps_dir(data_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print('use_cuda is', use_cuda)
    global_timer = timer()  # global timer
    # logger = setup_logs(args.logging_dir, run_name)  # setup logs
    device = torch.device("cuda:" + str(args.device) if use_cuda else "cpu")

    ## Loading the dataset
    params = {'num_workers': args.num_workers,
              'pin_memory': True} if use_cuda else {}

    print('===> loading train, validation and eval dataset')
    dataset = get_dataset(args=args, data_path=args.data_path, window_length=args.window_length)
    train_idx = np.random.choice(len(dataset), int(len(dataset) * (1 - args.test_split_ratio)), replace=False)
    test_idx = list(set(range(len(dataset))) - set(train_idx))

    train_dataset = data.dataset.Subset(dataset, train_idx)
    test_dataset = data.dataset.Subset(dataset, test_idx)

    train_dataset_path = path.join(data_dir, 'train_data.file')
    test_dataset_path = path.join(data_dir, 'test_data.file')

    print(f"Dumping training dataset to {train_dataset_path}")
    with open(train_dataset_path, 'wb') as fp:
        pickle.dump(train_dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dumping testing dataset to {test_dataset_path}")
    with open(test_dataset_path, 'wb') as fp:
        pickle.dump(test_dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   **params)  # set shuffle to True

    model = CDCK2(args.timestep, args.batch_size, args.window_length, in_features=dataset.n_features, device=device)
    is_data_parallel = False
    # if use_cuda:
    #     model = nn.DataParallel(model).cuda()
    #     is_data_parallel = True
    # else:
    #     model = model.to(device)
    model = model.to(device)

    # nanxin optimizer
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-4, amsgrad=True),
        args.n_warmup_steps)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('### Model summary below###\n {}\n'.format(str(model)))
    print('===> Model total parameter: {}\n'.format(model_params))

    print(f"Length of train dataset: {len(train_dataset)}")
    print(f"Length of test dataset: {len(test_dataset)}")
    ## Start training
    best_acc = 0
    best_loss = np.inf
    best_epoch = -1
    losses = None
    epoch = 0

    for epoch in range(1, args.epochs + 1):
        epoch_timer = timer()

        # Train and validate
        # trainXXreverse(args, model, device, train_loader, optimizer, epoch, args.batch_size)
        # val_acc, val_loss = validationXXreverse(args, model, device, validation_loader, args.batch_size)
        loss = train(args, model, device, train_loader, optimizer, epoch, args.batch_size, is_data_parallel)
        if losses is None:
            losses = {k: [] for k, v in loss.items()}

        for k, v in losses.items():
            losses[k].append(loss[k])
        # val_acc, val_loss = validation(args, model, device, validation_loader, args.batch_size)

        if epoch % args.save_every == 0:
            model_path = path.join(model_dir, f'epoch_{epoch}.pt')
            torch.save(model, model_path)
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

    model_path = path.join(model_dir, f'epoch_{epoch}.pt')
    torch.save(model, model_path)

    plt.figure()
    plt.title("Loss vs epoch")
    for k, v in losses.items():
        plt.plot(range(len(v)), v, label=k)

    plt.legend()
    plt.savefig(path.join(args.exps_dir, "cpc_losses.png"))
    plt.close()
    ## end
    end_global_timer = timer()
    print("################## Success #########################")
    print("Total elapsed time: %s" % (end_global_timer - global_timer))

    # Do some TSNE
    # dataset = training_set
    # dataset.set_window_length(args.tsne_window_length)
    # dataset = get_dataset(args=args, data_path=args.data_path, window_length=compress_ratio)
    loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                             **params)  # set shuffle to True

    projects = torch.tensor([])
    count = 0
    totals = [len(test_dataset) // 32, len(test_dataset) // 16, len(test_dataset) // 8, len(test_dataset) // 4, len(test_dataset) // 2, len(test_dataset)]
    total = max(totals)
    with torch.no_grad():
        bar = tqdm(total=total)
        for batch in loader:
            if count >= total:
                break

            hidden = CDCK2.init_hidden(len(batch))
            if is_data_parallel:
                batch = batch.cuda()
                hidden = hidden.cuda()
            else:
                batch = batch.to(device)
                hidden = hidden.to(device)

            y = model.predict(batch, hidden).detach().cpu()
            projects = torch.cat([projects, y])
            bar.update(y.shape[0])
            count += y.shape[0]

    for perplexity in [10, 30, 50, 100, 200]:
        for total in totals:
            projects_tmp = np.random.choice(projects.shape[0], total)
            projects_tmp = projects[projects_tmp, :]

            file_name = path.join(args.exps_dir, f'all_cpc_tsne_perplexity_{perplexity}_{str(total)}_samples.png')

            reduce_dims_and_plot(projects_tmp,
                                 y=None,
                                 title=None,
                                 file_name=file_name,
                                 perplexity=perplexity,
                                 library='Multicore-TSNE',
                                 perform_PCA=False,
                                 projected=None,
                                 figure_type='2d',
                                 show_figure=False,
                                 close_figure=True,
                                 text=None)

    if args.send_email:
        with GmailNotifier(username=args.email_user, password=args.email_password, to=args.email_to) as noti:
            noti.send_args_description(args.exps_dir, args)


if __name__ == '__main__':
    main()
