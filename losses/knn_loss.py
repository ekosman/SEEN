import time

import torch
import torch.nn as nn
from sklearn.neighbors import BallTree
import numpy as np


class KNNLoss(nn.Module):
    def __init__(self, k=2, leaf_size=2):
        super(KNNLoss, self).__init__()
        self.k = k
        self.leaf_size = leaf_size
        self.iteration = 0

    def forward(self, x):
        k = min(self.k + 1, x.shape[0])
        tree = BallTree(x.detach().cpu().numpy(), leaf_size=self.leaf_size)
        i = tree.query(x.detach().cpu().numpy(), return_distance=False, k=k)
        i = i[:, 1:]
        loss = 0
        for x_i, (x_, x_neighbors_indices) in enumerate(zip(x, i)):
            diff = x_ - x
            diff = diff.norm(p=2, dim=1)
            distances = torch.exp(-diff)
            neighbors = x_neighbors_indices

            neighbors_distances = distances[neighbors]
            distances_wo_x = distances[np.arange(len(x)) != x_i]

            denominator = distances_wo_x.sum()

            loss -= torch.log(neighbors_distances / denominator).mean()

            self.iteration += 1

        return loss / x.shape[0]


class ClassificationKNNLoss(nn.Module):
    def __init__(self, k=2, leaf_size=2):
        super(ClassificationKNNLoss, self).__init__()
        self.k = k
        self.leaf_size = leaf_size
        self.iteration = 0

    def forward(self, x, y):
        y = y.detach().cpu()
        k = min(self.k + 1, x.shape[0])
        tree = BallTree(x.detach().cpu().numpy(), leaf_size=self.leaf_size)
        i = tree.query(x.detach().cpu().numpy(), return_distance=False, k=k)
        i = i[:, 1:]
        loss = 0
        for x_i, (x_, x_neighbors_indices, y_) in enumerate(zip(x, i, y)):
            diff = x_ - x
            diff = diff.norm(p=2, dim=1)
            distances = torch.exp(-diff)

            neighbors = x_neighbors_indices[y[x_neighbors_indices] == y_]
            if len(neighbors) == 0:
                continue

            neighbors_distances = distances[neighbors]
            distances_wo_x = distances[np.arange(len(x)) != x_i]

            denominator = distances_wo_x.sum()

            loss -= torch.log(neighbors_distances / denominator).mean()

            self.iteration += 1

        return loss / x.shape[0]
