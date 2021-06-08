import time

import torch
import torch.nn as nn
from sklearn.neighbors import BallTree
import numpy as np
from os import makedirs, path
import matplotlib.pyplot as plt



# folder = 'hists'
# makedirs(folder, exist_ok=True)
from utils.utils import get_threshold_by_distance


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


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


if __name__ == '__main__':
    loss = ClassificationKNNLoss()
    x = torch.randint(0, 10, size=(10, 3)).float()
    y = torch.randint(0,3, size=(10,))

    print(loss(x, y))