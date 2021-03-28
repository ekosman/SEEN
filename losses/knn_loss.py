import torch
import torch.nn as nn
from sklearn.neighbors import BallTree
import numpy as np


class KNNLoss(nn.Module):
    def __init__(self, k=2, leaf_size=2):
        super(KNNLoss, self).__init__()
        self.k = k
        self.leaf_size = leaf_size

    def forward(self, x):
        x = x / x.norm(p=2, dim=1).reshape(-1, 1)
        k = min(self.k + 1, x.shape[0])
        tree = BallTree(x.detach().numpy(), leaf_size=self.leaf_size)
        i = tree.query(x.detach().numpy(), return_distance=False, k=k)
        i = i[:, 1:]
        loss = 0
        for x_, x_neighbors_indices in zip(x, i):
            distances = torch.exp(x_.T @ x.T)

            denominator = distances.sum()

            loss -= torch.log(distances[x_neighbors_indices] / denominator).sum()

        return loss


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
    # loss = KNNLoss()
    loss = ContrastiveLoss()
    x = torch.randint(0, 10, size=(10, 3)).float()

    print(loss(x))