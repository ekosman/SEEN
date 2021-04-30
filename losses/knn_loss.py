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
        # x = x / x.norm(p=2, dim=1).reshape(-1, 1)
        k = min(self.k + 1, x.shape[0])
        tree = BallTree(x.detach().cpu().numpy(), leaf_size=self.leaf_size)
        i = tree.query(x.detach().cpu().numpy(), return_distance=False, k=k)
        i = i[:, 1:]
        loss = 0
        for x_i, (x_, x_neighbors_indices) in enumerate(zip(x, i)):
            diff = x_ - x
            diff = diff.norm(p=2, dim=1)
            distances = torch.exp(-diff)

            if self.iteration % 5000 == 0:
                diff_ = list(reversed(sorted(diff.detach().cpu().numpy())))

                chosen, ds, threshold = get_threshold_by_distance(diff_)

                plt.figure()
                plt.title(f"fdsfsd {self.iteration}")
                plt.plot(diff_, label='distances')
                plt.plot([0, len(diff_) - 1], [diff_[0], diff_[-1]], color='r')
                plt.plot(ds, label='difference')
                plt.axvline(chosen, color='g', label=f"chosen threshold: {diff_[chosen]} @ {chosen}")
                plt.legend()
                plt.savefig("distances.png")
                plt.show()
                plt.close()

                plt.figure()
                plt.title(f"Iteration {self.iteration}, max distance: {diff.max()}")
                plt.hist(diff_, bins=50)
                plt.axvline(threshold)
                # plt.yscale('log')
                # file_name = path.join(folder, f"{i}.png")
                plt.savefig('distance_hist.png')
                plt.show()
                plt.close()

            self.iteration += 1

            neighbors_distances = distances[x_neighbors_indices]
            distances_wo_x = distances[np.arange(len(x)) != x_i]

            denominator = distances_wo_x.sum()

            loss -= torch.log(neighbors_distances / denominator).mean()

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
    # loss = KNNLoss()
    loss = ContrastiveLoss()
    x = torch.randint(0, 10, size=(10, 3)).float()

    print(loss(x))