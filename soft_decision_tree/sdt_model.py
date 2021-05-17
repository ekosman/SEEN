from collections import deque

import torch
import torch.nn as nn
import numpy as np
from queue import Queue
import networkx as nx
import matplotlib.pyplot as plt

from data_structures.tree_condition import TreeCondition


class Node:
    def __init__(self, depth, index, weights=None):
        self.left = None
        self.right = None
        self._class = None
        self.depth = depth
        self.index = index
        self.weights = weights
        self.visited = False

    @property
    def n_weights(self):
        return (abs(self.weights) > 0).sum().item()

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_condition(self, attr_names):
        return TreeCondition(self.weights, attr_names)

    def reset(self):
        """
        Clears the reset flag of the sub-tree
        """
        q = deque()
        q.append(self)

        while len(q) != 0:
            node = q.popleft()
            if node.left is not None:
                node.left.visited = False
                q.append(node.left)
            if node.right is not None:
                node.right.visited = False
                q.append(node.right)


class SDT(nn.Module):
    """Fast implementation of soft decision tree in PyTorch.

    Parameters
    ----------
    input_dim : int
      The number of input dimensions.
    output_dim : int
      The number of output dimensions. For example, for a multi-class
      classification problem with `K` classes, it is set to `K`.
    depth : int, default=5
      The depth of the soft decision tree. Since the soft decision tree is
      a full binary tree, setting `depth` to a large value will drastically
      increases the training and evaluating cost.
    lamda : float, default=1e-3
      The coefficient of the regularization term in the training loss. Please
      refer to the paper on the formulation of the regularization term.
    use_cuda : bool, default=False
      When set to `True`, use GPU to fit the model. Training a soft decision
      tree using CPU could be faster considering the inherent data forwarding
      process.

    Attributes
    ----------
    internal_node_num_ : int
      The number of internal nodes in the tree. Given the tree depth `d`, it
      equals to :math:`2^d - 1`.
    leaf_node_num_ : int
      The number of leaf nodes in the tree. Given the tree depth `d`, it equals
      to :math:`2^d`.
    penalty_list : list
      A list storing the layer-wise coefficients of the regularization term.
    inner_nodes : torch.nn.Sequential
      A container that simulates all internal nodes in the soft decision tree.
      The sigmoid activation function is concatenated to simulate the
      probabilistic routing mechanism.
    leaf_nodes : torch.nn.Linear
      A `nn.Linear` module that simulates all leaf nodes in the tree.
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            depth=5,
            lamda=1e-3,
            use_cuda=False):
        super(SDT, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.depth = depth
        self.lamda = lamda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self._validate_parameters()

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [
            self.lamda * (2 ** (-depth)) for depth in range(0, self.depth)
        ]

        # Initialize internal nodes and leaf nodes, the input dimension on
        # internal nodes is added by 1, serving as the bias.
        self.inner_nodes = nn.Linear(self.input_dim + 1, self.internal_node_num_, bias=False)

        self.leaf_nodes = nn.Parameter(torch.randn(size=(self.leaf_node_num_, self.output_dim)), requires_grad=True)

    def get_classes(self):
        return list(self.leaf_nodes.cpu().detach().argmax(dim=1).numpy())

    def get_tree(self):
        weights = self.inner_nodes.weight.cpu().detach().numpy()
        root = Node(depth=0, index=0, weights=weights[0, :])
        q = Queue()
        q.put(root)
        leaf_i = 0
        while not q.empty():
            node = q.get()
            if node.depth < self.depth:
                if node.depth == self.depth - 1:
                    weights_left = None
                    weights_right = None
                else:
                    weights_left = weights[node.index * 2 + 1, :]
                    weights_right = weights[node.index * 2 + 2, :]

                node.left = Node(node.depth + 1, node.index * 2 + 1, weights_left)
                node.right = Node(node.depth + 1, node.index * 2 + 2, weights_right)
                q.put(node.left)
                q.put(node.right)
            if node.depth == self.depth:
                node._class = self.leaf_nodes[leaf_i, :].argmax().item()
                leaf_i += 1

        return root

    @staticmethod
    def get_avg_height(node):
        q = Queue()
        q.put(node)

        sum_heights = 0
        leaves = 0

        while not q.empty():
            node = q.get()
            if node.left is None and node.right is None:
                leaves += 1
                sum_heights += node.depth
            else:
                q.put(node.left)
                q.put(node.right)

        return sum_heights / leaves

    def visualize(self):
        root = self.get_tree()

        #     prune
        self.prune(root)
        avg_height = self.get_avg_height(root)

        A = nx.DiGraph()

        A.add_node(0, _class=root._class)

        q = Queue()
        labels = {}
        q.put((root, 0))

        while not q.empty():
            node, node_i = q.get()
            labels[node_i] = node._class if node._class is not None else node.n_weights
            if node._class is not None:
                # leaf node
                continue

            else:
                left_node = node_i*2+1
                right_node = node_i*2+2
                A.add_node(left_node, _class=node.left._class)
                A.add_node(right_node, _class=node.right._class)
                A.add_edge(node_i, left_node)
                A.add_edge(node_i, right_node)
                q.put((node.left, left_node))
                q.put((node.right, right_node))

        print(f"Average height: {avg_height}")
        plt.title(f"Average height: {avg_height}")
        pos = nx.drawing.nx_agraph.graphviz_layout(A, prog='dot')
        nx.draw(A, pos, with_labels=True, arrows=True, labels=labels)
        return avg_height, root

    def prune_weights(self, threshold):
        self.inner_nodes.weight.data = torch.mul(
            torch.gt(torch.abs(self.inner_nodes.weight), threshold).float(), self.inner_nodes.weight
        )

    @staticmethod
    def prune(node):
        if node.left is None and node.right is None:
            # it's a leaf
            return

        SDT.prune(node.left)
        SDT.prune(node.right)

        if node.left._class == node.right._class and node.left._class is not None:
            node._class = node.left._class
            node.left = None
            node.right = None

    def forward(self, X, soft_paths=True):
        _mu, _penalty = self._forward(X)
        y_pred = _mu @ self.leaf_nodes if soft_paths else self.leaf_nodes[_mu.argmax(dim=1), :]

        # When `X` is the training data, the model also returns the penalty
        # to compute the training loss.
        if self.training:
            return y_pred, _penalty
        else:
            return y_pred

    def _forward(self, X):
        """Implementation on the data forwarding process."""

        batch_size = X.size()[0]
        X = self._data_augment(X)

        path_prob = torch.sigmoid(self.inner_nodes(X))
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        _penalty = torch.tensor(0.0).to(self.device)

        # Iterate through internal nodes in each layer to compute the final path
        # probabilities and the regularization term.
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            # Extract internal nodes in the current layer to compute the
            # regularization term
            _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)

            _mu = _mu * _path_prob  # update path probabilities

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.leaf_node_num_)

        return mu, _penalty

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        """
        Compute the regularization term for internal nodes in different layers.
        """

        penalty = torch.tensor(0.0).to(self.device)

        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = torch.sum(
                _path_prob[:, node] * _mu[:, node // 2], dim=0
            ) / torch.sum(_mu[:, node // 2], dim=0)

            coeff = self.penalty_list[layer_idx]

            penalty -= 0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha))

        return penalty

    def _data_augment(self, X):
        """Add a constant input `1` onto the front of each sample."""
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        X = torch.cat((bias, X), 1)

        return X

    def _validate_parameters(self):

        if not self.depth > 0:
            msg = ("The tree depth should be strictly positive, but got {}"
                   "instead.")
            raise ValueError(msg.format(self.depth))

        if not self.lamda >= 0:
            msg = (
                "The coefficient of the regularization term should not be"
                " negative, but got {} instead."
            )
            raise ValueError(msg.format(self.lamda))
