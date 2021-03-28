import torch.nn as nn
import numpy as np


class EventAbstractor(nn.Module):
    def __init__(self, num_attributes, embedding_dim, n_steps=1, activation='relu'):
        super(EventAbstractor, self).__init__()

        n_neurons = np.geomspace(num_attributes, embedding_dim, n_steps + 1).astype(int)
        n_neurons[0] = num_attributes
        n_neurons[-1] = embedding_dim

        modules = []
        for input_dim, output_dim in zip(n_neurons[:-1], n_neurons[1:]):
            modules.append(nn.Linear(input_dim, output_dim))

            activation_layer = None
            if activation == 'relu':
                activation_layer = nn.ReLU()
            elif activation == 'tanh':
                activation_layer = nn.Tanh()
            elif activation == 'sigmoid':
                activation_layer = nn.Sigmoid()

            if activation_layer is not None:
                modules.append(activation_layer)

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)
