import torch


def init_gru_hidden(num_layers, num_directions, batch_size, hidden_size):
    return torch.zeros(num_layers * num_directions, batch_size, hidden_size)