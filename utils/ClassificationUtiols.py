import torch


def onehot_coding(target, device, output_dim):
    """Convert the class labels into one-hot encoded vectors."""
    target_onehot = torch.FloatTensor(target.size()[0], output_dim).to(device)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.0)
    return target_onehot
