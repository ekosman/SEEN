import torch.nn as nn


class SDTLoss(nn.Module):
    def __init__(self):
        super(SDTLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        if type(outputs) == tuple:
            outputs, penalty = outputs
        else:
            penalty = 0

        return penalty + self.loss(outputs, targets)