import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        if type(outputs) == tuple:
            outputs, penalty = outputs
        else:
            penalty = 0

        outputs = outputs.argmax(dim=1)
        return (outputs == targets).float().sum()