"""Training and evaluating a soft decision tree on the MNIST dataset."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from losses.sdt_cross_entropy import SDTLoss
from soft_decision_tree.sdt_model import SDT
from six.moves import urllib

from utils.TorchUtils import TorchModel
from utils.callbacks import DefaultModelCallback
from utils.utils import register_logger

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


def onehot_coding(target, device, output_dim):
    """Convert the class labels into one-hot encoded vectors."""
    target_onehot = torch.FloatTensor(target.size()[0], output_dim).to(device)
    target_onehot.data.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1.0)
    return target_onehot


def get_data(batch_size):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    # Parameters
    register_logger()
    input_dim = 28 * 28  # the number of input dimensions
    output_dim = 10  # the number of outputs (i.e., # classes on MNIST)
    depth = 10  # tree depth
    lamda = 1e-3  # coefficient of the regularization term
    lr = 1e-3  # learning rate
    weight_decaly = 5e-4  # weight decay
    batch_size = 256  # batch size
    epochs = 51  # the number of training epochs
    log_interval = 10  # the number of batches to wait before printing logs
    use_cuda = torch.cuda.is_available()  # whether to use GPU

    # Model and Optimizer
    tree = SDT(input_dim, output_dim, depth, lamda, use_cuda)

    optimizer = torch.optim.Adam(tree.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decaly)

    # Load data
    train_loader, test_loader = get_data(batch_size)

    # Utils
    best_testing_acc = 0.0
    testing_acc_list = []
    training_loss_list = []
    criterion = SDTLoss()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = TorchModel(tree).to(device).train()
    model.register_callback(DefaultModelCallback(log_every=log_interval))

    model.fit(train_iter=train_loader,
              criterion=criterion,
              optimizer=optimizer,
              eval_iter=test_loader,
              epochs=epochs,
              evaluate_every=5)

    # for epoch in range(epochs):
    #
    #     # Training
    #     tree.train()
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #
    #         batch_size = data.size()[0]
    #         data, target = data.to(device), target.to(device)
    #         target_onehot = onehot_coding(target, device, output_dim)
    #
    #         output, penalty = tree.forward(data, is_training_data=True)
    #
    #         loss = criterion(output, target.view(-1))
    #         loss += penalty
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         # Print training status
    #         if batch_idx % log_interval == 0:
    #             pred = output.data.max(1)[1]
    #             correct = pred.eq(target.view(-1).data).sum()
    #
    #             msg = (
    #                 "Epoch: {:02d} | Batch: {:03d} | Loss: {:.5f} |"
    #                 " Correct: {:03d}/{:03d}"
    #             )
    #             print(msg.format(epoch, batch_idx, loss, correct, batch_size))
    #             training_loss_list.append(loss.cpu().data.numpy())
    #
    #     # Evaluating
    #     tree.eval()
    #     correct = 0.
    #
    #     for batch_idx, (data, target) in enumerate(test_loader):
    #         batch_size = data.size()[0]
    #         data, target = data.to(device), target.to(device)
    #
    #         output = F.softmax(tree.forward(data), dim=1)
    #
    #         pred = output.data.max(1)[1]
    #         correct += pred.eq(target.view(-1).data).sum()
    #
    #     accuracy = 100.0 * float(correct) / len(test_loader.dataset)
    #
    #     if accuracy > best_testing_acc:
    #         best_testing_acc = accuracy
    #
    #     msg = (
    #         "\nEpoch: {:02d} | Testing Accuracy: {}/{} ({:.3f}%) |"
    #         " Historical Best: {:.3f}%\n"
    #     )
    #     print(
    #         msg.format(
    #             epoch, correct,
    #             len(test_loader.dataset),
    #             accuracy,
    #             best_testing_acc
    #         )
    #     )
    #     testing_acc_list.append(accuracy)
