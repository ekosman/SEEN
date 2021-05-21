"""Training and evaluating a soft decision tree on the MNIST dataset."""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from soft_decision_tree.sdt_model import SDT
# from six.moves import urllib
from utils.ClassificationUtiols import onehot_coding

from utils.utils import register_logger


# opener = urllib.request.build_opener()
# opener.addheaders = [('User-agent', 'Mozilla/5.0')]
# urllib.request.install_opener(opener)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=51)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--entropy_lamda', type=float, default=1e-3)
    parser.add_argument('--sparsity_lamda', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    return parser.parse_args()


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
    args = get_args()
    register_logger()
    input_dim = 28 * 28  # the number of input dimensions
    output_dim = 10  # the number of outputs (i.e., # classes on MNIST)
    depth = args.depth  # tree depth
    lamda = args.entropy_lamda  # coefficient of the regularization term
    lr = args.lr  # learning rate
    weight_decay = args.weight_decay  # weight decay
    batch_size = args.batch_size  # batch size
    epochs = args.epochs  # the number of training epochs
    log_interval = args.log_interval  # the number of batches to wait before printing logs
    use_cuda = torch.cuda.is_available()  # whether to use GPU

    # Model and Optimizer
    tree = SDT(input_dim, output_dim, depth, lamda, use_cuda)

    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()

    optimizer = torch.optim.Adam(tree.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    # Load data
    train_loader, test_loader = get_data(batch_size)

    # Utils
    best_testing_acc = 0.0
    testing_acc_list = []
    training_loss_list = []
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if use_cuda else "cpu")
    tree = tree.to(device)

    # model = TorchModel(tree).to(device).train()
    # model.register_callback(DefaultModelCallback(log_every=log_interval))
    #
    # model.fit(train_iter=train_loader,
    #           criterion=criterion,
    #           optimizer=optimizer,
    #           eval_iter=test_loader,
    #           epochs=epochs,
    #           evaluate_every=5)
    heights = []
    for epoch in range(epochs):
        print(f"Classes: {tree.get_classes()}")
        # Training
        tree.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            dt.fit(data.flatten(1), target)
            tree.initialize_from_decision_tree(dt)

            batch_size = data.size()[0]
            data, target = data.to(device), target.to(device)

            output, penalty = tree.forward(data)

            # Loss
            loss = criterion(output, target.view(-1))

            # Penalty
            loss += penalty

            # L1
            fc_params = torch.cat([x.view(-1) for x in tree.inner_nodes.parameters()])
            l1_regularization = torch.norm(fc_params, 1)
            loss += args.sparsity_lamda * l1_regularization

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training status
            if batch_idx % log_interval == 0:
                pred = output.data.max(1)[1]
                correct = pred.eq(target.view(-1).data).sum()

                msg = (
                    "Epoch: {:02d} | Batch: {:03d} / {:03d} | Loss: {:.5f} |"
                    " Accuracy: {:03f}"
                )
                print(msg.format(epoch, batch_idx, len(train_loader), loss.item(), correct.item() / batch_size))
                training_loss_list.append(loss.cpu().data.numpy())

        plt.figure(figsize=(300, 10), dpi=80)
        avg_height, root = tree.visualize()
        root.accumulate_samples(data, 'MLE')
        leaves = root.get_leaves()
        for leaf in leaves:
            leaf.reset_path()
            leaf.tighten_with_accumulated_samples()
            conds = leaf.get_path_conditions(['a' for _ in range(28*28)])
        heights.append(avg_height)
        # plt.savefig(f"tree_epoch_{epoch}_avg_height_{avg_height}.png")
        # plt.close()
        # Evaluating
        tree.eval()
        correct = 0.

        plt.figure()
        weights = tree.inner_nodes.weight.cpu().detach().numpy().flatten()
        plt.hist(weights, bins=500)
        plt.yscale("log")
        plt.show()
        plt.close()

        for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
            batch_size = data.size()[0]
            data, target = data.to(device), target.to(device)

            output = F.softmax(tree.forward(data), dim=1)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1).data).sum().item()

        accuracy = 100.0 * float(correct) / len(test_loader.dataset)

        if accuracy > best_testing_acc:
            best_testing_acc = accuracy

        msg = (
            "\nEpoch: {:02d} | Testing Accuracy: {}/{} ({:.3f}%) |"
            " Historical Best: {:.3f}%\n"
        )
        print(
            msg.format(
                epoch, correct,
                len(test_loader.dataset),
                accuracy,
                best_testing_acc
            )
        )
        testing_acc_list.append(accuracy)

    plt.figure()
    plt.plot(heights)
    plt.xlabel("Epoch")
    plt.ylabel("Average height")
    plt.savefig("heights.png")
    plt.close()
