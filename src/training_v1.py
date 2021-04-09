import torch
import logging
import os
import torch.nn.functional as F

## Get the same logger from main"
from losses.knn_loss import KNNLoss
from network.cpc import CDCK2

logger = logging.getLogger("cdc")


def trainXXreverse(args, model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    for batch_idx, [data, data_r] in enumerate(train_loader):
        data = data.float().unsqueeze(1).to(device)  # add channel dimension
        data_r = data_r.float().unsqueeze(1).to(device)  # add channel dimension
        optimizer.zero_grad()
        hidden1 = model.init_hidden1(len(data))
        hidden2 = model.init_hidden2(len(data))
        acc, loss, hidden1, hidden2 = model(data, data_r, hidden1, hidden2)

        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), lr, acc, loss.item()))


def train_spk(args, cdc_model, spk_model, device, train_loader, optimizer, epoch, batch_size, frame_window):
    cdc_model.eval()  # not training cdc model
    spk_model.train()
    for batch_idx, [data, target] in enumerate(train_loader):
        data = data.float().unsqueeze(1).to(device)  # add channel dimension
        target = target.to(device)
        hidden = cdc_model.init_hidden(len(data))
        output, hidden = cdc_model.predict(data, hidden)
        data = output.contiguous().view((-1, 256))
        target = target.view((-1, 1))
        shuffle_indexing = torch.randperm(data.shape[0])  # shuffle frames
        data = data[shuffle_indexing, :]
        target = target[shuffle_indexing, :].view((-1,))
        optimizer.zero_grad()
        output = spk_model.forward(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        acc = 1. * pred.eq(target.view_as(pred)).sum().item() / len(data)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data) / frame_window, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), lr, acc, loss.item()))


knn_loss = None


def train(args, model, device, train_loader, optimizer, epoch, batch_size, is_data_parallel):
    global knn_loss
    if knn_loss is None:
        knn_loss = KNNLoss(k=batch_size // 64).to(device)

    model.train()
    total_loss = 0

    for batch_idx, data in enumerate(train_loader):
        hidden = CDCK2.init_hidden(len(data))
        if is_data_parallel:
            data = data.cuda()
            hidden = hidden.cuda()
        else:
            data = data.to(device)
            hidden = hidden.to(device)

        optimizer.zero_grad()

        acc, loss_1, hidden, output = model(data, hidden)

        loss_2 = knn_loss(output)

        # print(loss.device)
        # print(loss_1.device)

        loss = loss_1 + loss_2
        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            print(f"cpc loss: {loss_1.item()}")
            print(f"knn loss: {loss_2.item()}")
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, args.epochs, batch_idx * len(data), len(train_loader.dataset),
                                    100. * (batch_idx + 1) / len(train_loader), lr, acc, loss.item()))

        total_loss += loss.item()

    return total_loss / len(train_loader)


def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                                 run_name + '-model_best.pth')

    torch.save(state, snapshot_file)
    print("Snapshot saved to {}\n".format(snapshot_file))
