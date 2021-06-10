import time

from losses.knn_loss import KNNLoss
from network.cpc import CDCK2

knn_crt = None


def train(args, model, device, train_loader, optimizer, epoch, batch_size, is_data_parallel):
    global knn_crt
    if knn_crt is None:
        knn_crt = KNNLoss(k=args.k).to(device)

    model.train()
    total_loss = {
        'cpc': 0,
        'knn': 0,
        'total': 0,
    }
    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        hidden = CDCK2.init_hidden(len(data))
        if is_data_parallel:
            data = data.cuda()
            hidden = hidden.cuda()
        else:
            data = data.to(device)
            hidden = hidden.to(device)

        optimizer.zero_grad()

        acc, cpc_loss, hidden, output = model(data, hidden)

        knn_loss = knn_crt(output)

        # print(loss.device)
        # print(loss_1.device)

        loss = cpc_loss + knn_loss
        loss.backward()
        optimizer.step()
        lr = optimizer.update_learning_rate()
        if batch_idx % args.log_interval == 0:
            print(f"cpc loss: {cpc_loss.item()}")
            print(f"knn loss: {knn_loss.item()}")
            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}  {} seconds/iteration'.format(
                epoch,
                args.epochs,
                (batch_idx + 1) * len(data),
                len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                lr,
                acc,
                loss.item(),
                (time.time() - start_time)/(batch_idx + 1)))

        total_loss['cpc'] += cpc_loss.item()
        total_loss['knn'] += knn_loss.item()
        total_loss['total'] += loss.item()

    for k, v in total_loss.items():
        total_loss[k] /= len(train_loader)

    return total_loss
