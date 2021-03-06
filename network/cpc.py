import torch
import torch.nn as nn
import numpy as np

hidden_dim = 256


class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, padding, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.norm = nn.BatchNorm1d(out_features)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class CDCK2(nn.Module):
    def __init__(self, timestep, batch_size, seq_len, in_features, device):
        super(CDCK2, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep
        self.device = device

        strides = [2, 2, 2, 2, 2]
        paddings = [2, 2, 2, 2, 2]
        kernel_sizes = [5, 5, 5, 5, 5]
        conv_steps = [in_features, 16, 32, 64, 128, 256]
        self.embedding_dim = conv_steps[-1]
        self.encoder = nn.Sequential(
            *[ConvBlock(in_, out_, stride, padding, kernel_size) for in_, out_, stride, padding, kernel_size in
              zip(conv_steps[:-1], conv_steps[1:], strides, paddings, kernel_sizes)])

        self.gru = nn.GRU(self.embedding_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(hidden_dim, self.embedding_dim) for i in range(timestep)])
        self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.kaiming_normal_(self.gru.__getattr__(p), mode='fan_out', nonlinearity='relu')

        self.apply(_weights_init)

    def encode(self, x):
        return self.encoder(x).view(x.shape[0], -1)

    @staticmethod
    def init_hidden(batch_size):
        return torch.zeros(1, batch_size, hidden_dim)

    def forward(self, x, hidden):
        batch = x.size()[0]

        z = self.encoder(x)
        compress_ratio = x.shape[2] / z.shape[2]
        t_samples = torch.randint(int(self.seq_len / compress_ratio - self.timestep),
                                  size=(1,)).long()
        z = z.transpose(1, 2)
        nce = 0
        encode_samples = torch.empty((self.timestep, batch, self.embedding_dim)).float().to(self.device)
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, self.embedding_dim)
        forward_seq = z[:, :t_samples + 1, :]
        output, hidden = self.gru(forward_seq, hidden)
        c_t = output[:, -1, :].view(batch, hidden_dim)
        pred = torch.empty((self.timestep, batch, self.embedding_dim)).float().to(self.device)
        correct = 0
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                          torch.arange(0, batch).to(self.device)))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        accuracy = 1. * correct.item() / (batch * self.timestep)
        return accuracy, nce, hidden, output[:, -1, :]

    def predict(self, x, hidden):
        z = self.encoder(x)
        z = z.transpose(1, 2)
        output, hidden = self.gru(z, hidden)
        return output[:, -1, :]
