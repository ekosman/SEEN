import torch
import torch.nn as nn
import numpy as np

hidden_dim = 64


class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, padding, kernel_size):
        super(ConvBlock, self).__init__()
        self.encoder = nn.Sequential(  # downsampling factor = 160
            nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.encoder(x)


class CDCK2(nn.Module):
    def __init__(self, timestep, batch_size, seq_len, in_features):
        super(CDCK2, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.timestep = timestep

        strides = [5, 2, 2, 2, 2, 2]
        paddings = [3, 2, 1, 1, 1, 1]
        kernel_sizes = [10, 8, 4, 4, 4, 3]
        conv_steps = [in_features, 8, 16, 32, 64, 128, 256]
        self.embedding_dim = conv_steps[-1]
        self.encoder = nn.Sequential(*[ConvBlock(in_, out_, stride, padding, kernel_size) for in_, out_, stride, padding, kernel_size in
                                       zip(conv_steps[:-1], conv_steps[1:], strides, paddings, kernel_sizes)])

        self.gru = nn.GRU(self.embedding_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(hidden_dim, self.embedding_dim) for i in range(timestep)])
        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

        def _weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # initialize gru
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
        # print(f"1: {x.shape[0]}")
        t_samples = torch.randint(int(self.seq_len / 160 - self.timestep),
                                  size=(1,)).long()  # randomly pick time stamps
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # print(f"2: {z.shape[0]}")
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1, 2)
        # print(f"3: {z.shape[0]}")
        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.embedding_dim)).float()  # e.g. size 12*8*512
        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z[:, t_samples + i, :].view(batch, self.embedding_dim)  # z_tk e.g. size 8*512
        forward_seq = z[:, :t_samples + 1, :]  # e.g. size 8*100*512
        # print(f"4: {forward_seq.shape[0]}")
        output, hidden = self.gru(forward_seq, hidden)  # output size e.g. 8*100*256
        c_t = output[:, t_samples, :].view(batch, hidden_dim)  # c_t e.g. size 8*256
        pred = torch.empty((self.timestep, batch, self.embedding_dim)).float()  # e.g. size 12*8*512
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)  # Wk*c_t e.g. size 8*512
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
            correct = torch.sum(
                torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        nce /= -1. * batch * self.timestep
        accuracy = 1. * correct.item() / batch

        return accuracy, nce, hidden

    def predict(self, x, hidden):
        batch = x.size()[0]
        # input sequence is N*C*L, e.g. 8*1*20480
        z = self.encoder(x)
        # encoded sequence is N*C*L, e.g. 8*512*128
        # reshape to N*L*C for GRU, e.g. 8*128*512
        z = z.transpose(1, 2)
        output, hidden = self.gru(z, hidden)  # output size e.g. 8*128*256

        return output, hidden  # return every frame
        # return output[:,-1,:], hidden # only return the last frame per utt
