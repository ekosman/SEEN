import torch
import torch.nn as nn
import numpy as np

from network.events_encoder import EventsEncoder
from network.network_tools import init_gru_hidden


class EventsNetCPC(nn.Module):
    def __init__(self, events_attributes, embedding_dim, context_dim, samples_to_predict):
        super(EventsNetCPC, self).__init__()
        self.embedding_dim = embedding_dim
        self.context_dim = context_dim
        self.samples_to_predict = samples_to_predict
        self.encoder = EventsEncoder(events_attributes=events_attributes,
                                     embedding_dim=embedding_dim)

        self.num_gru_layers = 2
        self.bidirectional = False
        self.num_gru_directions = 2 if self.bidirectional else 1
        self.rnn = nn.GRU(embedding_dim, context_dim, num_layers=self.num_gru_layers, bidirectional=self.bidirectional,
                          batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(context_dim, embedding_dim) for _ in range(samples_to_predict)])

        self.softmax = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()

    def forward(self, x, hidden=None):
        batch_size = len(x)

        if hidden is None:
            hidden = init_gru_hidden(self.num_gru_layers, self.num_gru_directions, batch_size, self.context_dim)

        z = self.encoder(x)

        sequence_length = z.shape[1]
        t_samples = sequence_length - self.samples_to_predict
        # z = z.transpose(1, 2)
        nce = 0  # average over timestep and batch

        input_seq = z[:, :-self.samples_to_predict, :]
        predict_samples = z[:, -self.samples_to_predict:, :]
        output, hidden = self.rnn(input_seq, hidden)

        c_t = output[:, -1, :]

        return c_t

        # pred = []
        # for i in np.arange(self.samples_to_predict):
        #     linear = self.Wk[i]
        #     pred_t_plus_i = linear(c_t)
        #     embedding_t_plus_i = predict_samples[:, i, :]
        #     corr_t_plus_i = embedding_t_plus_i @ pred_t_plus_i.T
        #     pred.append(linear(c_t))
        # for i in np.arange(0, self.timestep):
        #     total = torch.mm(predict_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 8*8
        #     correct = torch.sum(
        #         torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch_size)))  # correct is a tensor
        #     nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        # nce /= -1. * batch_size * self.timestep
        # accuracy = 1. * correct.item() / batch_size
        #
        # return accuracy, nce, hidden
