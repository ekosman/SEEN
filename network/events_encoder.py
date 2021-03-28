import torch
import torch.nn as nn

from network.event_abstractor import EventAbstractor


class EventsEncoder(nn.Module):
    def __init__(self, events_attributes, embedding_dim, n_steps=3, activation='tanh'):
        super(EventsEncoder, self).__init__()
        self.encoder = nn.ModuleList()

        for key, num_attributes in enumerate(events_attributes):
            self.encoder.append(EventAbstractor(num_attributes=num_attributes,
                                                embedding_dim=embedding_dim,
                                                n_steps=n_steps,
                                                activation=activation, ))

    def forward_sequence(self, events):
        return torch.stack([self.encoder[e.type](e.to_tensor()) for e in events])

    def forward(self, events):
        return torch.stack([self.forward_sequence(seq) for seq in events])
