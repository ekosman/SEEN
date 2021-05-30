import torch
import torch.nn as nn
import numpy as np
from torch import optim
# from torchsummary import summary
from tqdm import tqdm
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, code_dim, steps):
        super(AutoEncoder, self).__init__()
        self.steps = np.linspace(input_dim, code_dim, steps + 1, dtype=int)
        self.enc, self.dec = self.build_recursive(steps + 1)
        self.enc = nn.Sequential(*self.enc)
        self.dec = nn.Sequential(*self.dec)

    def build_recursive(self, num_steps):
        if num_steps == 1:
            return [], []

        in_size = self.steps[-num_steps]
        mid_size = self.steps[-num_steps + 1]

        enc = [nn.Linear(in_size, mid_size, bias=True),
            nn.LeakyReLU()]
        dec = [nn.Linear(mid_size, in_size)]

        middle_enc, middle_dec = self.build_recursive(num_steps - 1)

        if len(middle_dec) != 0:
            middle_dec.append(nn.LeakyReLU())
        return enc + middle_enc, middle_dec + dec

    def forward(self, x, return_intermidiate=False):
        m = self.enc(x)
        y = self.dec(m)

        if return_intermidiate:
            return y, m

        return y

    @property
    def parameters_count(self):
        return sum([p.numel() for p in self.parameters()])

    def fit(self, dataset, batch_size, epochs=500, log_every=20, lr=5e-3):
        self.train()
        data_iter = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=1,  # 4, # change this part accordingly
                                             pin_memory=True)
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for iteration, batch in enumerate(data_iter):
                outputs = self(batch)
                mask = (batch != -1)
                loss = F.mse_loss(outputs, batch, reduction='none') * mask
                loss = loss.sum() / mask.sum()
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iteration % log_every == 0:
                    print(f"Epoch {epoch} / {epochs}     iteration {iteration} / {len(data_iter)}    Loss: {total_loss / (iteration + 1)}")
            losses.append(total_loss / (iteration + 1))

        return losses

    def calculate_intermidiate(self, dataset, batch_size=1000):
        self.eval()
        data_iter = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=1,  # 4, # change this part accordingly
                                                pin_memory=True)
        interm_all = None
        with torch.no_grad():
            for batch in tqdm(data_iter):
                outputs, interm = self(batch, True)

                if interm_all is not None:
                    interm_all = np.concatenate([interm_all, interm.detach().numpy()])
                else:
                    interm_all = interm

        return interm_all


if __name__ == '__main__':
    model = AutoEncoder(167, 50, 3)
    # summary(model, input_size=(167,))
