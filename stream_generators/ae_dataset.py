import torch
from torch.utils import data
import numpy as np


class AEDataset(data.Dataset):
	def __init__(self, samples):
		samples = np.stack(samples).astype(np.float)
		samples_filtered = samples * (samples != -1)
		mins = samples_filtered.min(axis=0)
		maxs = samples_filtered.max(axis=0)

		for i in range(samples.shape[1]):
			min_ = mins[i]
			max_ = maxs[i]
			col = samples[:, i]
			non_zero = (col[col != -1] - min_) / (max_ - min_)
			samples[col != -1, i] = non_zero

		self.samples = samples

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, item):
		return torch.tensor(self.samples[item]).float()
