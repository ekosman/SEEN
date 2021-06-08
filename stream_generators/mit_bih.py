import pandas as pd
import torch.utils.data as data
import numpy as np
import torch


class MITBIH(data.Dataset):
    def __init__(self, dataset_path):
        """
        Args:
            dataset_path (str): path to csv data file
        """
        super(MITBIH, self).__init__()

        # loader params
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)

    def __repr__(self):
        return f"""
===== ClipLoader =====
dataset_path = {self.dataset_path}
n_records = {len(self)}
"""

    def __len__(self):
        """
        Retrieves the length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Method to access the i'th sample of the dataset
        Args:
            index: the sample index

        Returns: the i'th sample of this dataset
        """
        x = self.df.values[index]
        y = x[-1]
        x = x[:-1]
        x = x[np.newaxis, :]
        return torch.tensor(x).float(), torch.tensor(y).long()


if __name__ == '__main__':
    dataset = MITBIH(dataset_path=r"F:\Downloads\archive\mitbih_train.csv")
    print(dataset[0])
