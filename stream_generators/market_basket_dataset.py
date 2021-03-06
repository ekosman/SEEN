import random
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class MarketBasketDataset(data.Dataset):
    def __init__(self,
                 dataset_path,
                 transform=None,
                 target_transform=None,
                 ):
        """
        Args:
            dataset_path (str): path to csv data file
        """
        super(MarketBasketDataset, self).__init__()

        # loader params
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.items = self.df.itemDescription.unique()
        self.items_to_idx = {item: idx for idx, item in enumerate(self.items)}
        self.itemsets = list(self.df.groupby(['Member_number', 'Date'])['itemDescription'].unique())
        self.set_itemsets = [set(x) for x in self.itemsets]
        self.transform = transform
        self.target_transform = target_transform

    @property
    def n_items(self):
        return len(self.items)

    def itemset_inclusion(self, itemset):
        itemset = set(itemset)
        return len([x for x in self.set_itemsets if x.issuperset(itemset)])

    def itemset_intersection(self, itemset):
        itemset = set(itemset)
        return len([x for x in self.set_itemsets if len(x.intersection(itemset)) != 0])

    def support(self, itemset):
        return self.itemset_inclusion(itemset) / len(self)

    def supp_exclude(self, include, exclude):
        return self.itemset_inclusion_exclusion(include, exclude) / len(self)

    def itemset_inclusion_exclusion(self, include, exclude):
        include = set(include)
        exclude = set(exclude)
        return len([x for x in self.set_itemsets if x.issuperset(include) and len(x.intersection(exclude)) == 0])

    def bond(self, itemset):
        return self.itemset_inclusion(itemset) / self.itemset_intersection(itemset)

    def __repr__(self):
        return f"""
===== ClipLoader =====
dataset_path = {self.dataset_path}
n_items = {self.n_items}
n_records = {len(self)}
"""

    def __len__(self):
        """
        Retrieves the length of the dataset.
        """
        return len(self.itemsets)

    def __getitem__(self, index):
        """
        Method to access the i'th sample of the dataset
        Args:
            index: the sample index

        Returns: the i'th sample of this dataset
        """
        in_ = deepcopy(self.itemsets[index])
        out_ = deepcopy(self.itemsets[index])
        if self.transform is not None:
            in_ = self.transform(in_)

        if self.target_transform is not None:
            out_ = self.target_transform(out_)

        return in_, out_


class BinaryEncodingTransform:
    def __init__(self, mapping):
        self.mapping = mapping
        self.vocab_size = len(self.mapping)

    def __call__(self, x):
        res = torch.zeros(self.vocab_size)
        idx = [self.mapping[x_] for x_ in x]
        res[idx] = 1
        return res


class RemoveItemsTransform:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if len(x) == 1:
            return x

        if random.random() < self.p:
            x = np.delete(x, np.random.randint(0, len(x)))

        return x
