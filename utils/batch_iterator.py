import random
from math import ceil, floor


class Batch:
    def __init__(self, iterable, batch_size, shuffle=False, return_last=False):
        self.iterable = iterable
        self.batch_size = batch_size
        self.indices = list(range(len(iterable)))
        self.indices_iter = None
        self.finish = False
        self.shuffule = shuffle
        self.return_last = return_last

    def __len__(self):
        length = len(self.iterable) / self.batch_size
        return ceil(length) if self.return_last else floor(length)

    def __iter__(self):
        random.shuffle(self.indices)
        self.indices_iter = iter(self.indices)
        self.finish = False
        return self

    def __next__(self):
        res = []

        while True:
            if self.finish:
                raise StopIteration

            if len(res) == self.batch_size:
                return res

            try:
                item_idx = self.indices_iter.__next__()
                res.append(self.iterable[item_idx])
            except StopIteration:
                self.finish = True

                if self.return_last or len(res) == self.batch_size:
                    if len(res) != 0:
                        return res

                raise StopIteration


if __name__ == '__main__':
    x = list(range(24))
    b = Batch(x, batch_size=6, return_last=False, shuffle=True)
    for i in b:
        print(i)