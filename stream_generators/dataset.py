import time
from math import floor
from datetime import datetime, timedelta

from stream_generators.StreamGeneratorSingleEventSubset import StreamGeneratorSingleEventSubset
from stream_generators.real_world_stream import RealWorldStream


class RealWorldDataset:
    def __init__(self, data, seq_len, stride=1, return_base_index=False):
        self.data = data
        self._seq_len = seq_len if isinstance(seq_len, timedelta) else timedelta(seconds=seq_len)
        self.stride = timedelta(seconds=stride)
        self.return_base_index = return_base_index
        self.base_time = data[0].datetime
        self.done = False

    @property
    def seq_len(self):
        return self._seq_len

    @seq_len.setter
    def seq_len(self, value):
        self._seq_len = timedelta(seconds=value)

    def __iter__(self):
        self.base = 0
        self.base_time = self.data[0].datetime
        self.done = False
        return self

    def __next__(self):
        if self.done:
            raise StopIteration

        current_date = self.base_time
        current_stride = current_date + self.stride
        current_base = self.forward(self.base, current_date)
        current_date_end = current_date + self.seq_len
        ret = []
        i = 0

        while True:
            if current_base + i == len(self.data):
                # Out of range of the data
                self.done = True
                return ret

            e = self.data[current_base + i]

            if e.datetime >= current_date_end:
                # Out of range of sequence length
                self.base_time = max(self.base_time + self.stride, e.datetime - self.stride)
                return ret

            if e.datetime >= current_date:
                ret.append(e)

            if e.datetime <= current_stride:
                self.base = current_base + i

            i += 1

    def forward(self, index, target_date):
        while self.data[index].datetime < target_date:
            index += 1

        return index


class CustomDataset:
    def __init__(self, data, seq_len, stride=1, return_base_index=False):
        self.data = data
        self._seq_len = seq_len
        self.stride = stride
        self.return_base_index = return_base_index
        self.base = 0

    @property
    def seq_len(self):
        return self._seq_len

    @seq_len.setter
    def seq_len(self, value):
        self._seq_len = value

    def __iter__(self):
        self.base = 0
        return self

    def __len__(self):
        return floor((len(self.data) - self.seq_len) / self.stride) + 1

    def __next__(self):
        current_base = self.base
        if current_base + self.seq_len > len(self.data):
            raise StopIteration()
        self.base += self.stride
        if self.return_base_index:
            return self.data[current_base: current_base + self.seq_len], current_base
        else:
            return self.data[current_base: current_base + self.seq_len]

    def __getitem__(self, index):
        base = index * self.stride
        if self.return_base_index:
            return self.data[base: base + self.seq_len], base
        else:
            return self.data[base: base + self.seq_len]


def get_custom_dataset(stream, data, seq_len, return_base_index=False, stride=1):
    if isinstance(stream, RealWorldStream):
        return RealWorldDataset(data=data, seq_len=seq_len, stride=1, return_base_index=return_base_index)
    elif isinstance(stream, StreamGeneratorSingleEventSubset):
        return CustomDataset(data=data, seq_len=seq_len, stride=stride, return_base_index=return_base_index)
    else:
        raise Exception(f"Dataset for {type(stream)} is not implemented")


if __name__ == '__main__':
    data = list(range(50))
    dataset = CustomDataset(data, 5, 5)
    print(len(dataset))
    for d in dataset:
        print(d)