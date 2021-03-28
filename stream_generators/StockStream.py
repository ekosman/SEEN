from functools import partial

import numpy as np
from tqdm import tqdm

from data_structures.Event import CONTINUOUS
from stream_generators.dataset import get_custom_dataset
from stream_generators.real_world_stream import RealWorldStream, Timestamp


class StockStream(RealWorldStream):
    def __init__(self,
                 data_csv,
                 length=None):
        header = [
            'symbol',
            'date',
            'bid',
            'high',
            'low',
            'close',
            'volume',
            'diff',
            'ma5',
            'ma10',
        ]
        attrs = [
            'bid',
            'volume',
            'diff',
            'ma5',
            'ma10',
        ]

        types = [CONTINUOUS] * len(attrs)

        super(StockStream, self).__init__(data_csv=data_csv,
                                          length=length,
                                          types=types,
                                          header=header,
                                          # header=None,
                                          attrs=attrs,
                                          type_name='symbol',
                                          timestamp_column='date',
                                          )

        self.need_type_aggregation = False

    def cvt_id_to_event(self, id):
        return self.event_types_actual[id]

    def cvt_id_to_attr(self, id):
        return self.attrs[id]

    # def get_event_type(self, symbol):
    #     return self.event_types_actual_2_idx[symbol]

    def parse_timestamp(self, timestamp):
        timestamp = str(timestamp)
        return Timestamp(int(timestamp[:4]), int(timestamp[4:6]), int(timestamp[6:8]), int(timestamp[8:10]), int(timestamp[10:12]), 0)


if __name__ == '__main__':
    data_csv = r'C:\Users\eitan\PycharmProjects\GAStockPrediction\bayes\data\stocks_all\all_stocks_01.txt'
    stream = StockStream(data_csv=data_csv, length=50000)
    iter(stream)
    dataset = get_custom_dataset(stream, stream.data, 2, return_base_index=False, stride=1)
    for e in tqdm(dataset):
        print(e)
