import logging
from functools import partial

import pandas as pd
from tqdm import tqdm
import numpy as np

from data_structures.Event import Event, CONTINUOUS, DISCRETE
from stream_generators.real_world_stream import RealWorldStream


class MaritimeStream(RealWorldStream):
    def __init__(self,
                 data_csv,
                 length=None):
        header = ['sourcemmsi',
                  'navigationalstatus',
                  'rateofturn',
                  'speedoverground',
                  'courseoverground',
                  'trueheading',
                  'lon',
                  'lat'
                  ]

        digitizers = [
            RealWorldStream.id_,
            partial(np.digitize, bins=np.linspace(-140, 140, 80)),
            partial(np.digitize, bins=np.linspace(0, 103, 80)),
            partial(np.digitize, bins=np.linspace(0, 360, 80)),
            partial(np.digitize, bins=np.linspace(0, 360, 80)),
            partial(np.digitize, bins=np.linspace(-7, -3, 30)),
            partial(np.digitize, bins=np.linspace(47, 50, 30)),
        ]

        attrs = ['navigationalstatus',
                 'rateofturn',
                 'speedoverground',
                 'courseoverground',
                 'trueheading',
                 'lon',
                 'lat'
                 ]

        types = [
            DISCRETE,
            CONTINUOUS,
            CONTINUOUS,
            CONTINUOUS,
            CONTINUOUS,
            CONTINUOUS,
            CONTINUOUS,
        ]

        super(MaritimeStream, self).__init__(data_csv=data_csv,
                                             length=length,
                                             digitizers=digitizers,
                                             types=types,
                                             header=header,
                                             attrs=attrs,
                                             type_name='sourcemmsi')


if __name__ == '__main__':
    data_csv = r'C:\Users\eitan\PycharmProjects\GAStockPrediction\bayes\data\Maritime\nari_dynamic.csv'
    stream = MaritimeStream(data_csv=data_csv, length=100)

    for row in tqdm(stream):
        pass
