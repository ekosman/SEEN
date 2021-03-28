from datetime import datetime
from functools import partial

import numpy as np
from tqdm import tqdm

from data_structures.Event import CONTINUOUS, DISCRETE
from stream_generators.real_world_stream import RealWorldStream, Timestamp


class DublinBusGPSStream(RealWorldStream):
    def __init__(self,
                 data_csv,
                 length=None):
        header = [
            'Timestamp',
            'Line ID',
            'Direction',
            'Journey Pattern ID',  # A given run on the journey pattern
            'Time Frame',
            # The start date of the production time table - in Dublin the production time table starts at 6am and ends at 3am
            'Vehicle Journey ID',
            'Operator',  # Bus operator, not the driver
            'Congestion',  # 0=no,1=yes
            'Lon WGS84',
            'Lat WGS84',
            'Delay',  # seconds, negative if bus is ahead of schedule
            'Block ID',  # a section ID of the journey pattern
            'Vehicle ID',
            'Stop ID',
            'At Stop',  # [0=no,1=yes]
        ]

        attrs = [
            'Line ID',
            'Operator',
            'Congestion',
            'Lon WGS84',
            'Lat WGS84',
            'Delay',
            'Block ID',
            'Stop ID',
            'At Stop',
        ]

        types = [
            DISCRETE,
            DISCRETE,
            DISCRETE,
            CONTINUOUS,
            CONTINUOUS,
            CONTINUOUS,
            DISCRETE,
            DISCRETE,
            DISCRETE,
        ]

        super(DublinBusGPSStream, self).__init__(data_csv=data_csv,
                                                 length=length,
                                                 types=types,
                                                 header=header,
                                                 attrs=attrs,
                                                 type_name='Vehicle ID',
                                                 timestamp_column='Timestamp')

        self.need_type_aggregation = False

    def get_timestamp(self, row):
        timestamp = super(DublinBusGPSStream, self).get_timestamp(row) // 10**6
        return timestamp

    def parse_timestamp(self, timestamp):
        timestamp = datetime.fromtimestamp(timestamp)
        return Timestamp(year=timestamp.year, month=timestamp.month, day=timestamp.day, hour=timestamp.hour, minute=timestamp.minute, second=timestamp.second)


if __name__ == '__main__':
    data_csv = r'C:\Users\eitan\PycharmProjectהשs\GAStockPrediction\bayes\data\siri.20130124.csv'
    stream = DublinBusGPSStream(data_csv=data_csv, length=358200)

    for row in tqdm(stream):
        print(row)
