import logging
import re
import pandas as pd
from tqdm import tqdm
from collections import namedtuple

from Rules.rules import Rule
from data_structures.Event import Event, CONTINUOUS, DISCRETE, id_fn

Timestamp = namedtuple(  # pylint: disable=C0103
    'Timestamp', ('year', 'month', 'day', 'hour', 'minute', 'second'))


class RealWorldStream:
    @staticmethod
    def id_(x):
        return x

    def __init__(self, data_csv, length, types, header, attrs, type_name, chunk_size=5000, timestamp_column=None, start_ts=0):
        self.data_csv = data_csv
        self.length = length
        self.types = types
        self.header = header
        self.attrs = attrs
        self.type_name = type_name
        self.chunk_size = chunk_size
        self.timestamp_column = timestamp_column
        self.start_ts = start_ts

        self.attr_2_idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.data_df = None
        self.i = 0
        self.data = None
        self.timestamps = None
        self.done = False
        self.event_types_actual = []
        self.event_types_actual_2_idx = dict()
        self.event_types = dict()
        self.filter = None

    def cvt_raw_names_to_code(self, name):
        pattern = r'(.*).(.*)'
        match = re.findall(pattern, name)[0]
        return f'E{self.event_types_actual_2_idx[match[0]]}.{self.attrs[match[1]]}'

    def set_filter(self, fn):
        self.filter = fn

    def __repr__(self):
        return f"""
======== {type(self)} ========
length: {self.__len__()}
duration = {self.timestamps[-1] - self.timestamps[0]}
"""

    @property
    def story_length(self):
        return len(self.timestamps)

    def cvt_descriptor_to_meaningful_name(self, descriptor):
        """
        Transforms and descriptor like E0.0 to Ball.x
        """
        event, attr = Rule.parse_event_and_attr(descriptor)
        return f"{self.cvt_id_to_event(event)}.{self.cvt_id_to_attr(attr)}"

    def cvt_event_to_id(self, event):
        return self.event_types_actual_2_idx[event]

    def cvt_id_to_event(self, id):
        """
        The general conversion function
        """
        return self.event_types_actual[id]

    def cvt_id_to_attr(self, id):
        """
        The general conversion function
        """
        return id

    def get_attr_type(self, attr_name):
        """
        Retrieves the attribute type from the name such as E0.2
        :param attr_name: a tuple describing both event and attribute descriptors.
                            a variable called E4.1 would be described as (4,1)
        :return:
        """
        return self.get_attr_type_from_idx(attr_name[1])

    def is_attr_discrete(self, attr_name):
        """

        :param attr_name:
        :return:
        """
        return self.get_attr_type(attr_name) == DISCRETE

    def get_reverse_mapping_for_network_states(self, include_time=False):
        if include_time:
            maps = [None] * len(self.subset)
            for e in self.subset:
                maps[self.get_event_hash(e)] = id_fn
        else:
            maps = []

        maps += [mapping for _, mapping in self.attrs_maps.items()]
        return maps

    def get_attr_type_from_idx(self, idx):
        return self.types[idx]

    def get_names_for_network_states(self, include_time=False):
        if include_time:
            names = [None] * len(self.subset)
            for e in self.subset:
                names[self.get_event_hash(e)] = f'E{e.type}.time'
        else:
            names = []
        names += [name for _, name in self.attr_names.items()]
        return names

    def get_events_of_network_states(self, include_time=False):
        if include_time:
            names = [None] * len(self.subset)
            for e in self.subset:
                names[self.get_event_hash(e)] = str(e.type)
        else:
            names = []

        pattern = r'E(\d+).(.*)'
        names += [re.findall(pattern, name)[0][0] for _, name in self.attr_names.items()]
        return names

    def dump_repr(self, fp):
        fp.write(repr(self))

    @property
    def n_attrs(self):
        return len([attr for e in self.subset for attr in e.attrs])

    @property
    def n_events(self):
        return len(self.subset)

    def get_event_hash(self, event_):
        return self.event_hash[event_.type]

    def get_attr_hash(self, event_, attr_index):
        return self.attr_hash[(event_.type, attr_index)]

    def get_attr_name(self, event_, attr_index):
        return f'E{str(event_.type)}.{attr_index}'

    def update_subset(self, subset):
        self.subset = [e for type_, e in self.event_types.items() if e.type in subset]
        self.attrs_maps, self.attr_hash, self.attr_names, self.event_hash = self.generate_hashes(self.subset)

    def generate_hashes(self, event_types):
        logging.info("Generating hashes...")
        attrs = dict()
        attrs_maps = dict()
        names = dict()
        events = dict()
        i = 0
        k = 0
        for e in event_types:
            events[e.type] = k
            k += 1
            for j, attr in enumerate(e.attrs):
                attrs[(e.type, j)] = i
                names[i] = self.get_attr_name(e, j)
                attrs_maps[(e.type, j)] = attr.reverse

                i += 1

        return attrs_maps, attrs, names, events

    def __iter__(self):
        if self.data is None:
            i = 0
            data_df = pd.read_csv(self.data_csv, sep=',', names=self.header, chunksize=self.chunk_size)
            data = []
            bar = tqdm(total=self.length)
            process_events = self.process_events if self.filter is None else self.process_events_with_filter
            while i < self.length:
                try:
                    rows = data_df.get_chunk()
                except StopIteration:
                    break

                new_data = process_events(rows)
                data += new_data

                i += len(new_data)
                bar.update(len(new_data))

            self.data = data
            self.timestamps = sorted(list(set([e.datetime for e in data])))

        self.i = 0
        return self

    def parse_timestamp(self, timestamp):
        return timestamp

    def update_event_types(self, type_):
        if type_ not in self.event_types_actual:
            self.event_types_actual.append(type_)
            self.event_types_actual_2_idx[type_] = len(self.event_types_actual_2_idx)
            self.event_types[self.event_types_actual_2_idx[type_]] = Event(type=self.event_types_actual_2_idx[type_], num_attributes=len(self.attrs)).set_types(self.types)
            return True

        return False

    def get_event_type(self, type_):
        return self.event_types_actual_2_idx[type_]

    def get_timestamp(self, row):
        return row[self.timestamp_column]

    def __len__(self):
        return len(self.data) if self.data is not None else self.length

    def generate_event_from_row(self, type_, row):
        global_event = self.event_types[type_]
        global_event.update_from_list([row[attr] for attr in self.attrs])
        timestamp = self.get_timestamp(row)
        if timestamp < self.start_ts:
            return None

        return Event(type=type_, timestamp=self.parse_timestamp(timestamp)).from_list(
            [(attr_type, row[attr], global_attribute)
            for attr_type, attr, global_attribute in zip(self.types, self.attrs, global_event.attrs)])

    def process_event(self, row):
        if row[self.type_name] is None:
            return None

        type_ = self.agg(row[self.type_name]) if self.need_type_aggregation else row[self.type_name]

        self.update_event_types(type_)
        type_ = self.get_event_type(type_)

        return self.generate_event_from_row(type_, row)

    def process_events(self, rows):
        process_event_fn = self.process_event
        return [event for event in (process_event_fn(row) for i, row in rows.iterrows()) if event is not None]

    def process_events_with_filter(self, rows):
        process_event_fn = self.process_event
        filter = self.filter
        return [event for event in (filter(process_event_fn(row)) for i, row in rows.iterrows()) if event is not None]

    def __next__(self):
        if self.i == len(self):
            raise StopIteration

        event = self.data[self.i]
        self.i += 1
        return event