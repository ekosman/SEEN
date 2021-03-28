import logging
from functools import partial

import pandas as pd
from tqdm import tqdm
import numpy as np

from data_structures.Event import Event, CONTINUOUS


class BarCrawlStream:
	def __init__(self,
				 data_csv,
				 pid_2_phone_csv,
				 length=None):
		pid_2_phone_df = pd.read_csv(pid_2_phone_csv)
		self.pid_2_phone = dict(zip(pid_2_phone_df['pid'], pid_2_phone_df['phonetype']))
		self.idx_2_pids = list(pid_2_phone_df['pid'])
		self.pid_2_idx = {phone: idx for idx, phone in enumerate(self.idx_2_pids)}
		self.data_csv = data_csv
		self.data_df = None
		self.length = length
		self.i = 0
		self.data = []
		self.done = False
		self.digitizer = partial(np.digitize, bins=np.linspace(-25, 25, 200))
		self.event_types = self.generate_event_types()

	def generate_event_types(self):
		types = ['BK7610',
				 'BU4707',
				 'CC6740',
				 'DC6359',
				 'DK3500',
				 'HV0618',
				 'JB3156',
				 'JR8022',
				 'MC7070',
				 'MJ8002',
				 'PC6771',
				 'SA0297',
				 'SF3079'
				 ]

		return [Event(type=self.pid_2_idx[pid], num_attributes=3) for pid in types]

	def get_names_for_network_states(self, include_time=False):
		if include_time:
			names = [None] * len(self.subset)
			for e in self.subset:
				names[self.get_event_hash(e)] = f'E{e.type}.time'
		else:
			names = []
		names += [name for _, name in self.attr_names.items()]
		return names

	def dump_repr(self, fp):
		fp.write(repr(self))

	def get_events_in_patterns(self):
		def get_events_in_pattern(pattern):
			events = set()
			for cond in pattern:
				events.update(cond.get_events())
			return events

		return [get_events_in_pattern(pattern) for pattern in self.patterns]

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
		self.subset = [e for e in self.event_types if e.type in subset]
		self.attr_hash, self.attr_names, self.event_hash = self.generate_hashes(self.subset)

	def generate_hashes(self, event_types):
		logging.info("Generating hashes...")
		attrs = dict()
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

				i += 1

		return attrs, names, events

	def __iter__(self):
		self.i = 0
		self.data_df = pd.read_csv(self.data_csv, sep=',', header=0, chunksize=1)
		return self

	def __len__(self):
		if self.length is not None:
			return self.length
		l = 0
		with open(self.data_csv, newline='') as csvfile:
			l = sum([1 for _ in csvfile])

		self.length = l
		return l

	def __repr__(self):
		return f"""
======== BarCrawlerStream ========
length: {self.__len__()}
	"""

	def __next__(self):
		if self.i == len(self):
			self.done = True
			self.data_df = None
			raise StopIteration

		if self.done:
			e = self.data[self.i]
		else:
			row = self.data_df.get_chunk()
			e = Event(type=self.pid_2_idx[row['pid'].values[0]]).from_list(
				[(CONTINUOUS, self.digitizer(row[col].values[0])) for col in ['x', 'y', 'z']])
			self.data.append(e)

		self.i += 1
		return e


if __name__ == '__main__':
	data_csv = r'C:\Users\eitan\PycharmProjects\GAStockPrediction\bayes\data\Bar_Crawl_Detecting_Heavy_Drinking_Data_Set\all_accelerometer_data_pids_13.csv'
	pid_2_phone_csv = r'C:\Users\eitan\PycharmProjects\GAStockPrediction\bayes\data\Bar_Crawl_Detecting_Heavy_Drinking_Data_Set\phone_types.csv'
	stream = BarCrawlStream(data_csv=data_csv, pid_2_phone_csv=pid_2_phone_csv, length=100)

	for row in tqdm(stream):
		pass

# data_df = pd.read_csv(data_csv, sep=',', header=0, chunksize=1, iterator=True)
# x = []
# y = []
# z = []
# for chunk in tqdm(data_df):
# 	x.append(chunk['x'].values[0])
# 	y.append(chunk['y'].values[0])
# 	z.append(chunk['z'].values[0])
#
# import numpy as np
#
# print(f"X: min: {np.min(x)}       max: {np.max(x)}")
# print(f"Y: min: {np.min(y)}       max: {np.max(y)}")
# print(f"Z: min: {np.min(z)}       max: {np.max(z)}")
