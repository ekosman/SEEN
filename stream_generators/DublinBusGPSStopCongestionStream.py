import logging
import re
from functools import partial

import pandas as pd
from tqdm import tqdm
import numpy as np

from data_structures.Event import Event, CONTINUOUS


class DublinBusGPSStopCongestionStream:
	ENTER_STOP_CONGESTION = 0
	LEAVE_STOP_CONGESTION = 1
	ENTER_STOP_NON_CONGESTION = 2
	LEAVE_STOP_NON_CONGESTION = 3
	WAIT_IN_STOP = 4
	CONGESTION = 5  # while not in a stop

	@staticmethod
	def id_(x):
		return x

	def __init__(self,
				 data_csv,
				 length=None):
		self.data_csv = data_csv
		self.data_df = None
		self.length = length
		self.i = 0
		self.data = []
		self.done = False
		self.digitizers = [
			partial(np.digitize, bins=np.linspace(-6.6, -6, 80)),
			partial(np.digitize, bins=np.linspace(52, 53.6, 80)),
			partial(np.digitize, bins=np.linspace(-1500, 1000, 80)),
			DublinBusGPSStopCongestionStream.id_,
			DublinBusGPSStopCongestionStream.id_,
			DublinBusGPSStopCongestionStream.id_,
			partial(np.digitize, bins=np.linspace(0, 100, 80)),
		]
		self.header = [
			'Timestamp',
			'Line ID',
			'Direction',
			'Journey Pattern ID',
			'Time Frame',
			'Vehicle Journey ID',
			'Operator',
			'Congestion',
			'Lon WGS84',
			'Lat WGS84',
			'Delay',
			'Block ID',
			'Vehicle ID',
			'Stop ID',
			'At Stop',
			'Speed',
		]
		self.event_types = [
			self.enter_stop_congestion_event(),
			self.leave_stop_congestion_event(),
			self.enter_stop_non_congestion_event(),
			self.leave_stop_non_congestion_event(),
			self.wait_in_stop_event(),
			self.congestion_event(),
		]
		self.event_types_actual = [
			DublinBusGPSStopCongestionStream.ENTER_STOP_CONGESTION,
			DublinBusGPSStopCongestionStream.LEAVE_STOP_CONGESTION,
			DublinBusGPSStopCongestionStream.ENTER_STOP_NON_CONGESTION,
			DublinBusGPSStopCongestionStream.LEAVE_STOP_NON_CONGESTION,
			DublinBusGPSStopCongestionStream.WAIT_IN_STOP,
			DublinBusGPSStopCongestionStream.CONGESTION]
		self.attrs = [
			# 'Direction',
			# 'Journey Pattern ID',
			# 'Time Frame',
			# The start date of the production time table - in Dublin the production time table starts at 6am and ends at 3am
			# 'Vehicle Journey ID',  # A given run on the journey pattern
			# 'Operator',  # Bus operator, not the driver
			'Lon WGS84',
			'Lat WGS84',
			'Delay',  # seconds, negative if bus is ahead of schedule
			'Block ID',  # a section ID of the journey pattern
			'Vehicle ID',
			'Stop ID',
			'Speed',
		]
		self.attr_2_idx = {attr: idx for idx, attr in enumerate(self.attrs)}

		self.vehicle_at_stop = dict()

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
		self.data_df = pd.read_csv(self.data_csv, sep=',', names=self.header, chunksize=1)
		return self

	def __len__(self):
		if self.data is not None:
			return len(self.data)
		if self.length is not None:
			return self.length
		l = 0
		with open(self.data_csv, newline='') as csvfile:
			l = sum([1 for _ in csvfile])

		self.length = l
		return l

	def __repr__(self):
		return f"""
======== DublinBusGPSStopCongestionStream ========
length: {self.__len__()}
	"""

	def generate_event(self, type_, attrs, row):
		if row is not None:
			return Event(type=type_).from_list(
				[(CONTINUOUS, self.digitizers[self.attr_2_idx[attr]](row[attr].values[0])) for attr in attrs])
		else:
			return Event(type=type_).from_list(
				[(CONTINUOUS, 1) for _ in attrs])

	def enter_stop_congestion_event(self, row=None):
		attrs = [
			'Stop ID',
			'Lon WGS84',
			'Lat WGS84',
			'Delay',  # seconds, negative if bus is ahead of schedule
			'Block ID',  # a section ID of the journey pattern
			'Vehicle ID',
			'Speed',
		]
		return self.generate_event(DublinBusGPSStopCongestionStream.ENTER_STOP_CONGESTION, attrs, row)

	def leave_stop_congestion_event(self, row=None):
		attrs = [
			'Stop ID',
			'Lon WGS84',
			'Lat WGS84',
			'Delay',  # seconds, negative if bus is ahead of schedule
			'Block ID',  # a section ID of the journey pattern
			'Vehicle ID',
			'Speed',
		]
		return self.generate_event(DublinBusGPSStopCongestionStream.LEAVE_STOP_CONGESTION, attrs, row)

	def enter_stop_non_congestion_event(self, row=None):
		attrs = [
			'Stop ID',
			'Lon WGS84',
			'Lat WGS84',
			'Delay',  # seconds, negative if bus is ahead of schedule
			'Block ID',  # a section ID of the journey pattern
			'Vehicle ID',
			'Speed',
		]
		return self.generate_event(DublinBusGPSStopCongestionStream.ENTER_STOP_NON_CONGESTION, attrs, row)

	def leave_stop_non_congestion_event(self, row=None):
		attrs = [
			'Stop ID',
			'Lon WGS84',
			'Lat WGS84',
			'Delay',  # seconds, negative if bus is ahead of schedule
			'Block ID',  # a section ID of the journey pattern
			'Vehicle ID',
			'Speed',
		]
		return self.generate_event(DublinBusGPSStopCongestionStream.LEAVE_STOP_NON_CONGESTION, attrs, row)

	def wait_in_stop_event(self, row=None):
		attrs = [
			'Stop ID',
			'Lon WGS84',
			'Lat WGS84',
			'Delay',  # seconds, negative if bus is ahead of schedule
			'Block ID',  # a section ID of the journey pattern
			'Vehicle ID',
			'Speed',
		]
		return self.generate_event(DublinBusGPSStopCongestionStream.WAIT_IN_STOP, attrs, row)

	def congestion_event(self, row=None):
		attrs = [
			'Lon WGS84',
			'Lat WGS84',
			'Delay',  # seconds, negative if bus is ahead of schedule
			'Block ID',  # a section ID of the journey pattern
			'Vehicle ID',
			'Speed',
		]
		return self.generate_event(DublinBusGPSStopCongestionStream.CONGESTION, attrs, row)

	def __next__(self):
		while True:
			if self.i == len(self):
				self.done = True
				self.data_df = None
				raise StopIteration

			if self.done:
				event = self.data[self.i]
			else:
				try:
					row = self.data_df.get_chunk()
				except StopIteration:
					self.done = True
					self.data_df = None
					raise StopIteration

				if row.isnull().values.any():
					continue

				congestion = row['Congestion'].values[0]
				at_stop = row['At Stop'].values[0]
				vehicle_id = row['Vehicle ID'].values[0]
				if vehicle_id not in self.vehicle_at_stop and at_stop == 0 and congestion == 0:
					event = None
				elif vehicle_id not in self.vehicle_at_stop and at_stop == 0 and congestion == 1:
					event = self.congestion_event(row)
				elif vehicle_id not in self.vehicle_at_stop and at_stop == 1 and congestion == 0:
					event = self.enter_stop_non_congestion_event(row)
				elif vehicle_id not in self.vehicle_at_stop and at_stop == 1 and congestion == 1:
					event = self.enter_stop_congestion_event(row)
				elif self.vehicle_at_stop[vehicle_id] == 0 and at_stop == 0 and congestion == 0:
					event = None
				elif self.vehicle_at_stop[vehicle_id] == 0 and at_stop == 0 and congestion == 1:
					event = self.congestion_event(row)
				elif self.vehicle_at_stop[vehicle_id] == 0 and at_stop == 1 and congestion == 0:
					event = self.enter_stop_non_congestion_event(row)
				elif self.vehicle_at_stop[vehicle_id] == 0 and at_stop == 1 and congestion == 1:
					event = self.enter_stop_congestion_event(row)

				elif self.vehicle_at_stop[vehicle_id] == 1 and at_stop == 0 and congestion == 0:
					event = self.leave_stop_non_congestion_event(row)
				elif self.vehicle_at_stop[vehicle_id] == 1 and at_stop == 0 and congestion == 1:
					event = self.leave_stop_congestion_event(row)
				elif self.vehicle_at_stop[vehicle_id] == 1 and at_stop == 1 and congestion == 0:
					event = self.wait_in_stop_event(row)
				elif self.vehicle_at_stop[vehicle_id] == 1 and at_stop == 1 and congestion == 1:
					event = self.wait_in_stop_event(row)
				else:
					raise Exception("Missed a case!")

				self.vehicle_at_stop[vehicle_id] = at_stop

				if event is None:
					continue

				self.data.append(event)

			self.i += 1
			return event


if __name__ == '__main__':
	data_csv = r'C:\Users\eitan\PycharmProjects\GAStockPrediction\bayes\data\dublin_with_time.csv'
	stream = DublinBusGPSStopCongestionStream(data_csv=data_csv, length=358200)

	for e in tqdm(stream):
		print(e)
