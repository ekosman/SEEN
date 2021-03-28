from datetime import timedelta
from functools import partial

import numpy as np
from tqdm import tqdm

from data_structures.Event import CONTINUOUS
from stream_generators.real_world_stream import RealWorldStream, Timestamp

players = [
	'Balls',
	'Nick Gertje',
	'Dennis Dotterweich',
	'Niklas Waelzlein',
	'Wili Sommer',
	'Philipp Harlass',
	'Roman Hartleb',
	'Erik Engelhardt',
	'Sandro Schneider',
	'Leon Krapf',
	'Kevin Baer',
	'Luca Ziegler',
	'Ben Mueller',
	'Vale Reitstetter',
	'Christopher Lee',
	'Leon Heinze',
	'Leo Langhans',
	'Referee',
]
players_to_id = {player: i for i, player in enumerate(players)}

sid_2_player = {
	4: 'Balls',
	8: 'Balls',
	10: 'Balls',
	12: 'Balls',
	13: 'Nick Gertje',
	14: 'Nick Gertje',
	97: 'Nick Gertje',
	98: 'Nick Gertje',
	47: 'Dennis Dotterweich',
	16: 'Dennis Dotterweich',
	49: 'Niklas Waelzlein',
	88: 'Niklas Waelzlein',
	19: 'Wili Sommer',
	52: 'Wili Sommer',
	53: 'Philipp Harlass',
	54: 'Philipp Harlass',
	23: 'Roman Hartleb',
	24: 'Roman Hartleb',
	57: 'Erik Engelhardt',
	58: 'Erik Engelhardt',
	59: 'Sandro Schneider',
	28: 'Sandro Schneider',
	61: 'Leon Krapf',
	62: 'Leon Krapf',
	99: 'Leon Krapf',
	100: 'Leon Krapf',
	63: 'Kevin Baer',
	64: 'Kevin Baer',
	65: 'Luca Ziegler',
	66: 'Luca Ziegler',
	67: 'Ben Mueller',
	68: 'Ben Mueller',
	69: 'Vale Reitstetter',
	38: 'Vale Reitstetter',
	71: 'Christopher Lee',
	40: 'Christopher Lee',
	73: 'Leon Heinze',
	74: 'Leon Heinze',
	75: 'Leo Langhans',
	44: 'Leo Langhans',
	105: 'Referee',
	106: 'Referee',
}


class SoccerStream(RealWorldStream):
	@staticmethod
	def id_(x):
		return x

	def __init__(self,
				 data_csv,
				 length=None):
		header = [
			'sid',
			'ts',
			'x',
			'y',
			'z',
			'|v|',
			'|a|',
			'vx',
			'vy',
			'vz',
			'ax',
			'ay',
			'az'
		]
		attrs = [
			'x',
			'y',
			'z',
			'|v|',
			'vx',
			'vy',
			'vz',
			'|a|',
			'ax',
			'ay',
		]
		digitizers = [
			partial(np.digitize, bins=np.linspace(-52500, 52500, 100)),  # x
			partial(np.digitize, bins=np.linspace(-34000, 34000, 60)),  # y
			partial(np.digitize, bins=np.linspace(0, 3.5 * 10 ** 7, 100)),  # z
			partial(np.digitize, bins=np.linspace(0, 1.2 * 10 ** 9, 100)),  # |v|
			partial(np.digitize, bins=np.linspace(-10000, 10000, 100)),  # vx
			partial(np.digitize, bins=np.linspace(-10000, 10000, 100)),  # vy
			partial(np.digitize, bins=np.linspace(-10000, 10000, 100)),  # vz
			partial(np.digitize, bins=np.linspace(-10000, 10000, 100)),  # |a|
			partial(np.digitize, bins=np.linspace(-10000, 10000, 100)),  # ax
			partial(np.digitize, bins=np.linspace(-10000, 10000, 100)),  # ay
		]

		types = [CONTINUOUS] * len(digitizers)

		super(SoccerStream, self).__init__(data_csv=data_csv,
										   length=length,
										   types=types,
										   header=header,
										   attrs=attrs,
										   type_name='sid',
										   timestamp_column='ts',
										   start_ts=10753295594424116 // 10 ** 12)

		self.need_type_aggregation = True

	def agg(self, sid):
		player = sid_2_player[sid]
		return players_to_id[player]

	def cvt_id_to_event(self, id):
		id = super(SoccerStream, self).cvt_id_to_event(id)
		return players[id]

	def cvt_id_to_attr(self, id):
		return self.attrs[id]

	def get_timestamp(self, row):
		timestamp = super(SoccerStream, self).get_timestamp(row) // 10 ** 12
		return int(timestamp)

	def parse_timestamp(self, seconds):
		hours, seconds = seconds // 3600, seconds % 3600
		minutes, seconds = seconds // 60, seconds % 60
		return Timestamp(year=1, month=1, day=1, hour=hours,
						 minute=minutes, second=seconds)


if __name__ == '__main__':
	data_csv = r'F:\Downloads\full-game\full-game'
	stream = SoccerStream(data_csv=data_csv, length=100000)

	for e in tqdm(stream):
		print(e)
