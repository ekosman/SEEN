import datetime

from tqdm import tqdm
import numpy as np


from stream_generators.SoccerStream import SoccerStream, players_to_id

ball_ids = [4, 8, 10, 12]


def distance(p1, p2):
	return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class FilteredSoccerStream(SoccerStream):

	def __init__(self, data_csv, length, radius, sample_rate=1):
		super(FilteredSoccerStream, self).__init__(data_csv, length)
		self.ball_pos = None
		self.radius = radius
		self.filtered = False
		self.set_filter(self.my_filter)
		self.current_time = None
		self.sample_rate = datetime.timedelta(seconds=1)
		self.sample_types = None

	def my_filter(self, e):
		if self.current_time is None:
			self.current_time = e.datetime + self.sample_rate
			self.sample_types = set()

		while self.current_time < e.datetime:
			self.current_time += self.sample_rate
			self.sample_types = set()

		type_ = self.event_types_actual[e.type]

		ret = None
		if type_ == players_to_id['Balls']:
			self.ball_pos = (e.attrs[0].raw_content / 1000, e.attrs[1].raw_content / 1000)
			ret = e
		else:
			p = (e.attrs[0].content / 1000, e.attrs[1].content / 1000)
			if self.ball_pos is not None and distance(self.ball_pos, p) <= self.radius:
				ret = e

		if type_ in self.sample_types:
			return None

		if ret is not None:
			self.sample_types.add(ret.type)

		return ret

	# def __iter__(self):
	# 	super(FilteredSoccerStream, self).__iter__()
	# 	if not self.filtered:
	# 		data = []
	# 		for i, e in enumerate(self.data):
	# 			type_ = self.event_types_actual[e.type]
	# 			if type_ == players_to_id['Balls']:
	# 				self.ball_pos = (e.attrs[0].raw_content / 1000, e.attrs[1].raw_content / 1000)
	# 				data.append(e)
	# 			else:
	# 				p = (e.attrs[0].content / 1000, e.attrs[1].content / 1000)
	# 				if self.ball_pos is not None and distance(self.ball_pos, p) <= self.radius:
	# 					data.append(e)
	#
	# 		self.data = data
	# 		self.filtered = True
	#
	# 	return self

	# def __next__(self):
	# 	ok = False
	#
	# 	while not ok:
	# 		e = super(FilteredSoccerStream, self).__next__()
	# 		if e.type == players_to_id['Balls']:
	# 			self.ball_pos = (e.attrs[0].content, e.attrs[1].content)
	# 			ok = True
	# 		else:
	# 			if self.ball_pos is None:
	# 				continue
	# 			else:
	# 				p = (e.attrs[0].content, e.attrs[1].content)
	# 				ok = distance(self.ball_pos, p) <= self.radius
	#
	# 	return e


if __name__ == '__main__':
	data_csv = r'F:\Downloads\full-game\full-game'
	stream = FilteredSoccerStream(data_csv=data_csv, length=1000000, radius=10)

	for e in tqdm(stream):
		print(e)
