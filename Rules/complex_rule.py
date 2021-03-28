import itertools
from collections import Counter
from datetime import timedelta

import numpy as np
from tqdm import tqdm

from Rules.rules import MultiVariableMath
from stream_generators.dataset import get_custom_dataset


def get_max_delta(events_list):
	sorted_list = sorted([e.datetime for e in events_list])
	return max(sorted_list) - min(sorted_list) + timedelta(seconds=1)


def get_filtered_sequence_and_length(sequence, idx, real_world):
	if real_world:
		results = [[sequence[i] for i in idxs] for idxs in itertools.product(*idx)]
		lengths = [get_max_delta(seq) for seq in results]
	else:
		results = [([sequence[i] for i in idxs], max(idxs) - min(idxs) + 1) for idxs in itertools.product(*idx)]
		lengths = [r[1] for r in results]
		results = [r[0] for r in results]

	return results, lengths


def get_unique_sequences(sequence, return_length=False, filter_events=None, return_end=False, real_world=False):
	types_sequence = [e.type for e in sequence]
	types = np.unique(types_sequence)

	# Get the idx of every event type in the sequence
	if filter_events is None:
		idx = [np.where(types_sequence == t)[0] for t in types]
	else:
		idx = [np.where(types_sequence == t)[0] for t in types if t in filter_events]

	if return_length:
		if len(idx) == 0:
			return [], []

		return get_filtered_sequence_and_length(sequence, idx, real_world)

	if return_end:
		if len(idx) == 0:
			return [], []

		results = [([sequence[i] for i in idxs], max(idxs)) for idxs in itertools.product(*idx)]
		ends = [r[1] for r in results]
		results = [r[0] for r in results]
		return results, ends

	if len(idx) == 0:
		return []

	results = [[sequence[i] for i in idxs] for idxs in itertools.product(*idx)]
	return results


class ComplexRule:
	def __init__(self, rules_list, bond, window=None):
		self.rules = rules_list
		self.bond = bond
		self.window = window

	@classmethod
	def from_file(cls, file, cvt_raw_names_to_code):
		with open(file, 'r') as fp:
			lines = fp.read().splitlines(keepends=False)

		window = lines[-1]
		lines = lines[:-1]
		rules = [MultiVariableMath.from_line(line) for line in lines]
		return cls(rules, 1, window)

	@property
	def sigma(self):
		return set.union(*[rule.sigma for rule in self.rules])

	def __repr__(self):
		s = '\n'.join([repr(rule) for rule in self.rules])
		s += '\n'
		s += f'Within {str(self.window)}'
		return s

	def reactive_score(self, stream, real_world):
		sigma = self.sigma

		def closure(sequence):
			sequence = [e for e in sequence if e.type in sigma]
			event_seq = [e.type for e in sequence]
			counter = Counter(event_seq)
			events_occ = np.prod([counter[t] for t in sigma])
			if events_occ == 0:
				return 0, 0

			n_matches = sum([
				all([len(rule.test_sequence(unique_sequence)) != 0 for rule in self.rules])
				for unique_sequence in get_unique_sequences(sequence=sequence, filter_events=sigma, real_world=real_world)
			])
			return n_matches, events_occ

		data = [e for e in stream]
		dataset = get_custom_dataset(stream=stream, data=data, seq_len=self.window, return_base_index=False, stride=1)
		# dataset = CustomDataset(data=data, seq_len=self.window)

		n_matches_event_occ = [closure(sequence) for sequence in tqdm(dataset)]
		matches = sum([a[0] for a in n_matches_event_occ])
		events_occ = sum([a[1] for a in n_matches_event_occ])
		score = matches / events_occ if events_occ != 0 else 0

		return score

	def get_max_window(self, sequence, real_world=False):
		"""
		Given a sequence, returns the maximum window size that satisfies all the conditions
		returns 0 otherwise
		:param sequence: sequence of events
		"""
		max_window = timedelta(seconds=0) if real_world else 0
		for unique_sequence, length in zip(*get_unique_sequences(sequence, return_length=True, filter_events=self.sigma, real_world=real_world)):
			max_window = length if length > max_window \
								and all([len(rule.test_sequence(unique_sequence)) != 0 for rule in self.rules]) \
								else max_window

		return max_window

	def get_activations(self, sequence, base_index):
		return set([
			end + base_index
			for unique_sequence, end in zip(*get_unique_sequences(sequence, return_end=True, filter_events=self.sigma, real_world=False))
			if all([len(rule.test_sequence(unique_sequence)) != 0 for rule in self.rules])
		])
