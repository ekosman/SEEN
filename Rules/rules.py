import itertools

from data_structures.Event import id_fn


class Rule:
	def __init__(self):
		self.cvt_id_2_event = None
		self.cvt_id_2_attr = None
		self.cvt_fn = id_fn

	def set_cvt_name_function(self, fn):
		self.cvt_fn = fn

	def set_cvt_id_2_event(self, fn):
		self.cvt_id_2_event = fn

	def cvt_id_2_attr(self, fn):
		self.cvt_id_2_attr = fn

	@property
	def sigma(self):
		return set()

	@staticmethod
	def get_event_occurrences(event_type, sequence):
		return [(i, e) for i, e in enumerate(sequence) if str(e.type) == str(event_type)]

	@staticmethod
	def descriptor_2_name(desc):
		return f"E{desc[0]}.{desc[1]}"

	@staticmethod
	def parse_event_and_attr(name):
		"""
		Parses the name of an event and attribute
		:param name: Name of an event and an attribute, e.g. E30.23
		:return: tuple - (event_name, attribute_name)
		"""
		import re
		format = r'E(\d+)[._](\d+)'
		res = re.match(format, name)
		if not res:
			return None, None

		res = res.group(1, 2)
		arg1 = int(res[0]) if res[0].isdigit() else res[0]
		arg2 = int(res[1]) if res[1].isdigit() else res[1]

		return arg1, arg2

	@staticmethod
	def parse_op(value1, op, value2, tol=0):
		if op == '>':
			return value1 > value2
		if op == '<':
			return value1 < value2
		if op == '=':
			return abs(value2 - value1) <= tol


class UnaryMath(Rule):
	def __init__(self, operand, operator, value, reverse_maps):
		super(UnaryMath, self).__init__()
		self.operand = operand
		self.operator = operator
		self.value = reverse_maps[operand](value)
		self.event_type, self.attr = self.parse_event_and_attr(operand)

	@property
	def sigma(self):
		return set([self.event_type])

	def is_unary(self):
		return True

	def is_null(self):
		return False

	def __eq__(self, other):
		if type(self) != type(other):
			return False
		return self.operand == other.operand and self.operator == other.operator and self.value == other.value

	def __hash__(self):
		return hash(self.__repr__())

	def __repr__(self):
		s = f"{self.cvt_fn(self.operand)} {self.operator} {str(self.value)}"
		return s

	@classmethod
	def from_gradient(cls, operand, gradient, value):
		operator = '<' if gradient < 0 else '>'
		return cls(operand, operator, value)

	def test_sequence(self, sequence):
		"""
		Checks if a sequence satisfies this condition
		:param sequence: A sequence of events
		:return: Occurrence indices of the condition
		"""
		occurrences = self.get_event_occurrences(self.event_type, sequence)
		return [i for i, e in occurrences if self.parse_op(e.attrs[self.attr].raw_content, self.operator, self.value)]


class MultiVariableMath(Rule):
	def __init__(self, coeffs, operator, independent_names, dependent_name, tol, base, dependent_range,
				 independent_ranges, reverse_maps):
		"""
		:param coeffs: coefficients of the independent variables
		:param operator: < > =
		:param independent_names: names of the independent variables
		:param dependent_name: name of the dependent variable
		:param tol: for equality operator, define the tolerance
		:param base: defines the function base
		:param dependent_range: defines the ranges in which each variable is considered
		:param independent_ranges: defines the ranges in which each variable is considered
		"""
		super(MultiVariableMath, self).__init__()
		self.coeffs = coeffs
		self.operator = operator
		self.independent_names = independent_names
		self.dependent_name = dependent_name
		self.base = base
		self.tol = tol
		self.dependent_range = (reverse_maps[dependent_name](dependent_range[0]), reverse_maps[dependent_name](dependent_range[1]))
		self.independent_ranges = [(reverse_maps[independent_name](independent_range[0]), reverse_maps[independent_name](independent_range[1]))
								   for independent_range, independent_name in zip(independent_ranges, independent_names)]
		self.independent_descs = [self.parse_event_and_attr(name) for name in independent_names]
		self.dependednt_desc = self.parse_event_and_attr(dependent_name)
		self.reverse_maps = reverse_maps

	@classmethod
	def from_line(cls, line):
		return

	@property
	def sigma(self):
		return set([self.dependednt_desc[0]] + [desc[0] for desc in self.independent_descs])

	def is_unary(self):
		return False

	def is_null(self):
		"""
		A rule is null if all coefficients are zero
		:return:
		"""
		return all([round(coeff, 3) == 0 for coeff in self.coeffs])

	def __eq__(self, other):
		if type(self) != type(other):
			return False

		eq_coeffs = all(self.coeffs == other.coeffs)
		eq_op = self.operator == other.operator
		eq_independent_names = self.independent_names == other.independent_names
		eq_base = self.base == other.base
		# Note that we ignore the tol here
		return eq_coeffs and eq_op and eq_independent_names and eq_base

	def __hash__(self):
		return hash(self.__repr__())

	def __repr__(self):
		return self.get_description_from_coeff_and_base(self.coeffs, self.base, self.dependent_name, self.independent_names)

	def get_description_from_coeff_and_base(self, coeffs, base, y_name, x_names):
		for f in self.base:
			f.set_cvt_name_function(self.cvt_fn)

		sub_descs = [f"{round(coeff, 3)} * {repr(f)}"
					for coeff, f in zip(coeffs, base) if round(coeff, 3) != 0]

		ranges_descs = [f"{inf} <= {self.cvt_fn(x_name)} <= {sup}" for x_name, (inf, sup) in zip(x_names, self.independent_ranges)]
		ranges_descs.append(f"{self.dependent_range[0]} <= {self.cvt_fn(y_name)} <= {self.dependent_range[1]}")
		ranges_repr = " AND ".join(ranges_descs)

		if self.operator == '=':
			return f"{self.cvt_fn(y_name)} {self.operator} {' + '.join(sub_descs)} AND {ranges_repr} (tol={self.tol})"

		return f"{self.cvt_fn(y_name)} {self.operator} {' + '.join(sub_descs)} AND {ranges_repr}"

	def evaluate_on_events(self, events):
		"""
		:param events: A list of events. The i'th entry is an event of the type listed in self.independent_descs
		:return:
		"""
		return sum([coeff * f.evaluate_on_events(events) for coeff, f in zip(self.coeffs, self.base)])

	def sort_independent_events_based_on_descs(self, events_dict):
		"""
		:param events_dict: Dictionary containing mapping between events names and their occurences
		:return: sorted list of occurences of events based on their descriptors
		"""
		return [events_dict[independent_event] for independent_event, independent_attr in self.independent_descs]

	def remove_indexes_from_events_combination(self, combination):
		return [e for i, e in combination]

	def get_indexes_from_events_combination(self, combination):
		return [i for i, e in combination]

	def check_ranges(self, lhs, rhs):
		return self.dependent_range[0] <= lhs <= self.dependent_range[1] \
			and all([range_[0] <= val <= range_[1] for val, range_ in zip(rhs, self.independent_ranges)])

	def test_sequence(self, sequence):
		"""
		Checks if a sequence satisfies this rule
		:param sequence: A sequence of events
		:return: Indices of the activations
		"""
		dependent_occurences = self.get_event_occurrences(self.dependednt_desc[0], sequence)
		if len(dependent_occurences) == 0:
			return []
		independent_occurences = {event_type: self.get_event_occurrences(event_type, sequence)
								for event_type, attr in self.independent_descs}
		independent_occurences = self.sort_independent_events_based_on_descs(independent_occurences)
		if any([len(x) == 0 for x in independent_occurences]) or len(dependent_occurences) == 0:
			# make sure all events exist in the sequence, otherwise the condition can't be active
			return []

		for (dependent_idx, dependent_occurence), independent_combination in itertools.product(dependent_occurences, itertools.product(*independent_occurences)):
			lhs = dependent_occurence[int(self.dependednt_desc[1])].raw_content
			all_indices = self.get_indexes_from_events_combination(independent_combination) + [dependent_idx]
			independent_combination = self.remove_indexes_from_events_combination(independent_combination)

			# check ranges
			if not self.check_ranges(lhs=lhs,
									rhs=[event_occurence[independent_attr].raw_content
												for event_occurence, (independent_event, independent_attr) in zip(independent_combination, self.independent_descs)]):
				continue

			rhs = self.evaluate_on_events(independent_combination)
			if self.parse_op(lhs, self.operator, rhs, self.tol):
				return [max(all_indices)]  # occurs at the beginning of the sequence, change it if desired

		return []


class DiscreteRule(Rule):
	def __init__(self, variables, values, reverse_maps):
		"""
		@:param variables
		@:param values
		"""
		super(DiscreteRule, self).__init__()
		self.values = [reverse_maps[var](val) for var, val in zip(variables, values)]
		self.variables = variables

		self.descs = [self.parse_event_and_attr(name) for name in variables]
		self.attrs_indexes = [desc[1] for desc in self.descs]

	@property
	def sigma(self):
		return set([desc[0] for desc in self.descs])

	def is_unary(self):
		return len(self.values) == 1

	def is_null(self):
		"""
		A rule is null if all coefficients are zero
		:return:
		"""
		return False

	def __eq__(self, other):
		if type(self) != type(other):
			return False

		eq_vars = self.variables == other.variables
		eq_values = self.values == other.values
		return eq_vars and eq_values

	def __hash__(self):
		return hash(self.__repr__())

	def __repr__(self):
		conds = [f'{self.cvt_fn(attr_name)} = {str(attr_value)}' for attr_name, attr_value in zip(self.variables, self.values)]
		return ' AND '.join(conds)

	def sort_independent_events_based_on_descs(self, events_dict):
		"""
		:param events_dict: Dictionary containing mapping between events names and their occurences
		:return: sorted list of occurences of events based on their descriptors
		"""
		return [events_dict[independent_event] for independent_event, independent_attr in self.descs]

	def remove_indexes_from_events_combination(self, combination):
		return [e for i, e in combination]

	def get_indexes_from_events_combination(self, combination):
		return [i for i, e in combination]

	def test_sequence(self, sequence):
		"""
		Checks if a sequence satisfies this rule
		:param sequence: A sequence of events
		:return: Indices of the activations
		"""
		occurences = {event_type: self.get_event_occurrences(event_type, sequence)
								for event_type, attr in self.descs}
		occurences = self.sort_independent_events_based_on_descs(occurences)
		if any([len(x) == 0 for x in occurences]):
			# make sure all events exist in the sequence, otherwise the condition can't be active
			return []

		for combination in itertools.product(*occurences):
			all_indices = self.get_indexes_from_events_combination(combination)
			combination = self.remove_indexes_from_events_combination(combination)
			combination_values = [event[int(attr)].raw_content for event, attr in zip(combination, self.attrs_indexes)]
			if combination_values == self.values:
				return [max(all_indices)]   # occurs at the beginning of the sequence, change it if desired

		return []
