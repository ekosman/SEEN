import itertools
import logging
import random
import re
import threading
from copy import deepcopy
from queue import Queue

import networkx as nx
import numpy as np
from tqdm import tqdm

from Rules.rules import Rule
from data_structures.Condition import Condition, TimeCondition
from data_structures.Event import Event, id_fn

# from stream_generators.dataset import get_custom_dataset

CONDITION_SAME_EVENT = 'same'
CONDITION_TWO_EVENTS = 'two'


class StreamGeneratorSingleEventSubset:
    def __init__(self,
                 provided_events=None,
                 provided_conditions=None,
                 provided_patterns=None,
                 conditions_per_pattern=5,
                 use_time_conditions=True,
                 use_attr_conditions=True,
                 min_length=50000,
                 num_events=20,
                 min_attributes=1,
                 max_attributes=5,
                 start_pattern_prob=0.96,
                 continue_pattern_prob=0.96,
                 patterns_count=10,
                 conditions_count=30,
                 subset=[0, 1]  # should contain numbers within range(num_events)
                 ):
        logging.info("Creating stream...")
        self.conditions_per_pattern = conditions_per_pattern
        self.min_length = min_length
        self.use_time_conditions = use_time_conditions
        self.use_attr_conditions = use_attr_conditions
        self.event_types = StreamGeneratorSingleEventSubset.generate_event_types(provided_events=provided_events,
                                                                                 min_attributes=min_attributes,
                                                                                 max_attributes=max_attributes,
                                                                                 num_events=num_events)
        self.subset = [e for e in self.event_types if e.type in subset]
        if provided_conditions is None:
            self.conditions = StreamGeneratorSingleEventSubset.generate_conditions(events=self.subset,
                                                                                   use_attr_conditions=use_attr_conditions,
                                                                                   use_time_conditions=use_time_conditions,
                                                                                   num_conditions=conditions_count
                                                                                   # num_conditions=self.n_attrs
                                                                                   )
        else:
            self.conditions = provided_conditions

        self.attrs_maps, self.attr_hash, self.attr_names, self.event_hash = self.generate_hashes(self.subset)
        if provided_patterns is None:
            # self.patterns = [StreamGeneratorSingleEventSubset.generate_pattern(conditions_per_pattern=self.conditions_per_pattern,
            #                                                              conditions=self.conditions)
            #                  for _ in range(patterns_count)]
            self.patterns = StreamGeneratorSingleEventSubset.generate_patterns(conditions=self.conditions,
                                                                               conditions_per_pattern=conditions_per_pattern,
                                                                               num_conditions=conditions_count,
                                                                               num_patterns=patterns_count)
        else:
            self.patterns = provided_patterns

        self.true_patterns_activations = [set() for _ in range(len(self.patterns))]
        self.pattern_index = 0
        self.current_pattern = None
        self.stop = False
        self.state = 'background'
        self.pattern_probs = [1 / len(self.patterns)] * len(self.patterns)
        self.start_pattern_prob = start_pattern_prob
        self.continue_pattern_prob = continue_pattern_prob
        self.current_conditions = None
        self.current_pattern = None
        self.pattern_i = None
        self.counters = [0] * len(self.patterns)
        self.patterns_chars_count = 0
        self.total_length = 1
        self.min_window = float('inf')
        self.max_window = 0
        self.pattern_start = 0
        self.data = []
        self.i = 0
        self.done = False
        self.thread = None
        self.kill_pill = False

    @property
    def story_length(self):
        return len(self)

    def cvt_descriptor_to_meaningful_name(self, descriptor):
        """
        Transforms and descriptor like E0.0 to Ball.x
        """
        event, attr = Rule.parse_event_and_attr(descriptor)
        return f"{self.cvt_id_to_event(event)}.{self.cvt_id_to_attr(attr)}"

    def cvt_id_to_event(self, id):
        """
        The general conversion function
        """
        return id

    def cvt_id_to_attr(self, id):
        """
        The general conversion function
        """
        return id

    def is_attr_discrete(self, attr_name):
        """

        :param attr_name:
        :return:
        """
        return False

    def get_reverse_mapping_for_network_states(self, include_time=False):
        if include_time:
            maps = [None] * len(self.subset)
            for e in self.subset:
                maps[self.get_event_hash(e)] = id_fn
        else:
            maps = []

        maps += [mapping for _, mapping in self.attrs_maps.items()]
        return maps

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

    def dump_patterns(self, fp):
        def dump_fn(pattern):
            s = '\n'.join([repr(rule) for rule in pattern])
            s += '\n'
            return s

        for pattern_i, pattern in enumerate(self.patterns):
            fp.write(f"------------Pattern {pattern_i}------------\n")
            fp.write(dump_fn(pattern))
            fp.write("\n\n")

    def dump_repr(self, fp):
        fp.write(repr(self))

    @staticmethod
    def generate_event_types(provided_events, min_attributes, max_attributes, num_events):
        provided_events_types = {e.type: e for e in provided_events} if provided_events is not None else dict()
        return [Event(i, np.random.randint(min_attributes, max_attributes + 1)) if i not in provided_events_types.keys()
                else provided_events_types[i]
               for i in range(num_events)]

    def get_events_in_patterns(self):
        def get_events_in_pattern(pattern):
            events = set()
            for cond in pattern:
                events.update(cond.get_events())
            return events

        return [get_events_in_pattern(pattern) for pattern in self.patterns]

    @staticmethod
    def generate_patterns(conditions, conditions_per_pattern, num_conditions, num_patterns):
        patterns = []
        used_conditions = set()
        conditions = set(conditions)
        if num_patterns is None:
            num_patterns = -1
        while len(used_conditions) < num_conditions or len(patterns) < num_patterns:
            if len(conditions) == 0:
                break
            pattern = StreamGeneratorSingleEventSubset.generate_pattern(conditions=conditions,
                                                                        conditions_per_pattern=conditions_per_pattern,
                                                                        pattern_id=len(patterns))
            if used_conditions.issuperset(pattern):
                continue

            logging.info(f"{len(pattern)} conditions")

            used_conditions = used_conditions.union(pattern)
            conditions = conditions - used_conditions
            patterns.append(pattern)

        return patterns

    @staticmethod
    def generate_pattern(conditions, conditions_per_pattern=None, pattern_id=-1):
        if pattern_id == -1:
            logging.info("Generating pattern from conditions...")
        else:
            logging.info(f"Generating pattern {pattern_id} from conditions...")

        counter = 0
        res = []
        unused_conditions = conditions.copy()

        time_graph = nx.DiGraph()

        if conditions_per_pattern is None:
            conditions_per_pattern = np.random.geometric(1/np.cbrt(len(conditions)))

        bar = tqdm(total=conditions_per_pattern)
        while counter < conditions_per_pattern:
            if len(unused_conditions) == 0:
                logging.info(f"No conditions left for generaing a pattern with {conditions_per_pattern} conditions")
                break
            cond = np.random.choice(list(unused_conditions))

            if type(cond) == TimeCondition:
                e1 = cond.op1
                e2 = cond.op2

                if time_graph.has_edge(e1, e2):
                    continue

                e1_exists = time_graph.has_node(e1)
                e2_exists = time_graph.has_node(e2)

                time_graph.add_edge(e1, e2)

                try:
                    nx.find_cycle(time_graph, e1, orientation='original')  # will raise exception if no cycle found
                    time_graph.remove_edge(e1, e2)
                    if not e1_exists:
                        time_graph.remove_node(e1)
                    if not e2_exists:
                        time_graph.remove_node(e2)
                    continue
                except nx.exception.NetworkXNoCycle:
                    pass

            # if not a time conditions, will be added anyways
            unused_conditions.remove(cond)
            res.append(cond)

            counter += 1
            bar.update(1)

        return res

    @staticmethod
    def generate_conditions(events, use_time_conditions, use_attr_conditions, num_conditions=100):
        """
        A condition is an object with fields:
        op1: operand 1
        op2: operand 2
        operand: operator in {>, <, =}
        """
        logging.info("Generating conditions...")
        operators = ['>', '<', '=']
        conds = []
        counter = 0
        bar = tqdm(total=num_conditions)
        while counter < num_conditions:
            e1 = np.random.choice(events)
            attr1 = np.random.choice(len(e1.attrs))
            e2 = np.random.choice(events)
            attr2 = np.random.choice(len(e2.attrs))

            if e1.type == e2.type and attr1 == attr2 or Condition.has_condition(conds, e1.type, attr1, e2.type, attr2):
                continue

            type_ = CONDITION_TWO_EVENTS

            if e1.type == e2.type:
                type_ = CONDITION_SAME_EVENT
                # type_ = CONDITION_TWO_EVENTS if random.random() < 0.5 else CONDITION_SAME_EVENT

            if use_attr_conditions:
                conds.append(Condition(op1_event=e1.type,
                                       op1_attr=attr1,
                                       op2_event=e2.type,
                                       op2_attr=attr2,
                                       operator=np.random.choice(operators),
                                       type_=type_))
                counter += 1
                bar.update(1)

            if use_time_conditions and e1.type != e2.type and not TimeCondition.has_condition(conds, e1.type, e2.type):
                conds.append(TimeCondition(op1_event=e1.type,
                                       op2_event=e2.type))
                counter += 1
                bar.update(1)

        return conds

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

    @staticmethod
    def randomize_pattern(pattern):
        for e in pattern:
            if e is not None:
                e.randomize()

    def patterns_generator_thread(self):
        """
        Runs the generator thread in the background
        """
        self.kill_pill = threading.Event()
        self.patterns_queue = Queue(maxsize=10)
        self.thread = threading.Thread(target=self.patterns_generator_thread_aux, args=(self.kill_pill,))
        self.thread.start()

    def stop_patterns_generator_thread(self):
        if self.thread is None:
            raise RuntimeError("thread is none")
        self.kill_pill.set()
        # self.thread.join()
        self.thread = None

    def patterns_generator_thread_aux(self, stop_event):
        while not stop_event.is_set():
            pattern_i = np.random.choice(len(self.patterns), p=self.pattern_probs)
            current_conditions = self.patterns[pattern_i]
            current_pattern = StreamGeneratorSingleEventSubset.conditions2pattern(
                conditions=current_conditions,
                events=self.event_types)
            self.patterns_queue.put((pattern_i, current_pattern))

        print("Stopping as you wish.")

    @staticmethod
    def conditions2pattern(conditions, events):
        time_conditions = list(filter(lambda cond: type(cond) == TimeCondition, conditions))
        other_conditions = list(filter(lambda cond: type(cond) != TimeCondition, conditions))
        events_in_conditions = set()
        events_in_conditions_with_time_constraints = set()

        # Collect events in time conditions
        for cond in conditions:
            events_in_conditions.add(cond.op1)
            events_in_conditions.add(cond.op2)
            if type(cond) == TimeCondition:
                events_in_conditions_with_time_constraints.add(cond.op1)
                events_in_conditions_with_time_constraints.add(cond.op2)

        pattern_length = len(events_in_conditions)
        pattern = [None] * pattern_length
        chosen_indices = np.random.choice(range(pattern_length), size=len(events_in_conditions_with_time_constraints), replace=False)
        events_in_conditions = list(events_in_conditions)
        events_in_conditions_with_time_constraints = list(events_in_conditions_with_time_constraints)

        # Check for a valid permutation of the events in the time constraints
        ok = True
        if len(chosen_indices) != 0:
            for permutation in itertools.permutations(chosen_indices):
                ok = True
                # pattern[list(permutation)] = Event.get_events_for_types(events, events_in_conditions_with_time_constraints)
                StreamGeneratorSingleEventSubset.set_events(pattern=pattern,
                                                            idxs=list(permutation),
                                                            events=Event.get_events_for_types(events, events_in_conditions_with_time_constraints))
                # pattern[list(permutation)] = events_in_conditions_with_time_constraints
                for cond in time_conditions:
                    ok &= cond.is_satisfied(pattern)
                    if not ok:
                        break
                if ok:
                    break

        if not ok:
            raise Exception("No combination found which satisfies all time conditions")

        # StreamGeneratorSingleEventSubset.randomize_pattern(pattern)

        events_to_set = list(set(events_in_conditions) - set(events_in_conditions_with_time_constraints))
        if len(events_to_set) != 0:
            random.shuffle(events_to_set)
            chosen_indices = [idx for idx in range(len(pattern)) if idx not in chosen_indices]
            chosen_indices = np.random.choice(chosen_indices, size=len(events_to_set), replace=False)
            # pattern[chosen_indices] = Event.get_events_for_types(events, events_to_set)
            StreamGeneratorSingleEventSubset.set_events(pattern=pattern,
                                                        idxs=chosen_indices,
                                                        events=Event.get_events_for_types(events,
                                                                                          events_to_set))
            pattern = [deepcopy(x) if x is not None else None for x in pattern]

        try:
            StreamGeneratorSingleEventSubset.randomize_pattern(pattern)
        except AttributeError as e:
            logging.info("==============================ERROR============================")
            logging.info(e)
            logging.info("==============================PATTERN============================")
            logging.info(pattern)
            logging.info("==============================EVENTS============================")
            logging.info(events)
            logging.info("==============================EVENTS TO SET============================")
            logging.info(Event.get_events_for_types(events, events_to_set))
            logging.info("==============================CHOSEN IDX============================")
            logging.info(chosen_indices)
            logging.info("==============================CONDITIONS============================")
            logging.info(conditions)
            logging.info("==============================BEFORE DEEP COPY============================")
            pattern[chosen_indices] = Event.get_events_for_types(events, events_to_set)
            logging.info(pattern)
            exit()

        ok = False
        while not ok:
            ok = Condition.pattern_meets_conditions(pattern, other_conditions)

        pattern = list(pattern)
        return pattern

    @staticmethod
    def set_events(pattern, idxs, events):
        for idx, e in zip(idxs, events):
            pattern[idx] = e

    def __iter__(self):
        self.i = 0
        return self

    def __repr__(self):
        s = f"""length: {self.__len__()}
counters: {str(self.counters)}
mean of occurences: {np.mean(self.counters)}
Random chars: {self.total_length - self.patterns_chars_count}
Pattern chars: {self.patterns_chars_count}
Random character probability: {(self.total_length - self.patterns_chars_count) / self.total_length}
Min window: {self.min_window}
Max window: {self.max_window}
"""
        return s

    def get_next_char(self):
        if not self.stop or self.state != 'background':
            self.total_length += 1
            if self.total_length >= self.min_length:
                self.stop = True

            if self.state == 'background' and np.random.rand() <= self.start_pattern_prob:
                self.state = 'pattern'
                self.pattern_index = 0
                # self.pattern_start = self.total_length
                self.pattern_i = np.random.choice(len(self.patterns), p=self.pattern_probs)
                self.counters[self.pattern_i] += 1
                self.current_pattern = StreamGeneratorSingleEventSubset.conditions2pattern(
                    conditions=self.patterns[self.pattern_i],
                    events=self.event_types)

            if self.state == 'pattern' and np.random.rand() <= self.continue_pattern_prob:
                if self.pattern_index == 0:
                    self.pattern_start = self.total_length
                if self.pattern_index == len(self.current_pattern) - 1:
                    self.min_window = min(self.min_window, self.total_length - self.pattern_start + 1)
                    self.max_window = max(self.max_window, self.total_length - self.pattern_start + 1)
                    self.state = 'background'
                    self.true_patterns_activations[self.pattern_i].add(self.total_length)

                self.pattern_index += 1
                self.patterns_chars_count += 1
                e = self.current_pattern[self.pattern_index - 1]
                e.timestamp = self.total_length - 1
                return e
            else:
                e = deepcopy(np.random.choice(self.event_types))
                e.randomize()
                e.timestamp = self.total_length - 1
                return e

        self.done = True
        raise StopIteration

    def __len__(self):
        return len(self.data) if self.done is not None else self.min_length

    def __next__(self):
        if self.done:
            if self.i == len(self):
                raise StopIteration

            self.i += 1
            return self.data[self.i - 1]

        else:
            c = self.get_next_char()
            self.data.append(c)
            return c

    def f1_score(self, complex_rules, dataset):
        logging.info("Calculating F1...")
        pred_activations = []
        logging.info("Calculating pred activations...")
        for i, complex_rule in enumerate(complex_rules):
            logging.info(f"Rule {i}")
            # dataset = CustomDataset(self.data, seq_len=complex_rule.window, return_base_index=True)
            # dataset = get_custom_dataset(stream=self, data=self.data, seq_len=complex_rule.window, return_base_index=True)
            pred_activations.append(set.union(*[complex_rule.get_activations(seq, base) for seq, base in tqdm(dataset)]))

        f1s = []
        precisions = []
        recalls = []
        logging.info("Calculating F1's")
        for i, true_pattern_activation in tqdm(enumerate(self.true_patterns_activations)):
            logging.info(f"Calculating F1 for rule {i}")
            tps = [len(set.intersection(true_pattern_activation, pred_pattern_activation))
                   for pred_pattern_activation in pred_activations
                   ]
            assoc = np.argmax(tps)
            chosen_activations = pred_activations[assoc]
            tp = tps[assoc]
            fp = len(set.difference(chosen_activations, true_pattern_activation))
            precision = tp / (tp + fp) if tp != 0 else 0
            recall = tp / len(true_pattern_activation)

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
            # fn = len(set.difference(true_pattern_activation, chosen_activations))
            f1s.append(f1)

        return f1s, precisions, recalls
