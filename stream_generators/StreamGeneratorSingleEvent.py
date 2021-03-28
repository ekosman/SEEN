import logging
import random
import numpy as np
from copy import deepcopy

from data_structures.Event import Event


from Condition import *


class StreamGeneratorSingleEvent:
    def __init__(self,
                 min_length=50000,
                 num_events=20,
                 max_attributes=5,
                 start_pattern_prob=0.96,
                 continue_pattern_prob=0.96,
                 max_pattern_length=10,
                 patterns_count=10
                 ):
        logging.info("Creating stream...")
        self.min_length = min_length

        self.event_types = [Event(i, max_attributes) for i in range(num_events)]
        self.conditions = StreamGeneratorSingleEvent.generate_conditions(self.event_types)
        self.attr_hash, self.attr_names, self.event_hash = self.generate_hashes(self.event_types)
        self.patterns = [StreamGeneratorSingleEvent.generate_pattern(conditions_per_pattern=max_pattern_length//3,
                                                                     conditions=self.conditions)
                         for _ in range(patterns_count)]

        self.max_pattern_len = max_pattern_length
        self.pattern_index = 0
        self.current_pattern = None
        self.stop = False
        self.state = 'background'
        self.pattern_probs = [1 / patterns_count] * len(self.patterns)
        self.start_pattern_prob = start_pattern_prob
        self.continue_pattern_prob = continue_pattern_prob
        self.current_conditions = None
        self.current_pattern = None
        self.pattern_i = None
        self.counters = [0] * patterns_count
        self.patterns_chars_count = 0
        self.total_length = 1
        self.min_window = float('inf')
        self.max_window = 0
        self.pattern_start = 0
        self.data = []
        self.i = 0
        self.done = False
        self.sample_patterns = []

    @staticmethod
    def generate_pattern(conditions, conditions_per_pattern=3):
        used_events = set()
        counter = 0
        res = []
        valid_conditions = conditions.copy()
        while counter < conditions_per_pattern:
            if len(valid_conditions) == 0:
                used_events = set()
                counter = 0
                res = []
                valid_conditions = conditions.copy()

            cond = np.random.choice(conditions)
            e1 = cond.op1
            e2 = cond.op2
            if e1.type not in used_events and e2.type not in used_events:
                res.append(cond)
                used_events.add(e1.type)
                used_events.add(e2.type)
                counter += 1

                types = [e1.type, e2.type]

                valid_conditions = [cond for cond in valid_conditions if cond.op1.type not in types and cond.op2.type not in types]

        return res

    @staticmethod
    def generate_conditions(events, num_conditions=100):
        """
        A condition is an object with fields:
        op1: operand 1
        op2: operand 2
        operand: operator in {>, <, =}
        """
        operators = ['>', '<', '=']
        conds = []
        counter = 0
        while counter < num_conditions:
            e1 = np.random.choice(events)
            attr1 = np.random.choice(len(e1.attrs))
            e2 = np.random.choice(events)
            attr2 = np.random.choice(len(e2.attrs))

            if e1 == e2 and attr1 == attr2 or Condition.has_condition(conds, e1, attr1, e2, attr2):
                continue

            type_ = CONDITION_TWO_EVENTS

            if e1 == e2:
                type_ = CONDITION_SAME_EVENT
                # type_ = CONDITION_TWO_EVENTS if random.random() < 0.5 else CONDITION_SAME_EVENT

            conds.append(Condition(op1_event=e1,
                                   op1_attr=attr1,
                                   op2_event=e2,
                                   op2_attr=attr2,
                                   operator=np.random.choice(operators),
                                   type_=type_))

            counter += 1

        return conds

    @property
    def n_attrs(self):
        return len([attr for e in self.event_types for attr in e.attrs])

    @property
    def n_events(self):
        return len(self.event_types)

    def get_event_hash(self, event_):
        return self.event_hash[event_.type]

    def get_attr_hash(self, event_, attr_index):
        return self.attr_hash[(event_.type, attr_index)]

    def get_attr_name(self, event_, attr_index):
        return f'event{str(event_.type)}  attr{attr_index}  '

    def generate_hashes(self, event_types):
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

    @staticmethod
    def generate_event(num_events, max_attributes):
        return [Event(max_attributes) for _ in range(num_events)]

    @staticmethod
    def conditions2pattern(conditions, length, events):
        pattern = [None] * length
        left_indexes = list(range(length))
        used_events = set()
        for cond in conditions:
            if cond.type == CONDITION_SAME_EVENT:
                e = cond.op1
                attr1 = cond.attr1
                attr2 = cond.attr2
                new_event = deepcopy(e)
                ok = False
                while not ok:
                    new_event.put_values([attr1, attr2])
                    ok = cond.is_satisfied_same_event(new_event)

                i = np.random.choice(left_indexes)
                left_indexes.remove(i)
                pattern[i] = new_event
                used_events.add(new_event.type)

            elif cond.type == CONDITION_TWO_EVENTS:
                e1 = cond.op1
                e2 = cond.op2
                attr1 = cond.attr1
                attr2 = cond.attr2
                e1 = deepcopy(e1)
                e2 = deepcopy(e2)
                ok = False
                while not ok:
                    e1.put_values([attr1])
                    e2.put_values([attr2])
                    ok = cond.is_satisfied_2_events(e1, e2)

                i, j = np.random.choice(left_indexes, size=2, replace=False)
                left_indexes.remove(i)
                left_indexes.remove(j)
                pattern[i] = e1
                pattern[j] = e2
                used_events.add(e1.type)
                used_events.add(e2.type)

        unused_events = [e for e in events if e.type not in used_events]
        for i in range(len(pattern)):
            if pattern[i] is None:
                e = np.random.choice(unused_events)
                e_copy = deepcopy(e)
                e_copy.randomize()
                pattern[i] = e_copy
                unused_events.remove(e)

        # appear = set()
        # for e in pattern:
        #     if e.type in appear:
        #         print(False)
        #         break
        #     appear.add(e.type)
        # else:
        #     print(True)

        return pattern

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

        s += "Events:\n"
        for e in self.event_types:
            s += repr(e)

        return s

    def get_next_char(self):
        if not self.stop or self.state != 'background':
            self.total_length += 1
            if self.total_length >= self.min_length:
                self.stop = True

            if self.state == 'background' and np.random.rand() <= self.start_pattern_prob:
                self.state = 'pattern'
                self.pattern_index = 0
                self.pattern_i = np.random.choice(len(self.patterns), p=self.pattern_probs)
                self.counters[self.pattern_i] += 1
                self.current_conditions = self.patterns[self.pattern_i]
                self.current_pattern = StreamGeneratorSingleEvent.conditions2pattern(conditions=self.current_conditions,
                                                                                     length=self.max_pattern_len,
                                                                                     events=self.event_types)
                if random.random() < 0.25:
                    self.sample_patterns.append(self.current_pattern)

                self.pattern_start = self.total_length

            if self.state == 'pattern' and np.random.rand() <= self.continue_pattern_prob:
                if self.pattern_index == len(self.current_pattern) - 1:
                    self.max_window = max(self.max_window, self.total_length - self.pattern_start + 1)
                    self.min_window = min(self.min_window, self.total_length - self.pattern_start + 1)
                    self.state = 'background'
                self.pattern_index += 1
                self.patterns_chars_count += 1
                return self.current_pattern[self.pattern_index - 1]
            else:
                return np.random.choice(self.event_types)

        self.done = True
        raise StopIteration

    def __len__(self):
        return len(self.data)

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


if __name__ == '__main__':
    event = Event(type=1, max_attributes=5)
    print(repr(event))

    streamer = StreamGeneratorSingleEvent(min_length=50000,
                               num_events=5,
                               max_attributes=10,
                               start_pattern_prob=0.96,
                               continue_pattern_prob=0.96,
                               max_pattern_length=10,
                               patterns_count=10)
    var = [x for x in streamer]
    print(repr(streamer))
