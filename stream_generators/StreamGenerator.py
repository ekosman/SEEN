import logging
import numpy as np

from data_structures.Event import Event


def get_pattern(events, length):
    return [np.random.choice(events) for _ in range(length)]


class StreamGenerator:
    def __init__(self,
                 min_length=50000,
                 num_events=5,
                 max_attributes=5,
                 start_pattern_prob=0.96,
                 continue_pattern_prob=0.96,
                 max_pattern_length=10,
                 patterns_count=10
                 ):
        logging.info("Creating stream...")
        self.min_length = min_length
        self.event_types = [Event(i, max_attributes) for i in range(num_events)]
        self.attr_hash, self.attr_names, self.event_hash = self.generate_hashes(self.event_types)
        self.patterns = [get_pattern(events=self.event_types, length=max_pattern_length) for _ in range(patterns_count)]
        # self.patterns = [[]]

        self.pattern_index = 0
        self.current_pattern = None
        self.stop = False
        self.state = 'background'
        self.pattern_probs = [1 / patterns_count] * len(self.patterns)
        self.start_pattern_prob = start_pattern_prob
        self.continue_pattern_prob = continue_pattern_prob
        self.current_pattern = None
        self.pattern_i = None
        self.counters = [0] * patterns_count
        self.patterns_chars_count = 0
        self.total_length = 1
        self.max_window = 0
        self.pattern_start = 0
        self.data = []
        self.i = 0
        self.done = False

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
                self.current_pattern = self.patterns[self.pattern_i]
                self.pattern_start = self.total_length

            if self.state == 'pattern' and np.random.rand() <= self.continue_pattern_prob:
                if self.pattern_index == len(self.current_pattern) - 1:
                    self.max_window = max(self.max_window, self.total_length - self.pattern_start + 1)
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

    streamer = StreamGenerator(min_length=50000,
                               num_events=5,
                               max_attributes=10,
                               start_pattern_prob=0.96,
                               continue_pattern_prob=0.96,
                               max_pattern_length=10,
                               patterns_count=10)
    var = [x for x in streamer]
    print(repr(streamer))
