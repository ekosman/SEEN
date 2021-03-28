import secrets
from datetime import datetime
from functools import partial

import numpy as np
import random

import torch

DISCRETE = 'discrete'
CONTINUOUS = 'continuous'


def id_fn(x):
    print('id called')
    return x


def reverse(val, min, a):
    # print('reverse called')
    val = float(val)
    return min + a * val


class Attribute:
    def __init__(self, values=None, type=DISCRETE):
        self.type = type
        self.values = values
        self._content = None
        self.global_attribute = None  # pointer to the global attribute descriptor

        self.digitizer = None  # will be available only for the global attribute
        self.map = None  # will be available only for the global attribute
        self.min = None  # will be available only for the global attribute
        self.max = None  # will be available only for the global attribute

    def __repr__(self):
        return f"{self.content}     {self.type}"

    def update_global(self, value, bins=100):
        if self.type == DISCRETE:
            self.digitizer = self.digitizer or id_fn
        else:
            value = float(value)
            if self.min is None:
                self.min = value
            if self.max is None:
                self.max = value
            self.max = max(self.max, value)
            self.min = min(self.min, value)
            if self.max == self.min:
                self.digitizer = id_fn
            else:
                self.digitizer = partial(np.digitize, bins=np.linspace(self.min, self.max, bins))
            self.map = partial(reverse, min=self.min, a=(self.max - self.min)/bins)

    def reverse(self, val):
        # print(f"reverse called!")
        if self.global_attribute is not None:
            # print("global is not non")
            # print(self.global_attribute.reverse)
            return self.global_attribute.reverse(val)

        if self.type == DISCRETE:
            # print(f"my type is {self.type}")
            # print(self.global_attribute.reverse)
            return val

        # global attribute is None and type is continuous

        if type(val) == np.ndarray:
            return np.array([self.map(v) for v in val])

        if type(val) == list:
            return [self.map(v) for v in val]

        else:
            return self.map(val)

    @property
    def raw_content(self):
        return self._content

    @property
    def content(self):
        if self.type == DISCRETE or self.global_attribute is None:
            return self._content
        else:
            return self.global_attribute.digitizer(self._content)

    @content.setter
    def content(self, value):
        self._content = value

    def set(self):
        """
        Sets a random value to the content of the attribute
        """
        self.content = np.random.choice(self.values)
        return self

    def from_list(self, type_, value, global_attribute):
        self.type = type_
        self.content = value
        self.global_attribute = global_attribute
        return self


class Event:
    def __init__(self, type=None, num_attributes=None, timestamp=None):
        """
        :param type: The event type (A,B,C...)
        """
        self.type = type
        self.timestamp = timestamp
        self.attrs = Event.generate_attrs(num_attributes) if num_attributes is not None else None
        self.global_event = None

    def to_tensor(self):
        return torch.tensor([attr.raw_content for attr in self.attrs], dtype=torch.float)

    @property
    def n_attrs(self):
        return len(self.attrs)

    @property
    def datetime(self):
        return datetime(year=self.timestamp.year,
                        month=self.timestamp.month,
                        day=self.timestamp.day,
                        hour=self.timestamp.hour,
                        minute=self.timestamp.minute,
                        second=self.timestamp.second)

    def __getitem__(self, item):
        return self.attrs[item]

    def set_types(self, types):
        for attr, type in zip(self.attrs, types):
            attr.type = type

        return self

    def from_list(self, l):
        self.attrs = [Attribute().from_list(type_, value, global_attribute) for type_, value, global_attribute in l]
        return self

    def update_from_list(self, values):
        for attr, value in zip(self.attrs, values):
            attr.update_global(value)

    def __repr__(self):
        s = f"""
Event type: {self.type}
Number of attributes: {self.__len__()}
Attributes:
"""
        for attr in self.attrs:
            s += repr(attr) + '\n'

        return s

    def randomize(self):
        for i in range(len(self.attrs)):
            self.attrs[i].set()

    def put_values(self, attrs_indexes):
        for i in attrs_indexes:
            self.attrs[i].set()

    @staticmethod
    def generate_attrs(num_attributes):
        return [Event.generate_attr() for _ in range(num_attributes)]

    @staticmethod
    def generate_attr():
        type_ = DISCRETE
        # type_ = np.random.choice([DISCRETE, CONTINUOS])
        values = None
        if type_ == DISCRETE:
            values = list(range(0, 30, 1))
            # values = [3, 4, 5, 6]
        if type_ == CONTINUOUS:
            values = (1, 10)

        return Attribute(values=values, type=type_).set()

    def __len__(self):
        return len(self.attrs)

    @staticmethod
    def get_events_for_types(events, types):
        return list(filter(lambda e: e.type in types, events))
"""
Event1 = 3 atributes
 - [0,1,2]
 - [0,1,2,3,4]
 - [0,1]
 
Event2 = 1 atributes
 - [0,1,2,3,4,5]
 
Event3 = 0 atributes

Event4 = 6 atributes
 - [0,1]
 - [0,1]
 - [0,1,2,3]
 - [0,1,2]
 - [0,1,2,3,4,5,6]
 - [0,1,2,3]
 
Event5 = 9 atributes
 1 - [0,1]
 2 - [0,1,2]
 3 - [0,1,2,3]
 4 - [0,1,2]
 5 - [0,1]
 6 - [0,1,2,3,4]
 7 - [0,1,2,3,4,5]
 8 - [0,1,2]
 9 - [0,1,2,3]
 
 
"""

# pattern1 = [
#             Event(type=4).from_list([(DISCRETE, 1), (DISCRETE, 0), (DISCRETE, 3), (DISCRETE, 0), (DISCRETE, 6), (DISCRETE, 1)]),
#             Event(type=1).from_list([(DISCRETE, 1), (DISCRETE, 2), (DISCRETE, 0)]),
#             Event(type=2).from_list([(DISCRETE, 5)]),
#             Event(type=5).from_list([(DISCRETE, 1), (DISCRETE, 2), (DISCRETE, 3), (DISCRETE, 1), (DISCRETE, 6), (DISCRETE, 1)]),
# ]