from data_structures.variable_transformations import Identity

CONDITION_SAME_EVENT = 'same'
CONDITION_TWO_EVENTS = 'two'
CONDITION_TIME = 'time'


class Condition:
    def __init__(self, op1_event, op1_attr, op2_event, op2_attr, operator, type_, f=Identity()):
        self.op1 = op1_event
        self.attr1 = op1_attr
        self.op2 = op2_event
        self.attr2 = op2_attr
        self.operator = operator
        self.type = type_
        self.f = f  # applied to the second operator, i.e. <op1.attr1> <op> f(<op2.attr2>)

    def get_events(self):
        return [self.op1, self.op2]

    def __repr__(self):
        op2_s = f"E{self.op2}.{self.attr2}"

        s = f"""
E{self.op1}.{self.attr1} {self.operator} {self.f.repr(op2_s)}
type: {self.type}
"""
        return f"E{self.op1}.{self.attr1} {self.operator} {self.f.repr(op2_s)}"
        # return s

    def is_satisfied_same_event(self, event):
        if self.operator == '>':
            return event.attrs[self.attr1].content > self.f(event.attrs[self.attr2].content)

        if self.operator == '<':
            return event.attrs[self.attr1].content < self.f(event.attrs[self.attr2].content)

        if self.operator == '=':
            return event.attrs[self.attr1].content == self.f(event.attrs[self.attr2].content)

        return True

    def is_satisfied_2_events(self, event1, event2):
        if self.operator == '>':
            return event1.attrs[self.attr1].content > self.f(event2.attrs[self.attr2].content)

        if self.operator == '<':
            return event1.attrs[self.attr1].content < self.f(event2.attrs[self.attr2].content)

        if self.operator == '=':
            return event1.attrs[self.attr1].content == self.f(event2.attrs[self.attr2].content)

        return True

    @staticmethod
    def has_condition(conditions, op1, attr1, op2, attr2, type=None):
        for cond in conditions:
            if type == CONDITION_TIME and cond.op1 == op1 and cond.op2 == op2:
                return True
            if cond.op1 == op1 and cond.attr1 == attr1 and cond.op2 == op2 and cond.attr2 == attr2:
                return True

        return False

    @staticmethod
    def get_condition_for_events(conditions, e1, e2):
        res = []
        for cond in conditions:
            if cond.op1.type == e1.type and cond.op2.type == e2.type:
                res.append(cond)

        return res

    @staticmethod
    def pattern_meets_conditions(pattern, conditions):
        ok = False
        while not ok:
            ok = True

            for cond in conditions:
                op1 = None
                op2 = None

                for e in pattern:
                    if e is None:
                        continue

                    if e.type == cond.op1:
                        op1 = e
                    if e.type == cond.op2:
                        op2 = e

                if cond.type == CONDITION_SAME_EVENT:
                    ok = cond.is_satisfied_same_event(op1)
                    if not ok:
                        op1.put_values([cond.attr1, cond.attr2])
                        break
                if cond.type == CONDITION_TWO_EVENTS:
                    ok = cond.is_satisfied_2_events(op1, op2)
                    if not ok:
                        op1.put_values([cond.attr1])
                        op2.put_values([cond.attr2])
                        break
                if type(cond) == TimeCondition:
                    idx1 = pattern.index(op1)
                    idx2 = pattern.index(op2)
                    if idx1 > idx2:
                        return False

        return ok


class TimeCondition:
    def __init__(self, op1_event, op2_event):
        super().__init__(op1_event, None, op2_event, None, None, None)
        self.op1 = op1_event
        self.op2 = op2_event

    def get_events(self):
        return [self.op1, self.op2]

    def __repr__(self):
        s = f"""
        {self.op1}.time < {self.op2}.time
        """
        return s

    @classmethod
    def has_condition(cls, conditions, op1, op2):
        conditions = [cond for cond in conditions if type(cond) == cls]
        for cond in conditions:
            if cond.op1 == op1 and cond.op2 == op2:
                return True

        return False

    def is_satisfied(self, pattern):
        op1 = None
        op2 = None
        for i, e in enumerate(pattern):
            if e is None:
                continue

            if e.type == self.op1.type:
                op1 = i
            if e.type == self.op2.type:
                op2 = i

        if op1 is None or op2 is None:
            return True

        return op1 < op2