class TreeCondition:
    def __init__(self, weights, names):
        self.weights = weights
        self.names = names

    def __repr__(self):
        ops = [f"{w.item()} * {name}" for w, name in zip(self.weights, self.names) if w.item() != 0]
        return ' + '.join(ops)
