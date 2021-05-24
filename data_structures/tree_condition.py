class TreeCondition:
    def __init__(self, weights, names, sign, bias):
        self.weights = weights
        self.names = names
        self.sign = sign
        self.bias = bias
        self.ops = [f"{w.item()} * {name}" for w, name in zip(self.weights, self.names) if w.item() != 0]

    def __repr__(self):
        return ' + '.join(self.ops) + f" {self.sign} {str(self.bias)}"

    @property
    def comprehensibility(self):
        return len(self.ops)
