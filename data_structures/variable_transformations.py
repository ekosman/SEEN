import numpy as np


class Identity:
	def __call__(self, x):
		return x

	def repr(self, x):
		return x


class Square:
	def __call__(self, x):
		return x**2

	def repr(self, x):
		return f"({x})^2"


class Sqrt:
	def __call__(self, x):
		return np.sqrt(x)

	def repr(self, x):
		return f"sqrt({x})"


class Linear:
	def __init__(self, a, b):
		self.a = a
		self.b = b

	def __call__(self, x):
		return self.a*x + self.b

	def repr(self, x):
		return f"{self.a} * {x} + {self.b}"


class Custom1:
	def __call__(self, x):
		return np.sqrt(x) * 3

	def repr(self, x):
		return f"3*sqrt({x})"


class Custom2:
	def __call__(self, x):
		return 0.05 * x**2

	def repr(self, x):
		return f"0.05 * ({x})^2"