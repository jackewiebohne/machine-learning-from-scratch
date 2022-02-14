from abc import ABC, abstractmethod
import numpy as np

# credit to https://e-dorigatti.github.io/math/deep%20learning/2020/04/07/autodiff.html
# see also: https://sidsite.com/posts/autodiff/
# for the foundations of this autodiff module

class Node(ABC):
	'''
	abstract base class
	'''
	@abstractmethod
	def differentiate(self, var):
		'''var is the variable wrt which we differentiate'''
		pass

	@abstractmethod
	def integrate(self, var):
		'''var is the variable wrt which we integrate'''
		pass

	@abstractmethod
	def compute(self):
		'''for simply calculating inputs'''
		pass

	@abstractmethod
	def __repr__(self):
		pass

	def __add__(self, other):
		return Sum(self, other)

	def __mul__(self, other):
		return Mul(self, other)

	# def __sub__(self, other):
	# 	return Sub(self, other)

	def __truediv__(self, other):
		return Div(self, other)

	def __pow__(self, other):
		return Pow(self, other)


class Const(Node):
	'''
	class for a constant
	'''
	def __init__(self, value):
		self.value = value
		self.power = 1
		self.raised = 0

	def compute(self):
		return self.value

	def differentiate(self, var):
		return Const(0)

	def integrate(self, var):
		if isinstance(var, Var):
			return Const(self) * var.value
		else:
			raise TypeError('provided variable is not a Variable. Use class Var to instantiate one')

	def __repr__(self):
		return f'{self.value}'


class Var(Node):
	'''
	class for a variable
	'''
	def __init__(self, name, value=None):
		self.name = name
		self.value = value
		self.power = 1
		self.raised = 0

	def compute(self):
		if self.value:
			return self.value
		else:
			raise ValueError('unassigned variable')

	def differentiate(self, var):
		# if self.power == 1:
		return Const(1) if self == var else Const(0)
		# else:
			# return self.power.value * var.value**(self.power.value - 1) if self == var else Const(0) #var.value?? but what if abstract?

	def integrate(self, var):
		if isinstance(var, Var):
			return (self**(self.power.value+1))/(self.power.value+1) if self==var else var.value * self.value # var.value?? but what if abstract?
		else:
			raise TypeError('provided variable is not a Variable. Use class Var to instantiate one')
	
	def __repr__(self):
		return f'variable name: {self.name}, variable value: {self.value}'


class Sum(Node):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def compute(self):
		return self.x.compute() + self.y.compute()

	def differentiate(self, var):
		return self.x.differentiate(var), self.y.differentiate(var)

	def integrate(self, var):
		return self.x.integrate(var) + self.y.integrate(var)

	def __repr__(self):
		return f'({self.x} + {self.y})'


class Mul(Node):
	def __init__(self, x, y):
		self.x, self.y = x, y

	def compute(self):
		return self.x.compute() * self.y.compute()

	def differentiate(self, var):
		return self.x.differentiate(var) * self.y + self.y.differentiate(var) * self.x

	def integrate(self, var):
		return #not implemented

	def __repr__(self):
		return f'({self.x} * {self.y})'


class Pow(Node):
	def __init__(self, x, y):
		self.x, self.y = x, y
		self.x.power = self.y.value
		print(self.x.power)
		self.y.raised += 1

	def compute(self):
		return self.x.compute() ** self.y.compute()

	def differentiate(self, var):
		if self.x == var:
			return self.x.power * self.x.value**(self.x.power - 1)
		elif self.y == var:
			if self.y.raised > 1:
				raise NotImplementedError('this library can only do simpler derivations with variables only raised to the first power') 
			else:
				return self.x.value ** self.y.value * np.log(self.x.value)
		else:
			print(var)

	def integrate(self, var):
		pass

	def __repr__(self):
		return f'{self.x}**{self.y}'



class Div(Node):
	# change to power
	def __init__(self, x, y):
		self.x, self.y = x, y

	def compute(self):
		return self.x.compute() / self.y.compute()

	def differentiate(self, var):
		pass
	# 	return self.x.differentiate(var) / self.y.value if self.x == var \
	# 	else self.x.value * Pow(self.y, ##need to implement negation here###).differentiate(var)

	def integrate(self, var):
		return #not implemented

	def __repr__(self):
		return f'({self.x} / {self.y})'


class Neg(Node):
	pass
	
x = Var('x', 1)
y = Var('y', 2)
q = Var('q', 6)
p = Const(6)
z = y**q
# t = x/y**Const(2)


# w = Const(1)+Const(3)
# print(Const(1) * x.value)
# print(Const(1)*Const(3))
# print(w.integrate(w)) # returns the desired typeerror
# print(z.integrate(x))
# w = Const(1)**Const(4)
print(z.differentiate(y))
# print(z.compute())
# print(t.differentiate(y))
