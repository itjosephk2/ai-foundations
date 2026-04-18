import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 2*x


x = np.array(range(5))
y = f(x)

print(x)
print(y)

# plt.plot(x, y)
# plt.show()

print((y[1] - y[0]) / (x[1] - x[0]))


def g(x):
    return 2*x**2


y = g(x)

print(x)
print(y)

plt.plot(x, y)
# plt.show()

print((y[1] - y[0]) / (x[1] - x[0]))
print((y[3] - y[2]) / (x[3] - x[2]))

p2_delta = 0.0001

x1 = 1
x2 = x1 + p2_delta  # add delta

y1 = g(x1)  # result at the derivation point
y2 = g(x2)  # result at the other, close point

approximate_derivative = (y2-y1)/(x2-x1)
print(approximate_derivative)
