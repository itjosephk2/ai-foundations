import numpy as np

a = [1, 2, 3]
b = [2, 3, 4]

print(np.array([a]))
print(np.expand_dims(np.array(a), axis=0))

a = np.array([a])
b = np.array([b]).T

print(b)

print(np.dot(a, b))