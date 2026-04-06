import math
import numpy as np

# An example output from the output layer of the neural network
softmax_output = [.7, .1, .2]
# Ground Truth
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

loss = -sum(math.log(softmax_output[i]) * target_output[i]
            for i in range(len(target_output))
            if target_output[i] != 0)

print(loss)
print('...')
print(math.log(1.))
print(- math.log(0.95))
print(- math.log(0.9))
print(- math.log(0.8))
print('...')
print(- math.log(0.2))
print(- math.log(0.1))
print(- math.log(0.05))
print(- math.log(0.01))
print('...')

b = 5.2
print(np.log(b))
print(math.e ** 1.6486586255873816)
print('...')

# selecting the most confident answers - hard coded version
softmax_outputs = [
                    [0.7, 0.1, 0.2],
                    [0.1, 0.5, 0.4],
                    [0.02, 0.9, 0.08]
                ]

class_targets = [0, 1, 1]

for targ_idx, distribution in zip(class_targets, softmax_outputs):
    print(distribution[targ_idx])

print('...')
# Numpy version
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = [0, 1, 1]

print(softmax_outputs[[0, 1, 2], class_targets])

# using array length to iterate rather
# than hardcoding the index numbers with softmax_outputs
print(softmax_outputs[
    range(len(softmax_outputs)), class_targets
])

# - log as per the math of the equation
print(-np.log(softmax_outputs[
    range(len(softmax_outputs)), class_targets
]))

# calculating average with numpy
neg_log = -np.log(softmax_outputs[
    range(len(softmax_outputs)), class_targets
])
average_loss = np.mean(neg_log)
print(average_loss)
