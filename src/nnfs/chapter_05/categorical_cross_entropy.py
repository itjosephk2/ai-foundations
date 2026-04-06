import math

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
