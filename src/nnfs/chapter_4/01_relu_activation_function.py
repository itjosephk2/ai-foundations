import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

# not using numpy
# for i in inputs:
#     # if i > 0:
#     #     output.append(i)
#     # else:
#     #     output.append(0)
#     output.append(max(0, i))

# using numpy
output = np.maximum(0, inputs)

print(output)

#Dense layer
class Layer_Dense:

    # Layer Initialisation
    def __init__(self, n_inputs,n_neurons):
        # Initialise weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# relu activation
class Activation_ReLu:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)

# create dataset
x, y = spiral_data(samples=100, classes=3)

# Create ReLu Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create Relu activation (to be used wit Dense Layer):
activation1 = Activation_ReLu()

# Make a forward pass of our training data through this layer
dense1.forward(x)

# forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)

print(activation1.output[:5])
