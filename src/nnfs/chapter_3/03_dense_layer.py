from nnfs.datasets import spiral_data

import matplotlib.pyplot as plt
import numpy as np
import nnfs

nnfs.init()

# n_inputs = 2
# n_neurons = 4

# weights = 0.01 * np.random.randn(n_inputs, n_neurons)
# biases = np.zeros((1, n_neurons))

# print(weights)
# print(biases)

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
    
# Create dataset
x, y = spiral_data(samples=100, classes=3)

# Create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Perform a forward pass  of our training data through this layer
dense1.forward(x)

# Let's see the output of the first fewsamples:
print(dense1.output[:5])