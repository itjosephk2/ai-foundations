import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense Layers
class Layer_Dense:

    # layer initialisation
    def __init__(self, n_inputs, n_neurons):
        # Initialise weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate outputs values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    
# Relu activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

# Softmax activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # Get unnormaised probabilities
        exp_values = np.exp(inputs - np.max (inputs, axis =1, keepdims=True))

        # Normalise them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
    
# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense_1 = Layer_Dense(2,3)

# Create ReLU activation (to be used with Dense layer):
activation_relu = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense_2 = Layer_Dense(3,3)

# Create Softmax activation (to be used with Dense layer):
activation_softmax = Activation_Softmax()

# Make a forward pass of our training data through this layer
dense_1.forward(X)

# Make a forward pass through activation function
# it takes the output of first dense layer here
activation_relu.forward(dense_1.output)

# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense_2.forward(activation_relu.output)

# Make a forward pass through activation function
# it takes the output of second dense layer here
activation_softmax.forward(dense_2.output)

# Let's see output of the first few samples:
print(activation_softmax.output[:5])