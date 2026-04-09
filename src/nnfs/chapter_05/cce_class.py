import numpy as np
import nnfs
from nnfs.datasets import spiral_data


# Dense Layer
class Layer_Dense:

    # Layer Initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        #  Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)


class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))

        # normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities


# Commom loss class
class Loss:

    #  Calculates the data and regularisation losses
    #  given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample loss
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return loss
        return data_loss


class Loss_Categorigorical_Crossentropy(Loss):

    # Forward Pass
    def forward(self, y_pred, y_true):

        # Number pf samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides tonot drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped*y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# create dataset
x, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 utput values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activatioin_relu = Activation_ReLU()

# Create second dense layer with 3 input features *(as we take output
# of the previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# create Softmax activation (to be used with Dense Layer):
activation_sm = Activation_Softmax()

# Create loss function
loss_function = Loss_Categorigorical_Crossentropy()

# perform a forward pass of our training data through this layer
dense1.forward(x)

# Perform a forward pass through ReLU activation function
# it takes the output of the firstdense layer here
activatioin_relu.forward(dense1.output)

# perform a forward pass through second Dense layer
# it takes outputs of activation function of first layer
dense2.forward(activatioin_relu.output)

# Perform a forward pass through softmax activation function
activation_sm.forward(dense2.output)

# Let's see output of the first few samples:
print(activation_sm.output[:5])

# Perform a forward pass through the loss function
# It should take the output of the softmax function and return loss
loss = loss_function.calculate(activation_sm.output, y)

# Print loss value
print('loss:', loss)

# Calculate accuracy from output of activation softmax and targets
# calculate values along first axis
predictions = np.argmax(activation_sm.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

# Print accuracy
print('acc:', accuracy)
