import numpy as np
# import matplotlib.pyplot as plyt
# import nnfs
# from nnfs.datasets import vertical_data


class DL:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Softmax:

    def forward(self, inputs):

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


class Loss:

    def calculation(self, y_hat, y):
        return np.mean(self.forward(y_hat, y))
