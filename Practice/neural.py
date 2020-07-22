import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X, Y = spiral_data(100,3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

#Forward input data into layer1
layer1 = Layer_Dense(2,4)
layer1.forward(X)
#Forward output of layer1 (dot product of inputs and weights) to the ReLU function
activation1 = Activation_ReLU()
activation1.forward(layer1.outputs)
print(activation1.output)
