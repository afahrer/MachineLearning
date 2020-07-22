import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def create_data(points, classes):
    X = np.zeros((points*classes,2))
    y = np.zeros(points*classes,dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        X[ix] = np.c_[np.random.randn(points)*.1 + class_number/3,np.random.rand(points)*.1 + 0.5]
        y[ix] = class_number
    return X, y

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_SoftMax:
    def forward(self, inputs):
        self.output = np.exp(inputs) / sum(np.exp(inputs))

#Create Dataset
X, y = create_data(100,3)
#Create Model
dense1 = Layer_Dense(100,3)

activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
activation2 = Activation_SoftMax()
activation2.forward(dense1.forward(X))
print(activation2.output)
#loss_function = Loss_CategorialCrossentropy()
