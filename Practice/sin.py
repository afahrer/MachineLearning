# The goal of this file is to create a neural network from scratch
# then use it to approximate the sin function
# by: savag3adam
import numpy as np
import math

# constant values that change the shape of the network
batch_size = 10
input_size = 1
output_size = 1
hidden_layer_size = 64

#Ensure consistent Random values (for testing)
np.random.seed(0)

class Layer:
   def __init__(self, n_inputs, n_neurons):
       # Shape of weights is (n_inputs, n_neurons)
       self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
       # Shape of biases is (n_neurons,)
       self.biases = np.zeros((n_neurons,))
   def forward(self,inputs):
       self.output = np.dot(inputs, self.weights) + self.biases

   def ReLu(self, inputs):
       self.output = np.maximum(0, np.dot(inputs, self.weights) + self.biases)

#Initialize data
X = np.random.uniform(-math.pi,math.pi,(batch_size,input_size))

#Initialize Layers
hidden_layer = Layer(input_size, hidden_layer_size)
hidden_layer1 = Layer(hidden_layer_size, hidden_layer_size)
output_layer = Layer(hidden_layer_size, output_size)
#Forward data and send outputs from input layer into the activation function
hidden_layer.ReLu(X)
hidden_layer1.ReLu(hidden_layer.output)
output_layer.forward(hidden_layer1.output)

#Print important values
print("Batch Size: ", batch_size)
print("Input Size: ", input_size)
print("Hidden Layer Size: ", hidden_layer_size)
print("Output Size: ", output_size, "\n")
print("Input Data: ", X, "\n")
print("Outputs: ",output_layer.output)

