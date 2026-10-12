import numpy as np

class dense_layer:
    def __init__ (self, n_inputs, n_neurons):
        #initialize weights nad inputs
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward (self, inputs):
        #calculate output values
        self.output = np.dot(inputs, self.weights) + self.biases



class activation_ReLU:
    def forward(self, inputs):
        #calculates outputs values from inputs
        self.output = np.maximum(0, inputs)


class activation_Softmax:
    def forward(self, inputs):
        # get unnormalized probabilities
        exp_values = np.exp(inputs -np.max(inputs, axis=1, keepdims=True))

        # Normalize for each sample
        probabilities = exp_values/ np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities