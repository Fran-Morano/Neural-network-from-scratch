import numpy as np

class Dense_layer:
    def __init__ (self, n_inputs, n_neurons):
        #initialize weights nad inputs
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward (self, inputs):
        #calculate output values
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        #calculates outputs values from inputs
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        #since we need to modify the original variable, let's make a copy of the values first
        self.dinputs = dvalues.copy()
        #zero gradient where the dvalues are negative
        self.dinputs[self.dinputs <=0] = 0
    



class Activation_Softmax:
    def forward(self, inputs):
        # get unnormalized probabilities
        exp_values = np.exp(inputs -np.max(inputs, axis=1, keepdims=True))

        # Normalize for each sample
        probabilities = exp_values/ np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        #Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        #enumerate outputs and gradients
        for index, (single_output,single_dvalues) in enumerate(zip(self.output,dvalues)):
                #flatten output array
                single_output = single_output.reshape(-1, 1)
                #calculate jacobian matriz of the output
                jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
                #calculate sample-wise gradient and add it to the array of sample gradientes
                self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss: #common loss class
    #calculates the data and regularization losses given model output and ground truth values
    def calculate(self, output, y):

        #calculate sample losses
        sample_losses = self.forward(output,y)

        data_loss = np.mean(sample_losses)

        return data_loss



class Loss_CategoricalCrossentropy(Loss): #cross entropy loss 

    def forward(self, y_pred, y_true):
        samples = len(y_pred) #numbers of samples in a batch

        #clip data to prevent division by 0, and clip from both side to not drag mean towards any edge
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        #Probabilities for target values
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true] #categorical labels
        elif len(y_true.shape) == 2:
            correct_confidences =np.sum(y_pred_clipped*y_true, axis=1) #one-hot coded values

        #losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
        

    def backward(self, dvalues, y_true):  #backward pass
        samples = len(dvalues) #amount of samples

        labels = len(dvalues[0]) #amount of labels for each sample

        #if labels are sparse turn them into 1-hot vectors
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        #calculate gradients
        self.dinputs = -y_true/dvalues
        #normalize the gradient
        self.dinputs = self.dinputs/samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples