import numpy as np

class Dense_layer:
    def __init__ (self, n_inputs, n_neurons):
        #initialize weights nad inputs
        self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward (self, inputs):
        #calculate output values
        self.output = np.dot(inputs, self.weights) + self.biases



class Activation_ReLU:
    def forward(self, inputs):
        #calculates outputs values from inputs
        self.output = np.maximum(0, inputs)



class Activation_Softmax:
    def forward(self, inputs):
        # get unnormalized probabilities
        exp_values = np.exp(inputs -np.max(inputs, axis=1, keepdims=True))

        # Normalize for each sample
        probabilities = exp_values/ np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities



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
        
