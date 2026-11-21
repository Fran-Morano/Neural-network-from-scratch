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
        self.dinputs[self.inputs <=0] = 0
    



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


class Optimizer_SGD:

    #Initizlize optimizer - set settings, learning rate of 1. is default for this optimzer
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    #Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. /( 1. + self.decay * self.iterations))


    #Update parameters
    def update_params(self, layer):
        if self.momentum:
            #If Layer does not contain momentum arrays, create them filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                #if there is no momentum arrays for weights, there isnt one for biases
                layer.bias_momentums = np.zeros_like(layer.biases)

            #Build weight updates with momentum 
            #Take previous updates multiplied by the retain factor and update with current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            #Build biases updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad:

    #Initizlize optimizer - set settings
    def __init__(self, learning_rate=1.0, decay=0. , epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    #Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))


    #Update parameters
    def update_params(self, layer):
        #If layer does not contain cache arrays create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        #Update cache with squared current gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        #Vanilla SGD paramter update + normalization
        #With square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)


    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSprop:

    #Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0. , epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    #Call once before any parameters update
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    #update parameters
    def update_params(self, layer):
        #If layer does not contain cache arrays, create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        #Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) * layer.dbiases**2

        #Vanilla SGD parameter update + normalization with squared rooted cache
        layer.weights += -self.current_learning_rate*layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate*layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    #Call once after any paramter updates
    def post_update_params(self):
        self.iterations +=1


class Optimizer_Adam:

    #Initizlize optimizer - set settings, learning rate of 1. is default for this optimzer
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    #Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. /( 1. + self.decay * self.iterations))


    #Update parameters
    def update_params(self, layer):
        #If Layer does not contain momentum arrays, create them filled with zeros
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        #Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1-self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1-self.beta_1) * layer.dbiases

        #get corrected momentun
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1**(self.iterations+1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1**(self.iterations+1))

        #Update cache 
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) * layer.dbiases**2
        
        #get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2**(self.iterations+1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2**(self.iterations+1))

        #vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate *  weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
        
       
    
    #Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1
