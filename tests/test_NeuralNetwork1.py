import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from NeuralNetwork import *
import nnfs
nnfs.init()
from nnfs.datasets import spiral_data

X, y = spiral_data(samples = 100, classes = 3) # dataset

dense1 = Dense_layer(2,3) # a dense layer of 2 inputs and 3 neurons

activation1 = Activation_ReLU() #activation fuction for the first dense layer

dense2 = Dense_layer(3,3) # second dense layer

activation2 = Activation_Softmax() # activation function for the second dense layer

loss_function = Loss_CategoricalCrossentropy() #implementing the loss function
#Now we connect everything

dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.output[:5])

loss = loss_function.calculate(activation2.output, y)

print(f"loss: {loss}")

predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) ==2:
    y = np.argmax(y, axis = 1)
accuracy = np.mean(predictions==y)
print(f"Accuravy: {accuracy}")