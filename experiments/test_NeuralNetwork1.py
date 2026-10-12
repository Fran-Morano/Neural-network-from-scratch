import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from NeuralNetwork import dense_layer, activation_ReLU, activation_Softmax
import nnfs
nnfs.init()
from nnfs.datasets import spiral_data

X, y = spiral_data(samples = 100, classes = 3) # dataset

dense1 = dense_layer(2,3) # a dense layer of 2 inputs and 3 neurons

activation1 = activation_ReLU() #activation fuction for the first dense layer

dense2 = dense_layer(3,3) # second dense layer

activation2 = activation_Softmax() # activation function for the second dense layer

#Now we connect everything

dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.output[:5])
