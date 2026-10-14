import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
import matplotlib.pyplot as plt
from NeuralNetwork import *
import nnfs
nnfs.init()
from nnfs.datasets import vertical_data, spiral_data
'''
With vertical data the random parameters get up to 93% accuracy
X, y = vertical_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()
'''

X, y = spiral_data(samples=100, classes=3) # with spiral data we only get up to 40% accuracy

dense1 = Dense_layer(2,3)
activation1 = Activation_ReLU()
dense2 = Dense_layer(3,3)
activation2 = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()


lowest_loss = 999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()


for i in range(10000):
    dense1.weights += 0.05 * np.random.randn(2,3)
    dense1.biases += 0.05 * np.random.randn(1,3)
    dense2.weights += 0.05 * np.random.randn(3,3)
    dense2.biases += 0.05 * np.random.randn(1,3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss <lowest_loss:
        print(f"New set of weights found, loss {loss}, acc  {accuracy}")
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()