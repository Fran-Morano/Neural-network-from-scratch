import numpy as np

inputs = [1 ,2 ,3, 4]
weights = [0.4, 1.3, -0.6, 0.3]
bias = 1.5

output = np.dot(inputs, weights) + bias


print(output)