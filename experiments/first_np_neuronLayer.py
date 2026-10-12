import numpy as np

inputs = [1 ,2 ,3, 4]
weights = [[0.4, 1.3, -0.6, 0.3],
            [2, 0.2, -1, 0],
            [-1.3, 1, -0.2, 0.4]]

bias=[1, 1.5, -0.5]

outputs = np.dot(weights,inputs) + bias
print(outputs)