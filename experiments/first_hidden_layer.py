import numpy as np

inputs = [[1 ,2 ,3, 4], 
          [2,-0.5, 1, 0],
          [3,0.2, -2, 1]]

        
weights1 = [[0.4, 1.3, -0.6, 0.3],
            [2, 0.2, -1, 0],
            [-1.3, 1, -0.2, 0.4]]

biases1 = [1, 2, 3]


weights2 = [[2, 1.5, -0.6],
            [0.2, -1.4, 3],
            [0, -2, 1.7]]

biases2 = [-1, 0.3, 2.1]

layer1_output = np.dot(inputs, np.array(weights1).T) + biases1

layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

print(layer2_output)