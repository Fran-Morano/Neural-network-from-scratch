import numpy as np

inputs = [[1 ,2 ,3, 4], 
          [2,-0.5, 1, 0],
          [3,0.2, -2, 1]]

        
weights = [[0.4, 1.3, -0.6, 0.3],
            [2, 0.2, -1, 0],
            [-1.3, 1, -0.2, 0.4]]

bias = [1, 2, 3]

output = np.dot(inputs, np.array(weights).T) + bias

print(output)