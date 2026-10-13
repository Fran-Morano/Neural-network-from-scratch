import math

softmax_output = [0.7, 0.1, 0.2] #example of the output layer of the neural network

target_output = [1,0,0] # ground truth

loss = 0
for i in range(len(softmax_output)):
    loss -= math.log(softmax_output[i])*target_output[i]

print(loss)

"""
because we are using one-hot target vectors, we could assume the categorical cross entropy loss function is just
math.log(softmax[0]), because the index 0 is where the target output is equal to 1
"""