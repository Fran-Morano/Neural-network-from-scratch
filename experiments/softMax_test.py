import numpy as np

layer_outputs = [1, 4.3, -2.3, 3.1] #its called layer outputs because the softmax activation function is mainly for the last layer

exp_values = np.exp(layer_outputs)
print(f"Exponentiated values = {exp_values}")

norm_values = exp_values / sum(exp_values)
print(f"Normalize Exponentiated values = {norm_values}")