inputs = [1 ,2 ,3, 4]
weights = [[0.4, 1.3, -0.6, 0.3],
            [2, 0.2, -1, 0],
            [-1.3, 1, -0.2, 0.4]]

bias=[1, 1.5, -0.5]


outputs = []
for n_weight, n_bias in zip(weights,bias):
    output = 0
    for input_weight, input in zip(n_weight, inputs):
        output+= input*input_weight
    output += n_bias
    outputs.append(output)
        
print(outputs)