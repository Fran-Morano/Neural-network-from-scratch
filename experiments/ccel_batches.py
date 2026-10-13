import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_target = [0, 1 , 1]#which index is the target for each sample
#this targets are sparse labels
'''
They coud also be one-hot target vectors
class_targets = np.array([[1, 0, 0],
 [0, 1, 0],
 [0, 1, 0]])
 to be sure this is not the case we can use if statments to check the dimensions

if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[range(len(softmax_outputs)), class_targets]

elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(softmax_outputs*class_targets,axis=1)

'''
confidence = -np.log(softmax_outputs[range(len(class_target)), class_target])
print(confidence)

average_confidence = np.mean(confidence)
print(average_confidence)