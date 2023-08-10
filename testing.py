import numpy as np

# a = np.zeros((3, 2))
# print(a)
# print(a.shape)
# b = [np.zeros(x.shape) for x in a]
# print(b)
# c = np.zeros_like(a)
# print(c)
# print(c[-1])
# print(b[-1])

import activation_functions
# # import numpy as np

# #softmax = activation_functions.softmax_function()
# relu = activation_functions.relu_function()
# #sigmoid = activation_functions.sigmoid_function()

# arr = np.array([[-1, 1, 2, 3, 5, 4], [4, 2, 6, 2, -6, -13.5]])
# #print(softmax.activate(arr))
# #print(softmax.derivative(arr))
# print(relu.activate(arr))
# print(relu.derivative(arr))
# #print(sigmoid.activate(arr))
# #print(sigmoid.derivative(arr))

def softmax(z):
    exps = np.exp(z - np.max(z))
    return exps / np.sum(exps)

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[4, 3], [2, 1]])
arr3 = np.array([[2, 2], [2, 2]])
arr4 = np.array([[1, 2], [3, 4], [5, 6]])
arr9 = np.array([23, 8, 2, 6, 3, 7, 1])
#print(np.argmax(arr9))
arr5 = [[1], [2], [3], [4]]
#print(softmax(arr5))

arr6 = [[3.19752616e-02],
       [5.51932336e-05],
       [4.06056380e-01],
       [1.92396888e-02],
       [1.13337678e-06],
       [8.52853102e-03],
       [3.63178411e-02],
       [9.78687679e-06],
       [3.41135023e-03],
       [1.16886951e-07]]





# print(arr4)
# print(arr4.transpose())

# # print((arr1 - arr2) / arr3)


# # https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

# # Backpropagate error and store in neurons
# def backward_propagate_error(network, expected):
# 	for i in reversed(range(len(network))):
# 		layer = network[i]
# 		errors = list()
# 		if i != len(network)-1:
# 			for j in range(len(layer)):
# 				error = 0.0
# 				# matrix multiplication as a for loop
# 				for neuron in network[i + 1]:
# 					# adds all of the weight gradients together
# 					error += (neuron['weights'][j] * neuron['delta'])
# 				errors.append(error)
# 		else:
# 			for j in range(len(layer)):
# 				neuron = layer[j]
# 				# list of all of the errors in the output
# 				errors.append(neuron['output'] - expected[j])
				
# 		# updates the gradient by taking the errors of the layer to the right multplied by the activation derivative
# 		for j in range(len(layer)):
# 			neuron = layer[j]
# 			neuron['delta'] = errors[j] * activation_derivative(neuron['output'])

# arr5 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
# arr6 = np.zeros_like(arr5)
# print(arr6)

