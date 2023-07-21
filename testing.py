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

# import activation_functions
# import numpy as np

# softmax = activation_functions.softmax_function()
# relu = activation_functions.relu_function()
# sigmoid = activation_functions.sigmoid_function()

# arr = np.array([-1, 1, 2, 3, 4])
# print(softmax.activate(arr))
# print(softmax.derivative(arr))
# print(relu.activate(arr))
# print(relu.derivative(arr))
# print(sigmoid.activate(arr))
# print(sigmoid.derivative(arr))



# arr1 = np.array([[1, 2], [3, 4]])
# arr2 = np.array([[4, 3], [2, 1]])
# arr3 = np.array([[2, 2], [2, 2]])
# arr4 = np.array([[1, 2], [3, 4], [5, 6]])


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

arr5 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
arr6 = np.zeros_like(arr5)
print(arr6)