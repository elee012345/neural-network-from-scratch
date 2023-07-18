# import numpy as np

# a = np.zeros((3, 2))
# print(a)
# print(a.shape)
# b = [np.zeros(x.shape) for x in a]
# print(b)
# c = np.zeros_like(a)
# print(c)
# print(c[-1])
# print(b[-1])

from activation_functions import activation_functions
import numpy as np

a = activation_functions()
softmax = a.softmax
relu = a.relu
sigmoid = a.sigmoid

arr = np.array([-1, 1, 2, 3, 4])
print(softmax.activate(arr))
print(softmax.derivative(arr))
print(relu.activate(arr))
print(relu.derivative(arr))
print(sigmoid.activate(arr))
print(sigmoid.derivative(arr))



arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[4, 3], [2, 1]])
arr3 = np.array([[2, 2], [2, 2]])

print(sum(arr1))

# print((arr1 - arr2) / arr3)