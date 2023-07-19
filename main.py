import mnist_loader
import network
from activation_functions import activation_functions
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
a = activation_functions()
softmax = a.softmax
net = network([784, 30, 10], softmax)
net.gradient_descent(training_data, 30, 10, 3.0, test_data=test_data)

