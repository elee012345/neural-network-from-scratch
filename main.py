import mnist_loader
from network import network
import activation_functions


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
relu = activation_functions.relu_function()
sigmoid = activation_functions.sigmoid_function()
net = network([784, 100, 50, 30, 10], sigmoid)
net.train(training_data, 200, 10, 3, testing_data=test_data)


