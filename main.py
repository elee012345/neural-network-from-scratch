import mnist_loader
from network import network
import activation_functions


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
softmax = activation_functions.softmax_function()
net = network([784, 30, 10], softmax)
net.train(training_data, 30, 10, 3.0, testing_data=test_data)


