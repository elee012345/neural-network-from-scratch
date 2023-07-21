import mnist_loader
from network import network
import activation_functions


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
relu = activation_functions.relu_function()
net = network([784, 30, 10], relu)
net.train(training_data, 30, 10, 1, testing_data=test_data)


