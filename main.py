import mnist_loader
from network import network
import activation_functions

# training_data expected outputs have already been one hotted/vectorized/whatever its called
# validation_data and test_data expected outputs have not been one hotted and are still just the digits
# very annoying
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
relu = activation_functions.relu_function()
sigmoid = activation_functions.sigmoid_function()
net = network([784, 30, 10], sigmoid)
net.train(training_data, 5, 10, 3, testing_data=test_data)


