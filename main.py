import mnist_loader
import network
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network([784, 30, 10])
net.gradient_descent(training_data, 30, 10, 3.0, test_data=test_data)

