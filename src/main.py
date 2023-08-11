import mnist_loader
from network import Network

# training_data expected outputs have already been one hotted/vectorized/whatever its called
# validation_data and test_data expected outputs have not been one hotted and are still just the digits
# very annoying
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = Network([784, 30, 10])
net.train(training_data, 5, 10, 1, testing_data=test_data)


