import mnist_loader
from network import Network
from activation_functions import Activation

# training_data expected outputs have already been one hotted/vectorized/whatever its called
# validation_data and test_data expected outputs have not been one hotted and are still just the digits
# very annoying
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# WHY IS THE RELU BROKEN
# DERIVATIVE IS BROKEN I THINK BECUSE THERE IS NO LEARNING
net = Network([784, 30, 10], hidden_activation=Activation.sigmoid)
net.train(training_data, 50, 10, 1, testing_data=test_data)


