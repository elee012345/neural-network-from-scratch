import numpy as np
import random


class network():
    

    def __init__(self, layers, activation_function):
        self.layers = layers

        self.activation_function = activation_function

        # apparently a gaussian/normal distribution works well lol
        # linear does as well, but in the example he uses normal so ¯\_(ツ)_/¯

        # ok nevermind there is a method to this madness
        # https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html
        # https://www.youtube.com/watch?v=vGsc_NbU7xc&t=965s

        # normal distributions aren't the best, they're okay, but for this simple network it's good enough for me
        # randn is random normal distribution
        # the parameters are the sizes of the resulting array

        # randn creates an ARRAY, NOT a single value

        # we ignore layer 0 because that's the input layer
        self.biases = [np.random.randn(1, b) for b in layers[1:]]
        # zip creates tuples of each layer to the next 
        # so in order something like [(0, 1), (1, 2), (2, 3)]
        # with real layers something like [(4, 10), (10, 15), (15, 3)] for a network of layers 4, 10, 15, 3
        # then np.random.randn makes an array of random weights from each neuron to the next

        # switch the j, k when initializing the lists because of this:
        # "By the way, it's this expression that motivates the quirk in the wljk notation mentioned earlier. If we used j to index the input neuron, and k to index the output neuron, then we'd need to replace the weight matrix in Equation (25) by the transpose of the weight matrix. That's a small change, but annoying, and we'd lose the easy simplicity of saying (and thinking) "apply the weight matrix to the activations".
        # because we're backpropagating and going backwards instead of forwards basically
        # or something like that lol

        # think of it as the weights going TO each neuron rather than FROM each neuron
        # it makes calculating things easier in code
        self.weights = [np.random.randn(k, j) for j, k in zip(layers[:-1], layers[1:])]

        self.z_vectors = []

        self.output = []

        # self.activations = [np.zeros(size) for size in self.layers]
        self.activations = []


    # for some reason this doesn't actually take you to the right place on the webpage
    # http://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output_from_a_neural_network

    # a is activations for a layer, so it is a list
    # a is also the initial inputs to the network
    def feedforward(self, a):
        # biases and then weights because biases is each neuron, and we'll loop through 
        # of the lists of weights as part of the bias

        # since we're feeding forward, we'll think of the current layer we're on as the one after 'a' (the parameter)
        # so we're calculating the resulting values of the layer after 'a' (the parameter) but assigning those values back to a (the current layer)
        # because making another variable is just too much work
        
        for b, w in zip(self.biases, self.weights):
            # think about how a dot product works:
            # multiply matrix rows by columns of the second matrix and add
            # this is the same as multiplying each weight by the activation and adding them together to see the final result
            # for reference: https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:matrices/x9e81a4f98389efdf:multiplying-matrices-by-matrices/a/multiplying-matrices

            # the sigmoid will work like a scalar multipled by a vector except it sigmoids instead

            # the weights are going to the current layer (not from) and the biases are for the current layer

            # each loop will multiply the array of weights by the previous activations and then add the bias and sigmoid that
            # for the current layer
            # remember that we initialized the weights going TO each neuron, NOT FROM each neuron
            z = np.dot(a, w) + b
            self.z_vectors.append(z)
            a = self.activation_function.activate(z)
            self.activations.append(a)

            # and now we loop through all of the weights and activations and stuff for all the layers until we get to the end
            # and get our final output

            # each loop calculates the activations for a single layer, not single neurons

        self.output = a

        # a is now the activations for the final (output) layer
        return a
    

    def train(self, training_data, epochs, mini_batch_size, learning_rate, testing_data=None):
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[batch:batch + mini_batch_size] for batch in range(0, len(training_data), mini_batch_size)]
            for batch in mini_batches:
                for inputs, desired_outputs in batch:
                    self.feedforward(inputs)
                    bias_gradients, weight_gradients = self.backpropagate(inputs, desired_outputs)
            # taking the gradient, multiplying it by the average of the learning rate for the mini batch size,
            # and subtracting it from the biases to update them according to the gradient
            self.biases = [
                bias - (learning_rate / mini_batch_size * bias_gradient)
                for bias, bias_gradient in zip(self.biases, bias_gradients)
            ]
            self.weights = [
                weight - (learning_rate / mini_batch_size * weight_gradient)
                for weight, weight_gradient in zip(self.weights, weight_gradients)
            ]


    # in this function we try to reduce the cost
    # really we call a function that reduces that cost but whatever

    # https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work

    def backpropagate(self, inputs, desired_outputs):
        weight_gradients = [np.zeros_like(self.weights)]
        bias_gradients = [np.zeros_like(self.biases)]
        
        activations = [inputs]

        # elementwise multiplication (activation for each neuron multiplied by cost derivative of each neuron)
        delta = self.cost_derivative(activations[-1], desired_outputs[-1]) * self.activation_function.derivative(self.z_vectors[-1])
        # the derivatives for biases and weights are basically the same except that
        # to get the weight derivative, you multply the bias derivative by the weight
        # hence why we don't calculate it separately for both the weights and biases
        bias_gradients[-1] = delta
        # need to transpose because weights are going from each layer's node to the previous layer's node
        # we want to go from the current layer's node to the next layer's node
        # so transpose the activations so that it goes the other way
        # ex of layers [2, 3]:
        # going from each output node to each input node...
        # [[1, 2], 
        #  [3, 4], 
        #  [5, 6]].transpose() 
        # ->
        # now from each input node to each output node :D
        # [[1, 3, 5]
        #  [2, 4, 6]]

        weight_gradients[-1] = np.dot(delta, self.activations[-2].transpose())

        # backprop and loop through each layer

        # need to multiply each node's values by each of the weights going from it to the 
        # next layer's nodes
        # and then multiply it by the derivative of the activation function going to it from the weighted inputs
        # from the previous layers
        for layer in range(len(self.layers)-2, 1, -1):
            


            # there are two ways to do this: 
            # we can take the gradients for each layer, actually update the layers and apply the gradients to them with gradient descent, and repeat
            # or we can recursively find all the gradients without applying them to each neuron and store all those values, 
            # and only once we have found all of the gradients apply them
            # we are going with the second choice


            # The error signal for a neuron in the hidden layer is calculated as the weighted error of each neuron in the output layer. Think of the error traveling back along the weights of the output layer to the neurons in the hidden layer.
            # The back-propagated error signal is accumulated and then used to determine the error for the neuron in the hidden layer, as follows:
            # error = (weight_k * error_j) * activation_derivative(output)
            # Where error_j is the error signal from the jth neuron in the output layer, weight_k is the weight that connects the kth neuron to the current neuron and output is the output for the current neuron.
            # https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

            activation_derivative = self.activation_function.derivative(self.z_vectors[layer])
            # need to transpose again
            # taking the nodes from the layer to the right and multiplying them and adding them up by delta/the gradient
            # then multiplying by the activation derivative
            delta = np.dot(self.weights[layer + 1].transpose(), delta) * activation_derivative
            bias_gradients[layer] = delta
            # take the activations of the neurons from the layer to the left (the previous layer)
            # and then multiply it by the delta and add it together and so on
            weight_gradients[layer] = np.dot(delta, self.activations[layer-1].transpose())
        
        return bias_gradients, weight_gradients




        # returns the gradients


        # first calculate the cost in relation to the activation of the outputs nodes
        # by taking the derivative of that activation times the weighted input going into it

        # then we find how each of the weights going into each of the output node values is affecting the cost
        # we do this by taking the activations of the layer to the left and multiplying them by the values that we calculated above

        # then we need to find the gradients of all the hidden nodes and weights and stuff
        # this is basically the same thing
        # with out output layer, though, we only had to evaluate the derivative of the cost there
        # with the hidden nodes, we take the values from the nodes in the layer to the right and use those to calculate the 
        # weight and bias gradients





    def cost_derivative(actual_output, desired_output):
        # the actual derivative of the cost function (squared mean error function)
        # is 2(output - desired output) but we multiply this value by the learning rate anyway
        # so it doesn't matter

        # there are different cost functions so i could make a thing like my activations class but whatever
        return (actual_output - desired_output)
    

