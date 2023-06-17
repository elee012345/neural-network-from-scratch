import numpy as np

class Network():
    

    def __init__(self, layers):
        self.layers = layers

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
        self.weights = [np.random.randn(k, j) for j, k in zip(layers[:-1], layers[1:])]


    # for some reason this doesn't actually take you to the right place on the webpage
    # http://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output_from_a_neural_network

    # a is activations for a layer, so it is a list
    def feedfoward(self, a):
        # biases and then weights because biases is each neuron, and we'll loop through 
        # of the lists of weights as part of the bias

        # since we're feeding forward, we'll think of the current layer we're on as the one after 'a' (the parameter)
        # so we're calculating the resulting values of the layer after 'a' (the parameter) but assigning those values back to a (the current layer)
        # because making another variable is just too much work
        for b, w in zip(self.biases, self.weights):
            # think about how a dot product works:
            # multiply matrix rows by columns of the second matrix and add
            # this is the same as multiplying each weight by the activation and adding them together to see the final result

            # the sigmoid will work like the hadamard product and sigmoid each value in the array individually

            # the weights are going to the current layer (not from) and the biases are for the current layer

            # each loop will multiply the array of weights by the previous activations and then add the bias and sigmoid that
            # for the current layer
            # remember that we initialized the weights going TO each neuron, NOT FROM each neuron
            a = sigmoid(np.dot(a, w) + b)
        # a is now the activations for the next layer
        return a
    
    


    


def sigmoid(z):
    return 1/(1+np.exp(-z))