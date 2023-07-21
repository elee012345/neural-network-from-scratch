import numpy as np
import random


class network():
    def __init__(self, layers, activation_function):
        self.layers = layers
        self.activation_function = activation_function
        self.biases = [np.random.randn(b, 1) for b in layers[1:]]
        self.weights = [np.random.randn(k, j) for j, k in zip(layers[:-1], layers[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = self.activation_function.activate(z)
        return a

    def train(self, training_data, epochs, mini_batch_size, learning_rate, testing_data=None):
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[batch:batch + mini_batch_size] for batch in range(0, len(training_data), mini_batch_size)]
            for batch in mini_batches:
                bias_gradients = [np.zeros(b.shape) for b in self.biases]
                weight_gradients = [np.zeros(w.shape) for w in self.weights]
                for inputs, desired_outputs in batch:
                    delta_bias_gradients, delta_weight_gradients = self.backpropagate(inputs, desired_outputs)
                    bias_gradients = [
                        b + db
                        for b, db in zip(bias_gradients, delta_bias_gradients)
                    ]
                    weight_gradients = [
                        w + dw
                        for w, dw in zip(weight_gradients, delta_weight_gradients)
                    ]
                self.biases = [
                    bias - (learning_rate / mini_batch_size * bias_gradient)
                    for bias, bias_gradient in zip(self.biases, bias_gradients)
                ]
                self.weights = [
                    weight - (learning_rate / mini_batch_size * weight_gradient)
                    for weight, weight_gradient in zip(self.weights, weight_gradients)
                ]

            print("Epoch " + str(epoch + 1) + " finished out of " + str(epochs))
            cost = self.test_progress(testing_data)
            print(f"Average costs {cost}")
            print(f"Average cost {sum(cost)/10}")
            print(f"eval: {self.evaluate(testing_data)} / {len(testing_data)}" ) 

    def test_progress(self, test_data):
        total = 0
        for inputs, desired_outputs in test_data:
            total += self.cost_derivative(self.feedforward(inputs), desired_outputs)
        average_cost = total/len(test_data[0])
        return average_cost

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def backpropagate(self, inputs, desired_outputs):
        
        

        weight_gradients = [np.zeros(w.shape) for w in self.weights]
        bias_gradients = [np.zeros(b.shape) for b in self.biases]

        z_vectors = []
        a = inputs
        activations = [inputs]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            z_vectors.append(z)
            a = self.activation_function.activate(z)
            activations.append(a)


        delta = self.cost_derivative(activations[-1], desired_outputs) * self.activation_function.derivative(z_vectors[-1])
        bias_gradients[-1] = delta
        weight_gradients[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, len(self.layers)):
            activation_derivative = self.activation_function.derivative(z_vectors[-layer])
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * activation_derivative
            bias_gradients[-layer] = delta
            weight_gradients[-layer] = np.dot(delta, activations[-layer-1].transpose())
        return bias_gradients, weight_gradients

    def cost_derivative(self, actual_output, desired_output):
        return (actual_output - desired_output)
    

