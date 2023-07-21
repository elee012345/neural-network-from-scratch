import numpy as np

# calculates results for entire layers, not individual nodes
class relu_function():
    # stores individual elements in lists which is annoying
    def activate(self, z):
        return np.maximum(z, 0)
    
    def derivative(self, z):
        for i in range(len(z)):
            z[i][0] = 1 if z[i] > 0 else 0
        return z
    
class sigmoid_function():
    def activate(self, z):
        return 1 / (1 + np.exp(-z))
    
    def derivative(self, z):
        activated = self.activate(z)
        return np.multiply(activated, ( 1 - activated))
    
class softmax_function():
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    def activate(self, z):
        exps = np.exp(z - np.max(z))
        return exps / np.sum(exps)
    # i have no idea if this works lol
    # stolen directly from https://github.com/SebLague/Neural-Network-Experiments/blob/main/Assets/Scripts/Neural%20Network/Activation/Activation.cs#L129
    def derivative(self, z):
        total = sum(np.exp(z))
        exp_z = np.exp(z)
        # np.multiply is hadamard product rather than dot product
        return (np.multiply(np.exp(exp_z), total) - np.multiply(exp_z, exp_z)) / np.multiply(total, total)