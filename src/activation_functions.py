import numpy as np
class Activation:
    # calculates results for entire layers, not individual nodes
    class relu:
        # stores individual elements in lists which is annoying
        @staticmethod
        def activate(z):
            return np.maximum(z, 0)
        
        @staticmethod
        def derivative(z):
            z[z <= 0] = 0
            z[z > 0] = 1
            return z
        
    class sigmoid:
        @staticmethod
        def activate(z):
            return 1 / (1 + np.exp(-z))
        
        @staticmethod
        def derivative(z):
            activated = 1 / (1 + np.exp(-z)) # sigmoid activation
            return np.multiply(activated, ( 1 - activated))
        
    class softmax():
        # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
        @staticmethod
        def activate(z):
            exps = np.exp(z - np.max(z))
            return exps / np.sum(exps)
        # i have no idea if this works lol
        # stolen directly from https://github.com/SebLague/Neural-Network-Experiments/blob/main/Assets/Scripts/Neural%20Network/Activation/Activation.cs#L129
        @staticmethod
        def derivative(z):
            total = sum(np.exp(z))
            exp_z = np.exp(z)
            # np.multiply is hadamard product rather than dot product
            return (np.multiply(np.exp(exp_z), total) - np.multiply(exp_z, exp_z)) / np.multiply(total, total)