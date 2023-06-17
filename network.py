import numpy as np

class Network():
    # apparently a gaussian/normal distribution works well lol
    # linear does as well, but in the example he uses normal so ¯\_(ツ)_/¯

    # ok nevermind there is a method to this madness
    # https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html
    # https://www.youtube.com/watch?v=vGsc_NbU7xc&t=965s

    # normal distributions aren't the best, they're okay, but for this simple network it's good enough for me

    def __init__(self, layers):
        self.biases = [np.random.randn(1, )]


def sigmoid(z):
    return 1/(1+np.exp(z))