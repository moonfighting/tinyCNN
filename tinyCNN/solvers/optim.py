import numpy as np


def sgd(w, dw, learning_rate = 0.01):
    w -= learning_rate * dw
    return w


def momentum(w, dw, learning_rate):
    pass