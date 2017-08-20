from layer import Layer
import layer
import numpy as np
import logging

class DataLayer(Layer):
    def __init__(self, output_shape, layer_name = ''):
        self.type = 'Data'
        self.output_shape = output_shape
        self.layer_name = layer_name
        self.X_lists = None
        self.y_lists = None
        self.batch_size = 1

    def forward(self, X_lists):
        return X_lists

    def backward(self, dout):
        return dout