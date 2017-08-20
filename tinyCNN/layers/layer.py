#coding:utf-8
import numpy as np
import scipy as sc
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

class Layer:
    """
    layer class, to be derived to different layers
    """
    def __init__(self, layer_name):
        self.type = 'BaseLayer'
        self.layer_name = layer_name
        self.output_shape = None
        self.cache = None
        self.output_data = None
    def setup(self, input_shape):
        """
        :param input_shape:
        :return:
        """
        pass
    def forward(self, X_lists):
        """
        :param X_lists:  input X lists
        :return: output_blob, the output blob
        """
        pass

    def backward(self, dout):
        pass



