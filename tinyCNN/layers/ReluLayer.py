from layer import Layer
import layer
import logging
import numpy as np


class ReluLayer(Layer):
    def __init__(self, layer_name = ''):
        Layer.__init__(self, layer_name = layer_name)
        self.cache = None
        self.type = 'relu'
        self.dX = None

    def setup(self, input_shape):
        self.output_shape = input_shape

    def forward(self, X_lists):
        """
        
        :param X_lists: input x lists 
        :return: the result of relu
        """
        assert X_lists, '%s layer\'t input is None' % self.layer_name
        assert len(X_lists) == 1, '%s layer has more than one input' % self.layer_name

        out, cache = self.relu_forward(X_lists[0])
        self.cache = cache
        self.output_data = out
        return tuple((out,) )

    def backward(self, dout):
        dx = self.relu_backward(dout, self.cache)
        self.dX = dx
        return dx

    def relu_forward(self, x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).
  
        Input:
        - x: Inputs, of any shape
  
        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        out = None
        #############################################################################
        # TODO: Implement the ReLU forward pass.                                    #
        #############################################################################
        pass
        out = np.maximum(x, 0)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        cache = x
        return out, cache

    def relu_backward(self, dout, cache):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).
  
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
  
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        dx = np.ones(x.shape)
        dx[x < 0] = 0
        dx = dout * dx
        #############################################################################
        # TODO: Implement the ReLU backward pass.                                   #
        #############################################################################
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dx