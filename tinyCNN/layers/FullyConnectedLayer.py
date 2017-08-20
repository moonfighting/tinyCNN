from layer import Layer
import layer
import logging
import numpy as np

class FullyConnectedLayer(Layer):
    def __init__(self, output_dim, bias = False, layer_name = '', input_shape = ()):
        Layer.__init__(self,  layer_name= layer_name)
        self.type = 'FullyConnected'
        self.bias = bias
        self.W = None
        self.b = None
        self.cache = None
        self.output_data = None
        self.output_dim = output_dim
        self.dW = None
        self.dX = None
        self.db = None
        if len(input_shape) > 0:
            self.setup(input_shape = input_shape)

    def setup(self, input_shape):
        C, H, W = input_shape

        input_dim = C * H * W
        self.W = np.random.normal(0, 0.01, size = (input_dim, self.output_dim))
        self.b = np.zeros(self.output_dim)
        self.output_shape = (self.output_dim, 1, 1)

    def forward(self, X_lists):

        assert len(X_lists) == 1, '%s layer has more than one input' % self.layer_name

        N = X_lists[0].shape[0]
        out_, cache_ = self.affine_forward(X_lists[0], self.W, self.b)
        self.output_data = out_
        self.cache = cache_
        out_ = out_.reshape(N, *self.output_shape)

        return tuple((out_,))

    def backward(self, dout):
        dX_, dW_, db_ = self.affine_backward(dout, self.cache)
        self.dX = dX_
        self.dW = dW_
        self.db = db_
        return dX_

    def affine_forward(self, x, w, b):
        """
        Computes the forward pass for an affine (fully-connected) layer.

        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.

        Inputs:
        - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
        - w: A numpy array of weights, of shape (D, M)
        - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        x_shape = x.shape
        x_shape_len = len(x.shape)
        N, C, H ,W  =x.shape

        input_dim = C * H * W
        #x_reshape = x.reshape(x.shape[0], -1);

        x_reshape = x.reshape(x.shape[0], input_dim)
        out = np.dot(x_reshape, w) + b
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        cache = (x, w, b)
        return out, cache

    def affine_backward(self, dout, cache):
        """
        Computes the backward pass for an affine layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, C, H, W)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        #############################################################################
        # TODO: Implement the affine backward pass.                                 #
        #############################################################################
        pass
        N, C, H, W = dout.shape
        dout_reshape = dout.reshape(N, -1)
        x_shape = x.shape
        x_shape_len = len(x.shape)

        D = 1
        for i in range(1, x_shape_len):
            D *= x_shape[i]
        x_reshape = x.reshape(x.shape[0], D)
        dx = np.dot(dout_reshape, w.T).reshape(x_shape)
        dw = np.dot(x_reshape.T, dout_reshape)
        db = np.sum(dout_reshape, axis=0)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dx, dw, db
