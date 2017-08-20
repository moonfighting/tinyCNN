from layer import Layer
import layer
import logging
import numpy as np

class SoftmaxLayer(Layer):
    def __init__(self, output_dim, bias=False, layer_name='', input_shape=()):
        Layer.__init__(self, layer_name=layer_name)
        self.type = 'Softmax'
        self.layer_name = layer_name
        self.probs = None
    def forward(self, X_list):
        """

        :param X_list:  input x, (N, C, H, W) , only one input
        :return:probs (N, D) D is the multi result of C * H * W
        """
        x = X_list
        N = x.shape[0]
        x.reshape(N, -1)
        self.probs = np.exp(x - np.max(x, axis = 1, keepdims = True))
        return self.probs

    def loss(self, X_list, y, is_training = False):
        """

        :param X_list: input X list, (N ,C, H, W)
        :param y:
        :return:
        """
        N, C, H, W = X_list.shape
        N_gt = y.shape[0]
        #print N, N_gt
        if N != N_gt:
            logging.error("data and groundtruth have different number: {0} vs {1}\n".format(N, N_gt))
            exit()
        _ = self.forward(X_list)
        loss_, dX = self.softmax_loss(X_list, y)
        dX = dX.reshape(N, C, H ,W)
        return loss_, dX

    def softmax_loss(self, X_list, y):
        """
        Computes the loss and gradient for softmax classification.

        Inputs:
        - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
          for the ith input.
        - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
          0 <= y[i] < C

        Returns a tuple of:
        - loss: Scalar giving the loss
        - dx: Gradient of the loss with respect to x
        """
        N, C, H ,W = X_list.shape
        #print N, C, H, W
        loss = -np.sum(np.log(self.probs[np.arange(N), y])) / N
        dx = self.probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        return loss, dx