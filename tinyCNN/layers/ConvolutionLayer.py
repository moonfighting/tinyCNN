from layer import Layer

import numpy as np

class ConvolutionLayer(Layer):
    """
    Convolution Layer
    """
    def __init__(self, filter_num = 1,
                        kernel_width = 1,
                        kernel_height = 1,
                        padding = 0,
                        stride = 1,
                        bias=False,
                        layer_name = '',
                        input_shape = ()
                    ):
        Layer.__init__(self, layer_name= layer_name)
        self.type = "Convolution"
        self.filter_num = filter_num
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.layer_name = layer_name
        self.output_shape = None
        self.cache = None
        self.output_data = None
        self.dW = None
        self.dX = None
        self.db = None
        if len(input_shape) > 0:
            self.setup(input_shape)


    def setup(self, input_shape):
        C, H, W = input_shape

        HO = 1 + (H + 2 * self.padding - self.kernel_height) / self.stride
        WO = 1 + (W + 2 * self.padding - self.kernel_width) / self.stride
        self.W = np.random.normal(0, 0.01, size = (self.filter_num, C, self.kernel_height, self.kernel_width))
        self.b = np.zeros(self.filter_num)
        self.output_shape = (self.filter_num, HO, WO)

    def forward(self, X_lists):
        """

        :param X_lists:  input x lists, ever x's shape is N, C, H, W
        :return:  output_blob: the convolution result of the Conv layer
        """
        assert len(X_lists) == 1, '{0} layer has more than one input'.format(self.layer_name)
        out, self.cache = self.conv_forward_naive(X_lists[0], self.W, self.b, {'stride': self.stride, 'pad': self.padding})
        self.output_data = out
        return tuple((out,))
    def backward(self, dout):
        dX_, dW_, db_ = self.conv_backward_naive(dout, self.cache)
        self.dW = dW_
        self.dX = dX_
        self.db = db_
        return dX_


    def conv_forward_naive(self, x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
    
        The input consists of N data points, each with C channels, height H and width
        W. We convolve each input with F different filters, where each filter spans
        all C channels and has height HH and width HH.
    
        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
          - 'pad': The number of pixels that will be used to zero-pad the input.
    
        Returns a tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        pad = conv_param['pad']
        stride = conv_param['stride']
        HO = 1 + (H + 2 * pad - HH) / stride
        WO = 1 + (W + 2 * pad - WW) / stride
        out = np.zeros((N, F, HO, WO))
        #############################################################################
        # TODO: Implement the convolutional forward pass.                           #
        # Hint: you can use the function np.pad for padding.                        #
        #############################################################################
        x_expanded = x
        if pad > 0:
            expanded_H, expanded_W = H + 2 * pad, W + 2 * pad
            x_expanded = np.zeros((N, C, expanded_H, expanded_W))
            for n in range(N):
                for c in range(C):
                    x_expanded[n, c, pad: H + pad, pad: pad + W] = x[n, c, :, :]

        for n in range(N):
            for f in range(F):
                for h_idx in range(HO):
                    for w_idx in range(WO):
                        start_h = h_idx * stride
                        start_w = w_idx * stride
                        out[n, f, h_idx, w_idx] = np.sum(
                            x_expanded[n, :, start_h:  start_h + HH, start_w: start_w + WW] * w[f, :, :, :]) + b[f]
        pass

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        cache = (x_expanded, w, b, conv_param)
        return out, cache


    def conv_backward_naive(self, dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
    
        Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
    
        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        #############################################################################
        # TODO: Implement the convolutional backward pass.                          #
        #############################################################################
        x, w, b, conv_param = cache
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        _, _, HO, WO = dout.shape
        stride = conv_param['stride']
        pad = conv_param['pad']
        dw = np.zeros_like(w)
        db = np.zeros_like(b)
        dx = np.zeros_like(x)

        for f in range(F):
            db[f] = np.sum(dout[:, f, :, :])
            for c in range(C):
                for hh in range(HH):
                    for ww in range(WW):
                        start_h = hh
                        start_w = ww
                        dw[f, c, hh, ww] = np.sum(
                            dout[:, f, :, :] * x[:, c, hh: hh + stride * HO: stride, ww: ww + WO * stride: stride])

        dout_H = H + HH - 1
        dout_W = W + WW - 1
        dout_expanded = np.zeros((N, F, dout_H, dout_W))
        for h_idx in range(HO):
            for w_idx in range(WO):
                dout_expanded[:, :, h_idx * stride + HH - 1, w_idx * stride + WW - 1] = dout[:, :, h_idx, w_idx]

        w_rotate = np.zeros_like(w)
        for f in range(F):
            for c in range(C):
                w_rotate[f, c, :, :] = np.rot90(w[f, c, :, :], 2)

        w_rotate_swap = w_rotate.swapaxes(0, 1)

        out, _ = self.conv_forward_naive(dout_expanded, w_rotate_swap, np.zeros(C), {'pad': 0, 'stride': 1})
        dx = out[:, :, pad: H - pad, pad: W - pad]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dx, dw, db
