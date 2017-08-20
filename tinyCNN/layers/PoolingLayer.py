from layer import Layer
import layer
import logging
import numpy as np
#
# class PoolingLayer2(Layer):
#     def __init__(self, kernel_size = 1, stride = 1, padding = 0):
#         Layer.__init__(self, layer_name= '')
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding

class PoolingLayer(Layer):

    """
    Poolint layers
    """
    #def __init__(self, kernel_size = 1, stride = 1, padding = 0, pooling_type = 'max', layer_name = ''):
    def __init__(self, kernel_size=1, stride=1, padding = 0,  pooling_type = 'max', layer_name = ''):
        Layer.__init__(self, layer_name = layer_name)
        self.type = pooling_type + '_pooling'
        self.layer_name = layer_name
        self.pooling_type = pooling_type
        self.cache = None
        self.dX = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def setup(self, input_shape):
        C, H, W = input_shape
        pooled_height = (H + 2 * self.padding - self.kernel_size) / self.stride + 1
        pooled_width = (W + 2 * self.padding - self.kernel_size) / self.stride + 1
        self.output_shape = (C, pooled_height, pooled_width)

    def forward(self, X_lists):
        """
        
        :param X_lists:  input X lists
        :return: 
        """
        assert X_lists, '%s layer\'t input is None' % self.layer_name
        assert len(X_lists) == 1, '%s layer has more than one input' % self.layer_name
        out_ = None
        if self.pooling_type == 'max':
            pool_param = {'pool_height': self.kernel_size, 'pool_width': self.kernel_size
                        , 'stride': self.stride}
            out_, cache_ = self.max_pool_forward_naive(X_lists[0], pool_param)
            self.cache  = cache_

        return tuple((out_, ))

    def backward(self, dout):
        """
        
        :param dout:  diff from top layer
        :return: dX
        """
        if self.pooling_type == 'max':
            dx = self.max_pool_backward_naive(dout, self.cache)
            self.dX = dx
        return self.dX

    def max_pool_forward_naive(self, x, pool_param):
      """
      A naive implementation of the forward pass for a max pooling layer.
    
      Inputs:
      - x: Input data, of shape (N, C, H, W)
      - pool_param: dictionary with the following keys:
        - 'pool_height': The height of each pooling region
        - 'pool_width': The width of each pooling region
        - 'stride': The distance between adjacent pooling regions
    
      Returns a tuple of:
      - out: Output data
      - cache: (x, pool_param)
      """
      out = None
      N,C,H,W = x.shape
      pool_height = pool_param['pool_height']
      pool_width = pool_param['pool_width']
      stride = pool_param['stride']
      out_height = (H - pool_height) / stride + 1
      out_width = (W - pool_width) / stride + 1
      #############################################################################
      # TODO: Implement the max pooling forward pass                              #
      #############################################################################
      out = np.zeros((N, C, out_height, out_width))
      for n in range(N):
        for c in range(C):
          for h in range(out_height):
            for w in range(out_width):
              start_h = h * stride
              start_w = w * stride
              out[n,c, h, w] = x[n, c, start_h : start_h + pool_height, start_w : start_w + pool_width ].max()

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      cache = (x, pool_param)
      return out, cache


    def max_pool_backward_naive(self, dout, cache):
      """
      A naive implementation of the backward pass for a max pooling layer.
    
      Inputs:
      - dout: Upstream derivatives
      - cache: A tuple of (x, pool_param) as in the forward pass.
    
      Returns:
      - dx: Gradient with respect to x
      """
      dx = None
      x, pool_param = cache
      dx = np.zeros_like(x)
      pool_width = pool_param['pool_width']
      pool_height = pool_param['pool_height']
      stride = pool_param['stride']
      N,C, out_height, out_width = dout.shape
      #############################################################################
      # TODO: Implement the max pooling backward pass                             #
      #############################################################################
      for n in range(N):
        for c in range(C):

          for ho in range(out_height):
            for wo in range(out_width):
              start_h = ho * stride
              start_w = wo * stride
              max_idx = x[n, c, start_h : start_h + pool_height, start_w : start_w + pool_width ].argmax()

              max_loc = (max_idx / pool_width, max_idx % pool_width)

              dx[n, c, start_h + max_loc[0], start_w + max_loc[1]] = dout[n,c,ho, wo]
      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return dx


if __name__ == '__main__':
    pool_layer = PoolingLayer(kernel_size = 1, stride = 1, padding = 0)