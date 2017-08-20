import sys
import tinyCNN
import tinyCNN.Net.Model
#import tinyCNN.layers.FullyConnectedLayer
import tinyCNN.layers.DataLayer
from tinyCNN.solvers import Solver
from tinyCNN.layers import LossLayer
import numpy as np


testNet = tinyCNN.Net.Model.tinyCNNModel()
testNet.add_layer(tinyCNN.layers.DataLayer(output_shape= (1, 28, 28), layer_name = 'input_data'))
testNet.add_layer(tinyCNN.layers.ConvolutionLayer(filter_num = 1, kernel_width = 5, kernel_height = 5,
                                                      padding = 0, stride = 1, bias = True, layer_name= 'conv1'), bottom_layer_name= 'input_data')


testNet2 = tinyCNN.Net.Model.tinyCNNModel()
testNet2.add_layer(tinyCNN.layers.DataLayer(output_shape = (1, 28, 28), layer_name = 'input_data'))
testNet2.add_layer(tinyCNN.layers.ConvolutionLayer(filter_num = 1, kernel_width = 5, kernel_height = 5,
                                                      padding = 0, stride = 1, bias = True, layer_name= 'conv1'), bottom_layer_name= 'input_data')
print 'testNet:', testNet.layer_name_dict['conv1'].W, testNet.layer_name_dict['conv1'].b

print 'testNet2:', testNet2.layer_name_dict['conv1'].W, testNet2.layer_name_dict['conv1'].b

testNet.save_model("tmp.model")

testNet2.load_model("tmp.model")
print 'testNet2:', testNet2.layer_name_dict['conv1'].W, testNet2.layer_name_dict['conv1'].b