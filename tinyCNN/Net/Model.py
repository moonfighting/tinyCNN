#coding:utf-8
import logging
import numpy as np
from ..layers import layer
import sys
version  = sys.version_info[0]
if version  < 3:
    import cPickle as pk
else:
    import pickle pk

logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt = '%a, %d %b %Y %H:%M:%S')
Layers_with_weights = ("Convolution", "FullyConnected")
Layers_need_Backpropagation = ['FullyConnected']
class tinyCNNModel(object):
    """
    使用index，和layer_name来组织网络结构，整体上来看，越往后add的layer，index越大
    layer_name_dict记录了每个名字对应的layer
    layer_bottom_dict记录了每个层的bottom_layer都有哪些
    layer_name_index_dict记录了每个layer对应的index
    layer_index_name_dict记录了每个index对应的layer
    在进行forward，按照index从小到大进行forward， 每个layer forward的输入为其bottom_layer的输出
    允许有多个输入，单个输出。
    """
    def __init__(self):
        self.layers = []
        self.layer_name_dict = {}
        self.layer_index_dict = {}
        self.layer_bottom_dict = {}
        self.layer_name_index_dict = {}
        self.layer_index_name_dict = {}
        self.index = 0
        self.losslayer = None
        pass


    def save_model(self, model_path):
        """
        
        :param model_path:  the model path to save in
        :return: 
        """
        with open(model_path, 'w') as fp:
            for layer_name in self.layer_name_dict:
                if self.layer_name_dict[layer_name].type in Layers_with_weights:
                    try:
                        pk.dump(self.layer_name_dict[layer_name], fp)
                    except:
                        logging.error('dump layer {0} failed'.format(layer_name))


    def load_model(self, model_path):
        """
        :param model_path: the model path to restore
        :return: 
        """
        with open(model_path, 'rb') as fp:
            model_layers = pk.load(fp)
            for layer_name in model_layers:
                self.layer_name_dict[layer_name] = model_layers[layer_name]
                


    def add_layer(self, layer, bottom_layer_name = ''):
        """
        :param layer: new layer to append into the Net
        :param bottom_layer_name: the bottom layer name of the added new layer
        :return: None
        """

        if bottom_layer_name not in self.layer_name_dict and layer.type != 'Data':
            logging.error('{0} not in Net\n'.format(bottom_layer_name))
            exit()
        if layer.type != 'Data' and bottom_layer_name != '':
            bottom_layer = self.layer_name_dict[bottom_layer_name]
            layer.setup(bottom_layer.output_shape)
            self.layer_bottom_dict[layer.layer_name] = bottom_layer_name
        self.layers.append(layer)
        self.layer_name_dict[layer.layer_name] = layer
        self.layer_index_dict[self.index] = layer
        self.layer_name_index_dict[layer.layer_name] = self.index
        self.layer_index_name_dict[self.index] = layer.layer_name
        self.index += 1

    def add_loss_layer(self, LossLayer, bottom_layer_name = ''):
        if bottom_layer_name not in self.layer_name_dict and layer.type != 'Data':
            logging.error('{0} not in Net\n'.format(bottom_layer_name))
            exit()
        self.losslayer = LossLayer

    def loss(self, X_lists, y, is_training = False):
        pass
        output = self.forward(X_lists)
        loss_ , dout = self.losslayer.loss(output[0], y, is_training)
        if is_training:
            self.backward(dout)
        return loss_, output

    def forward(self, X_lists):
        """
        model forward,
        :param X_lists:  input data, which is a mutable parameter， each X_list has a shape of (N, C, H, W)
        :return: forward_output  the result of the input's forward
        """
        output = self.forward_(X_lists)
        return output

    def forward_(self, X_lists):
        """
        model forward,
        :param X_lists:  input data, which is a mutable parameter, a tuple， each X_list has a shape of (N, C, H, W)
        :return: forward_output  the result of the input's forward without loss layer
        """
        if len(X_lists) < 1:
            logging.error("model has less than one input\n");
            exit()
        forward_output = self.forwardfromto_index(0, len(self.layers) - 1, X_lists)
        return forward_output

    def forwardfromto_name(self, start_layer_name, end_layer_name, X_lists):
        assert start_layer_name in self.layer_name_index_dict, 'can not find {0} in this Net'.format(start_layer_name)
        assert end_layer_name in self.layer_name_index_dict, 'can not find {0} in this Net'.format(end_layer_name)
        start_layer_index = self.layer_name_index_dict[start_layer_name]
        end_layer_index = self.layer_name_index_dict[end_layer_name]
        final_output = self.forwardfromto_index(start_layer_index, end_layer_index, X_lists)
        return final_output

    def forwardfromto_index(self, start_layer_index, end_layer_index, X_lists):
        start_layer = self.layer_index_dict[start_layer_index]
        final_output = None
        X_lists_iter = X_lists
        for layer_index in range(start_layer_index, end_layer_index + 1):
            output_blob = self.layer_index_dict[layer_index].forward(X_lists_iter)
            X_lists_iter = output_blob
        final_output = X_lists_iter

        return final_output


    def backward(self, dout):
        return self.backwardfromto_index(0, len(self.layers) - 1, dout)

    def backwardfromto_name(self, start_layer_name, end_layer_name, dout):
        assert start_layer_name in self.layer_name_index_dict, 'can not find {0} in this Net'.format(start_layer_name)
        assert end_layer_name in self.layer_name_index_dict, 'can not find {0} in this Net'.format(end_layer_name)
        start_layer_index = self.layer_name_index_dict[start_layer_name]
        end_layer_index = self.layer_name_index_dict[end_layer_name]
        return self.backwardfromto_index(self, start_layer_index, end_layer_index, dout)

    def backwardfromto_index(self, start_layer_index, end_layer_index, dout):
        dout_ = dout
        for layer_index in range(end_layer_index, start_layer_index - 1,  -1):
            dout_ = self.layer_index_dict[layer_index].backward(dout_)

        final_dout = dout_

        return final_dout


