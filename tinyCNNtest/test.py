import sys
import tinyCNN
import mnist
import tinyCNN.Net.Model
#import tinyCNN.layers.FullyConnectedLayer
import tinyCNN.layers.DataLayer
from tinyCNN.solvers import Solver
from tinyCNN.layers import LossLayer
import numpy as np

if __name__ == '__main__':

    ## model configuration ##
    testNet = tinyCNN.Net.Model.tinyCNNModel()
    testNet.add_layer(tinyCNN.layers.DataLayer(output_shape= (1, 28, 28), layer_name = 'input_data'))
    testNet.add_layer(tinyCNN.layers.ConvolutionLayer(filter_num = 20, kernel_width = 5, kernel_height = 5,
                                                      padding = 0, stride = 1, bias = True, layer_name= 'conv1'), bottom_layer_name= 'input_data')
    testNet.add_layer(tinyCNN.layers.PoolingLayer( kernel_size = 2, stride = 2, padding = 0,  layer_name= 'pool1'), bottom_layer_name= 'conv1')
    testNet.add_layer(tinyCNN.layers.ConvolutionLayer(filter_num= 50, kernel_height= 5, kernel_width= 5, padding= 0, stride= 1, bias = True, layer_name ='conv2'),
                      bottom_layer_name='pool1')
    testNet.add_layer(tinyCNN.layers.PoolingLayer(kernel_size = 2,  stride = 2, padding = 0,  layer_name= 'pool2'), bottom_layer_name= 'conv2')
    testNet.add_layer(tinyCNN.layers.FullyConnectedLayer(output_dim = 500, bias = True, layer_name = 'FC1'), bottom_layer_name= 'pool2')
    testNet.add_layer(tinyCNN.layers.ReluLayer(layer_name= 'relu1'), bottom_layer_name= 'FC1')
    testNet.add_layer(tinyCNN.layers.FullyConnectedLayer(output_dim = 10, bias=True, layer_name='FC2'),
                bottom_layer_name='relu1')
    testNet.add_loss_layer(LossLayer.SoftmaxLayer("prob"), bottom_layer_name= 'FC2')


    ######## get training and test lists ############

    mnist_train_datas, mnist_train_labels = mnist.read_mnist_data(mnist.mnist_train_data, mnist.mnist_train_label)
    mnist_test_datas, mnist_test_labels = mnist.read_mnist_data(mnist.mnist_test_data, mnist.mnist_test_label)
    #X_lists = np.random.random((4, 4, 1, 1))
    #X_lists = np.arange(4).reshape(1, 4, 1, 1) * 100
    #y = np.array([1,0,1,0])
    #print np.arange(4)
    #output = testNet.forward((X_lists, ))
    #loss, _= testNet.loss((X_lists, ), y, is_training= True)
    train_num, row, col = mnist_train_datas.shape
    test_num , _, _  = mnist_test_datas.shape

    X_train = mnist_train_datas.reshape((train_num, 1 , 28, 28))
    y_train = mnist_train_labels
    X_test = mnist_test_datas.reshape((test_num, 1, 28, 28))
    y_test = mnist_test_labels


    data_dict = {'X_train': X_train, 'y_train': y_train,
                 'X_val': X_test, 'y_val': y_test}
    train_solver = Solver(testNet, data_dict, learning_rate= 0.0001, optim_config=None, batch_size= 256,max_epoch= 100, optimizer='sgd')

    train_solver.train()
