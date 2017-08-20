from ..Net import Model
import numpy as np
import logging
import optim
from ..Net import Model
import pickle
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

class Solver(object):
    def __init__(self, model, data, **kargs):
        """
            Construct a new Solver instance.

            Required arguments:
            - model: A model object conforming to the API described above
            - data: A dictionary of training and validation data with the following:
              'X_train': Array of shape (N_train, d_1, ..., d_k) giving training images
              'X_val': Array of shape (N_val, d_1, ..., d_k) giving validation images
              'y_train': Array of shape (N_train,) giving labels for training images
              'y_val': Array of shape (N_val,) giving labels for validation images

            Optional arguments:
            - optimizer: A string giving the name of an update rule in optim.py.
              Default is 'sgd'.
            - optim_config: A dictionary containing hyperparameters that will be
              passed to the chosen update rule. Each update rule requires different
              hyperparameters (see optim.py) but all update rules require a
              'learning_rate' parameter so that should always be present.
            - lr_decay: A scalar for learning rate decay; after each epoch the learning
              rate is multiplied by this value.
            - batch_size: Size of minibatches used to compute loss and gradient during
              training.
            - num_epochs: The number of epochs to run for during training.
            - print_every: Integer; training losses will be printed every print_every
              iterations.
            - verbose: Boolean; if set to false then no output will be printed during
              training.
            """
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.model = model
        self.learning_rate = kargs['learning_rate']
        self.optimizer = kargs['optimizer']
        self.optim_config = kargs['optim_config']
        self.batch_size = kargs['batch_size']
        self.max_epoch = kargs['max_epoch']
        self.optimizer_fn = getattr(optim, self.optimizer)


    def _next_batch(self, X, y, iter, batch_size):
        """
        
        :param X:  input data
        :param y:   input label
        :param iter:  the iter index in the dataset
        :param batch_size: the batch size
        :return:  batch_data, batch_label
        """

        start_index = iter * batch_size
        end_index = start_index + batch_size
        num_ = X.shape[0]

        if start_index >= num_:
            logging.error('starting index greater than the num of input data')
        if end_index >= num_:
            return X[start_index: ], y[start_index: ]

        return X[start_index : end_index], y[start_index : end_index]


    def do_val(self):
        num_val = self.X_val.shape[0]
        iterations = num_val / self.batch_size
        if num_val % self.batch_size != 0:
            iterations += 1
        true_count = 0
        total_loss = 0
        for step in xrange(iterations):
            X_, y_ = self._next_batch(self.X_val,self.y_val, step, self.batch_size)
            loss,  output = self.model.loss((X_, ), y_, is_training = False)
            N, C, H, W = output[0].shape
            logits = np.squeeze(output[0], axis = (2, 3))
            prediced_label = np.argmax(logits, axis = 1)
            true_count += np.sum(prediced_label == y_)
            total_loss += loss

        return total_loss / float(iterations), true_count / float(num_val)




    def train(self):
        num_train = self.X_train.shape[0]

        iterations_per_epoch = num_train / self.batch_size
        if num_train % self.batch_size != 0:
            iterations_per_epoch += 1
        max_iterations = iterations_per_epoch * self.max_epoch

        best_accuracy  = 0
        best_loss = 10000
        for iter_step in xrange(max_iterations):
            iter_in_epoch = iter_step % iterations_per_epoch
            epoch_idx = iter_step / iterations_per_epoch
            X_, y_ = self._next_batch(self.X_train, self.y_train, iter_in_epoch, batch_size= self.batch_size )

            loss, _  = self.model.loss((X_, ), y_, is_training = True)
            for layer_name, layer in self.model.layer_name_dict.items():
                layer_type = self.model.layer_name_dict[layer_name].type
                if layer_type in Model.Layers_need_Backpropagation:
                    #print 'before update:'
                    #print self.model.layer_name_dict[layer_name].W
                    #print self.model.layer_name_dict[layer_name].b

                    # print 'gradients:'
                    # print self.model.layer_name_dict[layer_name].dW
                    # print self.model.layer_name_dict[layer_name].db
                    self.model.layer_name_dict[layer_name].W = self.optimizer_fn(self.model.layer_name_dict[layer_name].W,
                                                                                    self.model.layer_name_dict[layer_name].dW, self.learning_rate)
                    self.model.layer_name_dict[layer_name].b = self.optimizer_fn(self.model.layer_name_dict[layer_name].b,
                                                                                 self.model.layer_name_dict[layer_name].db, self.learning_rate)
                    # print 'after update:'
                    # print self.model.layer_name_dict[layer_name].W
                    # print self.model.layer_name_dict[layer_name].b

            print 'epoch %d, iter %d/%d, loss = %f' % (epoch_idx, iter_in_epoch, iterations_per_epoch, loss)

            if iter_in_epoch + 1 == iterations_per_epoch:
                val_loss, accuracy = self.do_val()
                print 'test loss = %f, accuracy = %f' % (val_loss, accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    print 'best_accuracy: %f' % best_accuracy
            # TO DO  save model

        print 'training done, best accuracy:%f' % best_accuracy

    def save_model(self, save_name):
        param_dict = {}
        for layer_name, layer in self.model.layer_name_dict.items():
            layer_type = self.model.layer_name_dict[layer_name].type
            if layer_type in Model.Layers_need_Backpropagation:
                param_dict[layer_name] = {'weights:',self.model.layer_name_dict[layer_name].W,
                                            'bias:', self.model.layer_name_dict[layer_name].b}

        fout = open(save_name, 'w')
        pickle.dump(param_dict, fout)