import sys
import os
import numpy as np
import struct
import cv2
mnist_train_data = 'mnist_data/mnist/train-images-idx3-ubyte'
mnist_train_label = 'mnist_data/mnist/train-labels-idx1-ubyte'
mnist_test_data = 'mnist_data/mnist/t10k-images-idx3-ubyte'
mnist_test_label = 'mnist_data/mnist/t10k-labels-idx1-ubyte'

def read_mnist_data(data_path, label_path):

    ## read datas ##
    fin = open(data_path, 'rb')
    buffer = fin.read()
    index = 0
    magicnumber, numImages, numRows, numColumns = struct.unpack_from('>IIII', buffer, index)
    index += struct.calcsize('>IIII')
    total_bits = numImages * numRows * numColumns
    bit_string = '>' + str(total_bits) + 'B'
    ims = struct.unpack_from(bit_string, buffer, index)
    imgs = np.array(ims, dtype = 'uint8').reshape(numImages, numRows, numColumns)

    # for i in range(imgs.shape[0]):
    #     if i % 1000 == 0:
    #         print 'saved %d images' % i
    #     cv2.imwrite('mnist_data/images/test/test_%s.jpg' % i, imgs[i])


    fin.close()


    ## read labels##
    fin = open(label_path)
    buffer = fin.read()
    index = 0
    magicnumber, numLabels = struct.unpack_from('>II', buffer, index)
    bit_string = '>' + str(numLabels) + 'B'
    index += struct.calcsize('>II')
    labels = struct.unpack_from(bit_string, buffer, index)
    labels = np.array(labels, dtype = 'int32').reshape(numLabels)
    fin.close()
    print 'read %d images and labels' % imgs.shape[0]
    return imgs, labels


if __name__ == '__main__':
    #read_mnist_data(mnist_train_data, mnist_train_label)
    read_mnist_data(mnist_test_data, mnist_test_label)

    #cv2.waitKey()