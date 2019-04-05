import numpy as np
from layers import *
import pdb


class ConvNet2(object):
    """
    A convolutional network.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_classes=10):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_classes: Number of scores to produce from the final affine layer.
        """
        self.params = {}

        #######################################################################
        # TODO: Initialize weights and biases for the convolutional neural    #
        # network. Weights should be initialized from a Gaussian distribution;#
        # biases should be initialized to zero. All weights and biases should #
        # be stored in the dictionary self.params.                            #
        #######################################################################

        filter_size = 5
        weight_scale = 1e-2

        self.params['W1'] = np.random.normal(scale=weight_scale, size=(6, input_dim[0], filter_size, filter_size))
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(16, 6, filter_size, filter_size))
        self.params['W3'] = np.random.normal(scale=weight_scale, size=((256, 128)))
        self.params['W4'] = np.random.normal(scale=weight_scale, size=((128, 10)))

        self.params['b1'] = np.zeros(6)
        self.params['b2'] = np.zeros(16)
        self.params['b3'] = np.zeros(128)
        self.params['b4'] = np.zeros(10)


    def loss(self, X, y=None, justLoss=False):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        # N = X.shape[0]
        # mode = 'test' if y is None else 'train'
        scores = None

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']


        conv_param = {'stride': 1, 'pad': 0}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        #********* begin forward pass ************

        #convolution->maxpool->relu
        conv1, conv_cache = conv_forward(X, W1, b1, conv_param)
        max1, maxpool_cache1 = max_pool_forward(conv1, pool_param)
        relu1, relu_cache1 = relu_forward(max1)

        ##convolution->maxpool->relu
        conv2, conv_cache2 = conv_forward(relu1, W2, b2, conv_param)
        max2, maxpool_cache2 = max_pool_forward(conv2, pool_param)
        relu2, relu_cache2 = relu_forward(max2)

       #affine->relu->affine
        scores, forward_cache = fc_forward(relu2, W3, b3)
        relu3, relu_cache3 = relu_forward(scores)
        scores, forward_cache2 = fc_forward(relu3, W4, b4)
        

        if y is None:
            return scores

        
        #calculate loss
        loss, grads = 0, {}

        loss, dscores = softmax_loss(scores, y)

        if justLoss:
            return loss

        #********* begin backward pass ************

        #affine backwards
        dx_4, grads['W4'], grads['b4'] = fc_backward(dscores, forward_cache2)
        
        #relu backwards->affine backwards
        dx_3 = relu_backward(dx_4, relu_cache3)
        dx_3, grads['W3'], grads['b3'] = fc_backward(dx_3, forward_cache)

        #relu backwards->pool backwards->convolve backwards
        dx_2 = relu_backward(dx_3, relu_cache2)
        dx_2 = max_pool_backward(dx_2, maxpool_cache2)
        dx_2, grads['W2'], grads['b2'] = conv_backward(dx_2, conv_cache2)

        #relu backwards->pool backwards->convolve backwards
        dx_1 = relu_backward(dx_2, relu_cache1)
        dx_1 = max_pool_backward(dx_1, maxpool_cache1)
        dx_1, grads['W1'], grads['b1'] = conv_backward(dx_1, conv_cache)
        

        return loss, grads
        
