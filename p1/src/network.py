import numpy as np
from layers import *
import pdb


class ConvNet(object):
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
        num_filters = 6
        hidden_dim = 784

        self.params['W1'] = np.random.normal(scale=weight_scale, size=(num_filters, input_dim[0], filter_size, filter_size))
        self.params['W2'] = np.random.normal(scale=weight_scale, size=(num_filters, 6, filter_size, filter_size))
        self.params['W3'] = np.random.normal(scale=weight_scale, size=(600, num_classes))

        # self.params['W3'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        # self.params['W4'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))

        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(num_filters)
        self.params['b3'] = np.zeros(num_classes)

        # self.params['b3'] = np.zeros(num_classes)
        # self.params['b4'] = np.zeros(num_classes)





    def loss(self, X, y=None):
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

        # filter_size = W1[2].shape
        # filter_size = 5

        conv_param = {'stride': 1, 'pad': 0}
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


        #######################################################################
        # TODO: Implement the forward pass for the convolutional neural net,  #
        # computing the class scores for X and storing them in the scores     #
        # variable.                                                           #
        #######################################################################

        #THIS IS COMPARISON CODE 
        # out_1, cache_1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        # out_2, cache_2 = affine_relu_forward(out_1, W2, b2)
        # out_3, cache_3 = affine_forward(out_2, W3, b3)
        # scores = out_3
        #END COMPARISON CODE

        conv1, conv_cache = conv_forward(X, W1, b1, conv_param)
        relu1, relu_cache1 = relu_forward(conv1)

        conv2, conv_cache2 = conv_forward(relu1, W2, b2, conv_param)
        relu2, relu_cache2 = relu_forward(conv2)

        scores, maxpool_cache = max_pool_forward(relu2, pool_param)
        scores, forward_cache = fc_forward(scores, W3, b3)
        


        if y is None:
            return scores

        loss, grads = 0, {}
        #######################################################################
        # TODO: Implement the backward pass for the convolutional neural net, #
        # storing the loss and gradients in the loss and grads variables.     #
        # Compute data loss using softmax, and make sure that grads[k] holds  #
        # the gradients for self.params[k].                                   #
        loss, dscores = softmax_loss(scores, y)
        # print(loss)

        #THIS IS COMPARISON CODE
        # loss += sum(0.5*self.reg*np.sum(W_tmp**2) for W_tmp in [W1, W2, W3])

        # dx_3, grads['W3'], grads['b3'] = fc_backward(dscores, cache_3)
        # dx_2, grads['W2'], grads['b2'] = affine_relu_backward(dx_3, cache_2)
        # dx_1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx_2, cache_1)

        # grads['W3'] += self.reg*self.params['W3']
        # grads['W2'] += self.reg*self.params['W2']
        # grads['W1'] += self.reg*self.params['W1']
        #END COMPARISON CODE


        dx_3, grads['W3'], grads['b3'] = fc_backward(dscores, forward_cache)
        dx_3 = max_pool_backward(dx_3, maxpool_cache)

        dx_2 = relu_backward(dx_3, relu_cache2)
        dx_2, grads['W2'], grads['b2'] = conv_backward(dx_3, conv_cache2)

        dx = relu_backward(dx_2, relu_cache1)
        dx, grads['W1'], grads['b1'] = conv_backward(dx, conv_cache)
        
        

        return loss, grads
        
