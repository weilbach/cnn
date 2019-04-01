import numpy as np
from layers import *
import pdb


class ConvNet1(object):
    """
    A convolutional network.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(1, 28, 28), num_classes=1):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_classes: Number of scores to produce from the final affine layer.
        """
        self.params = {}

        weight_scale = 1e-4

        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim[2], 1))
        self.params['b1'] = np.zeros(num_classes)




    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """

        mode = 'test' if y is None else 'train'
        scores = None


        W1, b1 = self.params['W1'], self.params['b1']

        scores, cache = fc_forward(X, W1, b1)

        if y is None:
            return scores

        loss, grads = 0, {}


        loss, dscores = l2_loss(scores, y)
        print(loss)


        _, grads['W1'], grads['b1'] = fc_backward(dscores, cache)

        return loss, grads
        
