import numpy as np
from scipy.signal import correlate
import pdb


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    

    rows = x.shape[0]
    cols = np.prod(x.shape[1:])
    x_reshaped = x.reshape(rows, cols)
    #change this to w for not toy data
    out = np.dot(x_reshaped, w) + b

    cache = (x, w, b)

    return out, cache



def fc_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    rows = x.shape[0]
    cols = np.prod(x.shape[1:])
    x_reshaped = x.reshape(rows, cols)

    dw = np.dot(x_reshaped.T, dout)
    dx = np.dot(dout, w.T)
    dx = np.reshape(dx, x.shape)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    result = np.maximum(0, x)

    result[result > 0] = 1
    dx = dout * result

    return dx


def conv_forward(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on
    both sides) along the height and width axes of the input. Be careful not to
    modfiy the original input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    N, C, H, W = x.shape
    F, C2, H2, W2 = w.shape

    pad = int(conv_param['pad'])
    stride = conv_param['stride']

    H_hat = 1 + (H + 2 * pad - H2) / stride
    W_hat = 1 + (W + 2 * pad - W2) / stride

    out = np.zeros((N, F, int(H_hat), int(W_hat)))
    
    x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant', constant_values=0)


    for i in range(0, N):
      img = x_padded[i]
      for f in range(0, F):
        filt = w[f]
        convolved = correlate(img, filt, mode='valid', method='fft') 
        convolved +=  b[f]
        out[i][f][:][:] = convolved

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward(dout, cache):
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

    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, H2, W2 = w.shape

    pad = int(conv_param['pad'])
    stride = conv_param['stride']

    x_padded = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant', constant_values=0)
    # print(x_padded.shape)
    #2,3,8,8

    db = np.zeros(b.shape)

    for i in range(0, F):
      b_sum = np.sum(dout[:, i, :, :])
      db[i] = b_sum


    dw = np.zeros(w.shape)
    for i in range(0, N):
      for f in range(0, F):
        filt = dout[i][f]
        convolved1 = correlate(x_padded[i], [filt], mode='valid', method='fft')
        dw[f] = np.add(dw[f], convolved1)



    dx = np.zeros(x.shape)
    for i in range(0, N):
      for c in range(0, C):
        dx_help = np.zeros((H, W))
        for f in range(0, F):
          filt = w[f][c]
          filt = np.rot90(filt)
          filt = np.rot90(filt)
          convolved1 = correlate(dout[i][f], filt, mode='full', method='fft')
          dx_help = np.add(dx_help, convolved1)
        
        dx[i][c] = dx_help

    
    
    # (Pdb) p dx.shape
    # (2, 3, 8, 8)
    # (Pdb) p dw.shape 
    # (5, 3, 3, 3)
    # (Pdb) p db.shape
    # (5,)
    # print(dx.shape)
    
    return dx, dw, db


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    N, C, H, W = x.shape
    
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    H_out = 1 + (H - pool_height) / stride 
    W_out = 1 + (W - pool_width) / stride 

  
    out = np.zeros((N, C, int(H_out), int(W_out)))
    for i in range(0, N):
      for c in range(0, C):
        h2 = 0
        for h in range(0, H - pool_height + 1, stride):
          w2 = 0
          for w in range(0, W - pool_width + 1, stride):
            out[i][c][h2][w2] = np.max(x[i,c, h:h+ pool_height, w:w+pool_width])
            w2 += 1
          h2 += 1

    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None

    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    dx = np.zeros((x.shape))
    
    for i in range(0, N):
      for c in range(0, C):
        h2 = 0
        for h in range(0, H - pool_height + 1, stride):
          w2 = 0
          for w in range(0, W - pool_width + 1, stride):
            pooled = x[i, c, h:h + pool_height, w:w + pool_width]
            temp = np.equal(pooled, np.max(pooled))
            dx[i, c, h:h+pool_height, w:w+pool_width] += dout[i, c, h2, w2] * temp
            w2 += 1
          h2 += 1
    

    return dx


def l2_loss(x, y):
    """
    Computes the loss and gradient of L2 loss.
    loss = 1/N * sum((x - y)**2)

    Inputs:
    - x: Input data, of shape (N, D)
    - y: Output data, of shape (N, D)

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    #think this is right 
    #well it's not right
    #now it is right thanks jsorenson

    loss, dx = None, None
    N, D = x.shape
    loss = 1/N * np.sum((x-y)**2)
    dx = np.zeros((N, D))
    # print('loss shapes')
    # print(x.shape)
    # print(y.shape)

    for i in range(0, N):
      for j in range(0, D):
        dx[i][j] = 2 * (x[i][j] - y[i][j])
    
    dx /= N

    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    # loss, dx = None, None
    N, C = x.shape

    loss = 0

    dx = np.zeros((N,C))

    for n in range(0, N):
      ind = y[n]
      probs = x[n] - np.max(x[n])
      probs = np.exp(probs)
      probs /= (np.sum(probs))
      loss += -np.log(probs[ind] + np.finfo(float).eps)
      dx[n] = probs
      dx[n][ind] -= 1
    
    dx /= N
    loss /= N

    return loss, dx
