import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # divide max value from all predictions to prevent float out of range
    predictions -= np.max(predictions)
    pred_exp = np.exp(predictions)

    if len(predictions.shape) > 1:
        res = pred_exp / np.sum(pred_exp, axis=1, keepdims=True)
    else:
        res = pred_exp / np.sum(pred_exp)
    return res


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # cross-entropy loss
    if type(target_index) is np.ndarray:
        res = - np.log(probs[np.arange(len(probs)), target_index])
    else:
        res = - np.log(probs[target_index])
    return res.mean()


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # Your final implementation shouldn't have any loops
    predictions = predictions.copy()
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    mask = np.zeros_like(predictions)
    if type(target_index) is np.ndarray:
        mask[np.arange(len(probs)), target_index] = 1
        d_preds = - (mask - softmax(predictions)) / mask.shape[0]
    else:
        mask[target_index] = 1
        d_preds = - (mask - softmax(predictions))
    return loss, d_preds


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    loss = np.sum(W**2) * reg_strength
    grad = 2 * W * reg_strength
    return loss, grad


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        res = np.maximum(X, 0)
        self.X = X
        return res

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # Your final implementation shouldn't have any loops
        d_result = (self.X > 0) * d_out
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # Your final implementation shouldn't have any loops
        W, B = self.W.value, self.B.value
        self.X = Param(X)
        out = np.dot(X, W) + B
        return out

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        X, W = self.X.value, self.W.value
        dW, dX = np.dot(X.T, d_out), np.dot(d_out, W.T)
        dB = np.dot(np.ones((X.shape[0], 1)).T, d_out)

        self.W.grad += dW
        self.B.grad += dB

        return dX

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding

        self.stride = None
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        
        # Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        self.stride = 1
        s = self.stride

        # compute output fields by formula: (W−F+2P)/S+1 from http://cs231n.github.io/convolutional-networks/
        out_height = int((height - self.filter_size + 2 * self.padding) / self.stride + 1)
        out_width = int((width - self.filter_size + 2 * self.padding) / self.stride + 1)

        pad_width = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        X = np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)

        # maybe, add cycle for each dimension element
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        for oh in range(out_height):
            for ow in range(out_width):
                # Implement forward pass for specific location
                for bs in range(batch_size):
                    for oc in range(self.out_channels):
                        out[bs, oh, ow, oc] = np.sum(X[bs, oh * s:oh * s + self.filter_size,
                                                     ow * s:ow * s + self.filter_size, :] *
                                                     self.W.value[:, :, :, oc]) + self.B.value[oc]
        self.X = X
        return out

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        X = self.X
        W = self.W.value

        filter_size, filter_size, channels, out_channels = W.shape
        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape
        dX = np.zeros_like(X)
        dW = np.zeros_like(W)
        dB = np.sum(d_out, (0, 1, 2))
        s = self.stride
        padding = self.padding

        # Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for c in range(channels):
            for oc in range(out_channels):
                for ho in range(out_height):
                    for wo in range(out_width):
                        # Implement backward pass for specific location
                        # Aggregate gradients for both the input and
                        # the parameters (W and B)
                        for hh in range(filter_size):
                            for ww in range(filter_size):
                                dW[hh, ww, c, oc] += np.dot(X[:, ho * s + hh, wo * s + ww, c].T, d_out[:, ho, wo, oc])
                        for hi in range(height):
                            for wi in range(width):
                                if (hi - ho * s >= 0) and (hi - ho * s < filter_size) and \
                                        (wi - wo * s >= 0) and (wi - wo * s < filter_size):
                                    dX[:, hi, wi, c] += np.dot(W[hi - ho * s, wi - wo * s, c, oc], d_out[:, ho, wo, oc])

            # raise Exception("Not implemented!")
        if padding != 0:
            dX = dX[:, padding:-padding, padding:-padding, :]  # bach to the initial input size

        self.B.grad += dB
        self.W.grad += dW

        return dX

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension

        out_height = int(np.ceil(1 + (height - self.pool_size) / self.stride))
        out_width = int(np.ceil(1 + (width - self.pool_size) / self.stride))

        out = np.zeros((batch_size, out_height, out_width, channels))
        s = self.stride
        for bs in range(batch_size):
            for ch in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        out[bs, oh, ow, ch] = np.amax(X[bs, oh * s:np.minimum(oh * s + self.pool_size, height),
                                                      ow * s:np.minimum(ow * s + self.pool_size, width), ch])
        self.X = X
        return out

    def backward(self, d_out):
        # Implement maxpool backward pass
        X = self.X
        batch_size, height, width, channels = self.X.shape
        s = self.stride
        out_height = int(np.ceil(1 + (height - self.pool_size) / self.stride))
        out_width = int(np.ceil(1 + (width - self.pool_size) / self.stride))
        dX = np.zeros_like(X)

        for bs in range(batch_size):
            for ch in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        X_pool = X[bs, oh * s:np.minimum(oh * s + self.pool_size, height),
                                   ow * s:np.minimum(ow * s + self.pool_size, width), ch]
                        dX_pool = np.zeros_like(X_pool)
                        ind_max = np.unravel_index(np.argmax(X_pool, axis=None), X_pool.shape)
                        dX_pool[ind_max] = 1
                        dX[bs, oh * s:np.minimum(oh * s + self.pool_size, height),
                           ow * s:np.minimum(ow * s + self.pool_size, width), ch] += dX_pool * d_out[bs, oh, ow, ch]
        return dX

    @staticmethod
    def params():
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, height*width*channels]
        self.X = X
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        # Implement backward pass
        X = self.X
        return d_out.reshape(X.shape)

    @staticmethod
    def params():
        # No params!
        return {}
