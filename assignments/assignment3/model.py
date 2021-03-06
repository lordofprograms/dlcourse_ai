import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        width, height, channels = input_shape
        filter_size = 3
        pool_size = 4
        padding = 1
        stride = pool_size
        # Create necessary layers
        self.conv1_layer = ConvolutionalLayer(channels, conv1_channels, filter_size, padding)
        self.relu1_layer = ReLULayer()
        self.max_pool1_layer = MaxPoolingLayer(pool_size, stride)
        self.conv2_layer = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size, padding)
        self.relu2_layer = ReLULayer()
        self.max_pool2_layer = MaxPoolingLayer(pool_size, stride)
        self.flattener = Flattener()
        fc_input = int(height / pool_size / pool_size * width / pool_size / pool_size * conv2_channels)
        self.fc_layer = FullyConnectedLayer(fc_input, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        params = self.params()
        W1, W2, W3 = params['W1'], params['W2'], params['W3']
        B1, B2, B3 = params['B1'], params['B2'], params['B3']
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        W1.grad, B1.grad = np.zeros_like(W1.value), np.zeros_like(B1.value)
        W2.grad, B2.grad = np.zeros_like(W2.value), np.zeros_like(B2.value)
        W3.grad, B3.grad = np.zeros_like(W3.value), np.zeros_like(B3.value)

        out = self.conv1_layer.forward(X)
        out = self.relu1_layer.forward(out)
        out = self.max_pool1_layer.forward(out)
        out = self.conv2_layer.forward(out)
        out = self.relu2_layer.forward(out)
        out = self.max_pool2_layer.forward(out)
        out = self.flattener.forward(out)
        out = self.fc_layer.forward(out)

        loss, d_preds = softmax_with_cross_entropy(out, y)

        # Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        d_out = self.fc_layer.backward(d_preds)
        d_out = self.flattener.backward(d_out)
        d_out = self.max_pool2_layer.backward(d_out)
        d_out = self.relu2_layer.backward(d_out)
        d_out = self.conv2_layer.backward(d_out)
        d_out = self.max_pool1_layer.backward(d_out)
        d_out = self.relu1_layer.backward(d_out)
        d_out = self.conv1_layer.backward(d_out)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        out = self.conv1_layer.forward(X)
        out = self.relu1_layer.forward(out)
        out = self.max_pool1_layer.forward(out)
        out = self.conv2_layer.forward(out)
        out = self.relu2_layer.forward(out)
        out = self.max_pool2_layer.forward(out)
        out = self.flattener.forward(out)
        out = self.fc_layer.forward(out)

        probs = softmax(out)
        y_pred = np.argmax(probs, axis=1)
        return y_pred

    def params(self):
        # Aggregate all the params from all the layers
        # which have parameters
        result = {
            "W1": self.conv1_layer.params()["W"],
            "B1": self.conv1_layer.params()["B"],
            "W2": self.conv2_layer.params()["W"],
            "B2": self.conv2_layer.params()["B"],
            "W3": self.fc_layer.params()["W"],
            "B3": self.fc_layer.params()["B"],
        }
        return result
