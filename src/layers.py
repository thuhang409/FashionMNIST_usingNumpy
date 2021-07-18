# coding: utf-8
from .functions import *
import numpy as np


class Convolutional:
    def __init__(self, W, b, stride=1, padding=1):
        self.W = W
        self.b = b
        self.stride = stride
        self.padding = padding
        self.cache = None

        self.dW = None
        self.db = None

    def forward(self, x):
        n_filters, d_filter, h_filter, w_filter = self.W.shape
        # print(x.shape)
        n_x, d_x, h_x, w_x = x.shape

        h_out = int((h_x - h_filter + 2*self.padding)/self.stride + 1)
        w_out = int((w_x - h_filter + 2*self.padding)/self.stride + 1)

        X_col = im2col_indices(x, h_filter, w_filter, stride=self.stride, padding=self.padding)

        W_col = self.W.reshape(n_filters, -1)
        out = W_col @ X_col + self.b
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)

        self.cache = (X_col, x)
        return out

    def backward(self, dout):
        n_filters, d_filter, h_filter, w_filter = self.W.shape
        X_col, x = self.cache

        db = np.sum(dout, axis=(0, 2, 3))
        self.db = db.reshape(n_filters, -1)

        dout_reshape = dout.transpose(1,2,3,0).reshape(n_filters, -1)
        W_reshape = self.W.reshape(n_filters, -1)

        dW = dout_reshape @ X_col.T
        self.dW = dW.reshape(self.W.shape)

        dx_col = W_reshape.T @ dout_reshape
        dx = col2im_indices(dx_col, x.shape, h_filter, w_filter, stride=self.stride, padding=self.padding)
        return dx

class MaxPooling:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

class Flatten:
    def __init__(self):
        self.shape = None
    def forward(self, x):
        self.shape = x.shape
        out = x.ravel().reshape(x.shape[0], -1)
        return out
    def backward(self, dout):
        dx = dout.ravel().reshape(self.shape)
        return dx

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Elu:
    def __init__(self, alpha = 1):
        self.alpha = alpha
        self.out = None
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = self.alpha * (np.exp(out[self.mask])-1)
        self.out = out
        return out

    def backward(self, dout):
        np.putmask(dout, self.mask, dout*(self.out + self.alpha))
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # output of softmax
        self.t = None  # correct label

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = len(self.t)
        dx = (self.y - self.t) / batch_size

        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.2):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask




