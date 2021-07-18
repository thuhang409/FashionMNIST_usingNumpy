# coding: utf-8
import numpy as np

from .layers import *
from collections import OrderedDict


class ThreeLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # initiate weights and bias
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, 600) / np.sqrt(2 / input_size)
        self.params['b1'] = np.zeros(600)
        self.params['W2'] = weight_init_std * np.random.randn(600, 300) / np.sqrt(2 / 600)
        self.params['b2'] = np.zeros(300)
        self.params['W3'] = weight_init_std * np.random.randn(300, output_size) / np.sqrt(2 / 300)
        self.params['b3'] = np.zeros(output_size)

        # create layers
        self.layers = OrderedDict()
        # Layer 1
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Elu()
        self.layers['Dropout1'] = Dropout()
        # Layer 2
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Elu()
        self.layers['Dropout2'] = Dropout()
        # Layer 3
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x:input data, t:correct labels
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db

        return grads
