# coding: utf-8
import numpy as np

from .layers import *
from collections import OrderedDict


class CNN:

    def __init__(self, D, hidden_size, output_size, weight_init_std = 0.01):
        
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(D, 1, 3, 3) / np.sqrt(2 / D)
        self.params['b1'] = np.zeros((D,1))
        self.params['gamma1'] = np.ones((1,D,1,1))
        self.params['beta1'] = np.zeros((1,D,1,1))
        self.params['W2'] = weight_init_std * np.random.randn(D*14*14, hidden_size) / np.sqrt(2 /D*14*14)
        self.params['b2'] = np.zeros((1, hidden_size))
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size) / np.sqrt(2 / hidden_size)
        self.params['b3'] = np.zeros((1, output_size))

        
        # create layers
        self.layers = OrderedDict()
        # Layer 1
        self.layers['Affine1'] = Convolutional(self.params['W1'], self.params['b1'])
        self.layers['BatchNorm1'] = BatchNorm(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['MaxPooling'] = MaxPooling()

        self.layers['Dropout1'] = Dropout()
        self.layers['Flatten'] = Flatten()
        # Layer 2
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Dropout2'] = Dropout()
        # Layer 3
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            # print(layer, x.shape)
        return x
        
    # x:input data, t:correct labels
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        # accuracy = np.sum(y == t) / float(x.shape[0])
        correct = np.sum(y == t) 
        return correct

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
            # print(layer, dout.shape)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['gamma1'] = self.layers['BatchNorm1'].dgamma
        grads['beta1'] = self.layers['BatchNorm1'].dbeta
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db


        return grads
