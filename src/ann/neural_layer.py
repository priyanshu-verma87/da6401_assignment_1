"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np

# Class for Layers
class NeuralLayer:
    # initialize the layer
    def __init__(self, input_dim, output_dim, weight_init="xavier"):
        """
        input_dim: number of input features
        output_dim: number of neurons in this layer
        weight_init: "random" or "xavier"
        """
        if weight_init == "zeros":
            self.W = np.zeros((input_dim, output_dim))
        elif weight_init == "xavier":
            lim = np.sqrt(1 / input_dim)
            self.W = np.random.randn(input_dim, output_dim) * lim
        elif weight_init == "random":
            self.W  = np.random.randn(input_dim, output_dim) * 0.01
        else:
            raise ValueError("Unsupported weight initialization")
        
        self.b = np.zeros((1, output_dim))

    # forward pass
    def forward(self, x):
        """
        x shape: (batch_size, input_dim)
        """
        self.input = x # we will store it for backward pass
        return x @ self.W + self.b
    
    # backward pass
    def backward(self, grad_out):
        """
        grad_out(dL/dz) shape: (batch_size, output_dim)
        """

        # gradient w.r.t weights: dL/dW = x^T @ dL/dZ
        self.gradW = self.input.T @ grad_out

        # gradient w.r.t bias: dL/db = sum over batch of dL/dZ
        self.gradb = np.sum(grad_out, axis=0, keepdims=True)

        # gradient w.r.t input: dL/dX = dL/dZ @ W^T
        grad_in = grad_out @ self.W.T

        return grad_in