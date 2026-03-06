"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh
"""

import numpy as np

# ReLU activation
class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_out):
        """
        grad_out: dL/da
        returns: dL/dz = dL/da * da/dz
        """
        grad_in = grad_out.copy()
        grad_in[self.input <= 0] = 0
        return grad_in
    

# Sigmoid activation
class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, grad_out):
        """
        grad_out: dL/da
        returns: dL/dz = dL/da * da/dz
        """
        return grad_out * self.output * (1 - self.output)
    

# Tanh activation
class Tanh:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_out):
        """
        grad_out: dL/da
        returns: dL/dz = dL/da * da/dz
        """
        return grad_out * (1 - self.output ** 2)


# Function to connect CLI string -> activation object
def get_activation(name):
    name = name.lower()

    if name == "relu":
        return ReLU()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    


