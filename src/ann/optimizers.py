"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSprop
"""

import numpy as np

class SGD:
    """
    Vanilla Stochastic Gradient Descent.
    Update rule:
        W = W - lr * grad
    """
    def __init__(self, learning_rate=0.01, weight_decay=0.0):
        self.lr = learning_rate
        self.weight_decay = weight_decay

    def lookahead(self, layer):
        pass  # Not required for SGD

    def update(self, layer):
        # L2 regularization
        if self.weight_decay > 0:
            layer.gradW += self.weight_decay * layer.W

        # Parameter update
        layer.W -= self.lr * layer.gradW
        layer.b -= self.lr * layer.gradb


class Momentum:
    """
    SGD with Momentum.
    Velocity update:
        v = gamma * v + lr * grad
    Parameter update:
        W = W - v
    """
    def __init__(self, learning_rate=0.01, weight_decay=0.0, gamma=0.9):
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.velocity = {} # stores velocity per layer

    def lookahead(self, layer):
        pass  # Not required for Momentum

    def update(self, layer):

        # Initialize velocity first time
        if layer not in self.velocity:
            self.velocity[layer] = {
                "vW": np.zeros_like(layer.W),
                "vb": np.zeros_like(layer.b)
            }

        # L2 regularization
        if self.weight_decay > 0:
            layer.gradW += self.weight_decay * layer.W

        vW = self.velocity[layer]["vW"]
        vb = self.velocity[layer]["vb"]

        # Update velocity
        vW = self.gamma * vW + self.lr * layer.gradW
        vb = self.gamma * vb + self.lr * layer.gradb

        # Update parameters
        layer.W -=  vW
        layer.b -=  vb

        # Store updated velocity
        self.velocity[layer]["vW"] = vW
        self.velocity[layer]["vb"] = vb


class NAG:
    """
    Nesterov Accelerated Gradient.

    Lookahead:
        W_lookahead = W - gamma * v_prev
    Update velocity:
           v = gamma * v_prev + lr * grad
    Final update:
           W = W - v
    """
    def __init__(self, learning_rate=0.01, weight_decay=0.0, gamma=0.9):
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.velocity = {}

    def lookahead(self, layer):
        """
        Move parameters to lookahead position before forward pass.
        """

        if layer not in self.velocity:
            self.velocity[layer] = {
                "vW": np.zeros_like(layer.W),
                "vb": np.zeros_like(layer.b)
            }

        vW = self.velocity[layer]["vW"]
        vb = self.velocity[layer]["vb"]

        # Move to lookahead position
        layer.W -= self.gamma * vW
        layer.b -= self.gamma * vb

    def update(self, layer):
        
        # Get previous velocity 
        vW = self.velocity[layer]["vW"]
        vb = self.velocity[layer]["vb"]

        # Restore original weights (undo lookahead shift)
        layer.W += self.gamma * vW
        layer.b += self.gamma * vb

        # L2 regularization
        if self.weight_decay > 0:
            layer.gradW += self.weight_decay * layer.W

        # Update velocity
        vW = self.gamma * vW + self.lr * layer.gradW
        vb = self.gamma * vb + self.lr * layer.gradb

        # Apply update
        layer.W -= vW
        layer.b -= vb

        # Store updated velocity
        self.velocity[layer]["vW"] = vW
        self.velocity[layer]["vb"] = vb


class RMSprop:
    """
    RMSProp Optimizer.
    Maintains exponentially weighted average of squared gradients:
        s = beta * s + (1 - beta) * grad^2
    Update rule:
        W = W - lr * grad / (sqrt(s) + eps)
    """
    def __init__(self, learning_rate=0.001, weight_decay=0.0, beta=0.9, eps=1e-8):
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.beta = beta
        self.eps = eps
        self.cache = {} # stores the squared gradient averages
    
    def lookahead(self, layer):
        pass  # Not required for RMSProp

    def update(self, layer):

        # Initialize cache if needed
        if layer not in self.cache:
            self.cache[layer] = {
                "sW": np.zeros_like(layer.W),
                "sb": np.zeros_like(layer.b)
            }
        
        # L2 regularization
        if self.weight_decay > 0:
            layer.gradW += self.weight_decay * layer.W

        sW = self.cache[layer]["sW"]
        sb = self.cache[layer]["sb"]

        # Update squared gradient average
        sW = self.beta * sW + (1 - self.beta) * (layer.gradW ** 2)
        sb = self.beta * sb + (1 - self.beta) * (layer.gradb ** 2)

        # Parameter update
        layer.W -= self.lr * layer.gradW / (np.sqrt(sW) + self.eps)
        layer.b -= self.lr * layer.gradb / (np.sqrt(sb) + self.eps)

        self.cache[layer]["sW"] = sW
        self.cache[layer]["sb"] = sb


# Function to connect CLI string -> optimizer object
def get_optimizer(name, learning_rate=0.01, weight_decay=0.0):

    name = name.lower()

    if name == "sgd":
        return SGD(learning_rate, weight_decay)
    elif name == "momentum":
        return Momentum(learning_rate, weight_decay)
    elif name == "nag":
        return NAG(learning_rate, weight_decay)
    elif name == "rmsprop":
        return RMSprop(learning_rate, weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")