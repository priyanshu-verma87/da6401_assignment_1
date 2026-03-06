"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

# mean squared error
class MeanSquaredError:
    def forward(self, y_true, y_pred):
        """
        y_true: (batch_size, output_dim)
        y_pred: (batch_size, output_dim)

        Returns scalar loss (averaged over batch)
        """
        self.y_true = y_true
        self.y_pred = y_pred

        # Compute mse = (1/N) * sum((y_true - y_pred)^2)
        loss = np.sum((y_true - y_pred) ** 2) / y_true.shape[0]
        return loss

    def backward(self, y_true, y_pred):
        """
        dL/dy_pred: (2/batch_size) * (y' - y)
        """
        batch_size = y_true.shape[0]
        return (2 / batch_size) * (y_pred - y_true)


class CrossEntropy:
    def forward(self, y_true, logits):
        """
        Computes Softmax + Cross Entropy loss.

        y_true : (batch_size,) integer class labels
        logits : (batch_size, num_classes)

        Returns: scalar mean loss over batch
        """

        # Subtract max per row to prevent overflow in exp 
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)

        # Compute exponentials
        exp_scores = np.exp(logits_shifted)

        # Normalize to get probabilities
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        self.batch_size = logits.shape[0]

        # Extract correct class probabilities
        correct_log_probs = -np.log(self.probs[np.arange(self.batch_size), y_true])

        # Return loss
        loss = np.mean(correct_log_probs)
        return loss

    def backward(self, y_true, y_pred):
        """
        Returns gradient of loss w.r.t logits: dL/dlogits
        """

        grad = self.probs.copy()

        # Compute dL/dlogits = (p - y)
        grad[np.arange(self.batch_size), y_true] -= 1
        grad = grad / self.batch_size

        return grad


# Function to connect CLI string -> objective function object
def get_loss(name):
    name = name.lower()

    if name == "mse":
        return MeanSquaredError()
    elif name == "cross_entropy":
        return CrossEntropy()
    else:
        raise ValueError(f"Unsupported loss function: {name}")