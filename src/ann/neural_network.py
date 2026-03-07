"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .neural_layer import NeuralLayer
from .activations import get_activation
from .objective_functions import get_loss
from .optimizers import get_optimizer

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """
    
    def __init__(self, cli_args):
        """
        Initialize the neural network.

        Args:
            cli_args: Command-line arguments for configuring the network
        """
        # store CLI configuration
        self.cli_args = cli_args

        # layers will be created later when input dimension is known
        self.layers = None
        
        
    def _initialize_layers(self, input_dim):

        hidden_sizes = self.cli_args.hidden_size
        activations = self.cli_args.activation
        weight_init = self.cli_args.weight_init
        output_dim = 10

        num_layers = len(hidden_sizes)

        # replicate activation if only one provided
        if len(activations) == 1:
            activations = activations * num_layers

        self.layers = []
        prev_dim = input_dim

        # Build the hidden layers
        for i in range(num_layers):

            self.layers.append(
                NeuralLayer(prev_dim, hidden_sizes[i], weight_init)
            )

            self.layers.append(
                get_activation(activations[i])
            )

            prev_dim = hidden_sizes[i]

        # output layer
        self.layers.append(
            NeuralLayer(prev_dim, output_dim, weight_init)
        )

        # initialize loss
        self.loss_fn = get_loss(self.cli_args.loss)

        # initialize optimizer
        self.optimizer = get_optimizer(
            self.cli_args.optimizer,
            learning_rate=self.cli_args.learning_rate,
            weight_decay=self.cli_args.weight_decay
        )


    def _ensure_layers(self, input_dim=None, weight_dict=None):

        if self.layers is not None:
            return

        if weight_dict is not None:
            input_dim = weight_dict["W1"].shape[0]

        if input_dim is None:
            raise ValueError("Cannot determine input dimension")

        self._initialize_layers(input_dim)


    # Forward 
    def forward(self, X):
        """
        Forward propagation through all layers.
        Args:
            X: Input data
        Returns:
            Output logits
        """
        self._ensure_layers(input_dim=X.shape[1])

        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    # Backward 
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Args:
            y_true: True labels
            y_pred: Predicted outputs
        Returns:
            return grad_w, grad_b
        """

        grad_W_list = []
        grad_b_list = []

        # Get gradient from loss (dL/dZ_last)
        grad = self.loss_fn.backward(y_true, y_pred)

        # Propagate through layers in reverse order
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

            # Only collect gradients for trainable layers
            if hasattr(layer, "W"):
                grad_W_list.append(layer.gradW)
                grad_b_list.append(layer.gradb)

        # Convert to object arrays 
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        
        return self.grad_W, self.grad_b
    
    # Update
    def update_weights(self):
        """
        Update weights using the optimizer.
        """
        for layer in self.layers:
            # Only update trainable layers
            if hasattr(layer, "W"):
                self.optimizer.update(layer)
    
    # Train
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """
        self._ensure_layers(input_dim=X_train.shape[1])

        n_samples = X_train.shape[0]

        for epoch in range(epochs):

            # Shuffle data
            permutation = np.random.permutation(n_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for i in range(0, n_samples, batch_size):

                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Lookahead for NAG
                for layer in self.layers:
                    if hasattr(layer, "W"):
                        self.optimizer.lookahead(layer)

                # Forward
                logits = self.forward(X_batch)

                # Compute loss
                loss = self.loss_fn.forward(y_batch, logits)

                # Backward
                self.backward(y_batch, logits)

                # Update
                self.update_weights()
    
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """
        # normalization
        if np.max(X) > 1.0:
            X = X.astype(np.float32) / 255.0
        else:
            X = X

        logits = self.forward(X)

        predictions = np.argmax(logits, axis=1)

        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='macro')
        recall = recall_score(y, predictions, average='macro')
        f1 = f1_score(y, predictions, average='macro')

        return accuracy, precision, recall, f1
    
    # Get and set weights 
    def get_weights(self):
        d = {}
        param_layer_idx = 1 
        for layer in self.layers:
            # Only extract if the layer has weights (Dense layers)
            if hasattr(layer, 'W'):
                d[f"W{param_layer_idx}"] = layer.W.copy()
                d[f"b{param_layer_idx}"] = layer.b.copy()
                param_layer_idx += 1
        return d

    def set_weights(self, weight_dict):

        self._ensure_layers(weight_dict=weight_dict)

        param_layer_idx = 1
        for layer in self.layers:
            # Only inject if the layer has weights
            if hasattr(layer, 'W'):
                w_key = f"W{param_layer_idx}"
                b_key = f"b{param_layer_idx}"
                
                if w_key in weight_dict:
                    layer.W = weight_dict[w_key].copy()
                if b_key in weight_dict:
                    layer.b = weight_dict[b_key].copy()
                
                param_layer_idx += 1
    