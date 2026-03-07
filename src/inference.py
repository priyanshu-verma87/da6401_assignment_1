"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import json
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data 


def parse_arguments():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate trained model.')
    
    # Dataset and basic training info
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'], help='Choose between mnist and fashion_mnist') 
    parser.add_argument('-e', '--epochs', type=int, default=15, help='Number of training epochs') 
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Mini-batch size') 
    
    # Optimizer and Loss
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mean_squared_error', 'cross_entropy'], help='Choice of mean_squared_error or cross_entropy.')
    parser.add_argument('-o', '--optimizer', type=str, default='momentum', choices=['sgd', 'momentum', 'nag', 'rmsprop'], help='Optimizer selection')
    
    # Architecture 
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='Weight decay for L2 regularization')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3, help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128, 128], help='Neurons in each hidden layer')
    parser.add_argument('-a', '--activation', type=str, nargs='+', default=['tanh'], choices=['sigmoid', 'tanh', 'relu'], help='Activation for hidden layers')
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=['random', 'xavier', 'zeros'], help='Weight initialization')
    
    # W&B and Pathing 
    parser.add_argument('-w_p', '--wandb_project', type=str, default='da6401_assignment_1', help='W&B project ID')
    parser.add_argument('-model_save_path', type=str, default='src/best_model.npy', help='Relative path for .npy weights')

    return parser.parse_args()


def load_model_weights(model_path):
    """
    Load trained model from disk.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
    Returns: Dictionary - logits, loss, accuracy, f1, precision, recall 
    """
    # Get raw logits
    logits = model.forward(X_test)
    
    # Compute Loss
    loss = model.loss_fn.forward(y_test, logits)
    
    # Compute Metrics
    acc, prec, rec, f1 = model.evaluate(X_test, y_test)
    
    return {
        "logits": logits,
        "loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def main():
    """
    Main inference function.
    """
    args = parse_arguments()
    
    # Load Data
    _, _, _, _, X_test, y_test = load_data(args.dataset)
    
    args.num_layers = len(args.hidden_size)
    if len(args.activation) == 1:
        args.activation = args.activation * args.num_layers
    
    # Initialize Model and Load Weights 
    model = NeuralNetwork(args)
    weights = load_model_weights(args.model_save_path)
    model.set_weights(weights)
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test)
    
    # Output metrics 
    print(f"Evaluation Results for {args.dataset}:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"Loss: {results['loss']:.4f}")

    print("Evaluation complete!")

    return results

if __name__ == '__main__':
    main()