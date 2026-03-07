"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import os
import numpy as np
import wandb
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data 


def parse_arguments():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description='Train a modular MLP using NumPy.')
    
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


# main function
def main():
    args = parse_arguments()
    
    # Initialize W&B
    run = wandb.init(project=args.wandb_project, config=args, mode="disabled") 
    args.__dict__.update(wandb.config)

    # Load all sets
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset) 
    num_classes = len(np.unique(y_train))

    # Ensure hidden_size is always a list and set num_layers accordingly
    args.num_layers = len(args.hidden_size)
    if len(args.activation) == 1:
        args.activation = args.activation * args.num_layers
    
    
    model = NeuralNetwork(args) 
    
    for epoch in range(args.epochs):

        # Train for one epoch
        model.train(X_train, y_train, epochs=1, batch_size=args.batch_size)

        # Validation metrics
        val_acc, _, _, val_f1 = model.evaluate(X_val, y_val)

        # Train accuracy
        train_acc, _, _, _ = model.evaluate(X_train, y_train)

        wandb.log({
            "epoch": epoch + 1,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "val_f1": val_f1
        })

        print(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}"
        )
        

    # Final Save 
    # best_weights = model.get_weights() 
    # np.save(args.model_save_path, best_weights) 

    # config_dict = vars(args)
    # config_path = os.path.join("src", "best_config.json")
    # with open(config_path, "w") as f:
    #     json.dump(config_dict, f, indent=4)

    print("Training complete!")


if __name__ == '__main__':
    main()