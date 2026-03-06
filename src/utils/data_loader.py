"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


def load_data(dataset_name, val_split=0.1):
    """
    Loads dataset, normalizes, flattens, and splits into train/validation/test.

    Args:
        dataset_name (str): "mnist" or "fashion_mnist"
        val_split (float): validation split ratio

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """

    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'fashion_mnist'.")

    # Normalize pixel values (0-255 to 0-1)
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Flatten images (28x28 to 784)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_split,
        random_state=42,
        stratify=y_train
    )

    return X_train, y_train, X_val, y_val, X_test, y_test