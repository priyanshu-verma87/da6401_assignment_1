# Neural Network from Scratch for MNIST & Fashion-MNIST

This project implements a **fully connected neural network (Multi-Layer Perceptron)** from scratch using **NumPy only**, without relying on deep learning frameworks such as PyTorch or TensorFlow for training.

The model supports multiple optimizers, activation functions, and loss functions and can be trained on **MNIST** and **Fashion-MNIST** datasets.

The project also integrates **Weights & Biases (W&B)** for experiment tracking and hyperparameter analysis.

---

# Features

- Neural network implemented from scratch using **NumPy**
- Supports multiple optimizers:
  - SGD
  - Momentum
  - NAG
  - RMSprop
- Multiple activation functions:
  - ReLU
  - Sigmoid
  - Tanh
- Loss functions:
  - Cross Entropy
  - Mean Squared Error (MSE)
- Configurable network architecture through CLI
- Mini-batch training
- W&B experiment tracking
- Separate scripts for **training** and **inference**

---

# Project Structure

## Project Structure

```
da6401_assignment_1/
├── models
├── notebooks
├── src/
│   ├── __pycache__/
│   │
│   ├── ann/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── activations.py
│   │   ├── neural_layer.py
│   │   ├── neural_network.py
│   │   ├── objective_function.py
│   │   └── optimizers.py
│   │
│   ├── utils/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   │
│   ├── __init__.py
│   ├── best_config.json
│   ├── best_model.npy
│   ├── inference.py
│   └── train.py
│
│
├── README.md
├── requirements.txt
└── sweeps.yaml
```

---

# Neural Network Architecture

The network is a **fully connected feedforward network** with configurable depth.

Example architecture:

```
Input Layer (784)
        ↓
Hidden Layer 1 (64 neurons, ReLU)
        ↓
Hidden Layer 2 (32 neurons, ReLU)
        ↓
Output Layer (10 neurons)
```

- Input size = **784** (flattened 28×28 image)
- Output size = **10** (digit classes)

---

# Installation

Clone the repository:

```bash
git clone <repo_link>
cd da6401_assignment_1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Required libraries:

- numpy
- sklearn
- torchvision
- wandb

---

# Running Training

Training is performed using the CLI in `train.py`.

Example command:

```bash
python -m src.train -d mnist -e 10 -b 64 -l cross_entropy -o adam -lr 0.001 -wd 0.0001 -nhl 2 -sz 64 32 -a relu relu -w_i xavier --wandb_project da6401_assignment1 --model_save_path models/model.npy
```

---

# Training Parameters

| Argument | Description |
|--------|-------------|
| `-d`, `--dataset` | Dataset: `mnist` or `fashion_mnist` |
| `-e`, `--epochs` | Number of training epochs |
| `-b`, `--batch_size` | Mini-batch size |
| `-l`, `--loss` | Loss function (`cross_entropy`, `mse`) |
| `-o`, `--optimizer` | Optimizer (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`) |
| `-lr`, `--learning_rate` | Learning rate |
| `-wd`, `--weight_decay` | L2 regularization weight decay |
| `-nhl`, `--num_layers` | Number of hidden layers |
| `-sz`, `--hidden_size` | Neurons in each hidden layer |
| `-a`, `--activation` | Activation for each hidden layer |
| `-w_i`, `--weight_init` | Weight initialization (`random`, `xavier`) |
| `--wandb_project` | W&B project name |
| `--model_save_path` | Relative path to save trained model |

---

# Running Inference

To evaluate a trained model:

```bash
python -m src.inference --model_path models/model.npy -d mnist -b 64 -nhl 2 -sz 64 32 -a relu relu -l cross_entropy -w_i xavier
```

This loads the trained weights and evaluates performance on the test dataset.

---

# Inference Output

The inference script returns:

- Test Loss
- Accuracy
- Precision
- Recall
- F1 Score

---

# Weights & Biases (Experiment Tracking)

This project uses **Weights & Biases** to track experiments.

Login to W&B:

```bash
wandb login
```

Each training run logs:

- training accuracy
- validation accuracy
- val_f1

---

# Data Processing

Datasets supported:

- MNIST
- Fashion-MNIST

Steps applied:

1. Download dataset
2. Normalize pixel values
3. Flatten images (28×28 to 784)
4. Train/Validation/Test split
5. Mini-batch training

---

# Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

---

# Key Concepts Implemented

This project implements core deep learning concepts:

- Forward propagation
- Backpropagation
- Gradient computation
- Weight initialization
- Optimization algorithms
- Numerical stability (softmax)

---

# W&B Report Link: 
https://api.wandb.ai/links/priyanshuvsp2841-indian-institute-of-technology-madras/2l7ovo82

# Github Link:
https://github.com/priyanshu-verma87/da6401_assignment_1

