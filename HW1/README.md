# Medical Image Classification with Linear Classifiers and Neural Networks

## Overview

This project involves implementing machine learning models for medical image classification using the OCTMNIST dataset. The dataset consists of optical coherence tomography (OCT) images for retinal disease diagnosis across four categories. The models range from simple linear classifiers to deep learning models, both with and without automatic differentiation.

## Implementation Details

### **1. Linear Classifiers without Machine Learning Libraries**
The first part of the project focuses on implementing linear classifiers without using external machine learning libraries (except for NumPy). It includes:

- **Perceptron Model**
  - Implement the `update_weights` method in the `Perceptron` class.
  - Train the perceptron for 20 epochs and evaluate its performance on training, validation, and test sets.
  - Plot training and validation accuracies over epochs.

- **Logistic Regression**
  - Implement the `update_weights` method in the `LogisticRegression` class.
  - Train logistic regression using stochastic gradient descent for 50 epochs.
  - Compare models using two learning rates: `0.01` and `0.001`.
  - Report final test accuracies and plot training/validation accuracies.

### **2. Multi-Layer Perceptron (MLP) Without Neural Network Libraries**
- Implement a feed-forward neural network with a single hidden layer (200 hidden units).
- Use **ReLU** activation for the hidden layer and **multinomial logistic loss** for classification.
- Train using stochastic gradient descent for 20 epochs with a learning rate of `0.001`.
- Initialize biases as zero and weights from a normal distribution (`μ = 0.1, σ² = 0.12`).
- Report final test accuracy and plot training loss, training accuracy, and validation accuracy.

### **3. Neural Network Implementation Using an Autodiff Toolkit**
The second part of the project involves using PyTorch (or an equivalent framework) to implement the same models with automatic differentiation.

- **Logistic Regression with PyTorch**
  - Implement the `train_batch` method and the `LogisticRegression` class.
  - Train the model using stochastic gradient descent (batch size = 16).
  - Tune the learning rate (`0.001`, `0.01`, `0.1`) and report the best configuration.
  - Plot training loss and validation accuracy.

- **Feed-Forward Neural Network with Dropout Regularization**
  - Implement the `FeedforwardNetwork` class with dropout.
  - Experiment with batch sizes (16 vs. 1024) and analyze their impact on training time and performance.
  - Train with different learning rates (`1, 0.1, 0.01, 0.001`) and compare results.
  - Evaluate overfitting by training with batch size 256 for 150 epochs.
  - Compare models using **L2 regularization (0.0001)** and **dropout (0.2)**.

### **4. Boolean Function Computation with Multi-Layer Perceptron**
- Demonstrate that a single perceptron cannot compute a given Boolean function.
- Show that the function can be computed using a multi-layer perceptron with two hidden units and hard threshold activations.
- Provide the required weights and biases.
- Extend the solution using **ReLU activations** and solve for `D = 2, A = B = 0`.

## Running the Code
- **Download Dataset:** `python download_octmnist.py`
- **Train Perceptron:** `python hw1-q1.py perceptron`
- **Train Logistic Regression:** `python hw1-q1.py logistic_regression -epochs 50 -learning_rate 0.01`
- **Train MLP (Manual Implementation):** `python hw1-q1.py mlp`
- **Train Logistic Regression with PyTorch:** `python hw1-q2.py`
- **Train MLP with PyTorch:** `python hw1.py mlp`

## Summary
This project explores different machine learning models for medical image classification, emphasizing both manual gradient computation and the use of automatic differentiation frameworks. The implementations allow for experimentation with different architectures, training strategies, and regularization techniques.
