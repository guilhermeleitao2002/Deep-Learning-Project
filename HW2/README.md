# Deep Learning - Homework 2

## Overview

This project focuses on implementing and analyzing different deep learning architectures, including self-attention mechanisms, convolutional neural networks (CNNs) for image classification, and encoder-decoder models for automatic speech recognition (ASR). The implementation includes mathematical derivations, optimization techniques, and model performance evaluations.

## Implementation Details

### **1. Efficient Self-Attention Approximation**
This section investigates the computational complexity of self-attention in transformers and explores a low-rank approximation to improve efficiency.

- **Computational Complexity Analysis**
  - Analyze the complexity of self-attention (`Softmax(QK^T)V`) with respect to sequence length (`L`).
  - Discuss why this complexity is problematic for long sequences.

- **McLaurin Series Approximation for Softmax**
  - Approximate the exponential function using the first three terms of the McLaurin series.
  - Derive a feature map `φ(q)` such that `exp(q^T k) ≈ φ(q)^T φ(k)`.
  - Determine the feature space dimensionality (`M`) as a function of hidden size (`D`).

- **Low-Rank Attention Approximation**
  - Reformulate the self-attention operation using the approximated feature maps.
  - Derive `Z ≈ D^(-1) Φ(Q) Φ(K)^T V`, where `D = Diag(Φ(Q)Φ(K)^T 1L)`.
  - Show how this approximation reduces complexity to linear in `L`.

### **2. Convolutional Neural Networks for Image Classification**
A convolutional neural network (CNN) is implemented to classify images from the **OCTMNIST** dataset. The implementation involves:

- **Baseline CNN Architecture**
  - Convolution layer (`8` channels, `3×3` kernel, stride `1`, padding to preserve image size).
  - ReLU activation.
  - Max pooling (`2×2` kernel, stride `2`).
  - Convolution layer (`16` channels, `3×3` kernel, stride `1`, no padding).
  - ReLU activation.
  - Max pooling (`2×2` kernel, stride `2`).
  - Fully connected layer (`320` output features).
  - ReLU activation.
  - Dropout (`p=0.7`).
  - Fully connected layers (`120` output features → Number of classes).
  - Output LogSoftmax layer.

- **Training and Hyperparameter Tuning**
  - Train the model for **15 epochs** using **SGD**.
  - Tune learning rate (`0.1`, `0.01`, `0.001`) using validation data.
  - Report the best configuration and plot **training loss** and **validation accuracy**.

- **Alternative CNN Architecture (Without Max Pooling)**
  - Modify convolution layers:
    - `conv1`: (`8` channels, `3×3` kernel, **stride `2`**, padding `1`).
    - `conv2`: (`16` channels, `3×3` kernel, **stride `2`**, no padding).
  - Implement a toggle (`no_maxpool`) to switch between models.
  - Train with the best hyperparameters from the baseline model.
  - Compare accuracy and computational performance.

- **Parameter Counting and Performance Justification**
  - Implement `get_number_trainable_params()` to compute the number of trainable parameters.
  - Compare performance between the two CNN architectures.

### **3. Automatic Speech Recognition (ASR)**
This section implements and evaluates different **decoder architectures** for automatic speech recognition (ASR) using the **LJ Speech Dataset**.

- **Recurrent-Based Decoder (LSTM)**
  - **Embedding layer** for tokenized text.
  - **Layer normalization** followed by an **LSTM**.
  - **Residual cross-attention** (query: LSTM output, key/value: encoder output).
  - **Normalization and linear classifier** (output vocabulary size = `35`).
  - Implement `_forward()` in `TextDecoderRecurrent`.
  - Train and compare **training loss**, **validation loss**, and **string similarity scores**.

- **Transformer-Based Decoder (Attention)**
  - **Embedding layer** + **position embeddings**.
  - **Residual self-attention** (query, key, value = token embeddings).
  - **Residual cross-attention** (query: attention output, key/value: encoder output).
  - **Normalization and linear classifier**.
  - Implement `_forward()` in `TextDecoderTransformer`.
  - Train and compare **training loss**, **validation loss**, and **string similarity scores**.

- **Comparison of LSTM vs. Transformer-Based Decoders**
  - Analyze how each architecture processes text (LSTM vs. attention).
  - Compare test results and discuss differences.

- **String Similarity Score Analysis**
  - Explain the meaning of different similarity scores.
  - Justify why each metric gives different values.

## Running the Code
- **Download Dataset:** `python download_octmnist.py`
- **Train Baseline CNN:** `python hw2-question-2-skeleton.py`
- **Train Alternative CNN:** `python hw2-question-2-skeleton.py --no_maxpool`
- **Train ASR with LSTM Decoder:** `run TextDecoderRecurrent`
- **Train ASR with Transformer Decoder:** `run TextDecoderTransformer`

## Summary
This project explores efficiency improvements in self-attention, CNN-based image classification, and encoder-decoder architectures for ASR. The implementations provide insights into computational trade-offs, hyperparameter tuning, and model performance comparisons.
