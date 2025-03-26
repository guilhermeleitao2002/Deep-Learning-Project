# Deep Learning - Homework 1 & 2

## Overview

These projects focus on implementing and analyzing deep learning models for image classification, self-attention mechanisms, and automatic speech recognition (ASR). The tasks cover fundamental and advanced techniques, including linear classifiers, multi-layer perceptrons (MLPs), convolutional neural networks (CNNs), transformers, and encoder-decoder architectures.

---

## **Homework 1: Image Classification with Linear Models and Neural Networks**
This project explores different machine learning models for classifying medical images from the **OCTMNIST** dataset.

### **1. Linear Classifiers (No ML Libraries)**
- **Perceptron**: Implemented and trained using basic linear algebra.
- **Logistic Regression**: Trained with stochastic gradient descent and tested with different learning rates (`0.01`, `0.001`).
- **Performance Evaluation**: Compared training, validation, and test accuracies.

### **2. Multi-Layer Perceptron (MLP)**
- Implemented an MLP with:
  - 200 hidden units.
  - **ReLU activation** and **cross-entropy loss**.
  - **SGD optimizer** with a learning rate of `0.001`.
- Trained for 20 epochs and evaluated performance.

### **3. Neural Network Implementation with Autodiff**
- **Logistic Regression** and **MLP** implemented using **PyTorch**.
- **Hyperparameter tuning**: Compared batch sizes, learning rates, and regularization techniques.
- **Dropout and L2 Regularization**: Analyzed their impact on overfitting.

### **4. Boolean Function Computation with Neural Networks**
- Demonstrated that a **single-layer perceptron** cannot compute certain Boolean functions.
- Designed a **multi-layer perceptron** to compute the function with **integer weights and biases**.

---

## **Homework 2: Efficient Attention, CNNs, and ASR**
This project extends Homework 1 by exploring self-attention mechanisms, CNNs for image classification, and ASR models.

### **1. Efficient Self-Attention Approximation**
- **Analyzed self-attention complexity** in transformers.
- **Used McLaurin series expansion** to approximate the softmax function.
- **Derived a low-rank attention approximation** to reduce complexity from quadratic to linear.

### **2. Convolutional Neural Networks (CNNs) for Image Classification**
- Implemented a **CNN for OCTMNIST classification**:
  - Convolutional layers, max pooling, dropout, fully connected layers.
  - Trained for **15 epochs** with **SGD**.
- **Alternative CNN (No Max Pooling)**:
  - Modified convolutional layers with different strides.
  - Compared accuracy and efficiency.
- **Parameter Counting & Performance Justification**:
  - Implemented `get_number_trainable_params()` to analyze network complexity.

### **3. Automatic Speech Recognition (ASR)**
- Implemented **decoder architectures** for ASR using **LJ Speech Dataset**.
- **Recurrent-based Decoder (LSTM)**:
  - Embedded token sequences and processed them through an LSTM and residual attention layers.
- **Transformer-based Decoder**:
  - Used **self-attention** and **cross-attention** to process text sequences.
- **Comparative Analysis**:
  - Evaluated **training loss, validation loss, and string similarity scores**.
  - Compared **LSTM vs. Transformer decoders** in transcription accuracy.

---

## **Running the Code**
- **Download Dataset:** `python download_octmnist.py`
- **Train Linear Classifiers & MLP (Homework 1):** `python hw1-q1.py`
- **Train CNN Models (Homework 2):** `python hw2-question-2-skeleton.py`
- **Train ASR Models (Homework 2):** `run TextDecoderRecurrent` / `run TextDecoderTransformer`

## **Summary**
These projects provide hands-on experience with different deep learning models, from basic classifiers to advanced architectures like CNNs, transformers, and ASR decoders. The implementations emphasize mathematical derivations, model optimization, and performance analysis.
