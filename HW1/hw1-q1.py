#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a

        y_hat_i = self.predict(x_i.reshape(1, -1))[0]

        if y_hat_i != y_i:
            self.W[y_i, :] += x_i
            self.W[y_hat_i, :] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        
        scores = np.dot(self.W, x_i)

        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / np.sum(exp_scores)
        
        target = np.zeros(self.W.shape[0])
        target[y_i] = 1
        
        gradient = np.outer(probabilities - target, x_i)
        
        self.W -= learning_rate * gradient


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size = 200):
        # Initialize an MLP with a single hidden layer.
        self.w1 = np.random.normal(loc = 0.1, scale = 0.1 ** 2, size = (hidden_size, n_features))
        self.w2 = np.random.normal(loc = 0.1, scale = 0.1 ** 2, size = (n_classes, hidden_size))

        self.b1 = np.zeros((hidden_size, 1))
        self.b2 = np.zeros((n_classes, 1))

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        z_1 = np.dot(self.w1, X.T) + self.b1

        x_1 = np.maximum(0, z_1)

        z_2 = np.dot(self.w2, x_1) + self.b2
        scores = np.exp(z_2 - np.max(z_2))/np.exp(z_2 - np.max(z_2)).sum()

        predicted_labels = scores.argmax(axis = 0)
        return predicted_labels

    def update_weight(self, x_i, y_i, learning_rate = 0.001):

        x_i = np.reshape(x_i, (-1, 1))

        z_1 = np.dot(self.w1, x_i) + self.b1
        x_1 = np.maximum(0, z_1)

        z_2 = np.dot(self.w2, x_1) + self.b2
        x_2 = np.exp(z_2 - np.max(z_2))/np.exp(z_2 - np.max(z_2)).sum()

        e_y = np.zeros(x_2.shape)
        e_y[y_i] = 1

        x_1_derivative = np.array(list(map(lambda p: list(map(lambda q: 1 if q >= 0 else 0, p)), z_1)))

        delta_2 = x_2 - e_y
        w2_up = np.dot(delta_2, x_1.T)

        delta_1 = np.dot(self.w2.T, delta_2) * x_1_derivative
        w1_up = np.dot(delta_1, x_i.T)

        self.w2 -= learning_rate * w2_up
        self.b2 -= learning_rate * delta_2
        self.w1 -= learning_rate * w1_up
        self.b1 -= learning_rate * delta_1

        return np.sum(e_y * np.log(x_2))


    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """

        loss = 0
        for x_i, y_i in zip(X, y):
            loss -= self.update_weight(x_i, y_i, learning_rate)

        return loss/len(X)


def plot(epochs, train_accs, val_accs, name='plot'):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.savefig(f'{name}.png')

def plot_loss(epochs, loss, name='plot_loss'):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.savefig(f'{name}.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-output', default='plot', type=str,
                        help="""Name of the file where the accuracy plot should be
                        saved without the extension. The loss plot will be saved to the same file name
                        with '_loss' concatenated to the end.""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, opt.output)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, opt.output + '_loss')


if __name__ == '__main__':
    main()
