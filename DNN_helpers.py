import numpy as np
from handlers import *
from create_network import *
import matplotlib.pyplot as plt


def initialize_random_parameters(layer_dims, X):
    parameters = {}
    parameters["W1"] = np.random.randn(layer_dims[0], X.shape[0]) * 0.01
    parameters["b1"] = np.zeros((layer_dims[0], 1))
    for i in range(1, len(layer_dims)):
        parameters["W" + str(i+1)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01              # i x i-1
        parameters["b" + str(i+1)] = np.zeros((layer_dims[i], 1))

    return parameters


def forward_propagation(parameters, X):
    forward = {}
    forward["A0"] = X
    total_layers = parameters.__len__() // 2
    for i in range(1, total_layers + 1):
        forward["Z" + str(i)] = np.dot(parameters["W" + str(i)], forward["A" + str(i-1)]) + parameters["b" + str(i)]
        if i == total_layers:
            forward["A" + str(i)] = sigmoid(forward["Z" + str(i)])
        else:
            forward["A" + str(i)] = relu(forward["Z" + str(i)])

    Yhat = forward["A" + str(i)]
    return forward, Yhat


def compute_cost(y, yhat):
    m = y.shape[1]
    cost = - np.sum(y * np.log(yhat) + (1-y) * (np.log(1-yhat))) / m
    return cost


def back_prop(Yhat, Y, forward, parameters):
    cache = {}
    cache["dZ4"] = Yhat - Y
    m = Y.shape[1]
    total_layers = parameters.__len__() // 2
    for i in range(1, total_layers + 1)[::-1]:
        cache["dW" + str(i)] = np.dot(cache['dZ' + str(i)], forward["A" + str(i-1)].T) / m
        cache["db" + str(i)] = np.sum(cache['dZ' + str(i)], axis=1, keepdims=True) / m
        if i == total_layers:
            cache["dZ" + str(i-1)] = np.dot(parameters["W" + str(i)].T, cache["dZ" + str(i)])\
                                     * sigmoid(forward["Z" + str(i-1)], derivative=True)
        if i != 1:
            cache["dZ" + str(i - 1)]   = np.dot(parameters["W" + str(i)].T, cache["dZ" + str(i)]) \
                                     * relu(forward["Z" + str(i-1)], derivative=True)

    return cache


def plot_cost_log(cost_log):
    plt.plot(cost_log)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Function")
    plt.show()


def update_parameters(parameters, cache, learning_rate=0.05):
    total_layers = parameters.__len__() // 2
    for i in range(1, total_layers+1):
        parameters["W" + str(i)] -= learning_rate * cache["dW" + str(i)]
        parameters["b" + str(i)] -= learning_rate * cache["db" + str(i)]

    return parameters

