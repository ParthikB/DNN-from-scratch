import numpy as np
from handlers import *
from create_network import *
import matplotlib.pyplot as plt


def normalize(X):
    m = X.shape[1]
    mu = np.mean(X)
    sigma = np.sum(X**2) / m
    return (X-mu)/sigma


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
    caches = []
    caches.append(None)
    for i in range(1, total_layers + 1):
        forward["Z" + str(i)] = np.dot(parameters["W" + str(i)], forward["A" + str(i-1)]) + parameters["b" + str(i)]
        if i == total_layers:
            forward["A" + str(i)] = sigmoid(forward["Z" + str(i)])
        else:
            forward["A" + str(i)] = relu(forward["Z" + str(i)])
        caches.append(forward["Z" + str(i)])

    Yhat = forward["A" + str(i)]
    return forward, Yhat, caches


def compute_cost(y, yhat):
    m = y.shape[1]
    cost = - np.sum(y * np.log(yhat) + (1-y) * (np.log(1-yhat))) / m
    return cost


def back_prop(Yhat, Y, forward, parameters, caches):
    grads = {}
    total_layers = parameters.__len__() // 2
    grads["dZ" + str(total_layers)] = Yhat - Y
    m = Y.shape[1]
    for i in range(1, total_layers + 1)[::-1]:
        grads["dW" + str(i)]   = np.dot(grads['dZ' + str(i)], forward["A" + str(i-1)].T) / m
        grads["db" + str(i)]   = np.sum(grads['dZ' + str(i)], axis=1, keepdims=True) / m
        grads["dA" + str(i-1)] = np.dot(parameters["W" + str(i)].T, grads["dZ" + str(i)])
        if i == total_layers:
            grads["dZ" + str(i-1)] = grads["dA" + str(i-1)] * sigmoid(forward["Z" + str(i-1)], derivative=True)
        if i != 1:
            # print(f"dA{str(i-1)} :", grads["dA" + str(i-1)], "shape :", grads["dA" + str(i-1)].shape)
            # print(f"dz{str(i-1)} :", caches[i-1], "shape :", caches[i-1].shape)
            grads["dZ" + str(i-1)] = grads["dA" + str(i-1)] * relu_backward(grads["dA" + str(i-1)], caches[i-1])

    return grads


def update_parameters(parameters, cache, learning_rate=0.05):
    # print("old", parameters)
    total_layers = parameters.__len__() // 2
    for i in range(1, total_layers+1):
        # print(cache)
        # print(learning_rate * cache["dW" + str(i)])
        parameters["W" + str(i)] -= learning_rate * cache["dW" + str(i)]
        parameters["b" + str(i)] -= learning_rate * cache["db" + str(i)]
        # print("---------------------------")
    # print("updated",parameters)
    # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return parameters


def accuracy_score(Yhat, Y):
    Yhat = np.where(Yhat < 0.5, 0, 1)
    accuracy = 100 - np.mean(np.abs(Yhat-Y) * 100)
    return accuracy


def predict(X, Y, parameters):
    forward, A2, caches = forward_propagation(parameters, X)
    A2 = np.where(A2 < 0.5, 0, 1)
    accuracy = accuracy_score(A2, Y)
    return accuracy, A2


def plot_cost_log(cost_log):
    plt.plot(cost_log)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Function")
    plt.show()