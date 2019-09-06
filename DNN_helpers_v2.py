import numpy as np
from handlers import *


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
        parameters["W" + str(i+1)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
        parameters["b" + str(i+1)] = np.zeros((layer_dims[i], 1))

    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = {"A" : A,
             "W" : W,
             "b" : b}
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for i in range(1, L):
        A, cache = linear_activation_forward(A, parameters["W" + str(i)], parameters["b" + str(i)], 'relu')
        caches.append(cache)

    A, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], 'sigmoid')
    caches.append(cache)

    return A, caches


def linear_backward(dZ, cache):
    A_prev = cache['A']
    W = cache['W']
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    L = len(caches)
    grads = {}
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
    for l in range(L-1)[::-1]:
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l+1)], grads["db" + str(l+1)] = linear_activation_backward(grads["dA" + str(l+1)], current_cache, 'relu')

    return grads


def compute_cost(y, yhat):
    m = y.shape[1]
    cost = - np.sum(y * np.log(yhat) + (1-y) * (np.log(1-yhat))) / m
    return cost


def update_parameters(parameters, grads, learning_rate=0.05):
    L = parameters.__len__() // 2
    for i in range(1, L+1):
        parameters["W" + str(i)] -= learning_rate * grads["dW" + str(i)]
        parameters["b" + str(i)] -= learning_rate * grads["db" + str(i)]

    return parameters


def accuracy_score(Yhat, Y):
    Yhat = np.where(Yhat < 0.5, 0, 1)
    accuracy = 100 - np.mean(np.abs(Yhat-Y) * 100)
    return accuracy


def predict(X, Y, parameters):
    AL, caches = L_model_forward(X, parameters)
    AL = np.where(AL < 0.5, 0, 1)
    accuracy = accuracy_score(AL, Y)
    return accuracy, AL


def plot_cost_log(cost_log, learning_rate):
    plt.plot(cost_log)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(f"Cost Function | Learning Rate : {learning_rate}")
    plt.show()