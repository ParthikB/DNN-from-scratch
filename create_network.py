import numpy as np
from handlers import sigmoid


def initialize_paramters(X):
    W1 = np.zeros((X.shape[0], 1))
    # W1 = np.random.randn(1, X.shape[0]) * 0.01
    b1 = 0
    parameters = {"W1" : W1, "b1" : b1}
    return parameters

def normalize(X):
    m = X.shape[1]
    mu = (1/m) * np.sum(X)
    sd = (1/m) * np.sum(X**2)
    X = (X - mu) / sd
    return X


def forward_prop(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]

    Z1 = np.dot(W1.T, X) + b1
    A1 = sigmoid(Z1)

    forward = {'Z1' : Z1, "A1" : A1}
    return forward


def compute_cost(forward, y):
    yhat = forward['A1']
    # print(yhat)
    m = y.shape[1]

    cost = - (1/m) * np.sum( y * np.log(yhat) + (1-y) * np.log(1-yhat) )

    return cost


def back_prop(forward, X, Y):
    A1 = forward["A1"]
    m = Y.shape[0]

    dZ1 = A1 - Y
    # print("dZ1        :",dZ1)
    dW1 = np.dot(X, dZ1.T) / m
    db1 = np.sum(dZ1) / m
    back = {'dW1' : dW1, "db1" : db1}
    return back


def update_parameters(parameters, back, learning_rate=0.009):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    dW1 = back['dW1']
    db1 = back['db1']

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    parameters = {"W1" : W1, "b1" : b1}
    return parameters

if __name__ == '__main__':
    print(sigmoid(0))