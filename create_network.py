from handlers import *


def layers_size(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y


def initialize_random_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


def forward_propagation(parameters, X):
    W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return cache, A2


def compute_cost(A2, Y):
    m = Y.shape[1]
    cost = - np.sum(Y * np.log(A2) + (1-Y) * np.log(1-A2)) / m
    return cost


def back_prop(X, Y, cache, parameters):
    A1, A2 = cache["A1"], cache["A2"]
    W1, W2 = parameters["W1"], parameters["W2"]
    m = Y.shape[1]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m    # 1 x 4 >> A1 = 4 x 200 >> dZ2 = 1 x 200
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))         # 4 x 200 >> w1 = 4 x 2 // X = 2 x 200
    dW1 = np.dot(dZ1, X.T) / m      # 4 x 1 >> dZ1 = 4 x 200 // X = 2 x 200
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    back = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return back


def update_parameters(parameters, back, learning_rate=0.005):
    parameters["W1"] -= learning_rate * back["dW1"]
    parameters["b1"] -= learning_rate * back["db1"]
    parameters["W2"] -= learning_rate * back["dW2"]
    parameters["b2"] -= learning_rate * back["db2"]

    return parameters


def predict(X, parameters):
    cache, A2 = forward_propagation(parameters, X)
    return A2