import numpy as np
from create_network import initialize_paramters, normalize
from handlers import *
import matplotlib.pyplot as plt

train_data = np.load("train.npy", allow_pickle=True)
test_data = np.load("train.npy", allow_pickle=True)

train_data = create_data(500, 100)
X = train_data[0]                                               # X = 2*200
Y = train_data[1]                                               # Y = 1*200
Xte = test_data[0]
Yte = test_data[1]

X = normalize(X)
plot_data(train_data)

parameters = initialize_paramters(X)

LEARNING_RATE = 0.005
COST_LOG = []

for i in range(5000):
    print("Iteration :", i)
    W, b = parameters["W1"], parameters["b1"]                   # W = 2*1   b = 1*1
    m = X.shape[1]                                              # m = 200

    # forward propagation
    Z = np.dot(W.T, X) + b                                      # Z = 1*200
    A = sigmoid(Z)                                              # A = 1*200

    # Compute cost
    cost = - np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) / m
    COST_LOG.append(cost)

    # Back propagation
    dZ = A - Y                                                  # dZ = 1*200
    dW = np.dot(X, dZ.T) / m                                    # dW = 2*1
    db = np.sum(dZ) / m

    # Updating parameters
    parameters["W1"] -= LEARNING_RATE * dW
    parameters["b1"] -= LEARNING_RATE * db



A = np.where(A<0.5, 0, 1)

train_accuracy = 100 - np.mean(np.abs(A - Y)) * 100
print(f"Accuracy : {train_accuracy} %")

plt.plot(COST_LOG)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title(f"Accuracy : {train_accuracy} %")
plt.show()

