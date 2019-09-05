import numpy as np
import matplotlib.pyplot as plt
from handlers import *
from create_network import *
from one_hidden_layer import *
from collections import Counter

def plot_decision_boundary(parameters,X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.5
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = predict(np.c_[xx.ravel(), yy.ravel()].reshape(2, -1) ,parameters)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap="cool")
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0], cmap=plt.cm.Spectral)


test_data = create_data(100, 100)
Xte, Yte = test_data[0], test_data[1]
# parameters = np.load("83.0%-parameters.npy", allow_pickle=True)

Yhat, cost_log, parameters = nn_model(Xte, Yte, parameters)

# train_data = create_data(1000, 100)
# X, Y = train_data[0], train_data[1]

# x_min, x_max = Xte[0, :].min() - 1, Xte[0, :].max() + 1
# # y_min, y_max = Xte[1, :].min() - 1, Xte[1, :].max() + 1
# # h = 1
# # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# # Z = predict(np.c_[xx.ravel(), yy.ravel()].reshape(2, -1), parameters)
# # Z = Z.reshape(xx.shape)
# # plt.contourf(xx, yy, Z, cmap="cool")
# # plt.show()

plot_decision_boundary(parameters, Xte, Yte)