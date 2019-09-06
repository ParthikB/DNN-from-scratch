import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_data(total_samples, range_of_data):
    X1, X2, Y = [], [], []

    for datapoints in range(total_samples):
        x2 = np.random.randint(1, range_of_data + 1)
        x1 = np.random.randint(1, range_of_data + 1)

        if x1 < range_of_data / 2:
            label = 0
        else:
            label = 1
        X1.append(x1)
        X2.append(x2)
        X = np.array([X1, X2])
        Y.append(label)

    return np.array(X).reshape(2, -1), np.array(Y).reshape(1, -1)


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def accuracy_score(Yhat, Y):
    return 100 - np.mean(np.abs(Yhat - Y)) * 100


def plot_data(data):
    X = data[0]
    Y = data[1]
    sns.scatterplot(X[0], X[1], hue=Y[0])
    plt.show()


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)


if __name__ == "__main__":
    X, Y = create_data(50, 100)
    np.save("test.npy", [X, Y])
    print([X.shape, Y.shape])
    print(X)
    print()
    print(Y)