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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def accuracy_score(Yhat, Y):
    return 100 - np.mean(np.abs(Yhat - Y)) * 100


def plot_data(data):
    X = data[0]
    Y = data[1]
    sns.scatterplot(X[0], X[1], hue=Y[0])
    plt.show()

# def best_fit_line():
#     for i in range(1, 101):
#         y =

if __name__ == "__main__":
    X, Y = create_data(50, 100)
    np.save("test.npy", [X, Y])
    print([X.shape, Y.shape])
    print(X)
    print()
    print(Y)