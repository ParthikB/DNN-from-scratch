from handlers import *
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from create_network import *


def plot_decision_boundary(model ,X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.5
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()].reshape(2, -1))
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap="cool")
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0], cmap=plt.cm.Spectral)


train_data = np.load("train.npy", allow_pickle=True)
test_data = np.load("train.npy", allow_pickle=True)
train_data = create_data(10000, 100)
test_data = create_data(200, 100)

X = train_data[0]                                               # X = 2*200
Y = train_data[1]                                               # Y = 1*200
Xte = test_data[0]
Yte = test_data[1]

print(train_data[0].shape)
print(test_data[0].shape)

model = dtc()

model.fit(X[0].reshape(-1, 1), Y[0])
preds = model.predict(Xte[0].reshape(-1, 1))

plot_decision_boundary(model, X, Y)

# print(preds)
# print(Yte)
print("Accuracy :", accuracy_score(Yte[0], preds))


