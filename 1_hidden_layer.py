import numpy as np
from handlers import *
from create_network import *
import matplotlib.pyplot as plt

# data = np.load('train.npy', allow_pickle=True)
train_data = create_data(2000, 100)
test_data = create_data(50, 100)

Xte, Yte = test_data[0], test_data[1]
X, Y = train_data[0], train_data[1]

n_x, n_h, n_y = layers_size(X, Y)    # 2 | 4 | 1

parameters = initialize_random_parameters(n_x, n_h, n_y)


def nn_model(X, Y, parameters):
    cost_log = []
    for i in range(4500):

        cache, A2 = forward_propagation(parameters, X)

        cost = compute_cost(A2, Y)
        cost_log.append(cost)

        back = back_prop(X, Y, cache, parameters)

        parameters = update_parameters(parameters, back, learning_rate=0.002)

    A2 = np.where(A2 < 0.5, 0, 1)

    return A2, cost_log, parameters


Yhat, cost_log, parameters = nn_model(X, Y, parameters)

Yte_hat = predict(Xte, parameters)

print("Train accuracy :", accuracy_score(Yhat, Y))
print("Test accuracy  :", accuracy_score(Yte_hat, Yte))


plt.plot(cost_log)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title(f"Minimum cost : {min(cost_log)} at iteration {cost_log.index(min(cost_log))}")
plt.show()