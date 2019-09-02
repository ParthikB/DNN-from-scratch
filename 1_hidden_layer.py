import numpy as np
from handlers import *
from create_network import *
import matplotlib.pyplot as plt

data = np.load('train.npy', allow_pickle=True)
# data = create_data(200, 100)
X, Y = data[0], data[1]

n_x, n_h, n_y = layers_size(X, Y)    # 2 | 4 | 1

parameters = initialize_random_parameters(n_x, n_h, n_y)

cost_log = []
for i in range(200):

    cache, A2 = forward_propagation(parameters, X)

    cost = compute_cost(A2, Y)
    cost_log.append(cost)

    back = back_prop(X, Y, cache, parameters)

    parameters = update_parameters(parameters, back)

plt.plot(cost_log)
plt.show()