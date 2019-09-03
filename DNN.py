from DNN_helpers import *
import matplotlib.pyplot as plt

train_data = create_data(500, 100)
test_data = create_data(100, 100)
X, Y, Xte, Yte = train_data[0], train_data[1], test_data[0], test_data[1]

LAYER_DIMS = [3, 4, 2, 1]
cost_log = []

parameters = initialize_random_parameters(LAYER_DIMS, X)

for epoch in range(500):
    print("epoch :", epoch)

    forward, Yhat = forward_propagation(parameters, X)

    cost = compute_cost(Y, Yhat)
    cost_log.append(cost)

    cache = back_prop(Yhat, Y, forward, parameters)

    parameters = update_parameters(parameters, cache)

plot_cost_log(cost_log)