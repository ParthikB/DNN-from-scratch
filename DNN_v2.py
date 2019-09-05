import matplotlib.pyplot as plt
from DNN_helpers_v2 import *
from handlers import *

train_data = create_data(10000, 100)
test_data = create_data(100, 100)
X, Y, Xte, Yte = train_data[0], train_data[1], test_data[0], test_data[1]
X = normalize(X)

LAYER_DIMS = [4, 4, 2, 1]
cost_log = []
accuracy_log = []
test_accuracy_log = []
epoch_log = []

parameters = initialize_random_parameters(LAYER_DIMS, X)

for epoch in range(1000):
    print(epoch)
    AL, caches = L_model_forward(X, parameters)

    cost = compute_cost(Y, AL)
    cost_log.append(cost)

    grads = L_model_backward(AL, Y, caches)

    parameters = update_parameters(parameters, grads)


plot_cost_log(cost_log)