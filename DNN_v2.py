import matplotlib.pyplot as plt
from DNN_helpers_v2 import *
from handlers import *

train_data = create_data(10000, 100)
test_data = create_data(100, 100)
X, Y, Xte, Yte = train_data[0], train_data[1], test_data[0], test_data[1]
# X = normalize(X)

LAYER_DIMS = [16, 16, 1]
LEARNING_RATE = 0.05
cost_log = []
accuracy_log = []
test_accuracy_log = []
epoch_log = []

parameters = initialize_random_parameters(LAYER_DIMS, X)

for epoch in range(2500):
    if epoch % 100 == 0:
        print(epoch)

    AL, caches = L_model_forward(X, parameters)

    cost = compute_cost(Y, AL)
    cost_log.append(cost)

    grads = L_model_backward(AL, Y, caches)

    parameters = update_parameters(parameters, grads, LEARNING_RATE)

    acc = accuracy_score(AL, Y)
    accuracy_log.append(acc)
    epoch_log.append(epoch)
# test_acc, AL_te = predict(Xte, Yte, parameters)

print("Train Accuracy :", accuracy_score(AL, Y))
# print("Test Accuracy :", test_acc)
plot_cost_log(cost_log, LEARNING_RATE)