from DNN_helpers import *
import matplotlib.pyplot as plt

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

for epoch in range(5):
    if epoch % 100 == 0:
        print("epoch :", epoch)

    forward, Yhat, caches = forward_propagation(parameters, X)

    cost = compute_cost(Y, Yhat)
    cost_log.append(cost)

    grads = back_prop(Yhat, Y, forward, parameters, caches)

    parameters = update_parameters(parameters, grads, learning_rate=0.09)


    accuracy = accuracy_score(Yhat, Y)
    test_accuracy, Yte_hat = predict(Xte, Yte, parameters)
    accuracy_log.append(accuracy)
    test_accuracy_log.append(test_accuracy)
    epoch_log.append(epoch+1)


print("Train accuracy :", accuracy)
print("Test accuracy  :", test_accuracy)

plt.plot(epoch_log, accuracy_log, "r", label="Train")
plt.plot(epoch_log, test_accuracy_log, label="Test")
plt.legend()
plt.ylim(0, 100)
plt.xlabel("epoch")
plt.ylabel("Accuracy")
plt.show()

# plot_cost_log(cost_log)
