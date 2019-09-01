import numpy as np
import cv2
from handlers import *
from create_network import *
import pandas as pd
import matplotlib.pyplot as plt


# data = np.load('train.npy', allow_pickle=True)
data = create_data(200, 100)
X, Y = data[0], data[1]
parameters = initialize_paramters(X)

X = normalize(X)

cost_log = []
for i in range(1000):
    forward = forward_prop(X, parameters)
    # print("Parameters :", parameters)
    # print("Z1         :",forward["Z1"])

    cost = compute_cost(forward, Y)
    # print("Cost       :", cost)
    # print(f"Cost of iteration {i+1} : {cost}")
    cost_log.append(cost)



    back = back_prop(forward, X, Y)
    # print("Gradients  :", back)

    parameters = update_parameters(parameters, back)
    # print("Update     :", parameters)
    # print()


A1 = forward['A1']
A1 = np.where(A1 < 0.5, 0, 1)

# print(Y)
# print(A1)
print ("Accuracy :", 100 - np.mean(np.abs(A1-Y))*100)

# print(cost_log)
plt.plot(cost_log)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()