from handlers import *
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


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

# print(preds)
# print(Yte)
print("Accuracy :", accuracy_score(Yte[0], preds))


