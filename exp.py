from handlers import *
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score

trainData= create_data(100, 100)
testData = create_data(10, 100)

X_train, Y_train = trainData[0], trainData[0]
X_test, Y_test = testData[0], testData[1]

print(trainData[0].shape)
print(testData[0].shape)

model = dtc()

model.fit(X_train, Y_train)
preds = model.predict(X_test)

print(preds)
print(Y_test)
# print(accuracy_score(Y_test, preds))


