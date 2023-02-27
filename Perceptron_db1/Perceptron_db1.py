import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size + 1)

    def predict(self, inputs):
        inputs = np.insert(inputs, 0, 1)
        z = np.dot(inputs, self.weights)
        y_hat = 1 if z >= 0 else -1
        return y_hat

    def train(self, inputs, labels, learning_rate=0.1, epochs=100):
        for _ in range(epochs):
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = (labels[i] - prediction)/2
                self.weights += learning_rate * error * np.insert(inputs[i], 0, 1)

def train_model(X_train, y_train, perceptron):
    train_for_0 = y_train.copy()[y_test == 1 or y_test == 2]
    train_for_1 = y_train.copy()[y_test == 0 or y_test == 2]
    train_for_2 = y_train.copy()[y_test == 0 or y_test == 1]
    perceptron.train(X_train, train_for_0)
    perceptron.train(X_train, train_for_1)
    perceptron.train(X_train, train_for_2)

perceptron = Perceptron(2)

X_train = np.load("Perceptron_db1/X_train.npy")
X_train = [[np.sqrt(np.power(x[0], 2) + np.power(x[1], 2)), np.arctan(x[1]/x[0])] for x in X_train]
y_train = np.load("Perceptron_db1/y_train.npy")

train_model(X_train, y_train, perceptron)

X_test = np.load("Perceptron_db1/X_test.npy")
X_test = [[np.sqrt(np.power(x[0], 2) + np.power(x[1], 2)), np.arctan(x[1]/x[0])] for x in X_test]
y_test = np.load("Perceptron_db1/y_test.npy")
y_test[y_test == 0] = -1

predictions = []
correct = 0
plt.subplot(211)
for i in range(len(X_test)):
    predictions.append(perceptron.predict(X_test[i]))
    if predictions[i] == y_test[i]:
        correct += 1
    else:
        plt.scatter(X_test[i][0], X_test[i][1], c="black")

print(correct)
print(f"{100*correct/len(X_test)}%")


plt.grid()
for i in range(len(X_test)):
    plt.scatter(X_test[i][0], X_test[i][1], c= "blue" if y_test[i] == -1 else "red", marker= "^" if y_test[i] == -1 else "s")

plt.subplot(212)
plt.grid()
for i in range(len(X_train)):
    plt.scatter(X_train[i][0], X_train[i][1], c= "#00FFFF" if y_train[i] == -1 else "#ffb09c", marker= "^" if y_train[i] == -1 else "s")
plt.show()