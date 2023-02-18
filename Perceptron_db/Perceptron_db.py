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

    def train(self, inputs, labels, learning_rate=0.5, epochs=10000):
        for _ in range(epochs):
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = (labels[i] - prediction)/2
                self.weights += learning_rate * error * np.insert(inputs[i], 0, 1)

perceptron = Perceptron(2)

X_train = np.load("Perceptron_db/X_train.npy")
y_train = np.load("Perceptron_db/y_train.npy")

perceptron.train(X_train, y_train)

X_test = np.load("Perceptron_db/X_test.npy")
y_test = np.load("Perceptron_db/y_test.npy")

predictions = []
correct = 0
for i in range(len(X_test)):
    x = perceptron.predict(X_test[i])
    predictions.append(x if x == 1 else 0)
    if predictions[i] == y_test[i]:
        correct += 1

print(correct)
print(f"{correct/len(X_test)}%")

plt.grid()
for i in range(len(X_train)):
    plt.scatter(X_train[i][0], X_train[i][1], c= "#00FFFF" if y_train[i] == 0 else "#ffb09c", marker= "^" if y_train[i] == 0 else "s")
plt.show()