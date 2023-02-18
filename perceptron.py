import numpy as np


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

perceptron = Perceptron(2)

a = np.load("X_train.npy")
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([-1, -1, -1, 1])

perceptron.train(inputs, labels)

print(perceptron.predict([0, 0]))
print(perceptron.predict([0, 1]))
print(perceptron.predict([1, 0]))
print(perceptron.predict([1, 1]))
