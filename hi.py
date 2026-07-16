from basic_definitions import step, relu, relu_derivative
import random

class NeuralNetwork:
    def __init__(self):
        self.w1 = random.uniform(-1, 1)
        self.b1 = random.uniform(-1, 1)

        self.w2 = random.uniform(-1, 1)
        self.b2 = random.uniform(-1, 1)

        self.w3 = random.uniform(-1, 1)
        self.w4 = random.uniform(-1, 1)
        self.b3 = random.uniform(-1, 1)

    def forward(self, x):
        self.z1 = x * self.w1 + self.b1
        self.z2 = x * self.w2 + self.b2
        self.h1 = relu(self.z1)
        self.h2 = relu(self.z2)
        self.y_hat = self.h1 * self.w3 + self.h2 * self.w4 + self.b3
        return self.y_hat

    def backward(self, data, epochs=20000, lr=0.0001):
        for epoch in range(epochs):
            total_loss = 0

            for x, y in data.items():
                x = x / 100.0
                y_hat = self.forward(x)
                loss = (y_hat - y) ** 2
                total_loss += loss

                l_y_hat = 2 * (y_hat - y)

                l_w3 = l_y_hat * self.h1
                l_w4 = l_y_hat * self.h2
                l_b3 = l_y_hat

                l_h1 = l_y_hat * self.w3
                l_h2 = l_y_hat * self.w4

                l_z1 = l_h1 * relu_derivative(self.z1)
                l_z2 = l_h2 * relu_derivative(self.z2)

                l_w1 = l_z1 * x
                l_b1 = l_z1

                l_w2 = l_z2 * x
                l_b2 = l_z2

                self.w1 -= lr * l_w1
                self.b1 -= lr * l_b1

                self.w2 -= lr * l_w2
                self.b2 -= lr * l_b2

                self.w3 -= lr * l_w3
                self.w4 -= lr * l_w4
                self.b3 -= lr * l_b3

            if epoch % 1000 == 0:
                print(f"epoch: {epoch}, loss: {total_loss / len(data):.6f}")

    def predict(self, x):
        x = x / 100.0
        y_hat = self.forward(x)
        prediction = step(y_hat)
        return prediction, y_hat

data = {
    17: 0,
    18: 1,
    19: 1,
    39: 1,
    40: 0
}

nn = NeuralNetwork()
nn.backward(data)

ages = [12, 34, 67, 1, 22]

for age in ages:
    pred, margin = nn.predict(age)
    print(f"Age: {age}, Allowed: {pred}, Margin: {margin}")