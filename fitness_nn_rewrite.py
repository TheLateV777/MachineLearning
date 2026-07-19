import math
import random


def relu(x):
    return x if x > 0 else 0.0


def relu_derivative(x):
    return 1.0 if x > 0 else 0.0


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class NeuralNetwork:
    def __init__(self, seed=7):
        random.seed(seed)
        self.W1 = [[random.uniform(-0.5, 0.5) for _ in range(4)] for _ in range(4)]
        self.b1 = [random.uniform(-0.5, 0.5) for _ in range(4)]
        self.W2 = [random.uniform(-0.5, 0.5) for _ in range(4)]
        self.b2 = random.uniform(-0.5, 0.5)

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(4):
            z = sum(self.W1[i][j] * x[j] for j in range(4)) + self.b1[i]
            self.z1.append(z)
            self.h.append(relu(z))
        self.z2 = sum(self.W2[i] * self.h[i] for i in range(4)) + self.b2
        self.y_hat = sigmoid(self.z2)
        return self.y_hat

    def compute_loss(self, data):
        eps = 1e-9
        total = 0.0
        for x, y in data:
            p = self.forward(x)
            total += -(y * math.log(p + eps) + (1 - y) * math.log(1 - p + eps))
        return total / len(data)

    def train(self, data, epochs=5000, lr=0.05):
        for epoch in range(epochs):
            random.shuffle(data)
            for x, y in data:
                p = self.forward(x)

                dz2 = p - y
                dW2 = [dz2 * h_i for h_i in self.h]
                db2 = dz2

                dz1 = []
                for i in range(4):
                    dz1_i = dz2 * self.W2[i] * relu_derivative(self.z1[i])
                    dz1.append(dz1_i)

                dW1 = []
                for i in range(4):
                    row = []
                    for j in range(4):
                        row.append(dz1[i] * x[j])
                    dW1.append(row)
                db1 = dz1

                for i in range(4):
                    self.W2[i] -= lr * dW2[i]
                self.b2 -= lr * db2

                for i in range(4):
                    for j in range(4):
                        self.W1[i][j] -= lr * dW1[i][j]
                    self.b1[i] -= lr * db1[i]

            if epoch % 1 == 0:
                loss = self.compute_loss(data)
                print(f"epoch={epoch:4d} loss={loss:.4f}")

    def predict(self, x):
        p = self.forward(x)
        return (1 if p >= 0.5 else 0), p


def normalize(push, pull, crunches, laps):
    return [push / 20.0, pull / 20.0, crunches / 20.0, laps / 20.0]


def make_dataset(n=300, seed=11):
    random.seed(seed)
    data = []
    for _ in range(n):
        push = random.randint(0, 20)
        pull = random.randint(0, 20)
        crunches = random.randint(0, 20)
        laps = random.randint(0, 20)
        score = push + pull + crunches + 10 * laps
        y = 1 if score >= 100 else 0
        data.append((normalize(push, pull, crunches, laps), y))
    return data


def evaluate(model, data):
    correct = 0
    for x, y in data:
        pred, _ = model.predict(x)
        if pred == y:
            correct += 1
    return correct / len(data)


if __name__ == "__main__":
    train_data = make_dataset(400, seed=21)
    test_data = make_dataset(120, seed=22)

    nn = NeuralNetwork(seed=7)
    nn.train(train_data, epochs=100, lr=0.05)

    print("\nAccuracy")
    print(f"train: {evaluate(nn, train_data):.3f}")
    print(f"test : {evaluate(nn, test_data):.3f}")

    print("\nTryouts")
    tryouts = [
        (1, 1, 1, 1),
        (3, 4, 5, 1),
        (10, 10, 10, 14),
        (20, 20, 20, 20),
    ]
    for t in tryouts:
        pred, prob = nn.predict(normalize(*t))
        print(f"input={t} pred={pred} prob={prob:.3f}")

    print(nn.W1,nn.W2,nn.b1,nn.b2)