# Developed by TheLateV777
# 1 input, 1 hidden layer: 2 neurons, 1 output neuron
# Intended to check age for application to army
# Uses Revised Training Procedure

from basic_definitions import step,relu,relu_derivative
import random

class NeuralNetwork:
    def __init__(self):
        self.w1 = random.uniform(-1,1)
        self.b1 = random.uniform(-1,1)

        self.w2 = random.uniform(-1,1)
        self.b2 = random.uniform(-1,1)

        self.w3 = random.uniform(-1,1)
        self.w4 = random.uniform(-1,1)
        self.b3 = random.uniform(-1,1)

        print(self.w1)
        print(self.w2)
        print(self.w3)
        print(self.w4)
        print(self.b1)
        print(self.b2)
        print(self.b3)
        print('-------')

    def forward(self,x):
        self.z1 = x * self.w1 + self.b1
        self.z2 = x * self.w2 + self.b2
        self.h1 = relu(self.z1)
        self.h2 = relu(self.z2)
        self.y_hat = self.h1 * self.w3 + self.h2 * self.w4 + self.b3
        return self.y_hat

    def backward(self,datax,epochs=200000,lr=0.01):
        for epoch in range(epochs):
            total_loss = 0
            for x,y in datax.items():
                x = x/100
                y_hat = self.forward(x)
                loss = (y_hat-y)**2
                total_loss += loss

                l_y_hat = 2 * (y_hat-y)

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
                print(f'epoch: {epoch}, loss: {total_loss/len(datax):.6f}')

    def predict(self,x):
        x = x/100
        y_hat = self.forward(x)
        prediction = step(y_hat)
        return prediction, y_hat

data = {1:0,
        2:0,
        7:0,
        -10:0,
        -100:0,
        14:0,
        17:0,
        18:1,
        19:1,
        21:1,
        22:1,
        30:1,
        34:1,
        39:1,
        40:0,
        45:0,
        46:0,
        67:0}

nn = NeuralNetwork()
nn.backward(data)

ages = [12,34,67,1,22] + list(data.keys())

print(nn.w1)
print(nn.w2)
print(nn.w3)
print(nn.w4)
print(nn.b1)
print(nn.b2)
print(nn.b3)

for age in range(-1000,1000):
    print(f'Age: {age}, Allowed: {nn.predict(age)[0]}, Margin: {nn.predict(age)[1]}')


















