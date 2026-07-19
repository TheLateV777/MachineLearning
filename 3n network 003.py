# Developed by TheLateV777
# Intended to test out new degree n polynomial graph as see graph
# Failed

from basic_definitions import step,relu,relu_derivative
import random
import tkinter as tk

root = tk.Tk()
root.title("Neural Network Graph")

width = 900
height = 600
canvas = tk.Canvas(root, width=width, height=height, bg="white")
canvas.pack()

x_scale = 1
y_scale = 1
x_offset = width // 2
y_offset = height // 2
size = 1

canvas.create_line(0, y_offset, width, y_offset, fill="gray")
canvas.create_line(x_offset, 0, x_offset, height, fill="gray")

class NeuralNetwork:
    def __init__(self):
        self.w1 = random.uniform(-1,1)
        self.b1 = random.uniform(-1,1)

        self.w2 = random.uniform(-1,1)
        self.b2 = random.uniform(-1,1)

        self.w3 = random.uniform(-1,1)
        self.w4 = random.uniform(-1,1)
        self.b3 = random.uniform(-1,1)

    def forward(self,x):
        self.z1 = x * self.w1 + self.b1
        self.z2 = x * self.w2 + self.b2
        self.h1 = relu(self.z1)
        self.h2 = relu(self.z2)
        self.y_hat = self.h1 * self.w3 + self.h2 * self.w4 + self.b3
        return self.y_hat

    def compute_loss(self, data):
        loss = 0
        for x,y in data.items():
            loss += (y-self.forward(x/100))**2
        return loss/len(data)

    def backward(self,data,epochs=200000,lr=0.01):
        for epoch in range(epochs):
            for name in ["w1", "w2", "w3", "w4", "b1", "b2", "b3"]:
                prev_loss = self.compute_loss(data)
                setattr(self, name, getattr(self, name) - lr)
                new_loss = self.compute_loss(data)
                if new_loss > prev_loss:
                    setattr(self, name, getattr(self, name) + 2 * lr)


            if epoch % 1000 == 0:
                print(min(new_loss,prev_loss), epoch)

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

canvas.create_oval(x_offset + 5, y_offset -5,x_offset + -1, y_offset --1,fill="gray")

for age in range(-1000,1000):
    print(f'Age: {age}, Allowed: {nn.predict(age)[0]}, Margin: {nn.predict(age)[1]}')
    canvas_x1 = x_offset + age * x_scale
    canvas_y1 = y_offset - nn.predict(age)[1] * y_scale

    canvas.create_oval(canvas_x1 - size, canvas_y1 - size,
                       canvas_x1 + size, canvas_y1 + size,
                       fill="red", outline="red")

root.mainloop()
















