# Developed by TheLateV777
# 1 input, 1 hidden layer: 2 neurons, 1 output neuron
# Intended to check age for application to army
# Failed Miserably
# Best Loss: 0.25

from basic_definitions import step,relu
import random

# Neuron class with only name change
class HiddenNeuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def get_output(self,received_input):
        return received_input*self.weight + self.bias

    def change_values(self,weight,bias):
        self.weight = weight
        self.bias = bias

# Neuron class modified to accept two inputs
class OutputNeuron:
    def __init__(self, weight1, weight2, bias):
        self.weight1 = weight1
        self.weight2 = weight2
        self.bias = bias

    def get_output(self,input1,input2):
        return input1 * self.weight1 + input2 * self.weight2 + self.bias

    def change_values(self,weight1,weight2,bias):
        self.weight1 = weight1
        self.weight2 = weight2
        self.bias = bias

def loss_fn(prediction,answer):
    total_loss = 0
    for i,j in zip(prediction,answer):
        total_loss += (i - j)**2
    return total_loss/len(prediction)

def generate():
    return round(random.randint(-500,500) * 0.01, 0)

info = {17:0,18:1,20:1,39:1,40:0}

best_loss = float('inf')
best_combo = (0,0,0,0,0,0,0)

n1 = HiddenNeuron(1,1)
n2 = HiddenNeuron(1,1)
n3 = OutputNeuron(1,1,1)

i = 0

while best_loss != 0:
    weight1 = generate()
    weight2 = generate()
    weight31 = generate()
    weight32 = generate()
    bias1 = generate()
    bias2 = generate()
    bias3 = generate()

    prediction = []
    answer = info.values()

    n1.change_values(weight1,bias1)
    n2.change_values(weight2,bias2)
    n3.change_values(weight31,weight32,bias3)

    for x in info.keys():
        n1_out = relu(n1.get_output(x))
        n2_out = relu(n2.get_output(x))
        n3_out = step(n3.get_output(n1_out,n2_out))
        prediction.append(n3_out)

    current_loss = loss_fn(prediction,answer)

    if current_loss < best_loss:
        best_loss = current_loss
        best_combo = (weight1,bias1,weight2,bias2,weight31,weight32,bias3)

    i += 1

print(best_combo)









