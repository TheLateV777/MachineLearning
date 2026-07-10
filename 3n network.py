# Developed by TheLateV777
# Intended to multiply numbers
# Failed Miserably
# Best Loss: 4.8

import random

class Neuron:
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
    return round(random.randint(-500,500) * 0.01,1)

n1 = Neuron(1,1,1)
n2 = Neuron(1,1,1)
n3 = Neuron(1,1,1)

info = {(1,1):1,(2,1):1,(3,2):6,(2,2):4,(-3,-3):9,(-2,2):4}

best_loss = float('inf')
best_combo = (0,0,0,0,0,0,0,0,0)
i = 0

while best_loss > 1:
    i += 1
    weight11 = generate()
    weight12 = generate()
    bias1 = generate()
    weight21 = generate()
    weight22 = generate()
    bias2 = generate()
    weight31 = generate()
    weight32 = generate()
    bias3 = generate()
    prediction = []
    answer = info.values()
    for x,y in info.keys():
        n1.change_values(weight11,weight12,bias1)
        n2.change_values(weight21,weight22,bias2)
        n3.change_values(weight31,weight32,bias3)
        prediction.append(n3.get_output(n1.get_output(x,y),n2.get_output(x,y)))

    current_loss = loss_fn(prediction,answer)

    if current_loss < best_loss:
        best_loss = current_loss
        best_combo = (weight11,weight12,bias1,weight21,weight22,bias2,weight31,weight32,bias3)

    if i % 10000 == 0:
        print(best_loss)
        print(best_combo)
        print(i)

print(best_loss)
print(best_combo)

n1.change_values(best_combo[0],best_combo[1],best_combo[2])
n2.change_values(best_combo[3],best_combo[4],best_combo[5])
n3.change_values(best_combo[6],best_combo[7],best_combo[8])

test = [(3,4),(4,5),(-1,-1),(10,10)]
ai_answer = []
expected = [12,20,1,100]

for x,y in test:
    ai_answer.append(n3.get_output(n1.get_output(x,y),n2.get_output(x,y)))

print('ai_answer:',ai_answer)
print('expected:',expected)
print('loss',loss_fn(ai_answer,expected))

