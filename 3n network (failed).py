import random

class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def get_output(self,received_input):
        return received_input*self.weight + self.bias

    def change_values(self,weight,bias):
        self.weight = weight
        self.bias = bias

def loss_fn(prediction,answer):
    total_loss = 0
    for i,j in zip(prediction,answer):
        total_loss += (i - j)**2
    return total_loss/len(prediction)

n = Neuron(1,1)

info = {1:0,3:0,6:0,10:1,11:1,13:1}
best_loss = float('inf')
best_pair = 0

for weight in range(-500,499,1):
    for bias in range(-500,499,1):
        n.change_values(round(weight * 0.1, 1), round(bias * 0.1, 1))
        prediction = []
        for key in info.keys():
            prediction.append(n.get_output(key))

        current_loss = loss_fn(prediction,info.values())

        if current_loss < best_loss:
            best_loss = current_loss
            best_pair = (n.weight,n.bias)

n.change_values(best_pair[0],best_pair[1])
print('weight',n.weight,'bias',n.bias)
test = [7,8,9,25,-10,-100]
for i in test:
    if n.get_output(i) > 0.5:
        print(i,'big')
    else:
        print(i,'small')

