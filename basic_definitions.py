# This is a basic neuron, with one input and one output

class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def get_output(self,received_input):
        return received_input*self.weight + self.bias

    def change_values(self,weight,bias):
        self.weight = weight
        self.bias = bias

# This is loss function, put the output given by your AI in 'prediction'
# and put the actual answer in 'answer' to check the efficiency of your AI

def loss_fn(prediction,answer):
    total_loss = 0
    for i,j in zip(prediction,answer):
        total_loss += (i - j)**2
    return total_loss/len(prediction)