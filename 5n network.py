# Developed by TheLateV777
# Intended to evaluate fitness by taking in several factors
# 4 inputs, 1 hidden layer: 4 neurons, 1 output

from basic_definitions import relu,step
import random

class NeuralNetwork:
    def __init__(self):
        self.w1 = 1
        self.b1 = 1

        self.w2 = 1
        self.b2 = 1

        self.w3 = 1
        self.b3 = 1

        self.w4 = 1
        self.b4 = 1

        self.w11 = 1
        self.w12 = 1
        self.w13 = 1
        self.w14 = 1
        self.b11 = 1

    def forward(self,a,b,c,d):
        z1 = a * self.w1 + self.b1
        z2 = b * self.w2 + self.b2
        z3 = c * self.w3 + self.b3
        z4 = d * self.w4 + self.b4

        h1 = relu(z1)
        h2 = relu(z2)
        h3 = relu(z3)
        h4 = relu(z4)

        y_hat = h1 * self.w11 + h2 * self.w12 + h3 * self.w13 + h4 * self.w14 + self.b11

        return y_hat

    def compute_loss(self,data):
        loss = 0
        for x,y in data.items():
            loss += (y - self.forward(x[0],x[1],x[2],x[3])) ** 2
        return loss

    def backward(self,data,epochs=20000,lr = 1):
        og_lr = lr
        for epoch in range(epochs):
            for var in ('w1','w2','w3','w4','w11','w12','w13','w14','b1','b2','b3','b4','b11'):
                p_loss = self.compute_loss(data)
                setattr(self, var, getattr(self, var) - lr)
                c_loss = self.compute_loss(data)
                if c_loss > p_loss:
                    setattr(self, var, getattr(self, var) + 2 * lr)

            lr -= og_lr/epochs*2

            if epoch%100 == 0:
                print(f'epoch: {epoch}\nloss: {min(c_loss,p_loss):.2f}\nvalues: {(self.w1,self.w2,self.w3,self.w4,self.b1,self.b2,self.b3,self.b4,self.w11,self.w12,self.w13,self.w14,self.b11)}')
                print(lr)

    def predict(self,data):
        for a,b,c,d in data:
            print(step(self.forward(a,b,c,d)))


# push-ups, pull-ups, crunches, laps

data = {}

for i in range(100):
    push = random.randint(0,20)
    pull = random.randint(0,20)
    crunches = random.randint(0,20)
    laps = random.randint(0,20)

    data[(push,pull,crunches,laps)] = 1 if push + pull + crunches + 10*laps >= 100 else 0

n = NeuralNetwork()

n.backward(data,50000)

tryouts = [(1,1,1,1),(3,4,5,1),(10,10,10,14),(100,100,100,100)]

n.predict(tryouts)

print('---------')

wins = 0

for x,y in data.items():
    wins += 1 if y == 1 else 0

print(f'wins: {wins}')

print('---------')

n.predict(data)