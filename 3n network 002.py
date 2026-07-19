# Developed by TheLateV777
# Intended to test out new degree n polynomial graph
# Failed

class NeuralNetwork:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.w3 = 1
        self.w4 = 1
        self.b = 1

    def forward(self,x):
        y_hat = self.w1 * x**4 + self.w2 * x**3 + self.w3 * x**2 + self.w4 * x + self.b
        return y_hat

    def compute_loss(self, data):
        loss = 0
        for x,y in data.items():
            loss += (y-self.forward(x))**2
        return loss

    def backward(self,data):
        lr = 0.001
        for epoch in range(20000):
            for name in ["w1", "w2", "w3", "w4", "b"]:
                prev_loss = self.compute_loss(data)
                setattr(self, name, getattr(self, name) - lr)
                new_loss = self.compute_loss(data)
                if new_loss > prev_loss:
                    setattr(self, name, getattr(self, name) + 2 * lr)

            if epoch%100 == 0:
                print(min(new_loss,prev_loss))

data = {2:0,
        4:0,
        10:0,
        16:0,
        17:0,
        18:1,
        19:1,
        22:1,
        26:1,
        30:1,
        34:1,
        36:1,
        40:1,
        41:0,
        43:0,
        45:0,
        50:0}

n = NeuralNetwork()
n.backward(data)

print(n.w1,n.w2,n.w3,n.w4,n.b)











