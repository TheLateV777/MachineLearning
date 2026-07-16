# Developed by TheLateV777
# Intended to check the graph produced by neural networks

import tkinter as tk
import random

root = tk.Tk()
root.title("Neural Network Graph")

width = 900
height = 600
canvas = tk.Canvas(root, width=width, height=height, bg="white")
canvas.pack()

w1 = 1.7
b1 = 0.06

w2 = -3.6
b2 = 1.9

w3 = 1.9
w4 = 1.3
b3 = -1.9

x_scale = 1
y_scale = 1
x_offset = width // 2
y_offset = height // 2
size = 5

# axes
canvas.create_line(0, y_offset, width, y_offset, fill="gray")     # x-axis
canvas.create_line(x_offset, 0, x_offset, height, fill="gray")    # y-axis

for i in range(-200, 200):
    z1 = i * w1 + b1
    z2 = i * w2 + b2

    y_hat = z1*w3 + z2*w4 + b3

    canvas_x1 = x_offset + i * x_scale
    canvas_y1 = y_offset - y_hat * y_scale

    canvas.create_oval(canvas_x1 - size, canvas_y1 - size,
                       canvas_x1 + size, canvas_y1 + size,
                       fill="black", outline="black")


root.mainloop()

#1.775646945635034
#-3.62639472978368
#1.9603756399677628
#1.358659136823935
#0.06987383496452057
#1.9784594883521116
#-1.9888615356419916