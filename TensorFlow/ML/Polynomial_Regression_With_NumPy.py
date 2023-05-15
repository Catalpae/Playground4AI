import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-1, 1, 100).reshape(100, 1)
y = 3 * np.power(x, 2) + 2 + 0.3 * np.random.rand(x.size).reshape(100, 1)
# print(np.random.rand(x.size).shape)
# plt.scatter(x, y)
# plt.show()

w, b = np.random.rand(1, 1), np.random.rand(1, 1)
lr = 0.001
for epoch_i in range(100):
    for training_step in range(200):
        y_pred = w * np.power(x, 2) + b
        loss = 0.5 * (y - y_pred) ** 2
        loss = loss.sum()

        grad_w = ((y_pred - y) * np.power(x, 2)).sum()
        grad_b = (y_pred - y).sum()
        w -= lr * grad_w
        b -= lr * grad_b

y_pred = w * np.power(x, 2) + b
plt.plot(x, y_pred, 'r-', label='prediction')
plt.scatter(x, y, color='blue', marker='o', label='ground truth')
plt.xlim(-1, 1)
plt.ylim(2, 6)
plt.legend()
plt.show()

print(w, b)
