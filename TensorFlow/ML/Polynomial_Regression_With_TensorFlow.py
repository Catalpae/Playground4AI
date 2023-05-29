import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)
# print(np.linspace(-1, 1, 100).shape)
x0 = np.linspace(-1, 1, 100).reshape(100, 1)
y0 = 3 * np.power(x0, 2) + 2 + 0.2 * np.random.rand(x0.size).reshape(100, 1)
x, y = tf.constant(x0), tf.constant(y0)    # 数据集
# plt.scatter(x0, y0)
# plt.show()

w0, b0 = np.random.rand(1, 1), np.random.rand(1, 1)  # 模型参数随机初始化


class PolyRegression:
    def __init__(self, **args):
        super(PolyRegression, self).__init__(*args)
        # print(w0.shape)
        self.w = tf.Variable(w0)
        self.b = tf.Variable(b0)

    def __call__(self, x_input):
        y_pred = tf.square(x_input) * self.w + self.b  # 广播机制
        return y_pred


def myloss(y_pred, y_output):
    loss = tf.reduce_mean(tf.square(y_output - y_pred))
    return loss


lr = 1e-2
mymodel = PolyRegression()    # 多项式回归模型


"""
梯度计算采用自动微分，梯度更新采用自定义方式
"""
def method1():

    @tf.function
    def train_step_1(x_input, y_output, model, epoch):
        for i in range(epoch):
            with tf.GradientTape() as tape:
                y_pred = model(x_input)
                loss = myloss(y_pred, y_output)
                w, b = model.w, model.b
                dw, db = tape.gradient(loss, [w, b])
                w.assign(w - lr * dw)
                b.assign(b - lr * db)

    train_step_1(x, y, mymodel, 1000)
    plt.scatter(x, y, c='b')
    plt.scatter(x, mymodel(x), c='r')
    plt.show()


"""
使用optimizer实现自动微分、自动梯度更新
    1. apply_gradients()
"""
def method2():
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    @tf.function
    def train_step_2(x_input, y_output, model, epoch):
        for i in range(epoch):
            with tf.GradientTape() as tape:
                y_pred = model(x_input)
                loss = myloss(y_pred, y_output)
            w, b = model.w, model.b
            dw, db = tape.gradient(loss, [w, b])
            optimizer.apply_gradients(grads_and_vars=zip([dw, db], [w, b]))

    train_step_2(x, y, mymodel, 1000)
    plt.scatter(x, y, c='b')
    plt.scatter(x, mymodel(x), c='r')
    plt.show()


"""
使用optimizer实现自动微分、自动梯度更新
    2. minimize()
"""
def method3():
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    @tf.function
    def train_step_3(x_input, y_output, model, epoch):
        for i in range(epoch):
            with tf.GradientTape() as tape:
                y_pred = model(x_input)
                loss = myloss(y_pred, y_output)
                optimizer.minimize(loss, var_list=[model.w, model.b], tape=tape)    # 注意minimize()的API使用方法

    train_step_3(x, y, mymodel, 1000)
    plt.scatter(x, y, c='b')
    plt.scatter(x, mymodel(x), c='r')
    plt.show()


def method4():
    learning_rate = 1e-2
    w = tf.Variable(w0, name='w')
    b = tf.Variable(b0, name='b')

    def myloss_02():
        y_pred = tf.square(x) * w + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
        return loss

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    @tf.function
    def train_step_4(epoch):
        for _ in range(epoch):
            optimizer.minimize(myloss_02, var_list=[w, b])

    train_step_4(1000)
    plt.scatter(x, y, c='b')
    y_pred = tf.square(x) * w + b
    plt.scatter(x, y_pred, c='r')
    plt.show()


if __name__ == '__main__':
    # method1()
    # method2()
    # method3()
    method4()