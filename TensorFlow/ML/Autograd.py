import numpy as np
import tensorflow as tf


def test1():
    x = tf.Variable(0.0, name='x', dtype=tf.float32)
    a = tf.constant(1.0)
    b = tf.constant(5.0)
    c = tf.constant(2.0)
    # print(x, a)

    with tf.GradientTape() as tape:
        y = a * tf.pow(x, 2) + b * x + c    # 运算完后，y变为常量（tf.Tensor）
        # print(y)
    dy_dx, dy_da = tape.gradient(y, [x, a])    # 梯度磁带自动监视tf.Variable，不自动监视tf.Tensor
    print(dy_dx, dy_da)


def test2():
    x = tf.Variable(2.0, name='x', dtype=tf.float32)
    a = tf.constant(2.0)
    b = tf.constant(5.0)
    c = tf.constant(3.0)

    with tf.GradientTape() as tape:
        tape.watch([a, b, c])   # 增加watch，使得对常量张量也可以求导
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx, dy_da, dy_db, dy_dc = tape.gradient(y, [x, a, b, c])
    print(dy_dx, dy_da, dy_db, dy_dc)


def test3():
    x = tf.Variable(1.0, name='x', dtype=tf.float32)
    a = tf.constant(1.0)
    b = tf.constant(2.0)
    c = tf.constant(3.0)

    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            y = a * tf.pow(x, 2) + b * x + c
        dy_dx = tape1.gradient(y, x)
    dy2_dx2 = tape2.gradient(dy_dx, x)
    print(dy2_dx2)


def test4():
    x = tf.constant([1, 2.0])
    # print(x)

    # with tf.GradientTape() as tape:
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = x * x
        z = y * y

    print(tape.gradient(y, x).numpy())
    print(tape.gradient(z, x).numpy())
    del tape


def test5():
    x = tf.Variable(2.0)
    # print(x)

    for epoch in range(2):
        with tf.GradientTape() as tape:
            y = x + 1
            # print(y)
        dy_dx = tape.gradient(y, x)
        print(dy_dx)

        x = x + 1   # 这个操作会把变量 x 变为常量，使得梯度磁带不再监控 x
        # x.assign_add(1)


def test6():
    x = tf.Variable([[1.0, 2.0],
                     [3.0, 4.0]], dtype=tf.float32)
    # print(x)

    with tf.GradientTape() as tape:
        x2 = x ** 2
        # print(x2)

        y = np.mean(x2, axis=0)   # 使用TensorFlow之外的算子，梯度磁带将无法记录梯度路径
        y1 = tf.reduce_mean(y, axis=0)
        # print(y, y1)

    print(tape.gradient(y1, x))


if __name__ == '__main__':
    test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
