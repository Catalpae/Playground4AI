import matplotlib.pyplot as plt
plt.ion()


def draw(x_data, y_data, pred_data, pause_time=0.1):
    plt.cla()
    plt.scatter(x_data.numpy(), y_data.numpy())
    plt.plot(x_data.numpy(), pred_data.numpy(), 'r-')
    plt.pause(pause_time)