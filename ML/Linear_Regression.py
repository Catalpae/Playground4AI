import numpy as np
import training_data as td

x_data, y_data = td.generate_square_func_data()
input_dim = output_dim = 1
weight, bias = np.random.rand(input_dim, output_dim), np.random.rand(output_dim)
# print(weight.shape, bias.shape)
# print(weight.size, bias.size)
for _ in range(10):
    pass
