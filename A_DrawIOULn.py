import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# 定义要绘制的函数
def IOU_ln(x_values):
    # y_values = []
    # for i in x_values:
    #     y_values.append(-np.log(i))

    y_values=-np.log(x_values)
    return y_values

def IOU_1(x_values):
    y_values = 1-x_values
    return y_values

# 生成 x 值
# x_values = np.linspace(-1 * np.pi, 1 * np.pi, 1000)  # 生成从 -2π 到 2π 的等间距的 1000 个点
x_values = np.linspace(0.0001, 1 , 1000)  # 生成从 -2π 到 2π 的等间距的 1000 个点


# 绘制函数图像
plt.plot(x_values, IOU_ln(x_values), label='IOU_ln')
plt.plot(x_values, IOU_1(x_values), label='IOU_1')

# 添加标题和标签
plt.title('IOU Loss Function')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# 显示图例
plt.legend()

# 显示图像
plt.show()