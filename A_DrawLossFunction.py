import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def BCELoss(pred, true):
    func = nn.BCELoss()
    # func = nn.BCEWithLogitsLoss()
    return func(pred, true)



def get_y_values(func, x_values, label=1):
    """
    func: 绘图函数
    x_values: 横坐标
    label: 损失函数中的true值
    """
    y_values = []
    for i in x_values:
        true = torch.tensor(label, dtype=torch.float)
        pred = torch.tensor(i, dtype=torch.float)
        y_values.append(func(pred, true))
    y_values=torch.tensor(y_values,dtype=torch.float)
    return y_values


# 横坐标 x_values
# x_values = np.linspace(-1 * np.pi, 1 * np.pi, 1000)  # 生成从 -2π 到 2π 的等间距的 1000 个点
x_values = torch.linspace(start=0.05 , end=0.95 , steps=1000, dtype=torch.float)

# 纵坐标 y_values
y_values_BCE_0 = get_y_values(func=BCELoss,x_values=x_values,label=0)
y_values_BCE_1 = get_y_values(func=BCELoss,x_values=x_values,label=1)
y_values_BCE_0_1 = get_y_values(func=BCELoss,x_values=x_values,label=0.1)
y_values_BCE_0_05 = get_y_values(func=BCELoss,x_values=x_values,label=0.05)
y_values_BCE_0_9 = get_y_values(func=BCELoss,x_values=x_values,label=0.9)
y_values_BCE_0_95 = get_y_values(func=BCELoss,x_values=x_values,label=0.95)
# y_values_BCE_0_8 = get_y_values(func=BCELoss,x_values=x_values,label=0.8)
# y_values_BCE_0_2 = get_y_values(func=BCELoss,x_values=x_values,label=0.2)

# 绘制函数图像
plt.plot(x_values.numpy(), y_values_BCE_0.numpy(), label='BCE-0')
plt.plot(x_values.numpy(), y_values_BCE_1.numpy(), label='BCE-1')
plt.plot(x_values.numpy(), y_values_BCE_0_1.numpy(), label='BCE-0.1')
plt.plot(x_values.numpy(), y_values_BCE_0_05.numpy(), label='BCE-0.05')
plt.plot(x_values.numpy(), y_values_BCE_0_9.numpy(), label='BCE-0.9')
plt.plot(x_values.numpy(), y_values_BCE_0_95.numpy(), label='BCE-0.95')
# plt.plot(x_values.numpy(), y_values_BCE_0_8.numpy(), label='BCE-0.8')
# plt.plot(x_values.numpy(), y_values_BCE_0_2.numpy(), label='BCE-0.2')

# 添加标题和标签
plt.title('Loss Function Plot')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# 显示图例
plt.legend()

# 显示图像
# plt.grid(True)
plt.show()
print("OK!")


