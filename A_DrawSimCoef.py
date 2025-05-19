import torch
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.weight'] = 'bold'
title_font = {
    #'fontsize': rcParams['axes.titlesize'], # 设置成和轴刻度标签一样的大小
    'fontsize': 13,
    #'fontweight': rcParams['axes.titleweight'], # 设置成和轴刻度标签一样的粗细
    'fontweight': 'bold',
    #'color': rcParams['axes.titlecolor'], # 设置成和轴刻度标签一样的颜色
    # 'color': 'black',
}
label_font = {
    #'fontsize': rcParams['axes.titlesize'], # 设置成和轴刻度标签一样的大小
    'fontsize': 12,
    #'fontweight': rcParams['axes.titleweight'], # 设置成和轴刻度标签一样的粗细
    'fontweight': 'bold',
    #'color': rcParams['axes.titlecolor'], # 设置成和轴刻度标签一样的颜色
    'color': 'red',
}


def SimCoef(sim, label, threshold):
    if sim<=threshold:
        coef=threshold
    else:
        if label==1:
            coef=sim*2
        else:
            coef=threshold*(1-sim)
    return coef



def get_y_values(func, x_values, label=1, threshold=0.5):
    """
    func: 绘图函数
    x_values: 横坐标
    label: true值: 0:unchg 1:chg
    """
    y_values = []
    for i in x_values:
        y_values.append(func(i, label, threshold))
    y_values=torch.tensor(y_values,dtype=torch.float)
    return y_values


# 横坐标 x_values
# x_values = np.linspace(-1 * np.pi, 1 * np.pi, 1000)  # 生成从 -2π 到 2π 的等间距的 1000 个点
x_values = torch.linspace(start=-1 , end=1 , steps=2000, dtype=torch.float)
x_values_1 = torch.linspace(start=-1 , end=0.5 , steps=2000, dtype=torch.float)

# 纵坐标 y_values
y_values_0 = get_y_values(func=SimCoef,x_values=x_values,label=0, threshold=0.5)
y_values_1 = get_y_values(func=SimCoef,x_values=x_values,label=1, threshold=0.5)
y_values_unchg_chg = torch.linspace(start=0.5 , end=0.5 , steps=2000, dtype=torch.float)
y_values_0_0_4 = get_y_values(func=SimCoef,x_values=x_values,label=0, threshold=0.4)
y_values_1_0_4 = get_y_values(func=SimCoef,x_values=x_values,label=1, threshold=0.4)

y_values_chg = torch.linspace(start=0.515 , end=0.515 , steps=2000, dtype=torch.float)
y_values_unchg = torch.linspace(start=0.485 , end=0.485 , steps=2000, dtype=torch.float)

# 绘制函数图像
# plt.plot(x_values.numpy(), y_values_0.numpy(), label='label:0 unchg')
# plt.plot(x_values.numpy(), y_values_1.numpy(), label='label:1 chg')
plt.plot(x_values.numpy(), y_values_0.numpy(), label='unchg-unchg', color='blue')
plt.plot(x_values.numpy(), y_values_1.numpy(), label='chg-chg', color='orange')
plt.plot(x_values.numpy(), y_values_unchg_chg.numpy(), label='chg-unchg', color='green')
plt.plot(x_values_1.numpy(), y_values_unchg.numpy(), color='blue')
plt.plot(x_values_1.numpy(), y_values_chg.numpy(), color='orange')
# plt.plot(x_values.numpy(), y_values_0_0_4.numpy(), label='Sim-Coef-0-0.4')
# plt.plot(x_values.numpy(), y_values_1_0_4.numpy(), label='Sim-Coef-1-0.4')

# 添加标题和标签
plt.title('Cosine Similarity - Attention Coefficient(\u03B1)', fontdict=title_font)
plt.xlabel('Cosine Similarity', fontdict=label_font)
plt.ylabel('Attention Coefficient (\u03B1)', fontdict=label_font)
plt.scatter(0.5, 0.5, label='threshold')

# 显示图例
plt.legend()

# 显示图像
# plt.grid(True)
plt.show()
print("OK!")


