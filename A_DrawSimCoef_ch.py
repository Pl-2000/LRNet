import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore")

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus']=False
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
    return y_values


# 横坐标 x_values
# x_values = np.linspace(-1 * np.pi, 1 * np.pi, 1000)  # 生成从 -2π 到 2π 的等间距的 1000 个点
x_values = np.linspace(start=-1 , stop=1 , num=2000)
x_values_1 = np.linspace(start=-1 , stop=0.5 , num=2000)

# 纵坐标 y_values
y_values_0 = get_y_values(func=SimCoef,x_values=x_values,label=0, threshold=0.5)
y_values_1 = get_y_values(func=SimCoef,x_values=x_values,label=1, threshold=0.5)
y_values_unchg_chg = np.linspace(start=0.5 , stop=0.5 , num=2000)
y_values_0_0_4 = get_y_values(func=SimCoef,x_values=x_values,label=0, threshold=0.4)
y_values_1_0_4 = get_y_values(func=SimCoef,x_values=x_values,label=1, threshold=0.4)

y_values_chg = np.linspace(start=0.515 , stop=0.515 , num=2000)
y_values_unchg = np.linspace(start=0.485 , stop=0.485 , num=2000)

# 绘制函数图像
# plt.plot(x_values, y_values_0, label='label:0 unchg')
# plt.plot(x_values, y_values_1, label='label:1 chg')
plt.plot(x_values, y_values_0, label='未变化-未变化', color='blue')
plt.plot(x_values, y_values_1, label='变化-变化', color='orange')
plt.plot(x_values, y_values_unchg_chg, label='变化-未变化', color='green')
plt.plot(x_values_1, y_values_unchg, color='blue')
plt.plot(x_values_1, y_values_chg, color='orange')
# plt.plot(x_values, y_values_0_0_4, label='Sim-Coef-0-0.4')
# plt.plot(x_values, y_values_1_0_4, label='Sim-Coef-1-0.4')

# 添加标题和标签
# plt.title('Cosine Similarity - Attention Coefficient(\u03B1)', fontdict=title_font)
# plt.xlabel('Cosine Similarity', fontdict=label_font)
# plt.ylabel('Attention Coefficient (\u03B1)', fontdict=label_font)
plt.title('余弦相似度 - 注意力系数(\u03B1)', fontdict=title_font)
plt.xlabel('余弦相似度', fontdict=label_font)
plt.ylabel('注意力系数 (\u03B1)', fontdict=label_font)
plt.scatter(0.5, 0.5, label='阈值')

# 显示图例
plt.legend()

# 显示图像
# plt.grid(True)
plt.show()
print("OK!")


