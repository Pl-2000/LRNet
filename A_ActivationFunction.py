import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# # 定义要绘制的函数
# def my_function(x,n):
#
#     # return 1/(1+np.exp(-n*x))
#     return np.power(n,x)
#
# # 生成 x 值
# # x_values = np.linspace(-1 * np.pi, 1 * np.pi, 1000)  # 生成从 -2π 到 2π 的等间距的 1000 个点
# x_values = np.linspace(0 , 1 , 1000)  # 生成从 -2π 到 2π 的等间距的 1000 个点
#
# # # 计算对应的 y 值
# # y_values = my_function(x_values,1)
#
# # # 绘制函数图像
# # plt.plot(x_values, my_function(x_values,1), label='sigmoid-1')
# # plt.plot(x_values, my_function(x_values,2), label='sigmoid-2')
# # plt.plot(x_values, my_function(x_values,0.5), label='sigmoid-0.5')
# # plt.plot(x_values, my_function(x_values,3), label='sigmoid-3')
# # plt.plot(x_values, my_function(x_values,4), label='sigmoid-4')
# # plt.plot(x_values, my_function(x_values,5), label='sigmoid-5')
# # plt.plot(x_values, my_function(x_values,10), label='sigmoid-10')
# # plt.plot(x_values, my_function(x_values,10), label='sigmoid-10')
#
# plt.plot(x_values, my_function(x_values,1), label='1^x')
# plt.plot(x_values, my_function(x_values,0.9), label='0.9^x')
# plt.plot(x_values, my_function(x_values,0.8), label='0.8^x')
# plt.plot(x_values, my_function(x_values,0.5), label='0.5^x')
# plt.plot(x_values, my_function(x_values,0.1), label='0.1^x')
#
# # 添加标题和标签
# plt.title('Function Plot')
# plt.xlabel('x-axis')
# plt.ylabel('y-axis')
#
# # 显示图例
# plt.legend()
#
# # 显示图像
# plt.show()



class HybridCDLoss:
    def __init__(self, label_smoothing_para_beta=0.05, hard_ratio_para_theta=0.5):
        """
        label_smoothing_para_beta: 标签平滑系数/软标签软化程度
        hard_ratio_para_theta: 软硬标签loss分配中硬标签占比
        """
        self.BCELoss = nn.BCELoss()
        self.beta=label_smoothing_para_beta
        self.theta=hard_ratio_para_theta

    def __call__(self, pred, true):
        pred=torch.squeeze(pred)
        hard_bce_loss = self.BCELoss(pred, true)
        soft_bce_loss = self.BCELoss(self.soft_label(pred),self.soft_label(true))

        smooth_para = 1.0
        pred_flat = pred.view(-1)
        true_flat = true.view(-1)
        intersection = (pred_flat * true_flat).sum()
        dic_loss = 1 - ((2. * intersection + smooth_para) / (pred_flat.sum() + true_flat.sum() + smooth_para))

        return self.theta*hard_bce_loss + (1-self.theta)*soft_bce_loss + dic_loss

    def soft_label(self,origin_label):
        target_shape=origin_label.shape
        low_limit=torch.full(target_shape, fill_value=self.beta).cuda()
        upper_limit=torch.full(target_shape, fill_value=1-self.beta).cuda()
        return torch.min(torch.max(origin_label,low_limit),upper_limit)

