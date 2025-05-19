import torch
import torch.nn as nn


class DiceLoss:
    def __init__(self):
        self.smooth_para = 1.0

    def __call__(self, pred, true):
        pred_flat = pred.view(-1)
        true_flat = true.view(-1)
        intersection = (pred_flat * true_flat).sum()
        dice_loss = 1 - ((2. * intersection + self.smooth_para) / (pred_flat.sum() + true_flat.sum() + self.smooth_para))

        return dice_loss


class IOULoss:
    def __init__(self):
        self.smooth_para = 1.0

    def __call__(self, pred, true):
        pred_flat = pred.view(-1)
        true_flat = true.view(-1)
        intersection = (pred_flat * true_flat).sum()
        iou_loss = -torch.log((intersection + self.smooth_para) / (pred_flat.sum() + true_flat.sum() - intersection + self.smooth_para))

        return iou_loss


class IOULoss_Edge:
    def __init__(self):
        self.smooth_para = 1.0

    def __call__(self, pred, true):
        pred_flat = pred.view(-1)/255
        true_flat = true.view(-1)
        intersection = (pred_flat * true_flat).sum()
        iou_loss = -torch.log((intersection + self.smooth_para) / (pred_flat.sum() + true_flat.sum() - intersection + self.smooth_para))

        return iou_loss


class HybridCDLoss:
    def __init__(self, label_smoothing_para_beta=0.05, hard_ratio_para_theta=0.5):
        """
        label_smoothing_para_beta: 标签平滑系数/软标签软化程度
        hard_ratio_para_theta: 软硬标签loss分配中硬标签占比
        """
        self.BCELoss = nn.BCELoss()
        self.DiceLoss = DiceLoss()
        self.beta=label_smoothing_para_beta
        self.theta=hard_ratio_para_theta

    def __call__(self, pred, true):
        pred=torch.squeeze(pred)
        hard_bce_loss = self.BCELoss(pred, true)
        soft_bce_loss = self.BCELoss(self.soft_label(pred),self.soft_label(true))

        dice_loss = self.DiceLoss(pred, true)

        return self.theta*hard_bce_loss + (1-self.theta)*soft_bce_loss + dice_loss

    def soft_label(self,origin_label):
        target_shape=origin_label.shape
        low_limit=torch.full(target_shape, fill_value=self.beta).cuda()
        upper_limit=torch.full(target_shape, fill_value=1-self.beta).cuda()
        return torch.min(torch.max(origin_label,low_limit),upper_limit)


class HybridCDLoss_IOU:
    def __init__(self, label_smoothing_para_beta=0.05, hard_ratio_para_theta=0.5):
        """
        label_smoothing_para_beta: 标签平滑系数/软标签软化程度
        hard_ratio_para_theta: 软硬标签loss分配中硬标签占比
        """
        self.BCELoss = nn.BCELoss()
        self.IOULoss = IOULoss()
        self.beta=label_smoothing_para_beta
        self.theta=hard_ratio_para_theta

    def __call__(self, pred, true):
        pred=torch.squeeze(pred)
        hard_bce_loss = self.BCELoss(pred, true)
        soft_bce_loss = self.BCELoss(self.soft_label(pred),self.soft_label(true))

        iou_loss = self.IOULoss(pred, true)

        return self.theta*hard_bce_loss + (1-self.theta)*soft_bce_loss + iou_loss

    def soft_label(self,origin_label):
        target_shape=origin_label.shape
        low_limit=torch.full(target_shape, fill_value=self.beta).cuda()
        upper_limit=torch.full(target_shape, fill_value=1-self.beta).cuda()
        return torch.min(torch.max(origin_label,low_limit),upper_limit)