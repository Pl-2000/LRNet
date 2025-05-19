import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

from VGG16withLOP import VGG16_layer1, VGG16_layer2, VGG16_layer3, VGG16_layer4, VGG16_layer5
from Decoder import decoder_layer5, decoder_layer4, decoder_layer3, decoder_layer2, decoder_layer1


class backbone_vgg16(nn.Module):
    def __init__(self):
        super(backbone_vgg16, self).__init__()
        self.features = nn.ModuleList(list(vgg16(pretrained=True).features)[:30]).eval()

    def forward(self, x):
        results = []
        for index, sub_module in enumerate(self.features):
            x = sub_module(x)
            if index in {3, 8, 15, 22, 29}:
                results.append(x)
        return results


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=(1, 1), bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=(1, 1), bias=False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        avg_ca = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_ca = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        return self.sigmod(avg_ca + max_ca)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=(7, 7), padding=(3, 3), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_sa = torch.mean(x, dim=1, keepdim=True)
        max_sa = torch.max(x, dim=1, keepdim=True, out=None)[0]

        x = torch.cat([avg_sa, max_sa], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SCA(nn.Module):
    """
    Spatial Cross Attention
    """

    def __init__(self, in_channels, cos_sim_threshold=0.4, label_threshold=0.5):
        """
        in_channels: 每个分支的通道数
        """
        super(SCA, self).__init__()
        self.sa1 = SpatialAttention()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.sa2 = SpatialAttention()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        self.relu = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(in_channels)

        self.cos_sim_threshold=cos_sim_threshold
        self.label_threshold=label_threshold

    def forward(self, x, x_t1, x_t2, pre_sa_weights=None):
        # T1,T2-SA1
        sa1_input = torch.abs(x_t1 - x_t2)
        sa1_input = self.relu(self.conv1(sa1_input))
        sa1_weights = self.sa1(sa1_input)

        # main_stream-SA2
        sa2_input = self.relu(self.conv2(x))
        sa2_weights = self.sa2(sa2_input)

        # SA1 & SA2 - cosine_similarity -> 相似性权重alpha，阈值为self.cos_sim_threshold
        cos_sim=F.cosine_similarity(sa1_input, sa2_input, dim=1)
        cos_sim_threshold = torch.full(cos_sim.shape, fill_value=self.cos_sim_threshold).cuda()
        alpha=torch.max(cos_sim,cos_sim_threshold)
        alpha=torch.unsqueeze(alpha,dim=1)

        # 对sa1_weights，sa2_weights变化与否（1？0？）进行判别
        # chg_chg_mask：1+1->注意力权重增加 alpha*2
        # unchg_unchg_mask：0+0->注意力权重减少 self.cos_sim_threshold * (1 - alpha)
        # chg_unchg_mask/unchg_chg_mask：0+1/1+0->注意力权重设置为定值 self.cos_sim_threshold
        label_threshold = torch.full(sa1_weights.shape, fill_value=self.label_threshold).cuda()
        sa1_chg_mask=torch.gt(sa1_weights,label_threshold)
        sa2_chg_mask=torch.gt(sa2_weights,label_threshold)
        sa1_unchg_mask = torch.logical_not(sa1_chg_mask)
        sa2_unchg_mask = torch.logical_not(sa2_chg_mask)
        chg_chg_mask=torch.logical_and(sa1_chg_mask,sa2_chg_mask)*2*alpha
        unchg_unchg_mask=torch.logical_and(sa1_unchg_mask,sa2_unchg_mask)*self.cos_sim_threshold*(1-alpha)
        chg_unchg_mask=torch.logical_and(sa1_chg_mask,sa2_unchg_mask)*self.cos_sim_threshold
        unchg_chg_mask = torch.logical_and(sa1_unchg_mask, sa2_chg_mask)*self.cos_sim_threshold

        # sa1_weights和sa2_weights取均值
        sa_weights_tmp = torch.mean(torch.cat([sa1_weights, sa2_weights], dim=1), dim=1, keepdim=True)

        # 根据不同的注意力权重计算新的权重值，最终得到总的注意力权重
        chg_chg = torch.mul(sa_weights_tmp, chg_chg_mask)
        unchg_unchg = torch.mul(sa_weights_tmp, unchg_unchg_mask)
        chg_unchg = torch.mul(sa_weights_tmp, chg_unchg_mask)
        unchg_chg = torch.mul(sa_weights_tmp, unchg_chg_mask)

        sa_weights = torch.sum(torch.cat([chg_chg, unchg_unchg, chg_unchg, unchg_chg], dim=1), dim=1, keepdim=True)
        if pre_sa_weights!=None:
            sa_weights=torch.mean(torch.cat([sa_weights, pre_sa_weights], dim=1), dim=1, keepdim=True)

        # 特征注意力加强
        ca_sa = sa2_input * sa_weights
        ca_sa = self.bn3(self.relu(self.conv3(ca_sa)))

        return ca_sa, sa_weights


class LRNet_1Input(nn.Module):
    def __init__(self, lrnet_cos_sim_threshold=0.4, lrnet_label_threshold=0.5):
        super(LRNet_1Input, self).__init__()

        self.backbone = backbone_vgg16()
        self.main_stream_layer1 = VGG16_layer1()  # 3-64
        self.main_stream_layer2 = VGG16_layer2()  # 64-128
        self.main_stream_layer3 = VGG16_layer3()  # 128-256
        self.main_stream_layer4 = VGG16_layer4()  # 256-512
        self.main_stream_layer5 = VGG16_layer5()  # 512-512

        self.output_layer5 = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))

        self.sca1 = SCA(64, cos_sim_threshold=lrnet_cos_sim_threshold, label_threshold=lrnet_label_threshold)
        self.sca2 = SCA(128, cos_sim_threshold=lrnet_cos_sim_threshold, label_threshold=lrnet_label_threshold)
        self.sca3 = SCA(256, cos_sim_threshold=lrnet_cos_sim_threshold, label_threshold=lrnet_label_threshold)
        self.sca4 = SCA(512, cos_sim_threshold=lrnet_cos_sim_threshold, label_threshold=lrnet_label_threshold)
        self.sca5 = SCA(512, cos_sim_threshold=lrnet_cos_sim_threshold, label_threshold=lrnet_label_threshold)

        # self.avgpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # self.avgpool4 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
        # self.avgpool8 = nn.MaxPool2d(kernel_size=8, stride=8, padding=0, dilation=1, ceil_mode=False)
        # self.avgpool16 = nn.MaxPool2d(kernel_size=16, stride=16, padding=0, dilation=1, ceil_mode=False)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.avgpool4 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0, ceil_mode=False)
        self.avgpool8 = nn.AvgPool2d(kernel_size=8, stride=8, padding=0, ceil_mode=False)
        self.avgpool16 = nn.AvgPool2d(kernel_size=16, stride=16, padding=0, ceil_mode=False)

        self.decoder_layer5 = decoder_layer5(512, 512)
        self.sa5 = SpatialAttention()
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_trans5 = nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))

        self.ca4 = ChannelAttention(in_channels=512 * 2, ratio=8)
        self.decoder_layer4 = decoder_layer4(512, 256)
        self.sa4 = SpatialAttention()
        self.bn4 = nn.BatchNorm2d(256)
        self.conv_trans4 = nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))

        self.ca3 = ChannelAttention(in_channels=256 * 2, ratio=8)
        self.decoder_layer3 = decoder_layer3(256, 128)
        self.sa3 = SpatialAttention()
        self.bn3 = nn.BatchNorm2d(128)
        self.conv_trans3 = nn.ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))

        self.ca2 = ChannelAttention(in_channels=128 * 2, ratio=8)
        self.decoder_layer2 = decoder_layer2(128, 64)
        self.sa2 = SpatialAttention()
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_trans2 = nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2))

        self.ca1 = ChannelAttention(in_channels=64 * 2, ratio=8)
        self.decoder_layer1 = decoder_layer1(64, 3)
        self.sa1 = SpatialAttention()
        self.bn1 = nn.BatchNorm2d(3)

        self.output = nn.Conv2d(3, 1, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_t1, x_t2 = torch.split(x,3,dim=1)
        t1_stream_layer1, t1_stream_layer2, t1_stream_layer3, t1_stream_layer4, t1_stream_layer5 = self.backbone(x_t1)
        t2_stream_layer1, t2_stream_layer2, t2_stream_layer3, t2_stream_layer4, t2_stream_layer5 = self.backbone(x_t2)

        main_stream_input = torch.abs(x_t1 - x_t2)
        main_stream_layer1 = self.main_stream_layer1(main_stream_input)

        main_stream_layer1, sa_weights1 = self.sca1(main_stream_layer1, t1_stream_layer1, t2_stream_layer1)
        pre_sa_weights2 = torch.mean(torch.cat([self.avgpool2(sa_weights1)], dim=1), dim=1, keepdim=True)
        main_stream_layer2 = self.main_stream_layer2(main_stream_layer1)
        main_stream_layer2, sa_weights2 = self.sca2(main_stream_layer2, t1_stream_layer2, t2_stream_layer2, pre_sa_weights2)
        pre_sa_weights3 = torch.mean(torch.cat([self.avgpool4(sa_weights1),self.avgpool2(sa_weights2)], dim=1), dim=1, keepdim=True)
        main_stream_layer3 = self.main_stream_layer3(main_stream_layer2)
        main_stream_layer3, sa_weights3 = self.sca3(main_stream_layer3, t1_stream_layer3, t2_stream_layer3, pre_sa_weights3)
        pre_sa_weights4 = torch.mean(torch.cat([self.avgpool8(sa_weights1), self.avgpool4(sa_weights2), self.avgpool2(sa_weights3)], dim=1), dim=1,keepdim=True)
        main_stream_layer4 = self.main_stream_layer4(main_stream_layer3)
        main_stream_layer4, sa_weights4 = self.sca4(main_stream_layer4, t1_stream_layer4, t2_stream_layer4, pre_sa_weights4)
        pre_sa_weights5 = torch.mean(torch.cat([self.avgpool16(sa_weights1), self.avgpool8(sa_weights2), self.avgpool4(sa_weights3), self.avgpool2(sa_weights4)], dim=1), dim=1, keepdim=True)
        main_stream_layer5 = self.main_stream_layer5(main_stream_layer4)
        main_stream_layer5, sa_weights5 = self.sca5(main_stream_layer5, t1_stream_layer5, t2_stream_layer5, pre_sa_weights5)

        decoder_layer5 = self.decoder_layer5(main_stream_layer5)
        decoder_layer5 = self.bn5(self.sa5(decoder_layer5) * decoder_layer5)
        decoder_layer5 = self.conv_trans5(decoder_layer5)

        decoder_cat4 = torch.cat([decoder_layer5, main_stream_layer4], dim=1)
        decoder_layer4 = self.ca4(decoder_cat4) * decoder_cat4
        decoder_layer4 = self.decoder_layer4(decoder_layer4)
        decoder_layer4 = self.bn4(self.sa4(decoder_layer4) * decoder_layer4)
        decoder_layer4 = self.conv_trans4(decoder_layer4)

        decoder_cat3 = torch.cat([decoder_layer4, main_stream_layer3], dim=1)
        decoder_layer3 = self.ca3(decoder_cat3) * decoder_cat3
        decoder_layer3 = self.decoder_layer3(decoder_layer3)
        decoder_layer3 = self.bn3(self.sa3(decoder_layer3) * decoder_layer3)
        decoder_layer3 = self.conv_trans3(decoder_layer3)

        decoder_cat2 = torch.cat([decoder_layer3, main_stream_layer2], dim=1)
        decoder_layer2 = self.ca2(decoder_cat2) * decoder_cat2
        decoder_layer2 = self.decoder_layer2(decoder_layer2)
        decoder_layer2 = self.bn2(self.sa2(decoder_layer2) * decoder_layer2)
        decoder_layer2 = self.conv_trans2(decoder_layer2)

        decoder_cat1 = torch.cat([decoder_layer2, main_stream_layer1], dim=1)
        decoder_layer1 = self.ca1(decoder_cat1) * decoder_cat1
        decoder_layer1 = self.decoder_layer1(decoder_layer1)
        decoder_layer1 = self.bn1(self.sa1(decoder_layer1) * decoder_layer1)

        output = self.output(decoder_layer1)
        output_layer5 = self.output_layer5(main_stream_layer5)

        return self.sigmoid(output), self.sigmoid(output_layer5)


if __name__ == "__main__":
    # net = TripleNet()
    # print(list(net.modules()))

    # from torchvision.models import resnet34
    # sub_model_vgg=vgg16(pretrained=True)
    # sub_model=resnet34(pretrained=True)
    print("True!")
