import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight * x


class MConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.mconv1 = BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.mconv2 = BasicConv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2)
        self.mconv3 = BasicConv2d(in_channel, out_channel, kernel_size=7, stride=1, padding=3)

        self.decay_weight2 = nn.Parameter(torch.zeros(1, 1, 1))
        self.decay_weight3 = nn.Parameter(torch.zeros(1, 1, 1))

        self.se = SEWeightModule(out_channel)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid
        self.catc = BasicConv2d(3 * out_channel, out_channel, 1, 1)
        self.res_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1),
                                       nn.BatchNorm2d(out_channel))
        self.res_back_conv = nn.Conv2d(out_channel, in_channel, 1, 1)

        nn.init.trunc_normal_(self.decay_weight2, a=0.5, b=0.75)
        nn.init.trunc_normal_(self.decay_weight3, a=0.25, b=0.5)

    def forward(self, x):
        ideneity = x
        x1 = self.mconv1(x)
        x2 = self.mconv2(x)
        x3 = self.mconv3(x)

        x1_ch = x1
        x2_ch = self.decay_weight2 * x2
        x3_ch = self.decay_weight3 * x3

        x_cat = torch.cat((x1_ch, x2_ch, x3_ch), dim=1)
        x_catc = self.catc(x_cat)
        x_se = self.se(x_catc)

        out = self.relu(x_se + self.res_conv(ideneity))

        return out