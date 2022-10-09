import torch
import torch.nn as nn
import torch.nn.functional as F
from .swintransformer import swin_small_patch4_window7_224
from .MCA import MConv


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


class SPA_Module(nn.Module):
    def __init__(self, in_ch):
        super(SPA_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_ch, 1, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fx = self.conv1(x)
        fx = self.bn1(fx)
        fx = self.relu1(fx)
        fx = self.conv2(fx)
        fx = self.sigmoid(fx)
        return fx * x


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


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_relu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_relu(output)

        return output


class aggregation(nn.Module):
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample8 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_upsample9 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

    def forward(self, x1, x2, x3, x4):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x4_1 = self.conv_upsample6(self.upsample(self.upsample(self.upsample(x1))))*\
               self.conv_upsample7(self.upsample(self.upsample(x2)))*\
               self.conv_upsample8(self.upsample(x3))*x4

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)


        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x4_2 = torch.cat((x4_1, self.conv_upsample9(self.upsample(x3_2))), 1)
        x4_2 = self.conv_concat4(x4_2)

        x = self.conv4(x4_2)
        x = self.conv5(x)

        return x


class RrConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = Conv(in_channel, out_channel, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)
        self.res_conv =nn.Conv2d(in_channel, out_channel, 1)
        self.res_back_conv = nn.Conv2d(out_channel, in_channel, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv(x)
        f1 = self.relu(x + self.res_conv(identity))
        f2 = torch.mul(F.sigmoid(self.res_back_conv(f1)) + 1, identity)
        x = self.conv(f2)
        x = self.relu(self.res_conv(f2) + x)
        return x


class MSRAformer(nn.Module):
    # swintransformer based encoder decoder
    def __init__(self, channel=32, pretrained=True):
        super(MSRAformer, self).__init__()
        # ---- swintransformer Backbone ----
        self.swintransformer = swin_small_patch4_window7_224(1000, pretrained=pretrained)

        self.conv1 = Conv(96, 96, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)
        self.conv2 = Conv(192, 192, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)
        self.conv3 = Conv(384, 384, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)
        self.conv4 = Conv(768, 768, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)

        self.mconv1 = MConv(96, channel)
        self.mconv2 = MConv(192, channel)
        self.mconv3 = MConv(384, channel)
        self.mconv4 = MConv(768, channel)


        self.agg1 = aggregation(channel)

        self.ra1_conv1 = RrConv(768, 32)
        self.ra1_conv2 = RrConv(32, 32)
        self.ra1_conv3 = RrConv(32, 32)
        self.ra1_conv4 = RrConv(32, 32)
        self.ra1_conv5 = Conv(32, 1, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)

        self.ra2_conv1 = RrConv(384, 32)
        self.ra2_conv2 = RrConv(32, 32)
        self.ra2_conv3 = RrConv(32, 32)
        self.ra2_conv4 = RrConv(32, 32)
        self.ra2_conv5 = Conv(32, 1, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)

        self.ra3_conv1 = RrConv(192, 32)
        self.ra3_conv2 = RrConv(32, 32)
        self.ra3_conv3 = RrConv(32, 32)
        self.ra3_conv4 = RrConv(32, 32)
        self.ra3_conv5 = Conv(32, 1, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)

        self.ra4_conv1 = RrConv(96, 32)
        self.ra4_conv2 = RrConv(32, 32)
        self.ra4_conv3 = RrConv(32, 32)
        self.ra4_conv4 = RrConv(32, 32)
        self.ra4_conv5 = Conv(32, 1, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)
        # The parameters of SPA module are determined according
        # to the size of input image and Swin Transformer
        self.spa1 = SPA_Module(96)
        self.spa2 = SPA_Module(192)
        self.spa3 = SPA_Module(384)
        self.spa4 = SPA_Module(768)


    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.swintransformer.patch_embed(x)
        H1, W1 = H, W
        x = self.swintransformer.pos_drop(x)
        x_list=[x]
        x1_list=[]

        # print(self.swintransformer.layers)

        for i, layer in enumerate(self.swintransformer.layers):
            x1, x, H, W = layer(x_list[i], H, W)
            x1_list.append(x1)
            x_list.append(x)

        x1 = x1_list[0].permute(0, 2, 1)
        x2 = x1_list[1].permute(0, 2, 1)
        x3 = x1_list[2].permute(0, 2, 1)
        x4 = x1_list[3].permute(0, 2, 1)

        x1 = x1.view(B, 96, H1, W1)
        x2 = x2.view(B, 192, H1 // 2, W1 // 2)
        x3 = x3.view(B, 384, H1//4, W1//4)
        x4 = x4.view(B, 768, H1 // 8, W1 // 8)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        x1_f = self.mconv1(x1)
        x2_f = self.mconv2(x2)
        x3_f = self.mconv3(x3)
        x4_f = self.mconv4(x4)

        x_agg = self.agg1(x4_f, x3_f, x2_f, x1_f)
        pre_res = F.interpolate(x_agg, scale_factor=4, mode='bilinear')

        Feature_map_4 = F.interpolate(x_agg, scale_factor=0.125, mode='bilinear')
        x = -1*(torch.sigmoid(Feature_map_4)) + 1
        x = x.expand(-1, 768, -1, -1)
        x = self.spa4(x)
        x = x.mul(x4)
        x = self.ra1_conv1(x)
        x = self.ra1_conv2(x)
        x = self.ra1_conv3(x)
        x = self.ra1_conv4(x)
        ra4_feat = self.ra1_conv5(x)
        x = ra4_feat + Feature_map_4
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')

        Feature_map_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(Feature_map_3)) + 1
        x = x.expand(-1, 384, -1, -1)
        x = self.spa3(x)
        x = x.mul(x3)
        x = self.ra2_conv1(x)
        x = self.ra2_conv2(x)
        x = self.ra2_conv3(x)
        x = self.ra2_conv4(x)
        ra3_feat = self.ra2_conv5(x)
        x = ra3_feat + Feature_map_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')

        Feature_map_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(Feature_map_2)) + 1
        x = x.expand(-1, 192, -1, -1)
        x = self.spa2(x)
        x = x.mul(x2)
        x = self.ra3_conv1(x)
        x = self.ra3_conv2(x)
        x = self.ra3_conv3(x)
        x = self.ra3_conv4(x)
        ra2_feat = self.ra3_conv5(x)
        x = ra2_feat + Feature_map_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')

        Feature_map_1 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(Feature_map_1)) + 1
        x = x.expand(-1, 96, -1, -1)
        x = self.spa1(x)
        x = x.mul(x1)
        x = self.ra4_conv1(x)
        x = self.ra4_conv2(x)
        x = self.ra4_conv3(x)
        x = self.ra4_conv4(x)
        ra3_feat = self.ra4_conv5(x)
        x = ra3_feat + Feature_map_1
        lateral_map_1 = F.interpolate(x, scale_factor=4, mode='bilinear')

        return pre_res, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1

