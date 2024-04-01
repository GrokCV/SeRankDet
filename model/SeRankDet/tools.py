# @Time    : 2023/6/15 19:02
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : tools.py
# @Software: PyCharm
from __future__ import print_function, division
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch


def conv_relu_bn(in_channel, out_channel, dirate=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=dirate,
            dilation=dirate,
        ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )


class dconv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(dconv_block, self).__init__()
        self.conv1 = conv_relu_bn(in_ch, out_ch, 1)
        self.dconv1 = conv_relu_bn(out_ch, out_ch // 2, 2)
        self.dconv2 = conv_relu_bn(out_ch // 2, out_ch // 2, 4)
        self.dconv3 = conv_relu_bn(out_ch, out_ch, 2)
        self.conv2 = conv_relu_bn(out_ch * 2, out_ch, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        dx1 = self.dconv1(x1)
        dx2 = self.dconv2(dx1)
        dx3 = self.dconv3(torch.cat((dx1, dx2), dim=1))
        out = self.conv2(torch.cat((x1, dx3), dim=1))
        return out


class CDC_conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        kernel_size=3,
        padding=1,
        dilation=1,
        theta=0.7,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.theta = theta

    def forward(self, x):
        norm_out = self.conv(x)
        [c_out, c_in, kernel_size, kernel_size] = self.conv.weight.shape
        kernel_diff = self.conv.weight.sum(2).sum(2)
        kernel_diff = kernel_diff[:, :, None, None]
        diff_out = F.conv2d(
            input=x,
            weight=kernel_diff,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=0,
        )
        out = norm_out - self.theta * diff_out
        return out
