# @Time    : 2023/12/26 13:14
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : SeRankDet.py
# @Software: PyCharm
from __future__ import print_function, division

import einops
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import einsum

from model.SeRankDet.tools import conv_relu_bn, CDC_conv


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CDC_conv(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# class new_conv_block(nn.Module):
#     """
#     Convolution Block
#     """
#
#     def __init__(self, in_ch, out_ch):
#         super(new_conv_block, self).__init__()
#         self.conv_layer = nn.Sequential(
#             conv_relu_bn(in_ch, in_ch, 1),
#             conv_relu_bn(in_ch, out_ch, 1),
#             conv_relu_bn(out_ch, out_ch, 1)
#         )
#         self.cdc_layer = nn.Sequential(
#             CDC_conv(in_ch, out_ch // 2),
#             nn.BatchNorm2d(out_ch // 2),
#             nn.ReLU(inplace=True)
#         )
#         self.dconv_layer = nn.Sequential(
#             conv_relu_bn(in_ch, out_ch, 2),
#             conv_relu_bn(out_ch, out_ch // 2, 4),
#             conv_relu_bn(out_ch // 2, out_ch // 2, 2)
#         )
#         self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
#         self.conv = nn.Conv2d(out_ch // 2, out_ch, 1)
#
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.cdcd_conv = conv_relu_bn(out_ch, out_ch, 1)
#         self.final_conv = conv_relu_bn(out_ch, out_ch, 1)
#
#
#     def forward(self, x):
#         conv_out = self.conv_layer(x)
#         cdc_out = self.cdc_layer(x)
#         dconv_out = self.dconv_layer(x)
#
#         cdcd_out = torch.concat([cdc_out, dconv_out], dim=1)
#         cdcd_out = self.cdcd_conv(cdcd_out)
#
#         attn = torch.concat([cdc_out, dconv_out], dim=1)
#         avg_attn = torch.mean(attn, dim=1, keepdim=True)
#         max_attn, _ = torch.max(attn, dim=1, keepdim=True)
#         agg = torch.concat([avg_attn, max_attn], dim=1)
#         sig = self.conv_squeeze(agg).sigmoid()
#         attn = cdc_out * sig[:, 0, :, :].unsqueeze(1) + dconv_out * sig[:, 1, :, :].unsqueeze(1)
#         attn = self.conv(attn)
#
#         out = conv_out * attn * self.gamma + conv_out + cdcd_out
#         # out = self.final_conv(out)
#         return out


class Neck(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Neck, self).__init__()
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, enc, dec):
        x = torch.cat([enc, dec], dim=1)
        out_x = torch.cat([enc, dec], dim=1)
        attn = torch.cat([enc, dec], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.concat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = enc * sig[:, 0, :, :].unsqueeze(1) + dec * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        out = out_x * attn
        return self.gamma * out + x


class new_conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(new_conv_block, self).__init__()
        self.conv_layer = nn.Sequential(
            conv_relu_bn(in_ch, in_ch, 1),
            conv_relu_bn(in_ch, out_ch, 1),
            conv_relu_bn(out_ch, out_ch, 1),
        )
        self.cdc_layer = nn.Sequential(
            CDC_conv(in_ch, out_ch), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.dconv_layer = nn.Sequential(
            conv_relu_bn(in_ch, in_ch, 2),
            conv_relu_bn(in_ch, out_ch, 4),
            conv_relu_bn(out_ch, out_ch, 2),
        )
        self.final_layer = conv_relu_bn(out_ch * 3, out_ch, 1)

    def forward(self, x):
        conv_out = self.conv_layer(x)
        cdc_out = self.cdc_layer(x)
        dconv_out = self.dconv_layer(x)
        out = torch.concat([conv_out, cdc_out, dconv_out], dim=1)
        out = self.final_layer(out)
        return out


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


# self.active = torch.nn.Sigmoid()
def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode="bilinear")
    return src


class PositionalEncoding(nn.Module):
    def __init__(self, image_size, embedding_size):
        super(PositionalEncoding, self).__init__()
        pos_enc = torch.zeros(image_size, embedding_size)
        position = torch.arange(0, image_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size)
        )
        pos_enc[:, 0::2].copy_(torch.sin(position * div_term))
        pos_enc[:, 1::2].copy_(torch.cos(position * div_term))
        self.pos_enc = nn.Parameter(pos_enc, requires_grad=False)
        self.mp = {64: 512, 128: 256, 256: 128, 512: 64, 1024: 32}

    def forward(self, x):
        b, c, h, w = x.size()
        max_feature = x.view(b, c, -1)
        _, topk_indices = torch.topk(max_feature, k=self.mp[c], dim=2)
        topk_indices, _ = torch.sort(topk_indices, dim=2)
        max_feature = torch.gather(max_feature, 2, topk_indices)

        indices_x = topk_indices // w
        indices_y = topk_indices % w
        pos_embed = self.pos_enc[indices_x, indices_y]
        max_feature = pos_embed + max_feature

        return max_feature


def compute_index(x, y, max_x):
    return x + y * max_x


class MaxChannel(nn.Module):
    def __init__(self, in_ch, num_embeddings):
        super(MaxChannel, self).__init__()
        self.fc1 = nn.Linear(in_features=in_ch, out_features=in_ch * 2, bias=False)
        self.fc2 = nn.Linear(in_features=in_ch, out_features=in_ch * 2, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.pos_enc = PositionalEncoding(num_embeddings, num_embeddings)
        # self.embedding_layer = nn.Embedding(num_embeddings, 1)
        # self.num_emb = num_embeddings
        self.mp = {64: 512, 128: 256, 256: 128, 512: 64, 1024: 32}

    def forward(self, x):
        b, c, h, w = x.size()
        # max_feature = x.view(b, c, -1)
        # _, topk_indices = torch.topk(max_feature, k=self.mp[c], dim=2)
        # topk_indices, _ = torch.sort(topk_indices, dim=2)
        # max_feature = torch.gather(max_feature, 2, topk_indices)
        max_feature = self.pos_enc(x)

        # indices_x = topk_indices // w
        # indices_y = topk_indices % w
        # indices = compute_index(indices_x, indices_y, 2)
        # output = self.embedding_layer(topk_indices // self.num_emb)
        # output = output.squeeze(dim=3)
        # max_feature = output + max_feature

        q = self.fc1(max_feature)
        k = self.fc2(max_feature)
        k = einops.rearrange(k, "b c m -> b m c")
        attend = torch.matmul(q, k)
        attend = (attend - torch.mean(attend)) / (torch.std(attend) + 1e-5)
        attention = self.attend(attend)
        new_x = einops.rearrange(x, "b c h w -> b c (h w)")
        res = torch.matmul(attention, new_x)
        return einops.rearrange(res, "b c (h w) -> b c h w", h=h) + x


class SeRankDet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3, out_ch=1, deep_supervision=True, **kwargs):
        super(SeRankDet, self).__init__()
        self.deep_supervision = deep_supervision
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = new_conv_block(in_ch, filters[0])
        self.Conv2 = new_conv_block(filters[0], filters[1])
        self.Conv3 = new_conv_block(filters[1], filters[2])
        self.Conv4 = new_conv_block(filters[2], filters[3])
        self.Conv5 = new_conv_block(filters[3], filters[4])

        self.max_channel1 = MaxChannel(512, 512)
        self.max_channel2 = MaxChannel(256, 256)
        self.max_channel3 = MaxChannel(128, 128)
        self.max_channel4 = MaxChannel(64, 64)
        self.max_channel5 = MaxChannel(32, 32)

        self.neck5 = Neck(filters[3], filters[4])
        self.neck4 = Neck(filters[2], filters[3])
        self.neck3 = Neck(filters[1], filters[2])
        self.neck2 = Neck(filters[0], filters[1])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        # --------------------------------------------------------------------------------------------------------------
        self.conv5 = nn.Conv2d(filters[4], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(filters[3], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(filters[2], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters[1], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1)
        # --------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        e1 = self.Conv1(x)
        e1 = self.max_channel1(e1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e2 = self.max_channel2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3 = self.max_channel3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4 = self.max_channel4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        e5 = self.max_channel5(e5)

        d5 = self.Up5(e5)
        # d5 = torch.cat((e4, d5), dim=1)
        d5 = self.neck5(e4, d5)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        # d4 = torch.cat((e3, d4), dim=1)
        d4 = self.neck4(e3, d4)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # d3 = torch.cat((e2, d3), dim=1)
        d3 = self.neck3(e2, d3)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        # d2 = torch.cat((e1, d2), dim=1)
        d2 = self.neck2(e1, d2)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        d_s1 = self.conv1(d2)
        d_s2 = self.conv2(d3)
        d_s2 = _upsample_like(d_s2, d_s1)
        d_s3 = self.conv3(d4)
        d_s3 = _upsample_like(d_s3, d_s1)
        d_s4 = self.conv4(d5)
        d_s4 = _upsample_like(d_s4, d_s1)
        d_s5 = self.conv5(e5)
        d_s5 = _upsample_like(d_s5, d_s1)
        if self.deep_supervision:
            outs = [d_s1, d_s2, d_s3, d_s4, d_s5, out]
        else:
            outs = out
        # d1 = self.active(out)

        return outs


if __name__ == "__main__":
    x = torch.rand(8, 3, 512, 512)
    model = SeRankDet()
    outs = model(x)
    for out in outs:
        print(out.size())
