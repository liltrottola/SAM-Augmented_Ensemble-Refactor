import torch
import torch.nn as nn
from functools import partial
from torch.nn import Softmax
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
import typing as t
import torch.fft as fft
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from torch import Tensor
import torch.nn.functional as F
from math import sqrt
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x




#EUCB(2)
def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
class EUCB(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False, reduce_channels=True):
        super(EUCB, self).__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        self.reduce_channels = reduce_channels
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels_offset = in_channels // scale ** 2
            out_channels_offset = 2 * groups
        else:
            in_channels_offset = in_channels
            out_channels_offset = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels_offset, out_channels_offset, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels_offset, out_channels_offset, 1)
            constant_init(self.scope, val=0.)

        # 添加通道减半的卷积层
        if self.reduce_channels:
            self.channel_reducer = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            normal_init(self.channel_reducer, std=0.01)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        # i=self.sample(x, offset)
        # print("lp",i.shape)
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        # i=self.sample(x, offset)
        # print("pl",i.shape)
        return self.sample(x, offset)

    def forward(self, x):
        # 先进行上采样
        if self.style == 'pl':
            upsampled = self.forward_pl(x)
        else:
            upsampled = self.forward_lp(x)

        # 如果需要减半通道数
        if self.reduce_channels:
            return self.channel_reducer(upsampled)
        return upsampled


#   Large-kernel grouped attention gate (SFAM)
class SFAM(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super(SFAM, self).__init__()

        if kernel_size == 1:
            groups = 1
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=groups,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)

        return x * psi


#-----------------------------------------------------------------------------------------------------------------------
class ConvMlp(nn.Module):
    def __init__(self, in_features, out_features, act_layer=nn.ReLU):
        super().__init__()

        hidden_features = in_features // 2  # 隐藏层通道数是输入通道数的一半
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)  # 1x1卷积，减少通道数
        self.act = act_layer()  # 激活函数（默认ReLU）
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)  # 1x1卷积，恢复输出通道数

    def forward(self, x):
        x = self.fc1(x)  # 经过第一个卷积
        x = self.act(x)  # 激活函数
        x = self.fc2(x)  # 经过第二个卷积
        return x

# 定义多尺度空洞卷积模块 (MSDC)
class MS(nn.Module):
    def __init__(self, in_channels):
        super(MS, self).__init__()
        # 使用空洞卷积来增加感受野，捕捉不同尺度的特征
        self.dilate1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )  # 3*3卷积， dilation=1（普通卷积）

        self.dilate3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )  # dilation=3

        self.dilate5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=5, padding=5),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )  # dilation=5

        self.dilate7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=7, padding=7),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )  # dilation=7

        # 多尺度卷积结果的处理
        self.mlp = ConvMlp(4*in_channels, in_channels)  # 将4倍的通道数压缩回原通道数

    def forward(self, x):
        dilate1_out = self.dilate1(x)  # dilation=1
        dilate2_out = self.dilate3(x)  # dilation=3
        dilate3_out = self.dilate5(x)  # dilation=5
        dilate4_out = self.dilate7(x)  # dilation=7

        cnn_out = torch.cat([dilate1_out, dilate2_out, dilate3_out, dilate4_out], dim=1)  # 拼接不同尺度的输出
        out = self.mlp(cnn_out)  # 通过MLP进行通道整合
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_ch, ratio=16, alpha=0.5):
        super().__init__()
        self.alpha = alpha

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_ch, in_ch // ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_ch // ratio, in_ch, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        mix_pool = self.alpha * avg_pool + (1 - self.alpha) * max_pool

        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        mix_out = self.fc2(self.relu(self.fc1(mix_pool)))

        out = avg_out + max_out + mix_out
        ca = self.sigmoid(out)  # [B, C, 1, 1]

        return x * ca   # ⭐ GradCAM 可视化这个激活即可

# 定义动态池化协同注意力模块 (ARFM)
class ARFM(nn.Module):
    def __init__(self, in_ch, ratio=16, alpha=0.5):
        """
        初始化函数
        Args:
            in_ch: 输入通道数
            alpha: 平衡参数 (0 <= alpha <= 1)，用于混合池化
        """
        super(ARFM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化，输出为1x1大小
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化，输出为1x1大小
        self.alpha = alpha  # 平衡参数，控制池化策略的加权

        # 通道注意力机制（通过1x1卷积）
        self.fc1 = nn.Conv2d(in_ch, in_ch // ratio, 1, bias=False)  # 降维
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_ch // ratio, in_ch, 1, bias=False)  # 恢复维度
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数，用于生成权重

        # 引入多尺度空洞卷积模块
        self.msdc = MS(in_ch)

        # 空间注意力计算（使用7x7卷积）
        self.conv1 = nn.Conv2d(2, 1, 7, padding=7 // 2, bias=False)  # 7x7卷积，用于计算空间注意力
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

        self.ca = ChannelAttention(in_ch, ratio, alpha)

    def forward(self, x):
        # 计算通道注意力
        input = x  # 输入保持不变，后续进行加权
        # avg_pool = self.avg_pool(x)  # 平均池化
        # max_pool = self.max_pool(x)  # 最大池化
        # mix_pool = self.alpha * avg_pool + (1 - self.alpha) * max_pool  # 使用alpha加权平均池化与最大池化的结果
        #
        # # 计算每种池化方式的通道注意力
        # avg_out = self.fc2(self.relu1(self.fc1(avg_pool)))  # 平均池化的通道注意力
        # max_out = self.fc2(self.relu1(self.fc1(max_pool)))  # 最大池化的通道注意力
        # mix_out = self.fc2(self.relu1(self.fc1(mix_pool)))  # 混合池化的通道注意力
        #
        # out_pool = self.sigmoid(avg_out + max_out + mix_out)  # 将所有池化方式的结果相加并通过sigmoid计算最终权重
        # ca_out = input * out_pool  # 对输入的特征图进行通道加权
        ca_out = self.ca(x)

        # 多尺度特征提取
        input = self.msdc(ca_out)  # 通过MSDC提取多尺度特征

        # 计算空间注意力
        avg_out = torch.mean(input, dim=1, keepdim=True)  # 在通道维度上进行平均池化
        max_out, _ = torch.max(input, dim=1, keepdim=True)  # 在通道维度上进行最大池化
        sa_out = torch.cat([avg_out, max_out], dim=1)  # 将两个池化结果拼接
        sa_out = self.conv1(sa_out)  # 通过7x7卷积计算空间注意力
        sa_out = self.sigmoid(sa_out)  # 通过sigmoid计算最终空间注意力

        out = input * sa_out  # 最终输出特征图与空间注意力加权后的特征图相乘
        return out

#--------------------------------------------------------------------------------------------------------------------------------
class LayerNorm(nn.Module):  # 定义一个 LayerNorm 类，继承自 nn.Module
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 初始化可学习的权重参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 初始化可学习的偏置参数
        self.eps = eps  # 设置一个非常小的常数用于数值稳定性
        self.data_format = data_format  # 数据格式，支持 "channels_last" 或 "channels_first"
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError  # 如果数据格式不正确，抛出异常
        self.normalized_shape = (normalized_shape,)  # 将 normalized_shape 转化为元组

    def forward(self, x):  # 定义前向传播函数
        if self.data_format == "channels_last":  # 如果数据格式为 "channels_last"
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)  # 使用 PyTorch 内置的 LayerNorm
        elif self.data_format == "channels_first":  # 如果数据格式为 "channels_first"
            u = x.mean(1, keepdim=True)  # 计算输入张量沿着通道维度的均值
            s = (x - u).pow(2).mean(1, keepdim=True)  # 计算输入张量的方差
            x = (x - u) / torch.sqrt(s + self.eps)  # 标准化输入
            x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 应用权重和偏置
            return x  # 返回标准化后的结果

class InceptionDWConv2d(nn.Module):  # 定义一个 Inception 风格的大核分解深度可分离卷积模块
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.25):
        super().__init__()
        gc = int(in_channels * branch_ratio)  # 计算每个卷积分支的通道数
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)  # 定义深度可分离卷积，处理空间维度
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)  # 深度可分离卷积，处理宽度维度
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)  # 深度可分离卷积，处理高度维度
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)  # 计算拆分输入特征图的索引，分成 4 个部分

    def forward(self, x):  # 定义前向传播函数
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)  # 将输入张量拆分成 4 个部分
        return torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)  # 将各个分支的卷积结果合并返回

class MSFD(nn.Module):  # 定义多尺度大核分解注意力（MSFD）模块
    def __init__(self, in_c):
        super().__init__()
        self.in_2c = 2 * in_c  # 定义输入通道数的 2 倍，用于通道扩展
        self.split_c = in_c // 4  # 将输入通道数分为 4 份
        self.i_c = self.split_c  # 设置每个分支的输入通道数
        self.norm = LayerNorm(in_c, data_format='channels_first')  # 实例化一个 LayerNorm 层
        self.scale = nn.Parameter(torch.zeros((1, in_c , 1, 1)), requires_grad=True)  # 定义一个可学习的缩放参数

        # 定义多个不同尺度的大核卷积
        self.LKA9 = nn.Sequential(  # 9x9 大核卷积分支
            InceptionDWConv2d(self.i_c, 3, band_kernel_size=9),  # 使用 9x9 大核卷积
            nn.Conv2d(self.i_c, self.i_c, 1, 1, 0))  # 使用 1x1 卷积调整输出通道
        self.LKA7 = nn.Sequential(  # 7x7 大核卷积分支
            InceptionDWConv2d(self.i_c, 3, band_kernel_size=7),  # 使用 7x7 大核卷积
            nn.Conv2d(self.i_c, self.i_c, 1, 1, 0))  # 使用 1x1 卷积调整输出通道
        self.LKA5 = nn.Sequential(  # 5x5 大核卷积分支
            InceptionDWConv2d(self.i_c, 3, band_kernel_size=5),  # 使用 5x5 大核卷积
            nn.Conv2d(self.i_c, self.i_c, 1, 1, 0))  # 使用 1x1 卷积调整输出通道
        self.LKA3 = nn.Sequential(  # 3x3 大核卷积分支
            InceptionDWConv2d(self.i_c, 3, band_kernel_size=3),  # 使用 3x3 核卷积
            nn.Conv2d(self.i_c, self.i_c, 1, 1, 0))  # 使用 1x1 卷积调整输出通道

        # 定义用于通道间的卷积操作，分别使用不同大小的卷积核
        self.k3 = nn.Conv2d(self.split_c, self.split_c, 3, 1, 1, groups=self.split_c)  # 3x3 卷积
        self.k5 = nn.Conv2d(self.split_c, self.split_c, 5, 1, 5 // 2, groups=self.split_c)  # 5x5 卷积
        self.k7 = nn.Conv2d(self.split_c, self.split_c, 7, 1, 7 // 2, groups=self.split_c)  # 7x7 卷积
        self.k9 = nn.Conv2d(self.split_c, self.split_c, 9, 1, 9 // 2, groups=self.split_c)  # 9x9 卷积

        # 用于调整输入通道数的 1x1 卷积操作
        self.proj_first = nn.Sequential(
            nn.Conv2d(in_c, self.in_2c, 1, 1, 0))  # 1x1 卷积，用于扩大通道数

        self.proj_last = nn.Sequential(  # 用于恢复原始通道数的 1x1 卷积
            nn.Conv2d(in_c, in_c, 1, 1, 0))  # 1x1 卷积，用于还原输出通道数


    def forward(self, x):  # 定义前向传播函数
        shortcut = x  # 保存输入特征图，用于跳跃连接
        x = self.norm(x)  # 对输入进行层归一化
        x = self.proj_first(x)  # 通过 1x1 卷积扩展通道数

        a, x = torch.chunk(x, 2, dim=1)  # 将通道维度分成两部分，一部分用于注意力计算，另一部分继续处理
        a_1, a_2, a_3, a_4 = torch.chunk(a, 4, dim=1)  # 将注意力部分进一步拆分为 4 个部分

        # 计算各个卷积分支的特征，并加权融合
        a = torch.cat([self.LKA3(a_1) * self.k3(a_1),
                       self.LKA5(a_2) * self.k5(a_2),
                       self.LKA7(a_3) * self.k7(a_3),
                       self.LKA9(a_4) * self.k9(a_4)], dim=1)  # 各个卷积分支结果拼接

        # 通过注意力加权后的特征对 x 进行变换，并加上跳跃连接
        x = self.proj_last(x * a) * self.scale + shortcut  # 应用 1x1 卷积，结合缩放参数并加上原输入
        return x  # 返回最终输出

#-----------------------------------------------------------------------------------------------------------------------


#   Efficient multi-scale convolutional attention decoding (EMCAD)
class RFA(nn.Module):
    def __init__(self, channels=[1152,576,288,144], kernel_sizes=[1, 3, 5], expansion_factor=6, dw_parallel=True,
                 add=True, lgag_ks=3, activation='relu6', input_resolution=[11, 22, 44, 88]):
        super(RFA, self).__init__()
        eucb_ks = 3  # kernel size for eucb

        self.eucb3 = EUCB(in_channels=channels[0], scale=2, style='pl')

        self.lgag3 = SFAM(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2, kernel_size=lgag_ks,
                          groups=channels[1] // 2)


        self.eucb2 = EUCB(in_channels=channels[1], scale=2, style='pl')

        self.lgag2 = SFAM(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2, kernel_size=lgag_ks,
                          groups=channels[2] // 2)

        self.eucb1 = EUCB(in_channels=channels[2], scale=2, style='pl')

        self.lgag1 = SFAM(F_g=channels[3], F_l=channels[3], F_int=int(channels[3] / 2), kernel_size=lgag_ks,
                          groups=int(channels[3] / 2))


        self.ARFM4 = ARFM(in_ch=channels[0])
        self.ARFM3 = ARFM(in_ch=channels[1])
        self.ARFM2 = ARFM(in_ch=channels[2])
        self.ARFM1 = ARFM(in_ch=channels[3])

        self.mkda4 = MSFD(channels[0])
        self.mkda3 = MSFD(channels[1])
        self.mkda2 = MSFD(channels[2])
        self.mkda1 = MSFD(channels[3])


    def forward(self, x, skips):

        #   4
        d4 = self.ARFM4(x)

        d4 = self.mkda4(d4)

        # EUCB3
        d3 = self.eucb3(d4)

        # SFAM3
        x3 = self.lgag3(g=d3, x=skips[0])

        # Additive aggregation 3
        d3 = d3 + x3

        #   3
        d3 = self.ARFM3(d3)

        d3 = self.mkda3(d3)

        # EUCB2
        d2 = self.eucb2(d3)

        # SFAM2
        x2 = self.lgag2(g=d2, x=skips[1])

        # Additive aggregation 2
        d2 = d2 + x2

        #   2
        d2 = self.ARFM2(d2)

        d2 = self.mkda2(d2)

        # EUCB1
        d1 = self.eucb1(d2)

        # SFAM1
        x1 = self.lgag1(g=d1, x=skips[2])

        # Additive aggregation 1
        d1 = d1 + x1

        #   1
        d1 = self.ARFM1(d1)

        d1 = self.mkda1(d1)

        return [d4, d3, d2, d1]
