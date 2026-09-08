"""Reusable building blocks shared across architectures."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
    def forward(self, x):
        w = self.conv(self.pool(x))
        return x * w

class NAFBlock(nn.Module):
    def __init__(self, channels: int, ffn_expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        dw_channels = channels * 2
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1)
        self.dwconv = nn.Conv2d(dw_channels, dw_channels, kernel_size=3, padding=1, groups=dw_channels)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channels // 2)
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1)

        ffn_channels = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.drop_path = drop_path

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        x = x + y * self.gamma
        return x
