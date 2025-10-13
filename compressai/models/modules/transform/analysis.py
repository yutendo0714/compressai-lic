import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import subpel_conv3x3
from ..layers.conv import conv1x1, conv3x3, conv, deconv
from ..layers.res_blk import *


class AnalysisTransform(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.analysis_transform = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, M, stride=2)
        )

    def forward(self, x):
        x = self.analysis_transform(x)

        return x


class HyperAnalysis(nn.Module):
    """
    Local reference
    """
    def __init__(self, M=192, N=192):
        super().__init__()
        self.M = M
        self.N = N
        self.reduction = nn.Sequential(
            conv3x3(M, N),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
            nn.GELU(),
            conv3x3(N, N),
            nn.GELU(),
            conv3x3(N, N, stride=2),
        )

    def forward(self, x):
        x = self.reduction(x)

        return x