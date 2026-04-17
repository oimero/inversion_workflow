"""ginn.model — 1D 膨胀卷积残差网络（Dilated ResNet）。

网络接收 2 通道输入（地震 + LFM），输出 1 通道阻抗残差。
阻抗残差结合 LFM 即得到完整声阻抗。
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class DilatedResBlock(nn.Module):
    """膨胀卷积残差块。

    结构：Conv-BN-ReLU-Conv-BN + identity skip → ReLU

    Parameters
    ----------
    channels : int
        输入与输出通道数（保持不变）。
    kernel_size : int
        卷积核长度。
    dilation : int
        膨胀倍率；padding = dilation 保证输入输出等长。
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # 保持序列等长

        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播。

        Parameters
        ----------
        x : Tensor
            输入张量，shape ``(B, C, T)``。

        Returns
        -------
        Tensor
            输出张量，shape ``(B, C, T)``，与输入等长。
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class DilatedResNet1D(nn.Module):
    """1D 膨胀卷积 ResNet — GINN 阻抗残差预测骨干网络。

    Architecture
    ------------
    ::

        Conv1d(in→hidden) → BN → ReLU
        → ResBlock(d=1) → ResBlock(d=2) → ... → ResBlock(d=128)
        → Conv1d(hidden→out, k=1)

    Parameters
    ----------
    in_channels : int
        输入通道数，默认 2（地震 + LFM）。
    hidden_channels : int
        残差块内部通道数，默认 64。
    out_channels : int
        输出通道数，默认 1（阻抗残差）。
    dilations : sequence of int
        各残差块的膨胀倍率。
    kernel_size : int
        残差块卷积核长度，默认 3。
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        out_channels: int = 1,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        # ── 头部：升维 ──
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # ── 主体：膨胀残差块 ──
        self.res_blocks = nn.Sequential(
            *[DilatedResBlock(hidden_channels, kernel_size=kernel_size, dilation=d) for d in dilations]
        )

        # ── 平滑层：消除膨胀卷积的网格效应 ──
        self.smooth = nn.Sequential(
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # ── 尾部：降维到输出通道 ──
        self.tail = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """前向传播。

        Parameters
        ----------
        x : Tensor
            输入张量，shape ``(B, in_channels, T)``。

        Returns
        -------
        Tensor
            阻抗残差，shape ``(B, out_channels, T)``。
        """
        x = self.head(x)
        x = self.res_blocks(x)
        x = self.smooth(x)
        x = self.tail(x)
        return x

    def count_parameters(self) -> int:
        """返回可训练参数总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
