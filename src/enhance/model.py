"""Backbone models for stage-2 resolution enhancement."""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn
from torch import Tensor


class DilatedResBlock(nn.Module):
    """1D dilated residual block."""

    def __init__(self, channels: int, *, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
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
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + identity)


class DilatedResNet1D(nn.Module):
    """Small 1D dilated ResNet predicting ``delta_log_ai``."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        hidden_channels: int = 64,
        out_channels: int = 1,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[
                DilatedResBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.smooth = nn.Sequential(
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.head(x)
        x = self.res_blocks(x)
        x = self.smooth(x)
        return self.tail(x)

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


__all__ = ["DilatedResBlock", "DilatedResNet1D"]
