"""wtie.learning.blocks: 1D 神经网络基础积木模块。

本模块提供卷积、下采样、上采样、输出头与长度对齐等可复用组件，
用于搭建子波学习相关的 1D 网络结构。

边界说明
--------
- 本模块不负责完整网络拓扑定义、训练流程或损失函数计算。
- 本模块仅封装前向算子组合与张量形状变换。

核心公开对象
------------
1. ConvBnLrelu1d: Conv1d + BatchNorm1d + LeakyReLU 基础块。
2. DoubleConv1d: 双卷积块，支持残差平均融合。
3. Down1d / Up1d: 1D 下采样与上采样模块。
4. OutConv1d: 1x1 Conv1d 输出映射层。
5. MatchSizeToRef1d: 按参考张量长度对齐时间采样点数。

Examples
--------
>>> import torch
>>> from wtie.learning.blocks import DoubleConv1d
>>> x = torch.randn(4, 8, 256)
>>> y = DoubleConv1d(8, 16, kernel_size=3, padding=1)(x)
>>> y.shape
torch.Size([4, 16, 256])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from wtie.utils.types_ import Tensor

_padding_mode = "zeros"  #'replicate' REPLICATE IS BROKEN!


class ConvBnLrelu1d(nn.Module):
    """1D 卷积标准块：卷积、归一化与激活。

    该模块按顺序执行 `Conv1d -> BatchNorm1d -> LeakyReLU`，
    输入输出张量均采用 NCL 约定，即 `(batch, channels, n_samples)`。

    Attributes
    ----------
    conv : torch.nn.Conv1d
        一维卷积层（`bias=False`）。
    bn : torch.nn.BatchNorm1d
        批归一化层。
    act : torch.nn.LeakyReLU
        激活函数。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        padding_mode: str = _padding_mode,
        inplace: bool = False,
    ) -> None:
        """初始化 1D 卷积标准块。

        Parameters
        ----------
        in_channels : int
            输入通道数。
        out_channels : int
            输出通道数。
        kernel_size : int
            卷积核长度（采样点数 `n`）。
        padding : int
            两侧填充点数。
        padding_mode : str, default=_padding_mode
            填充模式，默认使用 `zeros`。
        inplace : bool, default=False
            是否原地执行 LeakyReLU。
        """

        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,  # type: ignore
            bias=False,
        )

        self.bn = nn.BatchNorm1d(out_channels)

        self.act = nn.LeakyReLU(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        """执行前向计算。

        Parameters
        ----------
        x : Tensor
            输入张量，shape 为 `(batch, in_channels, n_samples)`。

        Returns
        -------
        Tensor
            输出张量，shape 为 `(batch, out_channels, n_samples_out)`。
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LinearLrelu(nn.Module):
    """线性层与 LeakyReLU 组合块。"""

    def __init__(self, in_features: int, out_features: int, inplace: bool = False):
        """初始化线性激活块。

        Parameters
        ----------
        in_features : int
            输入特征维度。
        out_features : int
            输出特征维度。
        inplace : bool, default=False
            是否原地执行 LeakyReLU。
        """
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.act = nn.LeakyReLU(inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        """执行前向计算。

        Parameters
        ----------
        x : Tensor
            输入特征张量。

        Returns
        -------
        Tensor
            线性变换与激活后的特征张量。
        """
        x = self.linear(x)
        x = self.act(x)
        return x


class DoubleConv1d(nn.Module):
    """双 1D 卷积块，可选残差平均融合。

    当 `residual=True` 时，第一层输出作为残差分支与第二层输出相加后再除以 2，
    以保持数值尺度稳定。

    Attributes
    ----------
    out_channels : int
        输出通道数。
    residual : bool
        是否启用残差平均融合。
    one : ConvBnLrelu1d
        第一层卷积块。
    two : ConvBnLrelu1d
        第二层卷积块。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        padding_mode: str = _padding_mode,
        residual: bool = True,
    ) -> None:
        """初始化双卷积块。

        Parameters
        ----------
        in_channels : int
            输入通道数。
        out_channels : int
            输出通道数。
        kernel_size : int
            卷积核长度（采样点数 `n`）。
        padding : int
            两侧填充点数。
        padding_mode : str, default=_padding_mode
            填充模式。
        residual : bool, default=True
            是否启用残差平均融合。
        """

        super().__init__()
        self.out_channels = out_channels

        self.residual = residual

        ckwargs = dict(kernel_size=kernel_size, padding=padding, padding_mode=padding_mode, inplace=False)

        self.one = ConvBnLrelu1d(in_channels, out_channels, **ckwargs)  # type: ignore
        self.two = ConvBnLrelu1d(out_channels, out_channels, **ckwargs)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        """执行双卷积前向计算。

        Parameters
        ----------
        x : Tensor
            输入张量，shape 为 `(batch, in_channels, n_samples)`。

        Returns
        -------
        Tensor
            输出张量，shape 为 `(batch, out_channels, n_samples_out)`。
        """
        x = self.one(x)
        if self.residual:
            res = x
        x = self.two(x)
        if self.residual:
            x += res
            x /= 2.0
        return x


class SingleConv1d(nn.Module):
    """单层 1D 卷积标准块包装器。"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int, padding_mode: str = _padding_mode
    ) -> None:
        """初始化单卷积块。

        Parameters
        ----------
        in_channels : int
            输入通道数。
        out_channels : int
            输出通道数。
        kernel_size : int
            卷积核长度（采样点数 `n`）。
        padding : int
            两侧填充点数。
        padding_mode : str, default=_padding_mode
            填充模式。
        """

        super().__init__()

        ckwargs = dict(kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)

        # self.out_channels = out_channels

        self.one = ConvBnLrelu1d(in_channels, out_channels, **ckwargs)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        """执行单卷积前向计算。

        Parameters
        ----------
        x : Tensor
            输入张量，shape 为 `(batch, in_channels, n_samples)`。

        Returns
        -------
        Tensor
            输出张量，shape 为 `(batch, out_channels, n_samples_out)`。
        """
        return self.one(x)


class Down1d(nn.Module):
    """1D 下采样块：最大池化后接双卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        kernel_size: int,
        padding: int,
        padding_mode: str = _padding_mode,
    ) -> None:
        """初始化 1D 下采样块。

        Parameters
        ----------
        in_channels : int
            输入通道数。
        out_channels : int
            输出通道数。
        factor : int
            下采样倍数，对应池化核长与步长。
        kernel_size : int
            双卷积的卷积核长度（采样点数 `n`）。
        padding : int
            双卷积填充点数。
        padding_mode : str, default=_padding_mode
            双卷积填充模式。
        """

        super().__init__()

        # self.out_channels = out_channels

        self.mp = nn.MaxPool1d(factor)
        self.conv = DoubleConv1d(in_channels, out_channels, kernel_size, padding, padding_mode)

    def forward(self, x: Tensor) -> Tensor:
        """执行下采样前向计算。

        Parameters
        ----------
        x : Tensor
            输入张量，shape 为 `(batch, in_channels, n_samples)`。

        Returns
        -------
        Tensor
            输出张量，shape 为 `(batch, out_channels, n_samples // factor)`（近似，受边界影响）。
        """
        x = self.mp(x)
        x = self.conv(x)
        return x


class Up1d(nn.Module):
    """1D 上采样块：最近邻上采样后接双卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        kernel_size: int,
        padding: int,
        padding_mode: str = _padding_mode,
    ) -> None:
        """初始化 1D 上采样块。

        Parameters
        ----------
        in_channels : int
            输入通道数。
        out_channels : int
            输出通道数。
        factor : int
            上采样倍数。
        kernel_size : int
            双卷积的卷积核长度（采样点数 `n`）。
        padding : int
            双卷积填充点数。
        padding_mode : str, default=_padding_mode
            双卷积填充模式。
        """

        super().__init__()

        # self.out_channels = out_channels

        self.up = nn.Upsample(scale_factor=factor, mode="nearest")
        # self.up = nn.Upsample(scale_factor=factor, mode='trilinear', align_corners=True)
        self.conv = DoubleConv1d(in_channels, out_channels, kernel_size, padding, padding_mode=padding_mode)

    def forward(self, x: Tensor) -> Tensor:
        """执行上采样前向计算。

        Parameters
        ----------
        x : Tensor
            输入张量，shape 为 `(batch, in_channels, n_samples)`。

        Returns
        -------
        Tensor
            输出张量，shape 为 `(batch, out_channels, n_samples * factor)`（近似，受边界影响）。
        """
        x = self.up(x)
        x = self.conv(x)
        return x


class OutConv1d(nn.Module):
    """1D 输出映射层，使用 1x1 卷积调整通道数。"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int, padding_mode: str = _padding_mode
    ) -> int:
        """初始化输出映射层。

        Parameters
        ----------
        in_channels : int
            输入通道数。
        out_channels : int
            输出通道数。
        kernel_size : int
            保留参数（待确认）；当前实现固定使用 `1x1` 卷积。
        padding : int
            卷积填充点数。
        padding_mode : str, default=_padding_mode
            填充模式。
        """
        super().__init__()
        self.out_channels = out_channels

        # self.conv1 = DoubleConv3d(in_channels, out_channels, kernel_size,
        # padding, padding_mode=padding_mode)

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=padding, padding_mode=padding_mode)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        """执行输出映射前向计算。

        Parameters
        ----------
        x : Tensor
            输入张量，shape 为 `(batch, in_channels, n_samples)`。

        Returns
        -------
        Tensor
            输出张量，shape 为 `(batch, out_channels, n_samples_out)`。
        """
        # x = self.conv1(x)
        x = self.conv(x)
        return x


class MatchSizeToRef1d(nn.Module):
    """将输入张量沿采样轴补零到与参考张量同长度。"""

    def __init__(self) -> None:
        """初始化长度对齐模块。"""
        super().__init__()

    def forward(self, x: Tensor, ref: Tensor) -> Tensor:
        """按参考张量长度对输入做对称填充。

        Parameters
        ----------
        x : Tensor
            待对齐张量，shape 为 `(batch, channels, n_samples_x)`。
        ref : Tensor
            参考张量，shape 为 `(batch, channels, n_samples_ref)`。

        Returns
        -------
        Tensor
            对齐后的张量，长度维与 `ref` 一致，即 `n_samples_ref`。

        Raises
        ------
        RuntimeError
            当 `x` 的长度大于 `ref` 导致负填充时，`torch.nn.functional.pad` 可能抛出。
        """
        # input shape is NCL
        diff = ref.size()[2] - x.size()[2]

        x = F.pad(x, [diff // 2, diff - diff // 2])
        # if you have padding issues, see
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        return x
