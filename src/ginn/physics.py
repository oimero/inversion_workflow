"""ginn.physics — 物理正演模块（全可微分）。

将声阻抗 AI 转换为合成地震记录 d_syn：
    AI → 反射率 r → r ✱ wavelet → d_syn

整个过程在 PyTorch 计算图内完成，支持自动微分反向传播。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ForwardModel(nn.Module):
    """物理正演：阻抗 → 反射率 → 合成地震记录。

    Parameters
    ----------
    wavelet : np.ndarray
        1D 子波数组，shape ``(n_w,)``。
        子波内部会自动查找绝对最大值位置以确定不对称 padding。

    Attributes
    ----------
    wavelet_kernel : Tensor
        注册为 buffer 的翻转子波卷积核，shape ``(1, 1, n_w)``，不参与梯度更新。
    pad_left : int
        反射率序列左侧填充长度。
    pad_right : int
        反射率序列右侧填充长度。
    """

    def __init__(self, wavelet: np.ndarray) -> None:
        super().__init__()

        wavelet = np.asarray(wavelet, dtype=np.float32).ravel()
        n_w = wavelet.size

        # 找到子波绝对最大值（近似中心）
        i_max = int(np.argmax(np.abs(wavelet)))
        self.pad_left = i_max
        self.pad_right = n_w - 1 - i_max

        # F.conv1d 执行的是互相关，需要翻转子波使其成为真正的卷积
        wavelet_flipped = wavelet[::-1].copy()
        kernel = torch.from_numpy(wavelet_flipped).view(1, 1, -1)
        self.register_buffer("wavelet_kernel", kernel)

    def reflectivity(self, impedance: Tensor) -> Tensor:
        """从声阻抗计算反射率。

        公式：``r[t] = (AI[t+1] - AI[t]) / (AI[t+1] + AI[t])``

        Parameters
        ----------
        impedance : Tensor
            声阻抗，shape ``(B, 1, T)``。

        Returns
        -------
        Tensor
            反射率，shape ``(B, 1, T-1)``。
        """
        ai_upper = impedance[..., :-1]  # AI[t]
        ai_lower = impedance[..., 1:]  # AI[t+1]
        # 加 eps 防止除零
        r = (ai_lower - ai_upper) / (ai_lower + ai_upper + 1e-10)
        return r

    def convolve(self, reflectivity: Tensor) -> Tensor:
        """将反射率与子波做一维卷积，输出等长合成地震记录。

        Parameters
        ----------
        reflectivity : Tensor
            反射率，shape ``(B, 1, T_r)``。

        Returns
        -------
        Tensor
            合成地震记录，shape ``(B, 1, T_r)``，与反射率等长。
        """
        # 不对称 padding，使输出与反射率等长
        r_padded = F.pad(reflectivity, (self.pad_left, self.pad_right), mode="constant", value=0.0)
        # conv1d + padding=0 → 输出长度 = len(r_padded) - n_w + 1 = len(r)
        d_syn = F.conv1d(r_padded, self.wavelet_kernel, padding=0)  # type: ignore
        return d_syn

    def forward(self, impedance: Tensor) -> Tensor:
        """完整正演：阻抗 → 合成地震记录。

        Parameters
        ----------
        impedance : Tensor
            声阻抗，shape ``(B, 1, T)``。

        Returns
        -------
        Tensor
            合成地震记录，shape ``(B, 1, T)``，与输入等长。
            反射率比阻抗少 1 点，末尾补零后卷积以保持等长。
        """
        r = self.reflectivity(impedance)  # (B, 1, T-1)

        # 在反射率末尾补 1 个零，使其长度回到 T
        r = F.pad(r, (0, 1), mode="constant", value=0.0)  # (B, 1, T)

        d_syn = self.convolve(r)  # (B, 1, T)
        return d_syn
