"""ginn.loss — 掩码 MAE 损失函数 + 残差正则化。

组合两项损失:
1. 掩码 MAE: 仅在层位顶底之间计算合成地震与观测地震的平均绝对误差
2. 残差 L2 正则: 防止网络输出的阻抗残差尺度发散（因反射率的比值不变性）
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GINNLoss(nn.Module):
    """GINN 组合损失: 掩码 MAE + 残差 L2 正则化。

    .. math::
        \\mathcal{L} = \\underbrace{\\frac{\\sum |d_{syn} - d_{obs}| \\cdot m}{\\sum m}}_{\\text{waveform MAE}}
        + \\lambda \\cdot \\underbrace{\\frac{\\sum \\Delta^2 \\cdot m}{\\sum m}}_{\\text{residual L2 reg}}

    Parameters
    ----------
    lambda_reg : float
        残差 L2 正则化权重。默认 0.1。
        物理意义: 强制阻抗不偏离低频模型太远, 解决反射率比值不变性
        导致的绝对尺度不确定性。

    Notes
    -----
    反射率公式 ``r = (AI[t+1] - AI[t]) / (AI[t+1] + AI[t])`` 对 AI 的全局
    缩放是不变的。如果不加约束, 网络可以输出任意大的残差 Δ, 只要相对
    变化产生正确的反射率即可。L2 正则化将 Δ 锚定在零附近, 确保阻抗
    物理量级与低频模型一致。
    """

    def __init__(self, lambda_reg: float = 0.1) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(
        self,
        d_syn: Tensor,
        d_obs: Tensor,
        mask: Tensor,
        residual: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """计算组合损失。

        Parameters
        ----------
        d_syn : Tensor
            合成地震记录, shape ``(B, 1, T)``。
        d_obs : Tensor
            观测地震记录, shape ``(B, 1, T)``。
        mask : Tensor
            布尔掩码, shape ``(B, 1, T)``, True 表示有效区域。
        residual : Tensor
            网络输出的阻抗残差 Δ, shape ``(B, 1, T)``。

        Returns
        -------
        total_loss : Tensor
            标量总损失。
        loss_dict : dict
            包含各分项的浮点值, 用于日志记录。
        """
        mask_f = mask.float()
        n_valid = mask_f.sum().clamp(min=1.0)

        # 波形 MAE
        waveform_mae = ((d_syn - d_obs).abs() * mask_f).sum() / n_valid

        # 残差 L2 正则化（仅在掩码区域内）
        residual_l2 = (residual.pow(2) * mask_f).sum() / n_valid

        # 总损失
        total_loss = waveform_mae + self.lambda_reg * residual_l2

        reg_term = self.lambda_reg * residual_l2

        loss_dict = {
            "total": total_loss.item(),
            "waveform_mae": waveform_mae.item(),
            "residual_l2": residual_l2.item(),
            "reg_term": reg_term.item(),
            "lambda_reg": float(self.lambda_reg),
        }

        return total_loss, loss_dict
