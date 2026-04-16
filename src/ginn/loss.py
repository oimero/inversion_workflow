"""ginn.loss — 掩码 MAE 损失函数 + 残差正则化。

组合三项损失：
1. 掩码 MAE：仅在层位顶底之间计算合成地震与观测地震的平均绝对误差
2. 残差 L2 正则：防止网络输出的阻抗残差尺度发散（因反射率的比值不变性）
3. 残差 TV 正则：抑制沿时间轴的一阶高频振荡
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GINNLoss(nn.Module):
    """GINN 组合损失：掩码 MAE + 残差 L2/TV 正则化。

    .. math::
        \\mathcal{L} = \\underbrace{\\frac{\\sum |d_{syn} - d_{obs}| \\cdot m}{\\sum m}}_{\\text{waveform MAE}}
        + \\lambda_{reg} \\cdot \\underbrace{\\frac{\\sum \\Delta^2 \\cdot m}{\\sum m}}_{\\text{residual L2 reg}}
        + \\lambda_{tv} \\cdot \\underbrace{\\frac{\\sum |\\Delta[t+1] - \\Delta[t]| \\cdot w}{\\sum w}}_{\\text{residual TV reg}}

    Parameters
    ----------
    lambda_reg : float
        残差 L2 正则化权重。默认 0.1。
        物理意义：强制阻抗不偏离低频模型太远，解决反射率比值不变性
        导致的绝对尺度不确定性。
    lambda_tv : float
        残差 TV 正则化权重。默认 0.0。
        物理意义：惩罚沿时间轴的一阶差分，抑制会被子波滤掉的高频 null-space ringing。

    Notes
    -----
    反射率公式 ``r = (AI[t+1] - AI[t]) / (AI[t+1] + AI[t])`` 对 AI 的全局
    缩放是不变的。如果不加约束，网络可以输出任意大的残差 Δ，只要相对
    变化产生正确的反射率即可。L2 正则化将 Δ 锚定在零附近，确保阻抗
    物理量级与低频模型一致。
    """

    def __init__(self, lambda_reg: float = 0.1, lambda_tv: float = 0.0) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_tv = lambda_tv

    def forward(
        self,
        d_syn: Tensor,
        d_obs: Tensor,
        waveform_mask: Tensor,
        residual_mask: Tensor,
        residual: Tensor,
        taper_weight: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """计算组合损失。

        Parameters
        ----------
        d_syn : Tensor
            合成地震记录，shape ``(B, 1, T)``。
        d_obs : Tensor
            观测地震记录，shape ``(B, 1, T)``。
        waveform_mask : Tensor
            波形误差掩码，shape ``(B, 1, T)``，True 表示参与 waveform loss 的区域。
        residual_mask : Tensor
            残差正则掩码，shape ``(B, 1, T)``，True 表示参与 residual L2 的区域。
        residual : Tensor
            网络输出的阻抗残差 Δ，shape ``(B, 1, T)``。
        taper_weight : Tensor
            residual 的 core+halo taper 权重，shape ``(B, 1, T)``。
            TV 项使用相邻点对的最小权重作为 pair-wise 支撑。

        Returns
        -------
        total_loss : Tensor
            标量总损失。
        loss_dict : dict
            包含各分项的浮点值，用于日志记录。
        """
        waveform_mask_f = waveform_mask.float()
        residual_mask_f = residual_mask.float()
        n_waveform_valid = waveform_mask_f.sum().clamp(min=1.0)
        n_residual_valid = residual_mask_f.sum().clamp(min=1.0)

        # 波形 MAE
        waveform_mae = ((d_syn - d_obs).abs() * waveform_mask_f).sum() / n_waveform_valid

        # 残差 L2 正则化（仅在掩码区域内）
        residual_l2 = (residual.pow(2) * residual_mask_f).sum() / n_residual_valid

        # 残差 TV 正则化（沿时间轴一阶差分，作用于 core+halo taper 支撑区）
        diff = residual[..., 1:] - residual[..., :-1]
        pair_weight = torch.minimum(taper_weight[..., 1:], taper_weight[..., :-1])
        pair_weight_sum = pair_weight.sum().clamp(min=1.0)
        residual_tv = (diff.abs() * pair_weight).sum() / pair_weight_sum

        # 总损失
        total_loss = waveform_mae + self.lambda_reg * residual_l2 + self.lambda_tv * residual_tv

        reg_term = self.lambda_reg * residual_l2
        tv_term = self.lambda_tv * residual_tv

        loss_dict = {
            "total": total_loss.item(),
            "waveform_mae": waveform_mae.item(),
            "residual_l2": residual_l2.item(),
            "residual_tv": residual_tv.item(),
            "reg_term": reg_term.item(),
            "tv_term": tv_term.item(),
            "lambda_reg": float(self.lambda_reg),
            "lambda_tv": float(self.lambda_tv),
        }

        return total_loss, loss_dict
