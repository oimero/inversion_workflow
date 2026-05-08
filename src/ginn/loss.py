"""ginn.loss — 掩码 MAE 损失函数 + 高频扰动正则化。

组合三项损失：
1. 掩码 MAE：仅在层位顶底之间计算合成地震与观测地震的平均绝对误差
2. 高频扰动 L2 正则：防止阻抗偏离 LFM 过远
3. 高频扰动 TV 正则：抑制沿时间轴的一阶高频振荡
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GINNLoss(nn.Module):
    """GINN 组合损失：掩码 MAE + 高频扰动 L2/TV 正则化。

    .. math::
        \\mathcal{L} = \\underbrace{\\frac{\\sum |d_{syn} - d_{obs}| \\cdot m}{\\sum m}}_{\\text{waveform MAE}}
        + \\lambda_{l2} \\cdot \\underbrace{\\frac{\\sum \\phi^2 \\cdot m}{\\sum m}}_{\\text{perturbation L2}}
        + \\lambda_{tv} \\cdot \\underbrace{\\frac{\\sum |\\phi[t+1] - \\phi[t]| \\cdot w}{\\sum w}}_{\\text{perturbation TV}}

    Parameters
    ----------
    lambda_l2 : float
        高频扰动 L2 正则化权重。默认 0.1。
        物理意义：强制阻抗不偏离低频模型太远，解决反射率比值不变性
        导致的绝对尺度不确定性。
    lambda_tv : float
        高频扰动 TV 正则化权重。默认 0.0。
        物理意义：惩罚沿时间轴的一阶差分，抑制会被子波滤掉的高频 null-space ringing。

    Notes
    -----
    反射率公式 ``r = (AI[t+1] - AI[t]) / (AI[t+1] + AI[t])`` 对 AI 的全局
    缩放是不变的。L2 正则化将高频扰动 ``phi`` 锚定在零附近，避免阻抗
    在满足波形拟合时偏离低频模型过远。
    """

    def __init__(self, lambda_l2: float = 0.1, lambda_tv: float = 0.0) -> None:
        super().__init__()
        self.lambda_l2 = lambda_l2
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
            高频扰动正则掩码，shape ``(B, 1, T)``，True 表示参与 perturbation L2 的区域。
        residual : Tensor
            与 LFM 合成阻抗的高频扰动 ``phi``，shape ``(B, 1, T)``。
        taper_weight : Tensor
            高频扰动的 core+halo taper 权重，shape ``(B, 1, T)``。
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

        # 高频扰动 L2 正则化（仅在掩码区域内）
        residual_l2 = (residual.pow(2) * residual_mask_f).sum() / n_residual_valid

        # 高频扰动 TV 正则化（沿时间轴一阶差分，作用于 core+halo taper 支撑区）
        diff = residual[..., 1:] - residual[..., :-1]
        pair_weight = torch.minimum(taper_weight[..., 1:], taper_weight[..., :-1])
        pair_weight_sum = pair_weight.sum().clamp(min=1.0)
        residual_tv = (diff.abs() * pair_weight).sum() / pair_weight_sum

        # 总损失
        total_loss = waveform_mae + self.lambda_l2 * residual_l2 + self.lambda_tv * residual_tv

        l2_term = self.lambda_l2 * residual_l2
        tv_term = self.lambda_tv * residual_tv

        loss_dict = {
            "total": total_loss.item(),
            "waveform_mae": waveform_mae.item(),
            "residual_l2": residual_l2.item(),
            "residual_tv": residual_tv.item(),
            "l2_term": l2_term.item(),
            "tv_term": tv_term.item(),
            "lambda_l2": float(self.lambda_l2),
            "lambda_tv": float(self.lambda_tv),
        }

        return total_loss, loss_dict


class ResolutionPretrainLoss(nn.Module):
    """Synthetic resolution-pretrain loss for deterministic well-guided sharpening.

    The target residual is not treated as a pure waveform side effect. Low-pass
    and high-pass residual components are both supervised, while spectrum/RMS
    terms keep the residual amplitude from collapsing to the zero-residual
    shortcut.
    """

    def __init__(
        self,
        *,
        lambda_waveform: float = 1.0,
        lambda_residual_lowpass: float = 0.2,
        lambda_residual_highpass: float = 1.0,
        lambda_spectrum: float = 0.05,
        lambda_rms: float = 0.05,
        lambda_rms_underfit: float = 0.0,
        residual_rms_floor: float = 0.7,
        lowpass_samples: int = 17,
        highpass_samples: int = 7,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        for name, value in {
            "lambda_waveform": lambda_waveform,
            "lambda_residual_lowpass": lambda_residual_lowpass,
            "lambda_residual_highpass": lambda_residual_highpass,
            "lambda_spectrum": lambda_spectrum,
            "lambda_rms": lambda_rms,
            "lambda_rms_underfit": lambda_rms_underfit,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if lowpass_samples < 1:
            raise ValueError(f"lowpass_samples must be >= 1, got {lowpass_samples}.")
        if highpass_samples < 1:
            raise ValueError(f"highpass_samples must be >= 1, got {highpass_samples}.")
        if residual_rms_floor < 0.0:
            raise ValueError(f"residual_rms_floor must be non-negative, got {residual_rms_floor}.")

        self.lambda_waveform = float(lambda_waveform)
        self.lambda_residual_lowpass = float(lambda_residual_lowpass)
        self.lambda_residual_highpass = float(lambda_residual_highpass)
        self.lambda_spectrum = float(lambda_spectrum)
        self.lambda_rms = float(lambda_rms)
        self.lambda_rms_underfit = float(lambda_rms_underfit)
        self.residual_rms_floor = float(residual_rms_floor)
        self.lowpass_samples = int(lowpass_samples)
        self.highpass_samples = int(highpass_samples)
        self.eps = float(eps)

    def forward(
        self,
        d_syn: Tensor,
        d_target: Tensor,
        waveform_mask: Tensor,
        pred_residual: Tensor,
        target_residual: Tensor,
        residual_mask: Tensor,
        taper_weight: Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        waveform_mae = _masked_mean((d_syn - d_target).abs(), waveform_mask, eps=self.eps)

        low_pred = _moving_average_1d(pred_residual, self.lowpass_samples)
        low_target = _moving_average_1d(target_residual, self.lowpass_samples)
        residual_lowpass = _masked_mean(F.smooth_l1_loss(low_pred, low_target, reduction="none"), residual_mask, eps=self.eps)

        high_pred = _highpass_1d(pred_residual, self.highpass_samples)
        high_target = _highpass_1d(target_residual, self.highpass_samples)
        residual_highpass = _masked_mean(
            F.smooth_l1_loss(high_pred, high_target, reduction="none"),
            residual_mask,
            eps=self.eps,
        )

        spectrum = _masked_spectrum_loss(pred_residual, target_residual, taper_weight, eps=self.eps)
        pred_rms, target_rms = _masked_rms_values(pred_residual, target_residual, residual_mask, eps=self.eps)
        pred_highpass_rms, target_highpass_rms = _masked_rms_values(high_pred, high_target, residual_mask, eps=self.eps)
        rms = F.smooth_l1_loss(pred_rms, target_rms)
        rms_underfit = F.smooth_l1_loss(
            torch.relu(self.residual_rms_floor * target_rms - pred_rms),
            torch.zeros_like(pred_rms),
        )

        waveform_term = self.lambda_waveform * waveform_mae
        residual_lowpass_term = self.lambda_residual_lowpass * residual_lowpass
        residual_highpass_term = self.lambda_residual_highpass * residual_highpass
        spectrum_term = self.lambda_spectrum * spectrum
        rms_term = self.lambda_rms * rms
        rms_underfit_term = self.lambda_rms_underfit * rms_underfit
        total = (
            waveform_term
            + residual_lowpass_term
            + residual_highpass_term
            + spectrum_term
            + rms_term
            + rms_underfit_term
        )

        pred_rms_mean = pred_rms.mean()
        target_rms_mean = target_rms.mean()
        pred_highpass_rms_mean = pred_highpass_rms.mean()
        target_highpass_rms_mean = target_highpass_rms.mean()

        loss_dict = {
            "total": total.item(),
            "waveform_mae": waveform_mae.item(),
            "residual_lowpass": residual_lowpass.item(),
            "residual_highpass": residual_highpass.item(),
            "spectrum": spectrum.item(),
            "rms": rms.item(),
            "rms_underfit": rms_underfit.item(),
            "waveform_term": waveform_term.item(),
            "residual_lowpass_term": residual_lowpass_term.item(),
            "residual_highpass_term": residual_highpass_term.item(),
            "spectrum_term": spectrum_term.item(),
            "rms_term": rms_term.item(),
            "rms_underfit_term": rms_underfit_term.item(),
            "pred_residual_rms": pred_rms_mean.item(),
            "target_residual_rms": target_rms_mean.item(),
            "residual_rms_ratio": (pred_rms_mean / target_rms_mean.clamp(min=self.eps)).item(),
            "pred_highpass_rms": pred_highpass_rms_mean.item(),
            "target_highpass_rms": target_highpass_rms_mean.item(),
            "highpass_rms_ratio": (pred_highpass_rms_mean / target_highpass_rms_mean.clamp(min=self.eps)).item(),
            "lambda_waveform": self.lambda_waveform,
            "lambda_residual_lowpass": self.lambda_residual_lowpass,
            "lambda_residual_highpass": self.lambda_residual_highpass,
            "lambda_spectrum": self.lambda_spectrum,
            "lambda_rms": self.lambda_rms,
            "lambda_rms_underfit": self.lambda_rms_underfit,
            "residual_rms_floor": self.residual_rms_floor,
        }
        return total, loss_dict


def _masked_mean(values: Tensor, mask: Tensor, *, eps: float) -> Tensor:
    weight = mask.to(dtype=values.dtype)
    return (values * weight).sum() / weight.sum().clamp(min=eps)


def _moving_average_1d(values: Tensor, window: int) -> Tensor:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = F.pad(values, (pad, pad), mode="replicate")
    return F.avg_pool1d(padded, kernel_size=window, stride=1)


def _highpass_1d(values: Tensor, window: int) -> Tensor:
    return values - _moving_average_1d(values, window)


def _masked_center(values: Tensor, mask: Tensor, *, eps: float) -> Tensor:
    weight = mask.to(dtype=values.dtype)
    denom = weight.sum(dim=-1, keepdim=True).clamp(min=eps)
    mean = (values * weight).sum(dim=-1, keepdim=True) / denom
    return (values - mean) * weight


def _masked_spectrum_loss(pred: Tensor, target: Tensor, mask: Tensor, *, eps: float) -> Tensor:
    pred_centered = _masked_center(pred, mask, eps=eps)
    target_centered = _masked_center(target, mask, eps=eps)
    pred_amp = torch.abs(torch.fft.rfft(pred_centered, dim=-1))
    target_amp = torch.abs(torch.fft.rfft(target_centered, dim=-1))
    scale = pred.shape[-1] ** 0.5
    return F.smooth_l1_loss(pred_amp / scale, target_amp / scale)


def _masked_rms_loss(pred: Tensor, target: Tensor, mask: Tensor, *, eps: float) -> Tensor:
    pred_rms, target_rms = _masked_rms_values(pred, target, mask, eps=eps)
    return F.smooth_l1_loss(pred_rms, target_rms)


def _masked_rms_values(pred: Tensor, target: Tensor, mask: Tensor, *, eps: float) -> tuple[Tensor, Tensor]:
    weight = mask.to(dtype=pred.dtype)
    denom = weight.sum(dim=-1).clamp(min=eps)
    pred_rms = torch.sqrt((pred.pow(2) * weight).sum(dim=-1) / denom + eps)
    target_rms = torch.sqrt((target.pow(2) * weight).sum(dim=-1) / denom + eps)
    return pred_rms, target_rms
