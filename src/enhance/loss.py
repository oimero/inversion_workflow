"""Synthetic-only losses for stage-2 resolution enhancement."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EnhancementLoss(nn.Module):
    """Delta-logAI supervision without waveform physics in the training loop."""

    def __init__(
        self,
        *,
        lambda_lowpass: float = 0.2,
        lambda_highpass: float = 1.0,
        lambda_rms: float = 0.05,
        lambda_rms_underfit: float = 0.0,
        rms_floor: float = 0.7,
        lowpass_samples: int = 17,
        highpass_samples: int = 7,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        for name, value in {
            "lambda_lowpass": lambda_lowpass,
            "lambda_highpass": lambda_highpass,
            "lambda_rms": lambda_rms,
            "lambda_rms_underfit": lambda_rms_underfit,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if lowpass_samples < 1 or highpass_samples < 1:
            raise ValueError("lowpass_samples and highpass_samples must be positive.")
        if rms_floor < 0.0:
            raise ValueError(f"rms_floor must be non-negative, got {rms_floor}.")
        self.lambda_lowpass = float(lambda_lowpass)
        self.lambda_highpass = float(lambda_highpass)
        self.lambda_rms = float(lambda_rms)
        self.lambda_rms_underfit = float(lambda_rms_underfit)
        self.rms_floor = float(rms_floor)
        self.lowpass_samples = int(lowpass_samples)
        self.highpass_samples = int(highpass_samples)
        self.eps = float(eps)

    def forward(self, pred_delta: Tensor, target_delta: Tensor, mask: Tensor) -> tuple[Tensor, dict[str, float]]:
        low_pred = _moving_average_1d(pred_delta, self.lowpass_samples)
        low_target = _moving_average_1d(target_delta, self.lowpass_samples)
        high_pred = pred_delta - _moving_average_1d(pred_delta, self.highpass_samples)
        high_target = target_delta - _moving_average_1d(target_delta, self.highpass_samples)

        lowpass = _masked_mean(F.smooth_l1_loss(low_pred, low_target, reduction="none"), mask, eps=self.eps)
        highpass = _masked_mean(F.smooth_l1_loss(high_pred, high_target, reduction="none"), mask, eps=self.eps)
        pred_rms, target_rms = _masked_rms_values(pred_delta, target_delta, mask, eps=self.eps)
        pred_high_rms, target_high_rms = _masked_rms_values(high_pred, high_target, mask, eps=self.eps)
        rms = F.smooth_l1_loss(pred_rms, target_rms)
        rms_underfit = F.smooth_l1_loss(
            torch.relu(self.rms_floor * target_rms - pred_rms),
            torch.zeros_like(pred_rms),
        )

        lowpass_term = self.lambda_lowpass * lowpass
        highpass_term = self.lambda_highpass * highpass
        rms_term = self.lambda_rms * rms
        rms_underfit_term = self.lambda_rms_underfit * rms_underfit
        total = lowpass_term + highpass_term + rms_term + rms_underfit_term

        pred_rms_mean = pred_rms.mean()
        target_rms_mean = target_rms.mean()
        pred_high_rms_mean = pred_high_rms.mean()
        target_high_rms_mean = target_high_rms.mean()
        return total, {
            "total": total.item(),
            "delta_lowpass": lowpass.item(),
            "delta_highpass": highpass.item(),
            "delta_rms": rms.item(),
            "delta_rms_underfit": rms_underfit.item(),
            "lowpass_term": lowpass_term.item(),
            "highpass_term": highpass_term.item(),
            "rms_term": rms_term.item(),
            "rms_underfit_term": rms_underfit_term.item(),
            "pred_delta_rms": pred_rms_mean.item(),
            "target_delta_rms": target_rms_mean.item(),
            "delta_rms_ratio": (pred_rms_mean / target_rms_mean.clamp(min=self.eps)).item(),
            "pred_highpass_rms": pred_high_rms_mean.item(),
            "target_highpass_rms": target_high_rms_mean.item(),
            "highpass_rms_ratio": (pred_high_rms_mean / target_high_rms_mean.clamp(min=self.eps)).item(),
        }


def compose_enhanced_ai(
    base_ai: Tensor, delta_log_ai: Tensor, *, ai_min: float | None = None, ai_max: float | None = None
) -> Tensor:
    """Compose enhanced AI from a positive base AI and predicted delta log-AI."""
    enhanced = torch.clamp(base_ai, min=1e-6) * torch.exp(delta_log_ai)
    if ai_min is not None or ai_max is not None:
        enhanced = torch.clamp(enhanced, min=ai_min, max=ai_max)
    return enhanced


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


def _masked_rms_values(pred: Tensor, target: Tensor, mask: Tensor, *, eps: float) -> tuple[Tensor, Tensor]:
    weight = mask.to(dtype=pred.dtype)
    denom = weight.sum(dim=-1).clamp(min=eps)
    pred_rms = torch.sqrt((pred.pow(2) * weight).sum(dim=-1) / denom + eps)
    target_rms = torch.sqrt((target.pow(2) * weight).sum(dim=-1) / denom + eps)
    return pred_rms, target_rms
