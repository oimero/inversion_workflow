"""ginn.loss — masked MAE + perturbation regularisation for GINN.

Composes three loss terms:
1. loss-mask MAE: mean absolute error between synthetic and observed seismic
   inside the eroded loss mask.
2. perturbation L2: discourages the predicted AI from drifting too far from
   the low-frequency model.
3. perturbation TV: penalises first-order oscillation along the trace axis.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GINNLoss(nn.Module):
    """GINN composite loss: masked MAE + L2/TV perturbation regularisation.

    .. math::
        \\mathcal{L} = \\underbrace{\\frac{\\sum |d_{syn} - d_{obs}| \\cdot m}{\\sum m}}_{\\text{waveform MAE}}
        + \\lambda_{l2} \\cdot \\underbrace{\\frac{\\sum \\phi^2 \\cdot w}{\\sum w}}_{\\text{perturbation L2}}
        + \\lambda_{tv} \\cdot \\underbrace{\\frac{\\sum |\\phi[t+1] - \\phi[t]| \\cdot w}{\\sum w}}_{\\text{perturbation TV}}

    Parameters
    ----------
    lambda_l2 : float
        Perturbation L2 weight (default 0.1).  Anchors ``phi`` near zero
        to resolve the absolute-scale ambiguity of the reflectivity formula.
    lambda_tv : float
        Perturbation TV weight (default 0.0).  Penalises first-order trace
        differences to suppress high-frequency null-space ringing.
    """

    def __init__(self, lambda_l2: float = 0.1, lambda_tv: float = 0.0) -> None:
        super().__init__()
        self.lambda_l2 = lambda_l2
        self.lambda_tv = lambda_tv

    def forward(
        self,
        d_syn: Tensor,
        d_obs: Tensor,
        loss_mask: Tensor,
        residual: Tensor,
        taper_weight: Tensor,
        *,
        pred_ai: Tensor | None = None,
        waveform_weight_scale: Tensor | None = None,
        anchor_target_log_ai: Tensor | None = None,
        anchor_weight: Tensor | None = None,
        well_influence: Tensor | None = None,
        lambda_log_ai_anchor: float = 0.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the composite loss.

        Parameters
        ----------
        d_syn : Tensor
            Synthetic seismic, shape ``(B, 1, T)``.
        d_obs : Tensor
            Observed seismic, shape ``(B, 1, T)``.
        loss_mask : Tensor
            Eroded waveform loss mask, shape ``(B, 1, T)``, True where
            waveform MAE is computed.
        residual : Tensor
            High-frequency perturbation ``phi``, shape ``(B, 1, T)``.
        taper_weight : Tensor
            Core+halo taper weights for perturbation L2/TV support,
            shape ``(B, 1, T)``.  The TV term uses the element-wise
            minimum of adjacent sample weights.
        pred_ai : Tensor or None
            Predicted AI, shape ``(B, 1, T)``.  Required when anchor
            supervision is active.
        waveform_weight_scale : Tensor or None
            Per-trace scale applied to the loss mask (well-control taper).
        anchor_target_log_ai : Tensor or None
            Anchor target log-AI, shape ``(B, 1, T)``.
        anchor_weight : Tensor or None
            Per-sample anchor confidence weights, shape ``(B, 1, T)``.
        well_influence : Tensor or None
            Per-trace well influence factor, shape ``(B, 1)``.
        lambda_log_ai_anchor : float
            Anchor loss weight. 0.0 disables anchor supervision.

        Returns
        -------
        total_loss : Tensor
            Scalar total loss.
        loss_dict : dict
            Per-term float values for logging.
        """
        loss_mask_f = loss_mask.to(device=d_syn.device, dtype=d_syn.dtype)
        if waveform_weight_scale is not None:
            scale = waveform_weight_scale.to(device=loss_mask_f.device, dtype=loss_mask_f.dtype)
            scale = scale.reshape(scale.shape[0], 1, 1)
            loss_mask_f = loss_mask_f * scale
        taper_weight_f = taper_weight.to(device=residual.device, dtype=residual.dtype)
        n_waveform_valid = loss_mask_f.sum().clamp(min=1.0)
        n_residual_valid = taper_weight_f.sum().clamp(min=1.0)

        # Waveform MAE (inside eroded loss mask).
        waveform_mae = ((d_syn - d_obs).abs() * loss_mask_f).sum() / n_waveform_valid

        # Perturbation L2 (core+halo taper support).
        residual_l2 = (residual.pow(2) * taper_weight_f).sum() / n_residual_valid

        # Perturbation TV (first-order trace difference over taper support).
        diff = residual[..., 1:] - residual[..., :-1]
        pair_weight = torch.minimum(taper_weight_f[..., 1:], taper_weight_f[..., :-1])
        pair_weight_sum = pair_weight.sum().clamp(min=1.0)
        residual_tv = (diff.abs() * pair_weight).sum() / pair_weight_sum

        l2_term = self.lambda_l2 * residual_l2
        tv_term = self.lambda_tv * residual_tv
        total_loss = waveform_mae + l2_term + tv_term

        anchor_loss = torch.zeros((), device=residual.device)
        anchor_term = torch.zeros((), device=residual.device)
        anchor_sample_count = torch.zeros((), device=residual.device)
        anchor_trace_count = torch.zeros((), device=residual.device)
        if (
            pred_ai is not None
            and anchor_target_log_ai is not None
            and anchor_weight is not None
            and lambda_log_ai_anchor > 0.0
        ):
            aw = anchor_weight.to(device=pred_ai.device, dtype=pred_ai.dtype)
            if well_influence is not None:
                influence = well_influence.to(device=pred_ai.device, dtype=pred_ai.dtype)
                influence = influence.reshape(influence.shape[0], 1, 1)
                aw = aw * influence
            positive_anchor = aw > 0.0
            anchor_sample_count = positive_anchor.sum().to(dtype=pred_ai.dtype)
            anchor_trace_count = positive_anchor.any(dim=-1).sum().to(dtype=pred_ai.dtype)
            anchor_denom = aw.sum().clamp(min=1.0)
            target = anchor_target_log_ai.to(device=pred_ai.device, dtype=pred_ai.dtype)
            pred_log_ai = torch.log(torch.clamp(pred_ai, min=1e-6))
            raw_anchor = F.smooth_l1_loss(pred_log_ai, target, reduction="none")
            anchor_loss = (raw_anchor * aw).sum() / anchor_denom
            anchor_term = float(lambda_log_ai_anchor) * anchor_loss
            total_loss = total_loss + anchor_term

        loss_dict = {
            "total": total_loss.item(),
            "waveform_mae": waveform_mae.item(),
            "residual_l2": residual_l2.item(),
            "residual_tv": residual_tv.item(),
            "l2_term": l2_term.item(),
            "tv_term": tv_term.item(),
            "log_ai_anchor": anchor_loss.item(),
            "log_ai_anchor_term": anchor_term.item(),
            "log_ai_anchor_traces": anchor_trace_count.item(),
            "log_ai_anchor_neighbors": 0.0,
            "anchor_sample_count": anchor_sample_count.item(),
            "lambda_l2": float(self.lambda_l2),
            "lambda_tv": float(self.lambda_tv),
            "lambda_log_ai_anchor": float(lambda_log_ai_anchor),
        }

        return total_loss, loss_dict
