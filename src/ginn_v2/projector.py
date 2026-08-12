"""Configurable physical body-scale projection for GINN V2 corrections."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor

from cup.lfm.math import LowpassSpec
from ginn_v2.losses import _gaussian_weights, masked_lfm_lowpass


@dataclass(frozen=True)
class BodyScaleProjector:
    """Keep a correction between the configured body and LFM scales.

    The interface deliberately accepts the physical sample coordinates and
    the support mask at each call.  This keeps the module independent of the
    time/depth adapter while using the same operation for network outputs and
    well targets.
    """

    smoothing_fwhm_m: float
    sample_step: float
    lowpass_spec: LowpassSpec

    def __post_init__(self) -> None:
        width = float(self.smoothing_fwhm_m)
        step = float(self.sample_step)
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("smoothing_fwhm_m must be finite and positive.")
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("sample_step must be finite and positive.")
        spec = self.lowpass_spec
        if (
            not spec.enabled
            or spec.cutoff_cycles_per_axis_unit is None
            or spec.order is None
            or spec.buffer_mode is None
            or spec.buffer_axis_units is None
        ):
            raise ValueError("BodyScaleProjector requires a complete enabled low-pass specification.")

    def project(
        self,
        raw_correction: Tensor,
        coordinates_m: Tensor,
        support_mask: Tensor,
    ) -> Tensor:
        """Project a finite correction tensor into the configured body band."""

        if raw_correction.ndim != 2 or not torch.is_floating_point(raw_correction):
            raise ValueError("raw_correction must be a floating (batch, samples) tensor.")
        if coordinates_m.ndim not in {1, 2} or not torch.is_floating_point(coordinates_m):
            raise ValueError("coordinates_m must be a floating one- or two-dimensional tensor.")
        if support_mask.shape != raw_correction.shape or support_mask.dtype != torch.bool:
            raise ValueError("support_mask must be boolean and match raw_correction.")
        if coordinates_m.ndim == 1 and coordinates_m.shape[0] != raw_correction.shape[1]:
            raise ValueError("coordinates_m sample count differs from raw_correction.")
        if coordinates_m.ndim == 2 and coordinates_m.shape != raw_correction.shape:
            raise ValueError("coordinates_m batch shape differs from raw_correction.")
        if not bool(torch.all(torch.isfinite(raw_correction)).item()):
            raise ValueError("raw_correction must contain only finite values.")

        coordinates = coordinates_m.to(device=raw_correction.device, dtype=raw_correction.dtype)
        if coordinates.ndim == 1:
            weights = _gaussian_weights(coordinates, fwhm_m=self.smoothing_fwhm_m)[0].to(
                device=raw_correction.device,
                dtype=raw_correction.dtype,
            )
            support_float = support_mask.to(dtype=raw_correction.dtype)
            denominator = torch.matmul(support_float, weights.T)
            numerator = torch.matmul(raw_correction * support_float, weights.T)
            smooth_support = denominator > 0.0
            smoothed = torch.where(
                smooth_support,
                numerator / torch.clamp(denominator, min=torch.finfo(raw_correction.dtype).eps),
                torch.zeros_like(raw_correction),
            )
        else:
            smooth_rows: list[Tensor] = []
            support_rows: list[Tensor] = []
            for row in range(raw_correction.shape[0]):
                weights = _gaussian_weights(
                    coordinates[row],
                    fwhm_m=self.smoothing_fwhm_m,
                )[0].to(device=raw_correction.device, dtype=raw_correction.dtype)
                support_float = support_mask[row].to(dtype=raw_correction.dtype)
                denominator = torch.matmul(weights, support_float)
                numerator = torch.matmul(weights, raw_correction[row] * support_float)
                valid = denominator > 0.0
                smooth_rows.append(
                    torch.where(
                        valid,
                        numerator / torch.clamp(denominator, min=torch.finfo(raw_correction.dtype).eps),
                        torch.zeros_like(raw_correction[row]),
                    )
                )
                support_rows.append(valid)
            smoothed = torch.stack(smooth_rows)
            smooth_support = torch.stack(support_rows)
        low, low_support = masked_lfm_lowpass(
            smoothed,
            smooth_support,
            sample_step=self.sample_step,
            spec=self.lowpass_spec,
        )
        valid = support_mask & smooth_support & low_support
        return torch.where(valid, smoothed - low, torch.zeros_like(raw_correction))

    def project_numpy(
        self,
        raw_correction: np.ndarray,
        coordinates_m: np.ndarray,
        support_mask: np.ndarray,
    ) -> np.ndarray:
        """Apply :meth:`project` to one NumPy trace without changing its axis."""

        values = np.asarray(raw_correction, dtype=np.float64)
        coordinates = np.asarray(coordinates_m, dtype=np.float64)
        support = np.asarray(support_mask, dtype=bool)
        if values.ndim != 1 or coordinates.ndim != 1 or support.shape != values.shape:
            raise ValueError("NumPy projector inputs must be matching one-dimensional arrays.")
        with torch.no_grad():
            result = self.project(
                torch.as_tensor(values, dtype=torch.float64)[None, :],
                torch.as_tensor(coordinates, dtype=torch.float64),
                torch.as_tensor(support, dtype=torch.bool)[None, :],
            )
        return result[0].cpu().numpy()


def project_well_target(
    model_axis_target: np.ndarray,
    target_mask: np.ndarray,
    lfm_log_ai: np.ndarray,
    lfm_valid_mask: np.ndarray,
    coordinates_m: np.ndarray,
    projector: BodyScaleProjector,
) -> np.ndarray:
    """Map a smoothed well target into the same body-band space as the model."""

    target = np.asarray(model_axis_target, dtype=np.float64)
    mask = np.asarray(target_mask, dtype=bool)
    lfm = np.asarray(lfm_log_ai, dtype=np.float64)
    lfm_mask = np.asarray(lfm_valid_mask, dtype=bool)
    coordinates = np.asarray(coordinates_m, dtype=np.float64)
    if any(value.ndim != 1 for value in (target, mask, lfm, lfm_mask, coordinates)):
        raise ValueError("Well projector inputs must be one-dimensional.")
    if not (target.shape == mask.shape == lfm.shape == lfm_mask.shape == coordinates.shape):
        raise ValueError("Well projector inputs must have matching shapes.")
    valid = mask & lfm_mask & np.isfinite(target) & np.isfinite(lfm)
    if not np.any(valid):
        raise ValueError("Well projector has no joint finite target/LFM support.")
    correction = np.zeros_like(target)
    correction[valid] = target[valid] - lfm[valid]
    projected = projector.project_numpy(correction, coordinates, valid)
    result = np.full(target.shape, np.nan, dtype=np.float64)
    result[valid] = lfm[valid] + projected[valid]
    return result


__all__ = ["BodyScaleProjector", "project_well_target"]
