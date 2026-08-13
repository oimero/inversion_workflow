"""Physical residual-scale projector used by Enhance V2."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from cup.well.scale_separation import gaussian_smooth_finite_runs_numpy
from ginn_v2.losses import _gaussian_weights


class ResidualScaleProjector:
    """Project a correction to the configured high-frequency residual band.

    The projector is deliberately independent of the GINN LFM low-pass
    contract.  It implements ``raw - GaussianSmooth(raw)`` on physical sample
    coordinates and applies the same support-aware operation to torch batches.
    """

    def __init__(self, *, smoothing_fwhm_m: float) -> None:
        value = float(smoothing_fwhm_m)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("smoothing_fwhm_m must be finite and positive.")
        self.smoothing_fwhm_m = value

    def project(self, raw: Tensor, coordinates_m: Tensor, support_mask: Tensor) -> Tensor:
        if raw.ndim != 2 or not torch.is_floating_point(raw):
            raise ValueError("raw must be a floating tensor with shape (batch, samples).")
        if support_mask.shape != raw.shape or support_mask.dtype != torch.bool:
            raise ValueError("support_mask must be boolean and match raw.")
        if coordinates_m.ndim == 1:
            if coordinates_m.shape != (raw.shape[1],):
                raise ValueError("coordinates_m sample count differs from raw.")
            coordinates = coordinates_m[None, :].expand(raw.shape[0], -1)
        elif coordinates_m.ndim == 2 and coordinates_m.shape == raw.shape:
            coordinates = coordinates_m
        else:
            raise ValueError("coordinates_m must have shape (samples,) or (batch, samples).")
        if not bool(torch.all(torch.isfinite(raw)).item()) or not bool(torch.all(torch.isfinite(coordinates)).item()):
            raise ValueError("raw and coordinates_m must be finite.")
        weights = _gaussian_weights(
            coordinates.to(device=raw.device, dtype=raw.dtype),
            fwhm_m=self.smoothing_fwhm_m,
        )
        support = support_mask.to(dtype=raw.dtype)
        weighted = weights * support[:, None, :]
        denominator = weighted.sum(dim=-1)
        numerator = torch.bmm(weighted, raw.unsqueeze(-1)).squeeze(-1)
        smooth = numerator / denominator.clamp(min=torch.finfo(raw.dtype).eps)
        valid = support_mask & (denominator > 0.0)
        return torch.where(valid, raw - smooth, torch.zeros_like(raw))

    def project_numpy(self, values: np.ndarray, coordinates_m: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
        values_array = np.asarray(values, dtype=np.float64)
        coordinates = np.asarray(coordinates_m, dtype=np.float64)
        support = np.asarray(support_mask, dtype=bool)
        if values_array.ndim != 1 or coordinates.shape != values_array.shape or support.shape != values_array.shape:
            raise ValueError("values, coordinates_m, and support_mask must be matching 1-D arrays.")
        if np.any(~np.isfinite(coordinates)) or np.any(np.diff(coordinates) <= 0.0):
            raise ValueError("coordinates_m must be finite and strictly increasing.")
        output = np.zeros_like(values_array)
        finite = support & np.isfinite(values_array)
        for start, stop in _finite_runs(finite):
            local = values_array[start:stop]
            smooth = gaussian_smooth_finite_runs_numpy(
                local,
                coordinates[start:stop],
                fwhm_m=self.smoothing_fwhm_m,
            )
            output[start:stop] = local - smooth
        return output.astype(np.float32)


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    return [
        (int(start), int(stop))
        for start, stop in np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
        if stop > start
    ]


__all__ = ["ResidualScaleProjector"]
