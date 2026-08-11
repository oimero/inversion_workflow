"""Physical-metre Gaussian body/residual scale separation."""

from __future__ import annotations

import math
import torch
from torch import Tensor

from cup.well.scale_separation import (
    BODY_SMOOTHING_FWHM_M,
    gaussian_smooth_finite_runs_numpy,
    gaussian_smooth_numpy,
)

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def gaussian_smooth_torch(values: Tensor, coordinates_m: Tensor, *, fwhm_m: float = BODY_SMOOTHING_FWHM_M) -> Tensor:
    """Differentiable Torch equivalent of :func:`gaussian_smooth_numpy`."""
    if not isinstance(values, Tensor) or not torch.is_floating_point(values) or values.ndim < 1:
        raise TypeError("values must be a floating tensor with shape [..., samples].")
    if not isinstance(coordinates_m, Tensor) or not torch.is_floating_point(coordinates_m):
        raise TypeError("coordinates_m must be a floating tensor.")
    shared_coordinates = coordinates_m.ndim == 1
    if shared_coordinates:
        coordinates_m = coordinates_m.expand(values.shape)
    if coordinates_m.shape != values.shape:
        raise ValueError("coordinates_m must have shape (samples,) or match values.")
    if not bool(torch.all(torch.isfinite(values)).item()) or not bool(torch.all(torch.isfinite(coordinates_m)).item()):
        raise ValueError("values and coordinates_m must contain only finite values.")
    flat_coordinates = coordinates_m.reshape((-1, coordinates_m.shape[-1]))
    if bool(torch.any(torch.diff(flat_coordinates, dim=-1) <= 0.0).item()):
        raise ValueError("coordinates_m must be strictly increasing along every trace.")
    width = float(fwhm_m)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("fwhm_m must be finite and positive.")
    sigma = width * _FWHM_TO_SIGMA
    flat_values = values.reshape((-1, values.shape[-1]))
    coordinates = flat_coordinates.to(device=values.device, dtype=values.dtype)
    if shared_coordinates:
        distance = coordinates[0, :, None] - coordinates[0, None, :]
        weights = torch.exp(-0.5 * torch.square(distance / sigma))
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        output = torch.matmul(flat_values, weights.T)
    else:
        distance = coordinates[:, :, None] - coordinates[:, None, :]
        weights = torch.exp(-0.5 * torch.square(distance / sigma))
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        output = torch.bmm(weights, flat_values.unsqueeze(-1)).squeeze(-1)
    return output.reshape(values.shape)


def depth_coordinates_from_twt(velocity_mps: Tensor, twt_s: Tensor) -> Tensor:
    """Integrate fixed velocity along TWT to physical depth coordinates."""
    if velocity_mps.ndim != 2 or twt_s.ndim != 1 or velocity_mps.shape[-1] != twt_s.numel():
        raise ValueError("velocity_mps/twt_s must have shapes (batch, samples)/(samples,).")
    if not bool(torch.all(torch.isfinite(velocity_mps)).item()) or not bool(
        torch.all(torch.isfinite(twt_s)).item()
    ):
        raise ValueError("velocity and TWT must contain only finite values.")
    if bool(torch.any(velocity_mps <= 0.0).item()) or bool(torch.any(torch.diff(twt_s) <= 0.0).item()):
        raise ValueError("velocity must be positive and TWT must be strictly increasing.")
    dz = 0.25 * (velocity_mps[:, :-1] + velocity_mps[:, 1:]) * torch.diff(twt_s)[None, :]
    return torch.cat((torch.zeros_like(velocity_mps[:, :1]), torch.cumsum(dz, dim=-1)), dim=-1)


__all__ = [
    "BODY_SMOOTHING_FWHM_M",
    "depth_coordinates_from_twt",
    "gaussian_smooth_finite_runs_numpy",
    "gaussian_smooth_numpy",
    "gaussian_smooth_torch",
]
