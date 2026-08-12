"""Physics and body-scale losses used by body inversion."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from cup.lfm.math import LowpassSpec


@dataclass(frozen=True)
class ShapeLossResult:
    loss: Tensor
    correlation: Tensor
    normalized_shape_loss: Tensor
    support_count: Tensor


@dataclass(frozen=True)
class GainDiagnostic:
    gain: Tensor
    raw_amplitude_residual: Tensor


def _validate_trace_pair(observed: Tensor, predicted: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if observed.shape != predicted.shape or observed.shape != mask.shape or observed.ndim != 2:
        raise ValueError("observed, predicted, and support mask must have matching shape (batch, samples).")
    if not torch.is_floating_point(observed) or not torch.is_floating_point(predicted):
        raise TypeError("observed and predicted must be floating tensors.")
    if observed.dtype != predicted.dtype:
        predicted = predicted.to(dtype=observed.dtype)
    if mask.dtype != torch.bool:
        raise TypeError("support mask must be boolean.")
    if not bool(torch.all(torch.isfinite(observed)).item()) or not bool(torch.all(torch.isfinite(predicted)).item()):
        raise ValueError("observed and predicted must contain only finite values.")
    support_count = torch.count_nonzero(mask, dim=-1)
    if bool(torch.any(support_count < 2).item()):
        raise ValueError("Every trace needs at least two support samples for a shape loss.")
    return observed, predicted, mask


def normalize_support(values: Tensor, support_mask: Tensor, *, epsilon: float = 1e-8) -> Tensor:
    """Normalize each trace on its own upstream valid support."""

    if values.ndim != 2 or support_mask.shape != values.shape or support_mask.dtype != torch.bool:
        raise ValueError("values and support_mask must have matching shape (batch, samples).")
    if not torch.is_floating_point(values) or not bool(torch.all(torch.isfinite(values)).item()):
        raise ValueError("values must be finite floating tensors.")
    count = torch.sum(support_mask.to(dtype=values.dtype), dim=-1, keepdim=True)
    if bool(torch.any(count < 2).item()):
        raise ValueError("Each trace needs at least two support samples for normalization.")
    weights = support_mask.to(dtype=values.dtype)
    mean = torch.sum(values * weights, dim=-1, keepdim=True) / count.to(dtype=values.dtype)
    centered = values - mean
    rms = torch.sqrt(torch.sum(torch.square(centered) * weights, dim=-1, keepdim=True) / count.to(dtype=values.dtype))
    if bool(torch.any(rms <= float(epsilon)).item()):
        raise ValueError("A supported trace has zero variance and cannot be shape-normalized.")
    return centered / rms


def waveform_shape_loss(
    observed: Tensor,
    predicted: Tensor,
    support_mask: Tensor,
    *,
    lambda_shape: float = 1.0,
) -> ShapeLossResult:
    """Compute correlation plus normalized Smooth-L1 on one trace support."""

    observed, predicted, support_mask = _validate_trace_pair(observed, predicted, support_mask)
    if not math.isfinite(float(lambda_shape)) or lambda_shape < 0.0:
        raise ValueError("lambda_shape must be finite and non-negative.")
    support = support_mask.to(dtype=observed.dtype)
    count = torch.sum(support, dim=-1)
    observed_norm = normalize_support(observed, support_mask)
    predicted_norm = normalize_support(predicted, support_mask)
    correlation = torch.sum(observed_norm * predicted_norm * support, dim=-1) / count
    normalized_error = F.smooth_l1_loss(
        predicted_norm,
        observed_norm,
        reduction="none",
    )
    normalized_error = torch.sum(normalized_error * support, dim=-1) / count
    loss = (1.0 - correlation) + float(lambda_shape) * normalized_error
    return ShapeLossResult(
        loss=loss.mean(),
        correlation=correlation,
        normalized_shape_loss=normalized_error,
        support_count=count,
    )


def analytic_gain_diagnostic(
    observed: Tensor,
    predicted: Tensor,
    support_mask: Tensor,
) -> GainDiagnostic:
    """Return non-negative least-squares gain and raw residual without grad."""

    observed, predicted, support_mask = _validate_trace_pair(observed, predicted, support_mask)
    with torch.no_grad():
        support = support_mask.to(dtype=observed.dtype)
        numerator = torch.sum(observed * predicted * support, dim=-1)
        denominator = torch.sum(torch.square(predicted) * support, dim=-1)
        if bool(torch.any(denominator <= 0.0).item()):
            raise ValueError("Analytic gain requires non-zero predicted energy on every support.")
        gain = torch.clamp(numerator / denominator, min=0.0)
        residual = observed - gain[:, None] * predicted
        raw = torch.sqrt(torch.sum(torch.square(residual) * support, dim=-1) / torch.sum(support, dim=-1))
        return GainDiagnostic(gain=gain.detach(), raw_amplitude_residual=raw.detach())


def _gaussian_weights(coordinates_m: Tensor, *, fwhm_m: float) -> Tensor:
    if coordinates_m.ndim == 1:
        coordinates = coordinates_m[None, :]
    elif coordinates_m.ndim == 2:
        coordinates = coordinates_m
    else:
        raise ValueError("coordinates_m must have shape (samples,) or (batch, samples).")
    if not torch.is_floating_point(coordinates) or not bool(torch.all(torch.isfinite(coordinates)).item()):
        raise ValueError("coordinates_m must be finite and floating.")
    if bool(torch.any(torch.diff(coordinates, dim=-1) <= 0.0).item()):
        raise ValueError("coordinates_m must be strictly increasing.")
    width = float(fwhm_m)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("fwhm_m must be finite and positive.")
    sigma = width / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    distance = coordinates[:, :, None] - coordinates[:, None, :]
    weights = torch.exp(-0.5 * torch.square(distance / sigma))
    return weights / torch.sum(weights, dim=-1, keepdim=True)


def masked_physical_lowpass(
    values: Tensor,
    coordinates_m: Tensor,
    support_mask: Tensor | None = None,
    *,
    cutoff_wavelength_m: float,
) -> tuple[Tensor, Tensor]:
    """Apply a differentiable physical-coordinate Gaussian low-pass.

    The cutoff is recorded in metres and the actual weights use coordinate
    distances.  No sample-count conversion is performed, which is important
    for depth axes and for surveys with non-unit line steps.
    """

    if values.ndim != 2 or not torch.is_floating_point(values):
        raise ValueError("values must be a floating tensor with shape (batch, samples).")
    if not bool(torch.all(torch.isfinite(values)).item()):
        raise ValueError("values must contain only finite values.")
    if support_mask is None:
        support = torch.ones_like(values, dtype=torch.bool)
    else:
        if support_mask.shape != values.shape or support_mask.dtype != torch.bool:
            raise ValueError("support_mask must be boolean and match values.")
        support = support_mask
    weights = _gaussian_weights(coordinates_m.to(device=values.device, dtype=values.dtype), fwhm_m=cutoff_wavelength_m)
    if weights.shape[0] == 1 and values.shape[0] != 1:
        weights = weights.expand(values.shape[0], -1, -1)
    if weights.shape != (values.shape[0], values.shape[1], values.shape[1]):
        raise ValueError("coordinates_m batch dimension differs from values.")
    support_float = support.to(dtype=values.dtype)
    weighted = weights * support_float[:, None, :]
    denominator = torch.sum(weighted, dim=-1)
    valid_output = denominator > 0.0
    if not bool(torch.any(valid_output).item()):
        raise ValueError("Low-pass support is empty for every output sample.")
    numerator = torch.bmm(weighted, values.unsqueeze(-1)).squeeze(-1)
    output = torch.zeros_like(values)
    output[valid_output] = numerator[valid_output] / denominator[valid_output]
    return output, valid_output


@lru_cache(maxsize=256)
def _lfm_filter_matrix(
    length: int,
    sample_step: float,
    cutoff_cycles_per_axis_unit: float,
    order: int,
    buffer_mode: str,
    buffer_axis_units: float,
) -> np.ndarray:
    """Return the exact linear operator used by the Step-7 Butterworth filter."""

    from scipy.signal import butter, sosfiltfilt

    sos = butter(
        int(order),
        float(cutoff_cycles_per_axis_unit),
        btype="lowpass",
        fs=1.0 / float(sample_step),
        output="sos",
    )
    if int(length) < 2:
        raise ValueError("LFM anchor run requires at least two samples.")
    basis = np.eye(int(length), dtype=np.float64)
    pad_samples = int(np.ceil(float(buffer_axis_units) / float(sample_step)))
    if buffer_mode == "none" or pad_samples == 0:
        padded = basis
        crop = slice(None)
    else:
        mode = "reflect" if buffer_mode == "reflect" else "edge"
        padded = np.pad(basis, ((pad_samples, pad_samples), (0, 0)), mode=mode)
        crop = slice(pad_samples, pad_samples + int(length))
    operator = np.ascontiguousarray(sosfiltfilt(sos, padded, axis=0, padtype=None)[crop])
    operator.setflags(write=False)
    return operator


def masked_lfm_lowpass(
    values: Tensor,
    support_mask: Tensor,
    *,
    sample_step: float,
    spec: LowpassSpec,
) -> tuple[Tensor, Tensor]:
    """Apply the differentiable equivalent of the selected Step-7 low-pass."""

    if values.ndim != 2 or support_mask.shape != values.shape or support_mask.dtype != torch.bool:
        raise ValueError("values and support_mask must have matching (batch, samples) shapes.")
    if not torch.is_floating_point(values) or not bool(torch.all(torch.isfinite(values)).item()):
        raise ValueError("values must be a finite floating tensor.")
    if (
        not spec.enabled
        or spec.cutoff_cycles_per_axis_unit is None
        or spec.order is None
        or spec.buffer_mode is None
        or spec.buffer_axis_units is None
    ):
        raise ValueError("LFM anchor requires the complete enabled Step-7 low-pass specification.")
    output_rows: list[Tensor] = []
    valid_rows: list[Tensor] = []
    for row in range(values.shape[0]):
        row_output = torch.zeros_like(values[row])
        row_valid = torch.zeros_like(support_mask[row])
        padded = torch.cat(
            (
                torch.zeros(1, dtype=torch.bool, device=support_mask.device),
                support_mask[row],
                torch.zeros(1, dtype=torch.bool, device=support_mask.device),
            )
        )
        runs = torch.nonzero(padded[1:] != padded[:-1], as_tuple=False).reshape(-1, 2)
        for start, stop in runs.tolist():
            if not bool(support_mask[row, start].item()):
                continue
            matrix = torch.as_tensor(
                _lfm_filter_matrix(
                    stop - start,
                    float(sample_step),
                    float(spec.cutoff_cycles_per_axis_unit),
                    int(spec.order),
                    str(spec.buffer_mode),
                    float(spec.buffer_axis_units),
                ).copy(),
                device=values.device,
                dtype=values.dtype,
            )
            row_output[start:stop] = matrix @ values[row, start:stop]
            row_valid[start:stop] = True
        output_rows.append(row_output)
        valid_rows.append(row_valid)
    valid = torch.stack(valid_rows)
    if not bool(torch.any(valid).item()):
        raise ValueError("LFM anchor has no filterable support.")
    return torch.stack(output_rows), valid


def lfm_anchor_loss(
    predicted_body: Tensor,
    lfm_log_ai: Tensor,
    lfm_valid_mask: Tensor,
    *,
    sample_step: float,
    lowpass_spec: LowpassSpec,
) -> Tensor:
    """Suppress drift using the exact low-pass response selected in Step 7."""

    if predicted_body.shape != lfm_log_ai.shape or predicted_body.shape != lfm_valid_mask.shape:
        raise ValueError("predicted_body, lfm_log_ai, and lfm_valid_mask must have matching shapes.")
    residual_low, residual_support = masked_lfm_lowpass(
        predicted_body - lfm_log_ai,
        lfm_valid_mask,
        sample_step=sample_step,
        spec=lowpass_spec,
    )
    support = residual_support & lfm_valid_mask
    if not torch.any(support):
        raise ValueError("LFM anchor has no valid support.")
    return F.smooth_l1_loss(
        residual_low[support],
        torch.zeros_like(residual_low[support]),
        reduction="mean",
    )


def short_wave_energy_ratio(
    values: Tensor,
    coordinates_m: Tensor,
    support_mask: Tensor | None = None,
    *,
    body_smoothing_fwhm_m: float,
) -> Tensor:
    """Return short-wave energy as a fraction of non-DC body variation."""

    if support_mask is None:
        support_mask = torch.ones_like(values, dtype=torch.bool)
    low, support = masked_physical_lowpass(
        values,
        coordinates_m,
        support_mask,
        cutoff_wavelength_m=body_smoothing_fwhm_m,
    )
    usable = support & support_mask
    high = values - low
    support_float = usable.to(dtype=values.dtype)
    numerator = torch.sum(torch.square(high) * support_float, dim=-1)
    count = torch.sum(support_float, dim=-1, keepdim=True)
    mean = torch.sum(values * support_float, dim=-1, keepdim=True) / count
    centered = values - mean
    denominator = torch.sum(torch.square(centered) * support_float, dim=-1)
    return torch.where(
        denominator > torch.finfo(values.dtype).eps,
        numerator / denominator,
        torch.zeros_like(denominator),
    )


def vertical_roughness(values: Tensor, coordinates_m: Tensor, support_mask: Tensor | None = None) -> Tensor:
    """Return RMS first derivative per metre for each trace."""

    if values.ndim != 2 or not torch.is_floating_point(values):
        raise ValueError("values must be a floating (batch, samples) tensor.")
    if support_mask is None:
        support_mask = torch.ones_like(values, dtype=torch.bool)
    if support_mask.shape != values.shape or support_mask.dtype != torch.bool:
        raise ValueError("support_mask must match values and be boolean.")
    if coordinates_m.ndim == 1:
        coordinates = coordinates_m[None, :].expand(values.shape[0], -1)
    elif coordinates_m.ndim == 2:
        coordinates = coordinates_m
    else:
        raise ValueError("coordinates_m must be one- or two-dimensional.")
    if coordinates.shape != values.shape:
        raise ValueError("coordinates_m shape differs from values.")
    pair_support = support_mask[:, 1:] & support_mask[:, :-1]
    if bool(torch.any(torch.count_nonzero(pair_support, dim=-1) == 0).item()):
        raise ValueError("Vertical roughness has an empty support interval.")
    derivative = (values[:, 1:] - values[:, :-1]) / (coordinates[:, 1:] - coordinates[:, :-1])
    weights = pair_support.to(dtype=values.dtype)
    return torch.sqrt(torch.sum(torch.square(derivative) * weights, dim=-1) / torch.sum(weights, dim=-1))


__all__ = [
    "GainDiagnostic",
    "ShapeLossResult",
    "analytic_gain_diagnostic",
    "lfm_anchor_loss",
    "masked_lfm_lowpass",
    "masked_physical_lowpass",
    "normalize_support",
    "short_wave_energy_ratio",
    "vertical_roughness",
    "waveform_shape_loss",
]
