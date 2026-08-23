"""GINN waveform, anchor, and visibility objectives."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from cup.utils.masks import true_runs

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
    if not torch.is_floating_point(coordinates):
        raise ValueError("coordinates_m must be floating.")
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
    if not torch.is_floating_point(values):
        raise ValueError("values must be a floating tensor.")
    if (
        not spec.enabled
        or spec.cutoff_cycles_per_axis_unit is None
        or spec.order is None
        or spec.buffer_mode is None
        or spec.buffer_axis_units is None
    ):
        raise ValueError("LFM anchor requires the complete enabled Step-7 low-pass specification.")
    grouped_rows: dict[tuple[int, int], list[int]] = {}
    support_cpu = support_mask.detach().cpu().numpy()
    for row, row_support in enumerate(support_cpu):
        for start, stop in true_runs(row_support):
            grouped_rows.setdefault((start, stop), []).append(row)
    if not grouped_rows:
        raise ValueError("LFM anchor has no filterable support.")
    output = torch.zeros_like(values)
    valid = torch.zeros_like(support_mask)
    for (start, stop), rows in grouped_rows.items():
        matrix = torch.as_tensor(
            np.array(
                _lfm_filter_matrix(
                    stop - start,
                    float(sample_step),
                    float(spec.cutoff_cycles_per_axis_unit),
                    int(spec.order),
                    str(spec.buffer_mode),
                    float(spec.buffer_axis_units),
                ),
                copy=True,
            ),
            device=values.device,
            dtype=values.dtype,
        )
        row_indices = torch.as_tensor(rows, device=values.device, dtype=torch.long)
        output[row_indices, start:stop] = values[row_indices, start:stop] @ matrix.T
        valid[row_indices, start:stop] = True
    return output, valid


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


def _masked_gaussian_smooth(
    values: Tensor,
    coordinates_m: Tensor,
    support_mask: Tensor,
    *,
    fwhm_m: float,
) -> tuple[Tensor, Tensor]:
    """Smooth masked traces, using convolution on regular physical axes."""

    if coordinates_m.ndim == 1:
        coordinates = coordinates_m[None, :].expand(values.shape[0], -1)
    else:
        coordinates = coordinates_m
    outputs: list[Tensor] = []
    supports: list[Tensor] = []
    sigma = float(fwhm_m) / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    epsilon = torch.finfo(values.dtype).eps
    for row in range(values.shape[0]):
        differences = torch.diff(coordinates[row])
        if bool(torch.any(differences <= 0.0).item()):
            raise ValueError("visibility coordinates must be strictly increasing.")
        step = torch.median(differences)
        regular = bool(
            torch.allclose(
                differences,
                step.expand_as(differences),
                rtol=1.0e-4,
                atol=max(float(step) * 1.0e-6, 1.0e-8),
            )
        )
        support_float = support_mask[row].to(dtype=values.dtype)
        if regular:
            half_width = max(1, int(math.ceil(4.0 * sigma / float(step))))
            offsets = torch.arange(
                -half_width,
                half_width + 1,
                device=values.device,
                dtype=values.dtype,
            ) * step
            kernel = torch.exp(-0.5 * torch.square(offsets / sigma))
            kernel = kernel / torch.sum(kernel)
            shaped_kernel = kernel[None, None, :]
            numerator = F.conv1d(
                (values[row] * support_float)[None, None, :],
                shaped_kernel,
                padding=half_width,
            )[0, 0]
            denominator = F.conv1d(
                support_float[None, None, :],
                shaped_kernel,
                padding=half_width,
            )[0, 0]
        else:
            weights = _gaussian_weights(coordinates[row], fwhm_m=fwhm_m)[0]
            numerator = weights @ (values[row] * support_float)
            denominator = weights @ support_float
        valid = denominator > epsilon
        outputs.append(torch.where(valid, numerator / torch.clamp(denominator, min=epsilon), torch.zeros_like(numerator)))
        supports.append(valid)
    return torch.stack(outputs), torch.stack(supports)


@dataclass(frozen=True)
class VisibilityCompensationConfig:
    """Configuration for separating trace gain and slow vertical visibility."""

    vertical_smoothing_fwhm_m: float = 300.0
    envelope_floor_fraction: float = 0.10
    minimum_visibility: float = 0.25
    maximum_visibility: float = 4.0

    def __post_init__(self) -> None:
        for name in (
            "vertical_smoothing_fwhm_m",
            "envelope_floor_fraction",
            "minimum_visibility",
            "maximum_visibility",
        ):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.envelope_floor_fraction >= 1.0:
            raise ValueError("envelope_floor_fraction must be below one.")
        if self.minimum_visibility >= 1.0 or self.maximum_visibility <= 1.0:
            raise ValueError("visibility bounds must straddle one.")
        if self.minimum_visibility >= self.maximum_visibility:
            raise ValueError("minimum_visibility must be below maximum_visibility.")


@dataclass(frozen=True)
class VisibilityCompensation:
    """Waveforms and nuisance fields produced by one compensation call."""

    balanced_observed: Tensor
    balanced_synthetic: Tensor
    compensated_synthetic: Tensor
    support_mask: Tensor
    trace_gain: Tensor
    vertical_visibility: Tensor
    observed_envelope: Tensor
    synthetic_envelope: Tensor
    observed_trace_rms: Tensor

    def __post_init__(self) -> None:
        shape = self.balanced_observed.shape
        for name in (
            "balanced_synthetic",
            "compensated_synthetic",
            "vertical_visibility",
            "observed_envelope",
            "synthetic_envelope",
        ):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must match the waveform shape.")
        if self.support_mask.shape != shape or self.support_mask.dtype != torch.bool:
            raise ValueError("support_mask must be boolean and match the waveforms.")
        if self.trace_gain.shape != (shape[0],) or self.observed_trace_rms.shape != (shape[0],):
            raise ValueError("trace_gain and observed_trace_rms must have one value per trace.")


class VerticalVisibilityCompensator:
    """Estimate a detached slow visibility field and expose compensated waveforms.

    The caller supplies physical vertical coordinates.  A scalar least-squares
    gain handles acquisition scale, while a unit-geometric-mean visibility field
    handles slow within-trace amplitude variation.  Both nuisance estimates are
    detached; gradients only update the predicted seismic waveform.
    """

    def __init__(self, config: VisibilityCompensationConfig) -> None:
        self.config = config

    @staticmethod
    def _validate(observed: Tensor, synthetic: Tensor, support_mask: Tensor, coordinates_m: Tensor) -> None:
        if observed.ndim != 2 or synthetic.shape != observed.shape:
            raise ValueError("observed and synthetic must have matching (batch, samples) shapes.")
        if support_mask.shape != observed.shape or support_mask.dtype != torch.bool:
            raise ValueError("support_mask must be boolean and match the waveforms.")
        if not torch.is_floating_point(observed) or not torch.is_floating_point(synthetic):
            raise TypeError("observed and synthetic must be floating tensors.")
        if not bool(torch.all(torch.isfinite(observed)).item()) or not bool(torch.all(torch.isfinite(synthetic)).item()):
            raise ValueError("observed and synthetic must be finite.")
        if coordinates_m.ndim == 1:
            valid_coordinates = coordinates_m.shape[0] == observed.shape[1]
        elif coordinates_m.ndim == 2:
            valid_coordinates = coordinates_m.shape == observed.shape
        else:
            valid_coordinates = False
        if not valid_coordinates or not torch.is_floating_point(coordinates_m):
            raise ValueError("coordinates_m must match the waveform sample axis.")
        if bool(torch.any(torch.count_nonzero(support_mask, dim=-1) < 3).item()):
            raise ValueError("Visibility compensation needs at least three supported samples per trace.")

    def compensate(
        self,
        observed: Tensor,
        synthetic: Tensor,
        support_mask: Tensor,
        coordinates_m: Tensor,
    ) -> VisibilityCompensation:
        self._validate(observed, synthetic, support_mask, coordinates_m)
        dtype = synthetic.dtype
        device = synthetic.device
        observed = observed.to(device=device, dtype=dtype)
        coordinates = coordinates_m.to(device=device, dtype=dtype)
        support = support_mask.to(device=device)
        support_float = support.to(dtype=dtype)
        epsilon = torch.finfo(dtype).eps

        with torch.no_grad():
            detached_synthetic = synthetic.detach()
            numerator = torch.sum(observed * detached_synthetic * support_float, dim=-1)
            denominator = torch.sum(torch.square(detached_synthetic) * support_float, dim=-1)
            if bool(torch.any(denominator <= epsilon).item()):
                raise ValueError("Visibility compensation encountered a zero-energy synthetic trace.")
            trace_gain = torch.clamp(numerator / denominator, min=epsilon)
            scaled_synthetic = trace_gain[:, None] * detached_synthetic

            observed_power, observed_support = _masked_gaussian_smooth(
                torch.square(observed),
                coordinates,
                support,
                fwhm_m=self.config.vertical_smoothing_fwhm_m,
            )
            synthetic_power, synthetic_support = _masked_gaussian_smooth(
                torch.square(scaled_synthetic),
                coordinates,
                support,
                fwhm_m=self.config.vertical_smoothing_fwhm_m,
            )
            envelope_support = support & observed_support & synthetic_support
            count = torch.sum(support_float, dim=-1)
            observed_trace_rms = torch.sqrt(
                torch.sum(torch.square(observed) * support_float, dim=-1) / count
            )
            synthetic_trace_rms = torch.sqrt(
                torch.sum(torch.square(scaled_synthetic) * support_float, dim=-1) / count
            )
            observed_floor = self.config.envelope_floor_fraction * observed_trace_rms[:, None]
            synthetic_floor = self.config.envelope_floor_fraction * synthetic_trace_rms[:, None]
            observed_envelope = torch.maximum(torch.sqrt(torch.clamp(observed_power, min=0.0)), observed_floor)
            synthetic_envelope = torch.maximum(torch.sqrt(torch.clamp(synthetic_power, min=0.0)), synthetic_floor)

            log_ratio = torch.log(torch.clamp(observed_envelope, min=epsilon)) - torch.log(
                torch.clamp(synthetic_envelope, min=epsilon)
            )
            smooth_log_ratio, ratio_support = _masked_gaussian_smooth(
                log_ratio,
                coordinates,
                envelope_support,
                fwhm_m=self.config.vertical_smoothing_fwhm_m,
            )
            final_support = envelope_support & ratio_support
            final_float = final_support.to(dtype=dtype)
            final_count = torch.sum(final_float, dim=-1, keepdim=True)
            if bool(torch.any(final_count < 3).item()):
                raise ValueError("Visibility compensation produced insufficient envelope support.")
            centered_log_visibility = smooth_log_ratio - (
                torch.sum(smooth_log_ratio * final_float, dim=-1, keepdim=True) / final_count
            )
            centered_log_visibility = torch.clamp(
                centered_log_visibility,
                min=math.log(self.config.minimum_visibility),
                max=math.log(self.config.maximum_visibility),
            )
            visibility = torch.where(final_support, torch.exp(centered_log_visibility), torch.ones_like(observed))
            balanced_observed = torch.where(
                final_support,
                observed / torch.clamp(observed_envelope, min=epsilon),
                torch.zeros_like(observed),
            )
            balanced_synthetic_detached_scale = trace_gain[:, None] / torch.clamp(synthetic_envelope, min=epsilon)

        balanced_synthetic = torch.where(
            final_support,
            balanced_synthetic_detached_scale * synthetic,
            torch.zeros_like(synthetic),
        )
        compensated_synthetic = trace_gain[:, None] * visibility * synthetic
        return VisibilityCompensation(
            balanced_observed=balanced_observed,
            balanced_synthetic=balanced_synthetic,
            compensated_synthetic=compensated_synthetic,
            support_mask=final_support,
            trace_gain=trace_gain,
            vertical_visibility=visibility,
            observed_envelope=observed_envelope,
            synthetic_envelope=synthetic_envelope,
            observed_trace_rms=observed_trace_rms,
        )

    @staticmethod
    def amplitude_loss(result: VisibilityCompensation, observed: Tensor) -> Tensor:
        support = result.support_mask
        scale = torch.clamp(
            result.observed_trace_rms[:, None],
            min=torch.finfo(observed.dtype).eps,
        )
        return F.smooth_l1_loss(
            (result.compensated_synthetic / scale)[support],
            (observed / scale)[support],
            reduction="mean",
        )


def local_standard_deviation(
    values: Tensor,
    coordinates_m: Tensor,
    support_mask: Tensor,
    *,
    smoothing_fwhm_m: float,
) -> tuple[Tensor, Tensor]:
    """Return a physical-window local standard deviation on masked support."""

    mean, mean_support = _masked_gaussian_smooth(
        values,
        coordinates_m,
        support_mask,
        fwhm_m=smoothing_fwhm_m,
    )
    second, second_support = _masked_gaussian_smooth(
        torch.square(values),
        coordinates_m,
        support_mask,
        fwhm_m=smoothing_fwhm_m,
    )
    support = support_mask & mean_support & second_support
    standard_deviation = torch.sqrt(torch.clamp(second - torch.square(mean), min=0.0))
    return torch.where(support, standard_deviation, torch.zeros_like(values)), support


__all__ = [
    "GainDiagnostic",
    "ShapeLossResult",
    "VerticalVisibilityCompensator",
    "VisibilityCompensation",
    "VisibilityCompensationConfig",
    "analytic_gain_diagnostic",
    "lfm_anchor_loss",
    "local_standard_deviation",
    "masked_lfm_lowpass",
    "masked_physical_lowpass",
    "normalize_support",
    "short_wave_energy_ratio",
    "waveform_shape_loss",
]
