"""Slow vertical visibility compensation for the GINN physics closure."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ginn_v2.losses import _gaussian_weights


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
    "VerticalVisibilityCompensator",
    "VisibilityCompensation",
    "VisibilityCompensationConfig",
    "local_standard_deviation",
]
