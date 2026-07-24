"""NumPy and differentiable Torch decoder for structured object profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cup.synthetic.core.calibration import PROFILE_METRICS
from cup.synthetic.core.records import SampleAxis

from ginn_v2.truth import RawSegmentParameters, SegmentTruth, ZoneTruth


@dataclass(frozen=True)
class DecoderResult:
    """Decoded high-resolution trace and the two coefficient checkpoints."""

    log_ai: Any
    projected_parameters: tuple[Any, ...]
    effective_parameters: tuple[Any, ...]
    projection_scales: tuple[float, ...]
    c0_adjustments: tuple[Any, ...]


def _zone_model(calibration: Any, zone_id: str) -> Mapping[str, Any]:
    if hasattr(calibration, "zone_models"):
        models = getattr(calibration, "zone_models")
    elif isinstance(calibration, Mapping) and "zone_models" in calibration:
        models = calibration["zone_models"]
    elif isinstance(calibration, Mapping) and "states" in calibration:
        models = {zone_id: calibration}
    else:
        raise TypeError(
            "decoder calibration must expose zone_models or one explicit zone model."
        )
    if not isinstance(models, Mapping) or zone_id not in models:
        raise KeyError(f"Decoder calibration lacks zone {zone_id!r}.")
    model = models[zone_id]
    if not isinstance(model, Mapping):
        raise TypeError(f"Decoder calibration zone {zone_id!r} must be a mapping.")
    for field in ("states", "ai_bounds"):
        if field not in model:
            raise ValueError(f"Decoder calibration zone {zone_id!r} lacks {field}.")
    return model


def _state_model(
    calibration: Any,
    *,
    zone_id: str,
    state: str,
) -> Mapping[str, Any]:
    model = _zone_model(calibration, zone_id)
    states = model["states"]
    if not isinstance(states, Mapping) or state not in states:
        raise KeyError(f"Decoder calibration lacks state {state!r} in zone {zone_id!r}.")
    state_model = states[state]
    if not isinstance(state_model, Mapping) or "coefficients" not in state_model:
        raise ValueError(f"Decoder calibration state {zone_id!r}/{state!r} is incomplete.")
    return state_model


def _distribution(
    state_model: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    coefficients = state_model["coefficients"]
    if not isinstance(coefficients, Mapping) or name not in coefficients:
        raise ValueError(f"Decoder calibration state lacks coefficient distribution {name!r}.")
    distribution = coefficients[name]
    if not isinstance(distribution, Mapping):
        raise TypeError(f"Decoder calibration distribution {name!r} must be a mapping.")
    return distribution


def _metric_bounds(state_model: Mapping[str, Any], name: str) -> tuple[float, float]:
    distribution = _distribution(state_model, name)
    for field in ("median", "robust_sigma"):
        if field not in distribution:
            raise ValueError(f"Decoder calibration distribution {name!r} lacks {field}.")
    center = float(distribution["median"])
    radius = 3.0 * float(distribution["robust_sigma"])
    if not np.isfinite(center) or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"Decoder calibration distribution {name!r} is invalid.")
    return center - radius, center + radius


def _coefficient_bounds(
    state_model: Mapping[str, Any],
    name: str,
) -> tuple[float, float]:
    distribution = _distribution(state_model, name)
    required = ("lower", "upper")
    missing = sorted(set(required).difference(distribution))
    if missing:
        raise ValueError(
            f"Decoder calibration distribution {name!r} lacks explicit bounds {missing}."
        )
    lower = float(distribution["lower"])
    upper = float(distribution["upper"])
    if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
        raise ValueError(f"Decoder calibration distribution {name!r} has invalid bounds.")
    return lower, upper


def _ai_bounds(model: Mapping[str, Any]) -> tuple[float, float]:
    bounds = model["ai_bounds"]
    if not isinstance(bounds, Mapping) or not {"p01", "p99"}.issubset(bounds):
        raise ValueError("Decoder calibration ai_bounds must explicitly contain p01 and p99.")
    lower = float(bounds["p01"])
    upper = float(bounds["p99"])
    if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
        raise ValueError("Decoder calibration ai_bounds are invalid.")
    return lower, upper


def _profile_numpy(parameters: np.ndarray, xi: np.ndarray) -> np.ndarray:
    return (
        parameters[0]
        + parameters[1] * (2.0 * xi - 1.0)
        + parameters[2] * np.sin(np.pi * xi)
    )


def _profile_metrics_numpy(xi: np.ndarray, profile: np.ndarray) -> dict[str, float]:
    endpoint_line = np.interp(xi, xi[[0, -1]], profile[[0, -1]])
    return {
        "profile_mean": float(np.mean(profile)),
        "endpoint_difference": float(profile[-1] - profile[0]),
        "peak_to_peak": float(np.ptp(profile)),
        "internal_extreme_amplitude": float(np.max(np.abs(profile - endpoint_line))),
    }


def _valid_profile_numpy(
    parameters: np.ndarray,
    *,
    xi: np.ndarray,
    state_model: Mapping[str, Any],
) -> bool:
    metrics = _profile_metrics_numpy(xi, _profile_numpy(parameters, xi))
    for name in PROFILE_METRICS:
        lower, upper = _metric_bounds(state_model, name)
        tolerance = 1e-10 * max(1.0, abs(lower), abs(upper))
        if not lower - tolerance <= metrics[name] <= upper + tolerance:
            return False
    return True


def _project_numpy(
    parameters: np.ndarray,
    *,
    xi: np.ndarray,
    state_model: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    candidate = np.asarray(parameters, dtype=np.float64).reshape(3)
    center = np.asarray(
        [
            float(_distribution(state_model, name)["median"])
            for name in ("c0", "c1", "c2")
        ],
        dtype=np.float64,
    )
    if _valid_profile_numpy(candidate, xi=xi, state_model=state_model):
        return candidate, 1.0
    if not _valid_profile_numpy(center, xi=xi, state_model=state_model):
        raise ValueError("Decoder calibration profile center is outside support.")
    lower = 0.0
    upper = 1.0
    for _ in range(48):
        midpoint = 0.5 * (lower + upper)
        trial = center + midpoint * (candidate - center)
        if _valid_profile_numpy(trial, xi=xi, state_model=state_model):
            lower = midpoint
        else:
            upper = midpoint
    interior_scale = max(0.0, lower - 1e-8)
    return center + interior_scale * (candidate - center), float(interior_scale)


def _condition_numpy(
    parameters: np.ndarray,
    *,
    xi: np.ndarray,
    background: np.ndarray,
    state: str,
    state_model: Mapping[str, Any],
    ai_bounds: tuple[float, float],
) -> tuple[np.ndarray, float]:
    output = np.asarray(parameters, dtype=np.float64).reshape(3).copy()
    shape = output[1] * (2.0 * xi - 1.0) + output[2] * np.sin(np.pi * xi)
    c0_lower, c0_upper = _coefficient_bounds(state_model, "c0")
    mean_distribution = _distribution(state_model, "profile_mean")
    mean_median = float(mean_distribution["median"])
    mean_sigma = float(mean_distribution["robust_sigma"])
    if not np.isfinite(mean_median) or not np.isfinite(mean_sigma) or mean_sigma <= 0.0:
        raise ValueError("Decoder calibration profile_mean distribution is invalid.")
    lower = max(
        c0_lower,
        mean_median - 3.0 * mean_sigma - float(np.mean(shape)),
        ai_bounds[0] - float(np.min(background + shape)),
    )
    upper = min(
        c0_upper,
        mean_median + 3.0 * mean_sigma - float(np.mean(shape)),
        ai_bounds[1] - float(np.max(background + shape)),
    )
    epsilon = 32.0 * np.finfo(np.float64).eps
    if state == "high_impedance":
        lower = max(lower, -float(np.mean(shape)) + epsilon)
    elif state == "low_impedance":
        upper = min(upper, -float(np.mean(shape)) - epsilon)
    if lower > upper:
        raise ValueError("Decoder cannot condition c0 inside its calibration support.")
    original = float(output[0])
    output[0] = float(np.clip(original, lower, upper))
    return output, abs(output[0] - original)


def _raw_segment(segment: SegmentTruth | RawSegmentParameters) -> RawSegmentParameters:
    if isinstance(segment, SegmentTruth):
        return segment.raw_parameters()
    if isinstance(segment, RawSegmentParameters):
        return segment
    raise TypeError("decoder segments must be SegmentTruth or RawSegmentParameters.")


def _numpy_scalar(value: Any, *, label: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1 or not np.isfinite(array.reshape(-1)[0]):
        raise ValueError(f"{label} must be one finite scalar.")
    return float(array.reshape(-1)[0])


def _projection_xi_numpy(
    axis: np.ndarray,
    *,
    top: float,
    bottom: float,
) -> np.ndarray:
    coordinates = axis[(axis >= top) & (axis <= bottom)]
    if coordinates.size < 2:
        return np.linspace(0.0, 1.0, 5, dtype=np.float64)
    return (coordinates - top) / (bottom - top)


def decode_numpy(
    zone: ZoneTruth,
    segments: Sequence[SegmentTruth | RawSegmentParameters],
    axis: SampleAxis,
    calibration: Any,
) -> DecoderResult:
    """Decode complete object descriptions into a high-resolution log-AI trace."""
    if not isinstance(zone, ZoneTruth):
        raise TypeError("decode_numpy.zone must be ZoneTruth.")
    if not isinstance(axis, SampleAxis):
        raise TypeError("decode_numpy.axis must be SampleAxis.")
    coordinates = np.asarray(axis.coordinates, dtype=np.float64)
    zone_valid = np.asarray(zone.zone_valid, dtype=bool)
    if zone_valid.shape != coordinates.shape:
        raise ValueError("decode_numpy zone_valid must match the decode axis.")
    if not segments:
        raise ValueError("decode_numpy requires at least one segment.")
    model = _zone_model(calibration, zone.zone_id)
    ai_bounds = _ai_bounds(model)
    background = zone.background_a + zone.background_b * (
        2.0 * (coordinates - zone.top) / (zone.bottom - zone.top) - 1.0
    )
    zone_mask = zone_valid & (coordinates >= zone.top) & (coordinates <= zone.bottom)
    output = np.full(coordinates.shape, np.nan, dtype=np.float64)
    output[zone_mask] = background[zone_mask]
    projected_parameters: list[np.ndarray] = []
    effective_parameters: list[np.ndarray] = []
    projection_scales: list[float] = []
    c0_adjustments: list[float] = []
    segment_values: list[RawSegmentParameters] = []
    for item in segments:
        segment = _raw_segment(item)
        if segment.zone_id != zone.zone_id:
            raise ValueError("decode_numpy segments must belong to the supplied zone.")
        top = _numpy_scalar(segment.top, label="decoder segment top")
        bottom = _numpy_scalar(segment.bottom, label="decoder segment bottom")
        if not top < bottom:
            raise ValueError("decoder segment top must be smaller than bottom.")
        state_model = _state_model(
            calibration,
            zone_id=zone.zone_id,
            state=segment.state,
        )
        raw = np.asarray(
            [
                _numpy_scalar(segment.c0, label="decoder c0"),
                _numpy_scalar(segment.c1, label="decoder c1"),
                _numpy_scalar(segment.c2, label="decoder c2"),
            ],
            dtype=np.float64,
        )
        xi_projection = _projection_xi_numpy(coordinates, top=top, bottom=bottom)
        projected, scale = _project_numpy(
            raw,
            xi=xi_projection,
            state_model=state_model,
        )
        zeta_projection = (top + xi_projection * (bottom - top) - zone.top) / (
            zone.bottom - zone.top
        )
        background_projection = zone.background_a + zone.background_b * (
            2.0 * zeta_projection - 1.0
        )
        effective, adjustment = _condition_numpy(
            projected,
            xi=xi_projection,
            background=background_projection,
            state=segment.state,
            state_model=state_model,
            ai_bounds=ai_bounds,
        )
        projected_parameters.append(projected)
        effective_parameters.append(effective)
        projection_scales.append(scale)
        c0_adjustments.append(adjustment)
        segment_values.append(
            RawSegmentParameters(
                zone_id=segment.zone_id,
                object_id=segment.object_id,
                state=segment.state,
                state_id=segment.state_id,
                top=top,
                bottom=bottom,
                c0=effective[0],
                c1=effective[1],
                c2=effective[2],
            )
        )
    for index, segment in enumerate(segment_values):
        top = float(segment.top)
        bottom = float(segment.bottom)
        xi = (coordinates - top) / (bottom - top)
        profile = _profile_numpy(
            np.asarray([segment.c0, segment.c1, segment.c2], dtype=np.float64),
            xi,
        )
        mask = zone_mask & (coordinates >= top)
        mask &= coordinates <= bottom if index == len(segment_values) - 1 else coordinates < bottom
        output[mask] = background[mask] + profile[mask]
    output[zone_mask] = np.clip(output[zone_mask], ai_bounds[0], ai_bounds[1])
    return DecoderResult(
        log_ai=output,
        projected_parameters=tuple(projected_parameters),
        effective_parameters=tuple(effective_parameters),
        projection_scales=tuple(projection_scales),
        c0_adjustments=tuple(c0_adjustments),
    )


def _torch_scalar(value: Any, *, reference: Any, label: str):
    import torch

    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or not torch.is_floating_point(value):
            raise TypeError(f"{label} must be a floating scalar tensor.")
        return value.to(device=reference.device, dtype=reference.dtype)
    result = torch.as_tensor(float(value), dtype=reference.dtype, device=reference.device)
    if not bool(torch.isfinite(result).item()):
        raise ValueError(f"{label} must be finite.")
    return result


def _profile_torch(parameters: Any, xi: Any):
    import torch

    return (
        parameters[0]
        + parameters[1] * (2.0 * xi - 1.0)
        + parameters[2] * torch.sin(torch.pi * xi)
    )


def _metric_bounds_tensor(
    state_model: Mapping[str, Any],
    name: str,
    *,
    reference: Any,
) -> tuple[Any, Any]:
    import torch

    lower, upper = _metric_bounds(state_model, name)
    return (
        torch.as_tensor(lower, dtype=reference.dtype, device=reference.device),
        torch.as_tensor(upper, dtype=reference.dtype, device=reference.device),
    )


def _valid_profile_torch(
    parameters: Any,
    *,
    xi: Any,
    state_model: Mapping[str, Any],
) -> bool:
    import torch

    profile = _profile_torch(parameters, xi)
    endpoint_line = profile[0] + (profile[-1] - profile[0]) * (
        (xi - xi[0]) / (xi[-1] - xi[0])
    )
    metrics = {
        "profile_mean": torch.mean(profile),
        "endpoint_difference": profile[-1] - profile[0],
        "peak_to_peak": torch.max(profile) - torch.min(profile),
        "internal_extreme_amplitude": torch.max(torch.abs(profile - endpoint_line)),
    }
    for name in PROFILE_METRICS:
        lower, upper = _metric_bounds_tensor(
            state_model,
            name,
            reference=parameters,
        )
        if not bool(((metrics[name] >= lower) & (metrics[name] <= upper)).item()):
            return False
    return True


def _project_torch(
    parameters: Any,
    *,
    xi: Any,
    state_model: Mapping[str, Any],
) -> tuple[Any, float]:
    import torch

    center = torch.stack(
        [
            _torch_scalar(
                _distribution(state_model, name)["median"],
                reference=parameters,
                label=f"decoder {name} center",
            )
            for name in ("c0", "c1", "c2")
        ]
    )
    if _valid_profile_torch(parameters, xi=xi, state_model=state_model):
        return parameters, 1.0
    if not _valid_profile_torch(center, xi=xi, state_model=state_model):
        raise ValueError("Decoder calibration profile center is outside support.")
    lower = 0.0
    upper = 1.0
    for _ in range(48):
        midpoint = 0.5 * (lower + upper)
        trial = center + midpoint * (parameters - center)
        if _valid_profile_torch(trial, xi=xi, state_model=state_model):
            lower = midpoint
        else:
            upper = midpoint
    scale = max(0.0, lower - 1e-8)
    return center + scale * (parameters - center), float(scale)


def _condition_torch(
    parameters: Any,
    *,
    xi: Any,
    background: Any,
    state: str,
    state_model: Mapping[str, Any],
    ai_bounds: tuple[float, float],
) -> tuple[Any, Any]:
    import torch

    shape = parameters[1] * (2.0 * xi - 1.0) + parameters[2] * torch.sin(torch.pi * xi)
    c0_lower, c0_upper = _coefficient_bounds(state_model, "c0")
    mean_distribution = _distribution(state_model, "profile_mean")
    mean_median = float(mean_distribution["median"])
    mean_sigma = float(mean_distribution["robust_sigma"])
    if not np.isfinite(mean_median) or not np.isfinite(mean_sigma) or mean_sigma <= 0.0:
        raise ValueError("Decoder calibration profile_mean distribution is invalid.")
    lower = torch.as_tensor(
        c0_lower,
        dtype=parameters.dtype,
        device=parameters.device,
    )
    lower = torch.maximum(
        lower,
        torch.as_tensor(
            mean_median - 3.0 * mean_sigma,
            dtype=parameters.dtype,
            device=parameters.device,
        )
        - torch.mean(shape),
    )
    lower = torch.maximum(
        lower,
        torch.as_tensor(ai_bounds[0], dtype=parameters.dtype, device=parameters.device)
        - torch.min(background + shape),
    )
    upper = torch.as_tensor(
        c0_upper,
        dtype=parameters.dtype,
        device=parameters.device,
    )
    upper = torch.minimum(
        upper,
        torch.as_tensor(
            mean_median + 3.0 * mean_sigma,
            dtype=parameters.dtype,
            device=parameters.device,
        )
        - torch.mean(shape),
    )
    upper = torch.minimum(
        upper,
        torch.as_tensor(ai_bounds[1], dtype=parameters.dtype, device=parameters.device)
        - torch.max(background + shape),
    )
    epsilon = 32.0 * np.finfo(np.float64).eps
    if state == "high_impedance":
        lower = torch.maximum(
            lower,
            -torch.mean(shape)
            + torch.as_tensor(epsilon, dtype=parameters.dtype, device=parameters.device),
        )
    elif state == "low_impedance":
        upper = torch.minimum(
            upper,
            -torch.mean(shape)
            - torch.as_tensor(epsilon, dtype=parameters.dtype, device=parameters.device),
        )
    if bool((lower > upper).item()):
        raise ValueError("Decoder cannot condition c0 inside its calibration support.")
    c0 = torch.minimum(torch.maximum(parameters[0], lower), upper)
    return torch.stack((c0, parameters[1], parameters[2])), torch.abs(c0 - parameters[0])


def _projection_xi_torch(axis: Any, *, top: Any, bottom: Any):
    import torch

    coordinates = axis[(axis >= top) & (axis <= bottom)]
    if coordinates.numel() < 2:
        return torch.linspace(
            0.0,
            1.0,
            5,
            dtype=axis.dtype,
            device=axis.device,
        )
    return (coordinates - top) / (bottom - top)


def decode_torch(
    zone: ZoneTruth,
    segments: Sequence[SegmentTruth | RawSegmentParameters],
    axis: Any,
    calibration: Any,
    *,
    background_a: Any | None = None,
    background_b: Any | None = None,
) -> DecoderResult:
    """Decode with Torch while retaining gradients for background and raw coefficients."""
    import torch

    if not isinstance(axis, torch.Tensor):
        raise TypeError("decode_torch.axis must be a torch.Tensor.")
    if axis.ndim != 1 or not torch.is_floating_point(axis):
        raise TypeError("decode_torch.axis must be a floating 1D tensor.")
    if not bool(torch.all(torch.isfinite(axis)).item()):
        raise ValueError("decode_torch.axis must be finite.")
    zone_valid = torch.as_tensor(zone.zone_valid, dtype=torch.bool, device=axis.device)
    if zone_valid.shape != axis.shape:
        raise ValueError("decode_torch zone_valid must match the decode axis.")
    if not segments:
        raise ValueError("decode_torch requires at least one segment.")
    model = _zone_model(calibration, zone.zone_id)
    ai_bounds = _ai_bounds(model)
    a = _torch_scalar(
        zone.background_a if background_a is None else background_a,
        reference=axis,
        label="decoder background_a",
    )
    b = _torch_scalar(
        zone.background_b if background_b is None else background_b,
        reference=axis,
        label="decoder background_b",
    )
    zeta = (axis - zone.top) / (zone.bottom - zone.top)
    background = a + b * (2.0 * zeta - 1.0)
    zone_mask = zone_valid & (axis >= zone.top) & (axis <= zone.bottom)
    output = torch.full_like(axis, float("nan"))
    output = torch.where(zone_mask, background, output)
    projected_parameters: list[Any] = []
    effective_parameters: list[Any] = []
    projection_scales: list[float] = []
    c0_adjustments: list[Any] = []
    segment_values: list[RawSegmentParameters] = []
    for item in segments:
        segment = _raw_segment(item)
        if segment.zone_id != zone.zone_id:
            raise ValueError("decode_torch segments must belong to the supplied zone.")
        top = _torch_scalar(segment.top, reference=axis, label="decoder segment top")
        bottom = _torch_scalar(segment.bottom, reference=axis, label="decoder segment bottom")
        if bool((top >= bottom).item()):
            raise ValueError("decoder segment top must be smaller than bottom.")
        state_model = _state_model(
            calibration,
            zone_id=zone.zone_id,
            state=segment.state,
        )
        raw = torch.stack(
            [
                _torch_scalar(segment.c0, reference=axis, label="decoder c0"),
                _torch_scalar(segment.c1, reference=axis, label="decoder c1"),
                _torch_scalar(segment.c2, reference=axis, label="decoder c2"),
            ]
        )
        xi_projection = _projection_xi_torch(axis, top=top, bottom=bottom)
        projected, scale = _project_torch(
            raw,
            xi=xi_projection,
            state_model=state_model,
        )
        projection_coordinates = top + xi_projection * (bottom - top)
        projection_zeta = (projection_coordinates - zone.top) / (zone.bottom - zone.top)
        background_projection = a + b * (2.0 * projection_zeta - 1.0)
        effective, adjustment = _condition_torch(
            projected,
            xi=xi_projection,
            background=background_projection,
            state=segment.state,
            state_model=state_model,
            ai_bounds=ai_bounds,
        )
        projected_parameters.append(projected)
        effective_parameters.append(effective)
        projection_scales.append(scale)
        c0_adjustments.append(adjustment)
        segment_values.append(
            RawSegmentParameters(
                zone_id=segment.zone_id,
                object_id=segment.object_id,
                state=segment.state,
                state_id=segment.state_id,
                top=top,
                bottom=bottom,
                c0=effective[0],
                c1=effective[1],
                c2=effective[2],
            )
        )
    for index, segment in enumerate(segment_values):
        top = _torch_scalar(segment.top, reference=axis, label="decoder segment top")
        bottom = _torch_scalar(segment.bottom, reference=axis, label="decoder segment bottom")
        xi = (axis - top) / (bottom - top)
        parameters = torch.stack((segment.c0, segment.c1, segment.c2))
        profile = _profile_torch(parameters, xi)
        mask = zone_mask & (axis >= top)
        if index == len(segment_values) - 1:
            mask = mask & (axis <= bottom)
        else:
            mask = mask & (axis < bottom)
        output = torch.where(mask, background + profile, output)
    output = torch.clamp(output, min=ai_bounds[0], max=ai_bounds[1])
    return DecoderResult(
        log_ai=output,
        projected_parameters=tuple(projected_parameters),
        effective_parameters=tuple(effective_parameters),
        projection_scales=tuple(projection_scales),
        c0_adjustments=tuple(c0_adjustments),
    )


__all__ = ["DecoderResult", "RawSegmentParameters", "decode_numpy", "decode_torch"]
