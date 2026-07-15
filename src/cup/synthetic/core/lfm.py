"""Domain-neutral metadata for low-frequency-model degradation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.signal import firwin

from cup.impedance import canonical_lowpass, decompose_log_ai
from cup.synthetic.core.random import RandomNamespace, ar1_irregular, named_rng
from cup.utils.statistics import centered_rms


LFM_VARIANT_IDS = ("canonical", "controlled_default")
LFM_COMPONENT_ORDER = (
    "constant_bias",
    "axis_trend",
    "zonewise_bias",
    "lateral_smooth_bias",
    "amplitude_scale",
    "local_missing_control_bias",
    "over_smoothing",
)


@dataclass(frozen=True)
class LfmPolicy:
    sample_domain: str
    axis_unit: str
    global_seed: int
    random_namespace: RandomNamespace
    realization_id: str
    horizon_coordinates: np.ndarray
    controlled_degraded: Mapping[str, Any]
    algorithm: str = "depth"
    zone_id_model: np.ndarray | None = None
    degradation_variant_id: str | None = None


@dataclass(frozen=True)
class LfmProducts:
    canonical_background_log_ai: np.ndarray
    target_increment_log_ai: np.ndarray
    controlled_degraded_log_ai: np.ndarray
    residual_vs_lfm_ideal: np.ndarray
    residual_vs_lfm_controlled_degraded: np.ndarray
    qc: Mapping[str, Any] | None = None


def _finite_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _axis_lowpass(
    values: np.ndarray,
    *,
    valid_mask: np.ndarray,
    sample_interval: float,
    wavelength: float,
    numtaps: int,
    beta: float,
) -> np.ndarray:
    nyquist = 0.5 / sample_interval
    cutoff = 1.0 / wavelength
    if not 0.0 < cutoff < nyquist:
        raise ValueError("LFM spatial cutoff is outside (0, Nyquist).")
    count = int(numtaps)
    if count < 3 or count % 2 == 0:
        raise ValueError("LFM numtaps must be odd and >= 3.")
    taps = firwin(count, cutoff / nyquist, window=("kaiser", float(beta)), scale=True)
    half = count // 2
    array = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(array)
    output = np.full_like(array, np.nan, dtype=np.float64)
    for lateral_index, row in enumerate(array):
        for start, stop in _finite_segments(valid[lateral_index]):
            segment = row[start:stop]
            padded = np.pad(segment, (half, half), mode="edge")
            output[lateral_index, start:stop] = np.convolve(padded, taps, mode="valid")
    return output


def _controlled_degraded_lfm(
    base: np.ndarray,
    *,
    policy: LfmPolicy,
    lateral_coordinates: np.ndarray,
    sample_axis: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    config = policy.controlled_degraded
    output = np.asarray(base, dtype=np.float64).copy()
    valid = np.asarray(valid_mask, dtype=bool).copy()
    if valid.shape != output.shape:
        raise ValueError("LFM valid_mask must match the base array.")
    valid &= np.isfinite(output)
    lateral = np.asarray(lateral_coordinates, dtype=np.float64)
    axis = np.asarray(sample_axis, dtype=np.float64)
    horizons = np.asarray(policy.horizon_coordinates, dtype=np.float64)

    def rng(purpose: str) -> np.random.Generator:
        return named_rng(
            global_seed=policy.global_seed,
            benchmark_version=policy.random_namespace.benchmark_version,
            generator_family=policy.random_namespace.generator_family,
            stream_purpose=purpose,
            realization_id=policy.realization_id,
        )

    output[valid] += rng("lfm_constant_bias").normal(
        0.0, float(config["constant_bias_sigma_log_ai"])
    )
    trend_amplitude = rng("lfm_vertical_trend").normal(
        0.0, float(config["linear_vertical_trend_sigma_log_ai"])
    )
    trend = trend_amplitude * np.linspace(-1.0, 1.0, axis.size)[None, :]
    output[valid] += np.broadcast_to(trend, output.shape)[valid]
    zone_rng = rng("lfm_zonewise_bias")
    for zone_index in range(horizons.shape[1] - 1):
        bias = zone_rng.normal(0.0, float(config["zonewise_bias_sigma_log_ai"]))
        mask = (axis[None, :] >= horizons[:, zone_index : zone_index + 1]) & (
            axis[None, :] <= horizons[:, zone_index + 1 : zone_index + 2]
        )
        output[mask & valid] += bias
    lateral_rng = rng("lfm_lateral_smooth_bias")
    field = lateral_rng.normal(size=lateral.size)
    requested = float(config["lateral_correlation_fraction"]) * max(
        float(lateral[-1]), 1.0
    )
    spacing = float(np.median(np.diff(lateral)))
    alpha = np.exp(-spacing / max(requested, spacing))
    for index in range(1, field.size):
        field[index] = alpha * field[index - 1] + np.sqrt(1.0 - alpha**2) * field[index]
    field *= float(config["lateral_smooth_bias_sigma_log_ai"]) / max(
        float(np.std(field)), np.finfo(np.float64).eps
    )
    output[valid] += np.broadcast_to(field[:, None], output.shape)[valid]
    finite_valid = valid & np.isfinite(output)
    if not np.any(finite_valid):
        return np.full_like(output, np.nan, dtype=np.float64)
    center = float(np.mean(output[finite_valid]))
    scale = np.exp(
        rng("lfm_amplitude_scale").normal(0.0, float(config["amplitude_scale_sigma"]))
    )
    output[finite_valid] = center + scale * (output[finite_valid] - center)
    local = dict(config["local_missing_control_bias"])
    if bool(local["enabled"]):
        local_rng = rng("lfm_local_missing_control_bias")
        center_l = local_rng.uniform(0.2, 0.8) * max(float(lateral[-1]), 1.0)
        center_z = local_rng.uniform(float(axis[0]), float(axis[-1]))
        width_l = max(
            float(local["lateral_width_fraction"]) * max(float(lateral[-1]), 1.0),
            spacing,
        )
        width_z = max(
            float(local["vertical_width_fraction"]) * float(axis[-1] - axis[0]),
            float(np.diff(axis[:2])[0]),
        )
        amplitude = local_rng.uniform(-1.0, 1.0) * float(local["max_abs_log_ai"])
        blob = np.exp(
            -0.5 * ((lateral[:, None] - center_l) / width_l) ** 2
            - 0.5 * ((axis[None, :] - center_z) / width_z) ** 2
        )
        output[valid] += np.asarray(amplitude * blob)[valid]
    smoothing = dict(config.get("over_smoothing") or {})
    if bool(smoothing.get("enabled", False)):
        over = _axis_lowpass(
            output,
            valid_mask=valid,
            sample_interval=float(np.diff(axis[:2])[0]),
            wavelength=float(smoothing["minimum_wavelength_m"]),
            numtaps=int(smoothing["numtaps"]),
            beta=float(smoothing["kaiser_beta"]),
        )
        blend = float(smoothing["blend"])
        smoothed = finite_valid & np.isfinite(over)
        output[smoothed] = (1.0 - blend) * output[smoothed] + blend * over[smoothed]
    output[~valid] = np.nan
    return output


def _normalized_axis(values: np.ndarray) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    span = float(axis[-1] - axis[0]) if axis.size > 1 else 0.0
    if span <= 0.0:
        return np.zeros_like(axis)
    return 2.0 * (axis - float(axis[0])) / span - 1.0


def _windowed_bump(size: int, *, center: float, width: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, size, dtype=np.float64)
    half = max(float(width), np.finfo(np.float64).eps) * 0.5
    distance = np.abs(x - float(center)) / half
    output = np.zeros_like(x)
    inside = distance < 1.0
    output[inside] = 0.5 * (1.0 + np.cos(np.pi * distance[inside]))
    return output


def _time_rng(policy: LfmPolicy, coefficient_name: str) -> np.random.Generator:
    variant_id = policy.degradation_variant_id or policy.realization_id
    return named_rng(
        global_seed=policy.global_seed,
        benchmark_version=policy.random_namespace.benchmark_version,
        generator_family=policy.random_namespace.generator_family,
        stream_purpose="lfm_degradation",
        realization_id=policy.realization_id,
        coefficient_name=coefficient_name,
        variant_id=variant_id,
    )


def _time_controlled_degraded_lfm(
    ideal: np.ndarray,
    target: np.ndarray,
    *,
    policy: LfmPolicy,
    lateral_coordinates: np.ndarray,
    sample_axis: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the frozen time-domain controlled degradation policy."""
    config = dict(policy.controlled_degraded)
    degraded = np.asarray(ideal, dtype=np.float64).copy()
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(degraded)
    target_array = np.asarray(target, dtype=np.float64)
    lateral = np.asarray(lateral_coordinates, dtype=np.float64)
    axis = np.asarray(sample_axis, dtype=np.float64)
    components: dict[str, float] = {}

    constant_bias = float(config.get("constant_bias_sigma_log_ai", 0.02)) * float(
        _time_rng(policy, "constant_log_ai_bias").normal()
    )
    degraded[valid] += constant_bias
    components["constant_log_ai_bias"] = constant_bias

    trend_bias = (
        float(config.get("linear_twt_trend_sigma_log_ai", 0.02))
        * float(_time_rng(policy, "linear_twt_trend_bias").normal())
        * np.broadcast_to(_normalized_axis(axis)[None, :], target_array.shape)
    )
    degraded[valid] += trend_bias[valid]
    components["linear_twt_trend_bias_rms"] = centered_rms(trend_bias, valid)

    if policy.zone_id_model is None:
        raise ValueError("time LFM policy requires zone_id_model.")
    zone_model = np.asarray(policy.zone_id_model)
    zone_bias = np.zeros_like(target_array)
    zone_sigma = float(config.get("zonewise_bias_sigma_log_ai", 0.03))
    for zone_value in sorted(int(value) for value in np.unique(zone_model[zone_model >= 0])):
        zone_bias[zone_model == zone_value] = zone_sigma * float(
            _time_rng(policy, f"zonewise_bias:{zone_value}").normal()
        )
    degraded[valid] += zone_bias[valid]
    components["zonewise_bias_rms"] = centered_rms(zone_bias, valid)

    lateral_sigma = float(config.get("lateral_smooth_bias_sigma_log_ai", 0.02))
    if lateral.size >= 2 and lateral_sigma > 0.0:
        requested_lx = float(config.get("lateral_correlation_fraction", 0.3)) * max(
            float(lateral[-1] - lateral[0]), np.finfo(np.float64).eps
        )
        spacing = float(np.median(np.diff(lateral)))
        effective_lx = max(requested_lx, 4.0 * spacing)
        field, field_qc = ar1_irregular(
            lateral,
            correlation_length_m=effective_lx,
            rng=_time_rng(policy, "lateral_smooth_bias_field"),
        )
        lateral_bias = lateral_sigma * field[:, None] * np.ones_like(target_array)
    else:
        requested_lx = float("nan")
        effective_lx = float("nan")
        field_qc = {}
        lateral_bias = np.zeros_like(target_array)
    degraded[valid] += lateral_bias[valid]
    components["lateral_smooth_bias_rms"] = centered_rms(lateral_bias, valid)

    amplitude_scale = float(
        np.exp(
            float(config.get("amplitude_scale_sigma", 0.05))
            * float(_time_rng(policy, "amplitude_scale_bias").normal())
        )
    )
    for lateral_index in range(degraded.shape[0]):
        row_valid = valid[lateral_index] & np.isfinite(degraded[lateral_index])
        if np.any(row_valid):
            mean = float(np.mean(degraded[lateral_index, row_valid]))
            degraded[lateral_index, row_valid] = mean + amplitude_scale * (
                degraded[lateral_index, row_valid] - mean
            )

    missing = dict(config.get("local_missing_control_bias") or {})
    if bool(missing.get("enabled", True)):
        rng = _time_rng(policy, "local_missing_control_bias")
        lateral_window = _windowed_bump(
            target_array.shape[0],
            center=float(rng.uniform(0.25, 0.75)),
            width=float(missing.get("lateral_width_fraction", 0.30)),
        )
        axis_window = _windowed_bump(
            target_array.shape[1],
            center=float(rng.uniform(0.25, 0.75)),
            width=float(missing.get("twt_width_fraction", 0.30)),
        )
        amplitude = float(missing.get("max_abs_log_ai", 0.04)) * float(
            np.clip(rng.normal(), -1.0, 1.0)
        )
        missing_bias = amplitude * lateral_window[:, None] * axis_window[None, :]
        degraded[valid] += missing_bias[valid]
        components["local_missing_control_bias_peak"] = amplitude
        components["local_missing_control_bias_rms"] = centered_rms(missing_bias, valid)

    smoothing = dict(config.get("over_smoothing") or {})
    over_enabled = bool(smoothing.get("enabled", False))
    over_cutoff = float("nan")
    blend = 0.0
    if over_enabled:
        over_cutoff = float(smoothing.get("cutoff_hz", 6.0))
        blend = float(smoothing.get("blend", 1.0))
        over = _axis_lowpass(
            degraded,
            valid_mask=valid,
            sample_interval=float(np.diff(axis[:2])[0]),
            wavelength=1.0 / over_cutoff,
            numtaps=int(smoothing.get("numtaps", 129)),
            beta=float(smoothing.get("kaiser_beta", 8.6)),
        )
        degraded[valid] = (1.0 - blend) * degraded[valid] + blend * over[valid]

    return degraded, {
        "lfm_status": "ok",
        "lfm_controlled_degraded_over_smoothing_enabled": over_enabled,
        "lfm_controlled_degraded_over_smoothing_cutoff_hz": over_cutoff,
        "lfm_controlled_degraded_over_smoothing_blend": blend,
        "lfm_amplitude_scale_bias": amplitude_scale,
        "lfm_lateral_requested_correlation_length_m": requested_lx,
        "lfm_lateral_effective_correlation_length_m": effective_lx,
        "lfm_lateral_empirical_correlation_length_m": float(
            field_qc.get("empirical_correlation_length_m", float("nan"))
        ),
        **{f"lfm_component_{key}": value for key, value in components.items()},
    }


def build_lfm_products(
    target_log_ai: np.ndarray,
    sample_axis: np.ndarray,
    canonical_contract: object,
    *,
    lateral_coordinates: np.ndarray,
    valid_mask: np.ndarray,
    policy: LfmPolicy,
) -> LfmProducts:
    """Build canonical and controlled LFM products in their frozen order."""
    ideal, target_increment = decompose_log_ai(
        target_log_ai,
        sample_axis,
        canonical_contract,
        valid_mask=valid_mask,
    )
    if policy.algorithm == "time":
        degraded, qc = _time_controlled_degraded_lfm(
            ideal,
            target_log_ai,
            policy=policy,
            lateral_coordinates=lateral_coordinates,
            sample_axis=sample_axis,
            valid_mask=valid_mask,
        )
    elif policy.algorithm == "depth":
        degraded = _controlled_degraded_lfm(
            ideal,
            policy=policy,
            lateral_coordinates=lateral_coordinates,
            sample_axis=sample_axis,
            valid_mask=valid_mask,
        )
        qc = {}
    else:
        raise ValueError(f"Unsupported LFM algorithm: {policy.algorithm!r}")
    degradation = canonical_lowpass(
        degraded - ideal,
        sample_axis,
        canonical_contract,
        valid_mask=valid_mask,
    )
    degraded = ideal + degradation
    degraded[~np.asarray(valid_mask, dtype=bool)] = np.nan
    valid = np.asarray(valid_mask, dtype=bool)
    ideal[~valid] = np.nan
    degraded[~valid] = np.nan
    residual_ideal = np.asarray(target_log_ai) - ideal
    residual_degraded = np.asarray(target_log_ai) - degraded
    residual_ideal[~valid] = np.nan
    residual_degraded[~valid] = np.nan
    if policy.algorithm == "time":
        contract = canonical_contract
        qc.update(
            {
                "lfm_ideal_cutoff_hz": float(contract.cutoff),
                "lfm_ideal_numtaps": 0,
                "lfm_ideal_filter_family": contract.implementation,
                "lfm_valid_sample_count": int(np.count_nonzero(valid)),
                "lfm_ideal_rms": centered_rms(ideal, valid),
                "lfm_controlled_degraded_rms": centered_rms(degraded, valid),
                "lfm_degradation_rms": centered_rms(degraded - ideal, valid),
                "residual_vs_lfm_ideal_rms": centered_rms(residual_ideal, valid),
                "residual_vs_lfm_controlled_degraded_rms": centered_rms(
                    residual_degraded, valid
                ),
            }
        )
    return LfmProducts(
        canonical_background_log_ai=ideal,
        target_increment_log_ai=target_increment,
        controlled_degraded_log_ai=degraded,
        residual_vs_lfm_ideal=residual_ideal,
        residual_vs_lfm_controlled_degraded=residual_degraded,
        qc=qc,
    )


def build_lfm_degradation_metadata(
    sample_domain: str,
    *,
    axis_unit: str,
    component_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the shared LFM degradation sequence without fixing its operator."""
    domain = str(sample_domain).strip().casefold()
    if domain not in {"time", "depth"}:
        raise ValueError(f"Unsupported LFM sample domain: {sample_domain!r}")
    unit = str(axis_unit).strip()
    if not unit:
        raise ValueError("LFM axis_unit must be non-empty.")
    values = dict(component_values or {})
    return {
        "sample_domain": domain,
        "axis_unit": unit,
        "variant_ids": list(LFM_VARIANT_IDS),
        "component_order": list(LFM_COMPONENT_ORDER),
        "component_values": values,
        "canonicalization": "canonical_lowpass_difference_once",
    }


def validate_lfm_degradation_metadata(
    value: Mapping[str, Any],
    *,
    sample_domain: str,
) -> dict[str, Any]:
    """Validate the shared LFM component/variant metadata in a manifest."""
    if not isinstance(value, Mapping):
        raise ValueError("lfm_degradation must be a mapping.")
    expected = build_lfm_degradation_metadata(
        sample_domain,
        axis_unit="s" if str(sample_domain).casefold() == "time" else "m",
    )
    for key in (
        "sample_domain",
        "axis_unit",
        "variant_ids",
        "component_order",
        "component_values",
        "canonicalization",
    ):
        if key not in value:
            raise ValueError(f"lfm_degradation lacks required field: {key}")
    if str(value["sample_domain"]).casefold() != expected["sample_domain"]:
        raise ValueError("lfm_degradation sample_domain does not match reader.")
    if str(value["axis_unit"]) != expected["axis_unit"]:
        raise ValueError("lfm_degradation axis_unit does not match reader.")
    if list(value["variant_ids"]) != list(LFM_VARIANT_IDS):
        raise ValueError("lfm_degradation variant_ids do not match v4.")
    if list(value["component_order"]) != list(LFM_COMPONENT_ORDER):
        raise ValueError("lfm_degradation component_order does not match v4.")
    if not isinstance(value["component_values"], Mapping):
        raise ValueError("lfm_degradation component_values must be a mapping.")
    if str(value["canonicalization"]) != "canonical_lowpass_difference_once":
        raise ValueError("lfm_degradation canonicalization is not supported.")
    return {str(key): item for key, item in value.items()}


__all__ = [
    "LFM_COMPONENT_ORDER",
    "LFM_VARIANT_IDS",
    "LfmPolicy",
    "LfmProducts",
    "build_lfm_products",
    "build_lfm_degradation_metadata",
    "validate_lfm_degradation_metadata",
]
