"""Controlled low-frequency-model contract shared by time and depth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.signal import firwin

from cup.impedance import canonical_lowpass, decompose_log_ai
from cup.synthetic.core.random import RandomNamespace, ar1_irregular, named_rng
from cup.synthetic.schemas import LFM_DEGRADATION_CONTRACT_VERSION
from cup.utils.statistics import centered_rms


LFM_VARIANT_IDS = ("canonical", "controlled_default")
LFM_COMPONENT_ORDER = (
    "constant_bias", "axis_trend", "zonewise_bias", "lateral_smooth_bias",
    "amplitude_scale", "local_missing_control_bias", "over_smoothing",
    "canonicalize_degradation_once",
)
LFM_POLICY_FIELDS = (
    "constant_bias_sigma_log_ai",
    "axis_trend_sigma_log_ai",
    "zonewise_bias_sigma_log_ai",
    "lateral_smooth_bias_sigma_log_ai",
    "lateral_correlation_fraction",
    "amplitude_scale_sigma",
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
    zone_id_model: np.ndarray | None = None
    degradation_variant_id: str = "controlled_default"


@dataclass(frozen=True)
class LfmProducts:
    canonical_background_log_ai: np.ndarray
    target_increment_log_ai: np.ndarray
    controlled_degraded_log_ai: np.ndarray
    residual_vs_lfm_ideal: np.ndarray
    residual_vs_lfm_controlled_degraded: np.ndarray
    qc: Mapping[str, Any]


def _rng(policy: LfmPolicy, coefficient_name: str) -> np.random.Generator:
    return named_rng(
        global_seed=policy.global_seed,
        **policy.random_namespace.keys(),
        stream_purpose="lfm_degradation",
        realization_id=policy.realization_id,
        variant_id="controlled_default",
        coefficient_name=coefficient_name,
    )


def _finite_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    return list(zip(
        np.flatnonzero(~padded[:-1] & padded[1:]).tolist(),
        np.flatnonzero(padded[:-1] & ~padded[1:]).tolist(),
    ))


def _axis_lowpass(
    values: np.ndarray, *, valid: np.ndarray, sample_interval: float,
    cycles_per_axis_unit: float, numtaps: int, beta: float,
) -> np.ndarray:
    nyquist = 0.5 / sample_interval
    if not 0.0 < cycles_per_axis_unit < nyquist:
        raise ValueError("over-smoothing cutoff must lie in (0, Nyquist)")
    if numtaps < 3 or numtaps % 2 == 0:
        raise ValueError("over-smoothing numtaps must be odd and >= 3")
    taps = firwin(numtaps, cycles_per_axis_unit / nyquist, window=("kaiser", beta))
    half = numtaps // 2
    output = np.full_like(values, np.nan, dtype=np.float64)
    for trace, row in enumerate(values):
        for start, stop in _finite_segments(valid[trace] & np.isfinite(row)):
            output[trace, start:stop] = np.convolve(
                np.pad(row[start:stop], (half, half), mode="edge"), taps, mode="valid"
            )
    return output


def _roi_normalized_axis(axis: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float, float]:
    columns = np.any(valid, axis=0)
    if not np.any(columns):
        raise ValueError("controlled LFM requires a non-empty public ROI")
    lower = float(axis[np.flatnonzero(columns)[0]])
    upper = float(axis[np.flatnonzero(columns)[-1]])
    if upper <= lower:
        raise ValueError("controlled LFM public ROI must span multiple axis samples")
    return 2.0 * (axis - lower) / (upper - lower) - 1.0, lower, upper


def _compact_cosine(coordinate: np.ndarray, *, center: float, full_width: float) -> np.ndarray:
    distance = np.abs(np.asarray(coordinate) - center) / (0.5 * full_width)
    result = np.zeros_like(distance, dtype=np.float64)
    inside = distance < 1.0
    result[inside] = 0.5 * (1.0 + np.cos(np.pi * distance[inside]))
    return result


def _controlled_degradation(
    ideal: np.ndarray, *, policy: LfmPolicy, lateral_m: np.ndarray,
    sample_axis: np.ndarray, valid_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = dict(policy.controlled_degraded)
    degraded = np.asarray(ideal, dtype=np.float64).copy()
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(degraded)
    lateral = np.asarray(lateral_m, dtype=np.float64).reshape(-1)
    axis = np.asarray(sample_axis, dtype=np.float64).reshape(-1)
    if degraded.shape != valid.shape or degraded.shape != (lateral.size, axis.size):
        raise ValueError("controlled LFM axes and public mask do not match")
    if policy.zone_id_model is None:
        raise ValueError("controlled LFM requires projected zone_id_model")
    zone_model = np.asarray(policy.zone_id_model)
    if zone_model.shape != degraded.shape:
        raise ValueError("projected zone_id_model shape does not match LFM")
    components: dict[str, np.ndarray] = {}

    constant = float(config["constant_bias_sigma_log_ai"]) * float(_rng(policy, "constant_bias").normal())
    field = np.full_like(degraded, constant)
    degraded[valid] += field[valid]
    components["constant_bias"] = field

    normalized_axis, roi_lower, roi_upper = _roi_normalized_axis(axis, valid)
    trend_scalar = float(config["axis_trend_sigma_log_ai"]) * float(_rng(policy, "axis_trend").normal())
    field = np.broadcast_to(trend_scalar * normalized_axis[None, :], degraded.shape)
    degraded[valid] += field[valid]
    components["axis_trend"] = field

    field = np.zeros_like(degraded)
    for zone in sorted(int(value) for value in np.unique(zone_model[valid & (zone_model >= 0)])):
        value = float(config["zonewise_bias_sigma_log_ai"]) * float(_rng(policy, f"zonewise_bias:{zone}").normal())
        field[zone_model == zone] = value
    degraded[valid] += field[valid]
    components["zonewise_bias"] = field

    spacing = float(np.median(np.diff(lateral)))
    requested = float(config["lateral_correlation_fraction"]) * float(lateral[-1] - lateral[0])
    effective = max(requested, 4.0 * spacing)
    lateral_field, correlation_qc = ar1_irregular(
        lateral, correlation_length_m=effective, rng=_rng(policy, "lateral_smooth_bias")
    )
    field = float(config["lateral_smooth_bias_sigma_log_ai"]) * np.broadcast_to(lateral_field[:, None], degraded.shape)
    degraded[valid] += field[valid]
    components["lateral_smooth_bias"] = field

    amplitude_scale = float(np.exp(float(config["amplitude_scale_sigma"]) * _rng(policy, "amplitude_scale").normal()))
    before_scale = degraded.copy()
    for trace in range(degraded.shape[0]):
        row_valid = valid[trace]
        if np.any(row_valid):
            center = float(np.mean(degraded[trace, row_valid]))
            degraded[trace, row_valid] = center + amplitude_scale * (degraded[trace, row_valid] - center)
    components["amplitude_scale"] = degraded - before_scale

    local = dict(config["local_missing_control_bias"])
    center_lateral = center_axis = amplitude = float("nan")
    field = np.zeros_like(degraded)
    if bool(local["enabled"]):
        lateral_normalized = (lateral - lateral[0]) / (lateral[-1] - lateral[0])
        axis_normalized = (axis - roi_lower) / (roi_upper - roi_lower)
        center_lateral = float(_rng(policy, "local_missing_center_lateral").uniform(0.25, 0.75))
        center_axis = float(_rng(policy, "local_missing_center_axis").uniform(0.25, 0.75))
        amplitude = float(local["max_abs_log_ai"]) * float(np.clip(_rng(policy, "local_missing_amplitude").normal(), -1.0, 1.0))
        field = amplitude * _compact_cosine(
            lateral_normalized, center=center_lateral, full_width=float(local["lateral_width_fraction"])
        )[:, None] * _compact_cosine(
            axis_normalized, center=center_axis, full_width=float(local["axis_width_fraction"])
        )[None, :]
        degraded[valid] += field[valid]
    components["local_missing_control_bias"] = field

    smoothing = dict(config["over_smoothing"])
    before_smoothing = degraded.copy()
    if bool(smoothing["enabled"]):
        filtered = _axis_lowpass(
            degraded, valid=valid, sample_interval=float(np.median(np.diff(axis))),
            cycles_per_axis_unit=float(smoothing["cycles_per_axis_unit"]),
            numtaps=int(smoothing["numtaps"]), beta=float(smoothing["kaiser_beta"]),
        )
        blend = float(smoothing["blend"])
        degraded[valid] = (1.0 - blend) * degraded[valid] + blend * filtered[valid]
    components["over_smoothing"] = degraded - before_smoothing
    degraded[~valid] = np.nan

    qc: dict[str, Any] = {
        "lfm_status": "ok", "lfm_constant_bias": constant,
        "lfm_axis_trend_scalar": trend_scalar, "lfm_amplitude_scale": amplitude_scale,
        "lfm_lateral_requested_correlation_length_m": requested,
        "lfm_lateral_effective_correlation_length_m": effective,
        "lfm_lateral_empirical_correlation_length_m": float(correlation_qc["empirical_correlation_length_m"]),
        "lfm_local_kernel_center_lateral_normalized": center_lateral,
        "lfm_local_kernel_center_axis_normalized": center_axis,
        "lfm_local_kernel_lateral_width_fraction": float(local["lateral_width_fraction"]),
        "lfm_local_kernel_axis_width_fraction": float(local["axis_width_fraction"]),
        "lfm_local_kernel_amplitude": amplitude,
        "lfm_over_smoothing_enabled": bool(smoothing["enabled"]),
        "lfm_over_smoothing_cycles_per_axis_unit": float(smoothing["cycles_per_axis_unit"]),
        "lfm_over_smoothing_numtaps": int(smoothing["numtaps"]),
        "lfm_over_smoothing_kaiser_beta": float(smoothing["kaiser_beta"]),
        "lfm_over_smoothing_blend": float(smoothing["blend"]),
    }
    qc.update({f"lfm_component_{name}_rms": centered_rms(value, valid) for name, value in components.items()})
    return degraded, qc


def build_lfm_products(
    target_log_ai: np.ndarray, sample_axis: np.ndarray, canonical_contract: object, *,
    lateral_coordinates: np.ndarray, valid_mask: np.ndarray, policy: LfmPolicy,
) -> LfmProducts:
    ideal, increment = decompose_log_ai(target_log_ai, sample_axis, canonical_contract, valid_mask=valid_mask)
    raw, qc = _controlled_degradation(
        ideal, policy=policy, lateral_m=lateral_coordinates, sample_axis=sample_axis, valid_mask=valid_mask
    )
    valid = np.asarray(valid_mask, dtype=bool)
    degradation = canonical_lowpass(raw - ideal, sample_axis, canonical_contract, valid_mask=valid)
    degraded = ideal + degradation
    residual_ideal = np.asarray(target_log_ai) - ideal
    residual_degraded = np.asarray(target_log_ai) - degraded
    for array in (ideal, increment, degraded, residual_ideal, residual_degraded):
        array[~valid] = np.nan
    qc.update({
        "lfm_component_canonicalize_degradation_once_rms": centered_rms(
            degradation, valid
        ),
        "lfm_valid_sample_count": int(np.count_nonzero(valid)),
        "lfm_ideal_rms": centered_rms(ideal, valid),
        "lfm_controlled_degraded_rms": centered_rms(degraded, valid),
        "lfm_degradation_rms": centered_rms(degraded - ideal, valid),
        "residual_vs_lfm_ideal_rms": centered_rms(residual_ideal, valid),
        "residual_vs_lfm_controlled_degraded_rms": centered_rms(residual_degraded, valid),
    })
    return LfmProducts(ideal, increment, degraded, residual_ideal, residual_degraded, qc)


def normalized_lfm_component_values(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return only the materialized policy fields consumed by science v2."""
    missing = [key for key in LFM_POLICY_FIELDS if key not in value]
    if missing:
        raise ValueError(f"controlled LFM policy lacks fields: {missing}")
    return {
        key: dict(value[key]) if isinstance(value[key], Mapping) else value[key]
        for key in LFM_POLICY_FIELDS
    }


def build_lfm_degradation_metadata(sample_domain: str, *, axis_unit: str, component_values: Mapping[str, Any]) -> dict[str, Any]:
    domain = str(sample_domain).casefold()
    if (domain, axis_unit) not in {("time", "s"), ("depth", "m")}:
        raise ValueError("LFM domain/unit must be time/s or depth/m")
    return {
        "contract_version": LFM_DEGRADATION_CONTRACT_VERSION,
        "sample_domain": domain, "axis_unit": axis_unit,
        "variant_ids": list(LFM_VARIANT_IDS), "component_order": list(LFM_COMPONENT_ORDER),
        "component_values": normalized_lfm_component_values(component_values),
        "canonicalization": "canonical_lowpass_difference_once",
    }


def validate_lfm_degradation_metadata(value: Mapping[str, Any], *, sample_domain: str) -> dict[str, Any]:
    for key in ("contract_version", "sample_domain", "axis_unit", "variant_ids", "component_order", "component_values", "canonicalization"):
        if key not in value:
            raise ValueError(f"lfm_degradation lacks required field: {key}")
    if not isinstance(value["component_values"], Mapping):
        raise ValueError("lfm_degradation component_values must be a mapping")
    if not value["component_values"]:
        raise ValueError("lfm_degradation component_values must not be empty")
    unexpected = sorted(set(value["component_values"]) - set(LFM_POLICY_FIELDS))
    if unexpected:
        raise ValueError(
            f"lfm_degradation component_values has non-scientific fields: {unexpected}"
        )
    normalized_lfm_component_values(value["component_values"])
    expected = build_lfm_degradation_metadata(
        sample_domain,
        axis_unit="s" if sample_domain == "time" else "m",
        component_values=value["component_values"],
    )
    for key in ("contract_version", "sample_domain", "axis_unit", "variant_ids", "component_order", "canonicalization"):
        if value[key] != expected[key]:
            raise ValueError(f"lfm_degradation {key} does not match science v2")
    return dict(value)


__all__ = ["LFM_COMPONENT_ORDER", "LFM_POLICY_FIELDS", "LFM_VARIANT_IDS", "LfmPolicy", "LfmProducts", "build_lfm_products", "build_lfm_degradation_metadata", "normalized_lfm_component_values", "validate_lfm_degradation_metadata"]
