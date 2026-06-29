"""Field-conditioned truth generation for ``object_coefficients_v1``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from cup.seismic.observability import acoustic_reflectivity_from_log_ai, forward_log_ai
from cup.synthetic.calibration import (
    PROFILE_METRICS,
    ImpedanceCalibration,
    STATE_NAMES,
    object_profile_metrics,
)
from cup.synthetic.forward import antialias_taps, categorical_model_grids, downsample_continuous
from cup.synthetic.random import ar1_irregular, named_rng


@dataclass(frozen=True)
class GenerationScenario:
    scenario_id: str
    duration_mode: str
    geometry_family: str
    geometry_direction: str
    correlation_length_fraction: float
    coefficient_sigma_multiplier: float
    thickness_log_sigma: float
    variant_id: str = ""


@dataclass(frozen=True)
class GeneratedSection:
    realization_id: str
    scenario: GenerationScenario
    lateral_m: np.ndarray
    inline_float: np.ndarray
    xline_float: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    twt_highres_s: np.ndarray
    twt_model_s: np.ndarray
    truth_log_ai_highres: np.ndarray
    model_target_log_ai: np.ndarray
    reflectivity_highres: np.ndarray
    reflectivity_model: np.ndarray
    seismic_model_consistent: np.ndarray
    rgt_highres: np.ndarray
    rgt_model: np.ndarray
    state_id_highres: np.ndarray
    object_id_highres: np.ndarray
    object_xi_highres: np.ndarray
    zone_id_highres: np.ndarray
    geometry_event_mask_highres: np.ndarray
    boundary_mask_highres: np.ndarray
    boundary_fraction_model: np.ndarray
    boundary_mask_model: np.ndarray
    state_fraction_model: np.ndarray
    dominant_object_id_model: np.ndarray
    zone_id_model: np.ndarray
    valid_mask_model: np.ndarray
    forward_valid_mask_highres: np.ndarray
    forward_valid_mask_model: np.ndarray
    object_catalog: list[dict[str, Any]]
    object_lateral_coefficients: list[dict[str, Any]]
    qc: dict[str, Any]


class GenerationRejected(ValueError):
    """A complete realization that failed one or more frozen QC rules."""

    def __init__(
        self,
        reasons: Sequence[str],
        *,
        diagnostics: Mapping[str, Any],
        details: Sequence[Mapping[str, Any]],
    ) -> None:
        unique_reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
        super().__init__(";".join(unique_reasons))
        self.reasons = unique_reasons
        self.diagnostics = dict(diagnostics)
        self.details = [dict(item) for item in details]


def _rng_keys(
    calibration: ImpedanceCalibration,
    *,
    global_seed: int,
    stream_purpose: str,
    realization_id: str,
    zone_id: str = "",
    object_id: str = "",
    coefficient_name: str = "",
    variant_id: str = "",
) -> dict[str, Any]:
    return {
        "global_seed": int(global_seed),
        "benchmark_version": (
            "synthoseis_lite_v2"
            if str(calibration.generator_family).endswith("_v2")
            else "synthoseis_lite_v1"
        ),
        "generator_family": calibration.generator_family,
        "stream_purpose": stream_purpose,
        "realization_id": realization_id,
        "zone_id": zone_id,
        "object_id": object_id,
        "coefficient_name": coefficient_name,
        "variant_id": variant_id,
    }


def _truncated_normal(
    distribution: Mapping[str, Any],
    rng: np.random.Generator,
) -> float:
    center = float(distribution["median"])
    sigma = float(distribution["robust_sigma"])
    lower = float(distribution["lower"] if "lower" in distribution else distribution["p01"])
    upper = float(distribution["upper"] if "upper" in distribution else distribution["p99"])
    for _ in range(64):
        value = center + sigma * rng.normal()
        if lower <= value <= upper:
            return float(value)
    return float(np.clip(center, lower, upper))


def _sample_object_sequence(
    calibration: ImpedanceCalibration,
    *,
    zone_id: str,
    realization_id: str,
    global_seed: int,
    minimum_fraction: float,
    variant_id: str,
) -> list[dict[str, Any]]:
    model = calibration.zone_models[zone_id]
    initial_rng = named_rng(
        **_rng_keys(
            calibration,
            global_seed=global_seed,
            stream_purpose="state_sequence",
            realization_id=realization_id,
            zone_id=zone_id,
            variant_id=variant_id,
        )
    )
    state = int(initial_rng.choice(3, p=np.asarray(model["initial_probabilities"], dtype=np.float64)))
    transition = np.asarray(model["transition_matrix"], dtype=np.float64)
    objects: list[dict[str, Any]] = []
    consumed = 0.0
    for object_index in range(512):
        object_id = f"{zone_id}:{object_index}"
        state_name = STATE_NAMES[state]
        state_model = model["states"][state_name]
        duration_rng = named_rng(
            **_rng_keys(
                calibration,
                global_seed=global_seed,
                stream_purpose="duration",
                realization_id=realization_id,
                zone_id=zone_id,
                object_id=object_id,
                variant_id=variant_id,
            )
        )
        log_duration = _truncated_normal(state_model["log_duration"], duration_rng)
        duration = max(float(np.exp(log_duration)), float(minimum_fraction))
        if consumed + duration >= 1.0:
            remaining = 1.0 - consumed
            if remaining < minimum_fraction and objects:
                objects[-1]["base_duration"] += remaining
            else:
                objects.append(
                    {
                        "object_id": object_id,
                        "object_index": object_index,
                        "state_id": state,
                        "state": state_name,
                        "base_duration": remaining,
                        "right_censored": True,
                    }
                )
            break
        objects.append(
            {
                "object_id": object_id,
                "object_index": object_index,
                "state_id": state,
                "state": state_name,
                "base_duration": duration,
                "right_censored": False,
            }
        )
        consumed += duration
        state = int(initial_rng.choice(3, p=transition[state]))
    if not objects or not np.isclose(sum(item["base_duration"] for item in objects), 1.0, atol=1e-8):
        raise ValueError(f"invalid_state_sequence:{zone_id}")
    return objects


def _sample_background(
    calibration: ImpedanceCalibration,
    *,
    zone_id: str,
    realization_id: str,
    global_seed: int,
    variant_id: str,
) -> tuple[float, float]:
    model = calibration.zone_models[zone_id]["background"]
    output = []
    for name in ("background_a", "background_b"):
        distribution = model[name]
        rng = named_rng(
            **_rng_keys(
                calibration,
                global_seed=global_seed,
                stream_purpose="zone_background",
                realization_id=realization_id,
                zone_id=zone_id,
                coefficient_name=name,
                variant_id=variant_id,
            )
        )
        output.append(_truncated_normal(distribution, rng))
    return float(output[0]), float(output[1])


def _project_profile_coefficients(
    coefficients: np.ndarray,
    *,
    xi: np.ndarray,
    state_model: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    """Contract one coefficient vector toward its calibrated center until valid."""
    candidate = np.asarray(coefficients, dtype=np.float64).reshape(3)
    coordinate = np.asarray(xi, dtype=np.float64).reshape(-1)
    coefficient_models = state_model["coefficients"]
    center = np.asarray(
        [float(coefficient_models[name]["median"]) for name in ("c0", "c1", "c2")],
        dtype=np.float64,
    )

    def metric_bounds(name: str) -> tuple[float, float]:
        distribution = coefficient_models[name]
        center_value = float(distribution["median"])
        radius = 3.0 * float(distribution["robust_sigma"])
        return center_value - radius, center_value + radius

    def valid(values: np.ndarray) -> bool:
        profile = (
            values[0]
            + values[1] * (2.0 * coordinate - 1.0)
            + values[2] * np.sin(np.pi * coordinate)
        )
        metrics = object_profile_metrics(coordinate, profile)
        for name in PROFILE_METRICS:
            lower_bound, upper_bound = metric_bounds(name)
            tolerance = 1e-10 * max(1.0, abs(lower_bound), abs(upper_bound))
            if not lower_bound - tolerance <= metrics[name] <= upper_bound + tolerance:
                return False
        return True

    if valid(candidate):
        return candidate, 1.0
    if not valid(center):
        raise ValueError("invalid_impedance_calibration:profile_center_outside_bounds")
    lower = 0.0
    upper = 1.0
    for _ in range(48):
        midpoint = 0.5 * (lower + upper)
        trial = center + midpoint * (candidate - center)
        if valid(trial):
            lower = midpoint
        else:
            upper = midpoint
    interior_scale = max(0.0, lower - 1e-8)
    return center + interior_scale * (candidate - center), float(interior_scale)


def _condition_c0_to_ai_bounds(
    coefficients: np.ndarray,
    *,
    xi: np.ndarray,
    background: np.ndarray,
    state: str,
    state_model: Mapping[str, Any],
    ai_bounds: Mapping[str, Any],
) -> tuple[np.ndarray, float]:
    """Condition the location coefficient on profile, state and absolute AI bounds."""
    output = np.asarray(coefficients, dtype=np.float64).reshape(3).copy()
    coordinate = np.asarray(xi, dtype=np.float64).reshape(-1)
    background_values = np.asarray(background, dtype=np.float64).reshape(-1)
    shape = output[1] * (2.0 * coordinate - 1.0) + output[2] * np.sin(np.pi * coordinate)
    c0_model = state_model["coefficients"]["c0"]
    mean_model = state_model["coefficients"]["profile_mean"]
    lower = float(c0_model["lower"])
    upper = float(c0_model["upper"])
    mean_radius = 3.0 * float(mean_model["robust_sigma"])
    lower = max(
        lower,
        float(mean_model["median"]) - mean_radius - float(np.mean(shape)),
        float(ai_bounds["p01"]) - float(np.min(background_values + shape)),
    )
    upper = min(
        upper,
        float(mean_model["median"]) + mean_radius - float(np.mean(shape)),
        float(ai_bounds["p99"]) - float(np.max(background_values + shape)),
    )
    state_epsilon = 32.0 * np.finfo(np.float64).eps
    if state == "high_impedance":
        lower = max(lower, -float(np.mean(shape)) + state_epsilon)
    elif state == "low_impedance":
        upper = min(upper, -float(np.mean(shape)) - state_epsilon)
    if lower > upper:
        raise ValueError("invalid_object_profile:no_feasible_c0")
    original = float(output[0])
    output[0] = float(np.clip(original, lower, upper))
    return output, abs(float(output[0]) - original)


def _object_lateral_parameters(
    calibration: ImpedanceCalibration,
    *,
    objects: list[dict[str, Any]],
    zone_id: str,
    realization_id: str,
    global_seed: int,
    lateral_m: np.ndarray,
    scenario: GenerationScenario,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n_objects = len(objects)
    n_lateral = lateral_m.size
    coefficients = np.empty((n_objects, n_lateral, 3), dtype=np.float64)
    thickness_weights = np.empty((n_objects, n_lateral), dtype=np.float64)
    field_qc: list[dict[str, Any]] = []
    requested_lx = float(scenario.correlation_length_fraction * lateral_m[-1])
    spacing = float(np.median(np.diff(lateral_m)))
    effective_lx = max(requested_lx, 4.0 * spacing)
    for object_index, item in enumerate(objects):
        state_model = calibration.zone_models[zone_id]["states"][item["state"]]
        for coefficient_index, coefficient_name in enumerate(("c0", "c1", "c2")):
            distribution = state_model["coefficients"][coefficient_name]
            base_rng = named_rng(
                **_rng_keys(
                    calibration,
                    global_seed=global_seed,
                    stream_purpose=f"coefficient_{coefficient_name}",
                    realization_id=realization_id,
                    zone_id=zone_id,
                    object_id=item["object_id"],
                    coefficient_name=coefficient_name,
                    variant_id=scenario.variant_id,
                )
            )
            base = _truncated_normal(distribution, base_rng)
            field, qc = ar1_irregular(
                lateral_m,
                correlation_length_m=effective_lx,
                rng=named_rng(
                    **_rng_keys(
                        calibration,
                        global_seed=global_seed,
                        stream_purpose="coefficient_lateral",
                        realization_id=realization_id,
                        zone_id=zone_id,
                        object_id=item["object_id"],
                        coefficient_name=coefficient_name,
                        variant_id=scenario.variant_id,
                    )
                ),
            )
            values = base + (
                scenario.coefficient_sigma_multiplier
                * float(distribution["robust_sigma"])
                * field
            )
            coefficients[object_index, :, coefficient_index] = np.clip(
                values,
                float(distribution["lower"]),
                float(distribution["upper"]),
            )
            field_qc.append(
                {
                    "zone_id": zone_id,
                    "object_id": item["object_id"],
                    "field": coefficient_name,
                    "requested_lx_m": requested_lx,
                    "effective_lx_m": effective_lx,
                    **qc,
                }
            )
        thickness_field, qc = ar1_irregular(
            lateral_m,
            correlation_length_m=effective_lx,
            rng=named_rng(
                **_rng_keys(
                    calibration,
                    global_seed=global_seed,
                    stream_purpose="thickness_lateral",
                    realization_id=realization_id,
                    zone_id=zone_id,
                    object_id=item["object_id"],
                    variant_id=scenario.variant_id,
                )
            ),
        )
        thickness_weights[object_index] = (
            float(item["base_duration"]) * np.exp(scenario.thickness_log_sigma * thickness_field)
        )
        field_qc.append(
            {
                "zone_id": zone_id,
                "object_id": item["object_id"],
                "field": "thickness",
                "requested_lx_m": requested_lx,
                "effective_lx_m": effective_lx,
                **qc,
            }
        )
    thickness_weights /= np.sum(thickness_weights, axis=0, keepdims=True)
    return coefficients, thickness_weights, field_qc


def _apply_geometry_event(
    objects: list[dict[str, Any]],
    weights: np.ndarray,
    scenario: GenerationScenario,
    lateral_m: np.ndarray,
    *,
    minimum_fraction: np.ndarray,
    minimum_wedge_target_fraction: np.ndarray,
) -> tuple[np.ndarray, int | None, np.ndarray | None]:
    if scenario.geometry_family == "none":
        return weights, None, None
    candidates = [
        index for index, item in enumerate(objects) if item["state"] in {"low_impedance", "high_impedance"}
    ]
    coordinate = lateral_m / float(lateral_m[-1])
    if scenario.geometry_direction == "right_to_left":
        coordinate = 1.0 - coordinate
    if scenario.geometry_family == "wedge":
        multiplier = 0.25 + 1.5 * coordinate
    elif scenario.geometry_family == "pinchout":
        location = 0.35 if "035" in scenario.variant_id else 0.65
        width = 0.25
        start = max(0.0, location - width)
        scaled = np.clip((coordinate - start) / max(width, 1e-9), 0.0, 1.0)
        smooth = scaled * scaled * (3.0 - 2.0 * scaled)
        multiplier = 1.0 - smooth
    else:
        raise ValueError(f"Unsupported geometry family: {scenario.geometry_family}")
    center = 0.5 * (len(objects) - 1)
    ordered_candidates = sorted(candidates, key=lambda index: (abs(index - center), index))
    target = None
    required_other_fraction = (len(objects) - 1) * np.asarray(
        minimum_fraction,
        dtype=np.float64,
    )
    for candidate in ordered_candidates:
        candidate_weight = weights[candidate] * multiplier
        if np.any(1.0 - candidate_weight + 1e-12 < required_other_fraction):
            continue
        if (
            scenario.geometry_family == "wedge"
            and np.any(candidate_weight + 1e-12 < minimum_wedge_target_fraction)
        ):
            continue
        target = candidate
        break
    if target is None:
        raise GenerationRejected(
            ["missing_geometry_event_target"],
            diagnostics={
                "global_reversal_fraction": float("nan"),
                "global_clipping_fraction": float("nan"),
                "maximum_object_reversal_fraction": float("nan"),
                "maximum_object_clipping_fraction": float("nan"),
                "maximum_object_profile_violation_fraction": float("nan"),
                "rejected_object_count": 0,
            },
            details=[
                {
                    "reason": "missing_geometry_event_target",
                    "zone_id": "",
                    "object_id": "",
                    "state": "",
                    "event_target": False,
                    "count": len(candidates),
                    "denominator": len(objects),
                    "fraction": len(candidates) / max(len(objects), 1),
                    "threshold": float("nan"),
                }
            ],
        )
    target_weight = weights[target] * multiplier
    remaining = np.maximum(1.0 - target_weight, 0.0)
    others = np.delete(weights, target, axis=0)
    other_sum = np.sum(others, axis=0)
    if np.any(other_sum <= 0.0):
        raise ValueError("invalid_layer_duration")
    result = weights.copy()
    result[target] = target_weight
    for index in range(result.shape[0]):
        if index != target:
            result[index] = weights[index] / other_sum * remaining
    result /= np.sum(result, axis=0, keepdims=True)
    return result, target, multiplier


def _enforce_minimum_thickness(
    weights: np.ndarray,
    *,
    minimum_fraction: np.ndarray,
    exempt_index: int | None,
    zone_id: str,
) -> np.ndarray:
    """Normalize positive weights while preserving a per-column duration floor."""
    source = np.asarray(weights, dtype=np.float64)
    floor = np.asarray(minimum_fraction, dtype=np.float64).reshape(-1)
    if source.ndim != 2 or source.shape[1] != floor.size:
        raise ValueError("invalid_layer_duration")
    output = np.empty_like(source)
    included = [index for index in range(source.shape[0]) if index != exempt_index]
    for lateral_index in range(source.shape[1]):
        exempt_weight = 0.0 if exempt_index is None else float(source[exempt_index, lateral_index])
        available = 1.0 - exempt_weight
        required = len(included) * float(floor[lateral_index])
        if available + 1e-12 < required:
            raise GenerationRejected(
                ["invalid_layer_duration"],
                diagnostics={
                    "global_reversal_fraction": float("nan"),
                    "global_clipping_fraction": float("nan"),
                    "maximum_object_reversal_fraction": float("nan"),
                    "maximum_object_clipping_fraction": float("nan"),
                    "maximum_object_profile_violation_fraction": float("nan"),
                    "rejected_object_count": len(included),
                },
                details=[
                    {
                        "reason": "invalid_layer_duration",
                        "zone_id": zone_id,
                        "object_id": "",
                        "state": "",
                        "event_target": False,
                        "count": required,
                        "denominator": available,
                        "fraction": required / max(available, np.finfo(np.float64).eps),
                        "threshold": 1.0,
                        "lateral_index": lateral_index,
                    }
                ],
            )
        raw = source[included, lateral_index]
        raw_sum = float(np.sum(raw))
        if raw_sum <= 0.0:
            raise ValueError("invalid_layer_duration")
        residual = max(available - required, 0.0)
        output[included, lateral_index] = float(floor[lateral_index]) + residual * raw / raw_sum
        if exempt_index is not None:
            output[exempt_index, lateral_index] = exempt_weight
    return output


def generate_field_section(
    calibration: ImpedanceCalibration,
    *,
    realization_id: str,
    scenario: GenerationScenario,
    global_seed: int,
    lateral_m: np.ndarray,
    inline_float: np.ndarray,
    xline_float: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    horizon_twt_s: np.ndarray,
    output_dt_s: float,
    wavelet: np.ndarray,
    vertical_oversampling_factor: int = 8,
    minimum_truth_samples: int = 4,
    max_global_reversal_fraction: float = 0.10,
    max_object_reversal_fraction: float = 0.25,
    max_global_clipping_fraction: float = 0.005,
    max_object_clipping_fraction: float = 0.02,
    vertical_axis_origin: float | None = None,
    context_extent: float | None = None,
) -> GeneratedSection:
    """Generate one field-conditioned realization and its main closed forward model."""
    lateral = np.asarray(lateral_m, dtype=np.float64).reshape(-1)
    horizons = np.asarray(horizon_twt_s, dtype=np.float64)
    if horizons.shape != (lateral.size, len(calibration.ordered_horizons)):
        raise ValueError("horizon_twt_s shape does not match lateral samples and calibrated horizons.")
    if np.any(np.diff(horizons, axis=1) <= 0.0):
        raise ValueError("crossing_horizons")
    factor = int(vertical_oversampling_factor)
    truth_dt = float(output_dt_s) / factor
    if not np.isclose(truth_dt, calibration.truth_dt_s, rtol=0.0, atol=1e-12):
        raise ValueError("impedance_calibration_source_mismatch:truth_dt")
    wavelet_values = np.asarray(wavelet, dtype=np.float64).reshape(-1)
    context_s = (
        (wavelet_values.size // 2) * float(output_dt_s)
        if context_extent is None
        else float(context_extent)
    )
    if not np.isfinite(context_s) or context_s < 0.0:
        raise ValueError("context_extent must be finite and non-negative.")
    if vertical_axis_origin is None:
        start_s = np.floor((float(np.min(horizons[:, 0])) - context_s) / truth_dt) * truth_dt
        end_s = np.ceil((float(np.max(horizons[:, -1])) + context_s) / truth_dt) * truth_dt
    else:
        origin = float(vertical_axis_origin)
        start_s = origin + np.floor(
            (float(np.min(horizons[:, 0])) - context_s - origin) / float(output_dt_s)
        ) * float(output_dt_s)
        end_s = origin + np.ceil(
            (float(np.max(horizons[:, -1])) + context_s - origin) / float(output_dt_s)
        ) * float(output_dt_s)
    n_model_intervals = int(np.ceil((end_s - start_s) / output_dt_s))
    n_highres = n_model_intervals * factor + 1
    twt_highres = start_s + np.arange(n_highres, dtype=np.float64) * truth_dt
    twt_model = twt_highres[::factor]
    n_lateral = lateral.size
    log_ai = np.full((n_lateral, n_highres), np.nan, dtype=np.float64)
    rgt = np.full_like(log_ai, np.nan)
    state_id = np.full((n_lateral, n_highres), -1, dtype=np.int8)
    object_id = np.full((n_lateral, n_highres), -1, dtype=np.int32)
    object_xi = np.full_like(log_ai, np.nan)
    zone_id_grid = np.full((n_lateral, n_highres), -1, dtype=np.int16)
    geometry_event_mask = np.zeros((n_lateral, n_highres), dtype=bool)
    boundary = np.zeros((n_lateral, n_highres), dtype=bool)
    object_catalog: list[dict[str, Any]] = []
    object_lateral_coefficients: list[dict[str, Any]] = []
    field_qc: list[dict[str, Any]] = []
    rejection_details: list[dict[str, Any]] = []
    next_global_object_id = 0
    total_clipping_count = 0
    total_truth_sample_count = 0
    total_reversal_count = 0
    total_reversal_column_count = 0

    for zone_index, zone in enumerate(calibration.zones):
        zone_id = str(zone["zone_id"])
        zone_durations = horizons[:, zone_index + 1] - horizons[:, zone_index]
        reference_duration = float(np.median(zone_durations))
        minimum_fraction = minimum_truth_samples * truth_dt / reference_duration
        objects = _sample_object_sequence(
            calibration,
            zone_id=zone_id,
            realization_id=realization_id,
            global_seed=global_seed,
            minimum_fraction=minimum_fraction,
            variant_id=scenario.variant_id,
        )
        coefficients, thickness_weights, qc_rows = _object_lateral_parameters(
            calibration,
            objects=objects,
            zone_id=zone_id,
            realization_id=realization_id,
            global_seed=global_seed,
            lateral_m=lateral,
            scenario=scenario,
        )
        field_qc.extend(qc_rows)
        minimum_fraction_by_lateral = minimum_truth_samples * truth_dt / zone_durations
        minimum_wedge_target_fraction = 2.0 * truth_dt / zone_durations
        thickness_weights, event_target, event_multiplier = _apply_geometry_event(
            objects,
            thickness_weights,
            scenario,
            lateral,
            minimum_fraction=minimum_fraction_by_lateral,
            minimum_wedge_target_fraction=minimum_wedge_target_fraction,
        )
        thickness_weights = _enforce_minimum_thickness(
            thickness_weights,
            minimum_fraction=minimum_fraction_by_lateral,
            exempt_index=event_target,
            zone_id=zone_id,
        )
        a, b = _sample_background(
            calibration,
            zone_id=zone_id,
            realization_id=realization_id,
            global_seed=global_seed,
            variant_id=scenario.variant_id,
        )
        cumulative = np.vstack((np.zeros(n_lateral), np.cumsum(thickness_weights, axis=0)))
        for local_object_index, item in enumerate(objects):
            global_object_id = next_global_object_id
            next_global_object_id += 1
            reversal_count = 0
            valid_columns = 0
            clipping_count = 0
            sample_count = 0
            profile_violation_count = 0
            maximum_profile_violation_ratio = 0.0
            maximum_profile_violation_metric = ""
            maximum_profile_violation_value = float("nan")
            maximum_profile_violation_lower = float("nan")
            maximum_profile_violation_upper = float("nan")
            profile_projection_count = 0
            profile_projection_sum = 0.0
            minimum_profile_projection = 1.0
            c0_conditioning_count = 0
            c0_conditioning_adjustment_sum = 0.0
            maximum_c0_conditioning_adjustment = 0.0
            minimum_object_truth_samples = np.iinfo(np.int32).max
            maximum_object_truth_samples = 0
            for lateral_index in range(n_lateral):
                top = horizons[lateral_index, zone_index]
                bottom = horizons[lateral_index, zone_index + 1]
                object_top = top + cumulative[local_object_index, lateral_index] * (bottom - top)
                object_bottom = top + cumulative[local_object_index + 1, lateral_index] * (bottom - top)
                mask = (twt_highres >= object_top) & (
                    twt_highres < object_bottom if local_object_index + 1 < len(objects) else twt_highres <= object_bottom
                )
                indices = np.flatnonzero(mask)
                if indices.size == 0:
                    if local_object_index == event_target:
                        minimum_object_truth_samples = 0
                    continue
                minimum_object_truth_samples = min(
                    minimum_object_truth_samples,
                    int(indices.size),
                )
                maximum_object_truth_samples = max(
                    maximum_object_truth_samples,
                    int(indices.size),
                )
                xi = (twt_highres[indices] - object_top) / max(object_bottom - object_top, truth_dt)
                zeta = (twt_highres[indices] - top) / max(bottom - top, truth_dt)
                if indices.size >= 2:
                    qc_xi = xi
                    qc_zeta = zeta
                else:
                    qc_xi = np.linspace(0.0, 1.0, 5, dtype=np.float64)
                    qc_twt = object_top + qc_xi * (object_bottom - object_top)
                    qc_zeta = (qc_twt - top) / max(bottom - top, truth_dt)
                projected, projection = _project_profile_coefficients(
                    coefficients[local_object_index, lateral_index],
                    xi=qc_xi,
                    state_model=calibration.zone_models[zone_id]["states"][item["state"]],
                )
                background = a + b * (2.0 * zeta - 1.0)
                background_qc = a + b * (2.0 * qc_zeta - 1.0)
                projected, c0_adjustment = _condition_c0_to_ai_bounds(
                    projected,
                    xi=qc_xi,
                    background=background_qc,
                    state=item["state"],
                    state_model=calibration.zone_models[zone_id]["states"][item["state"]],
                    ai_bounds=calibration.zone_models[zone_id]["ai_bounds"],
                )
                c0, c1, c2 = projected
                object_lateral_coefficients.append(
                    {
                        "realization_id": realization_id,
                        "scenario_id": scenario.scenario_id,
                        "zone_id": zone_id,
                        "local_object_index": int(local_object_index),
                        "calibration_object_id": item["object_id"],
                        "object_id": int(global_object_id),
                        "state": item["state"],
                        "state_id": int(item["state_id"]),
                        "event_target": bool(local_object_index == event_target),
                        "lateral_index": int(lateral_index),
                        "lateral_m": float(lateral[lateral_index]),
                        "c0": float(c0),
                        "c1": float(c1),
                        "c2": float(c2),
                        "thickness_fraction": float(
                            thickness_weights[local_object_index, lateral_index]
                        ),
                        "object_top_s": float(object_top),
                        "object_bottom_s": float(object_bottom),
                        "profile_projection_scale": float(projection),
                        "c0_conditioning_adjustment": float(c0_adjustment),
                    }
                )
                profile_projection_count += int(projection < 1.0 - 1e-12)
                profile_projection_sum += projection
                minimum_profile_projection = min(minimum_profile_projection, projection)
                c0_conditioning_count += int(c0_adjustment > 1e-12)
                c0_conditioning_adjustment_sum += c0_adjustment
                maximum_c0_conditioning_adjustment = max(
                    maximum_c0_conditioning_adjustment,
                    c0_adjustment,
                )
                values = background + c0 + c1 * (2.0 * xi - 1.0) + c2 * np.sin(np.pi * xi)
                residual_profile_qc = (
                    c0
                    + c1 * (2.0 * qc_xi - 1.0)
                    + c2 * np.sin(np.pi * qc_xi)
                )
                metrics = object_profile_metrics(qc_xi, residual_profile_qc)
                metric_violation = False
                for metric_name in PROFILE_METRICS:
                    distribution = calibration.zone_models[zone_id]["states"][item["state"]][
                        "coefficients"
                    ][metric_name]
                    center_value = float(distribution["median"])
                    radius = 3.0 * float(distribution["robust_sigma"])
                    lower = center_value - radius
                    upper = center_value + radius
                    metric_value = float(metrics[metric_name])
                    tolerance = 1e-10 * max(1.0, abs(lower), abs(upper))
                    if lower - tolerance <= metric_value <= upper + tolerance:
                        continue
                    metric_violation = True
                    scale = max(upper - lower, np.finfo(np.float64).eps)
                    distance = lower - metric_value if metric_value < lower else metric_value - upper
                    ratio = float(distance / scale)
                    if ratio > maximum_profile_violation_ratio:
                        maximum_profile_violation_ratio = ratio
                        maximum_profile_violation_metric = metric_name
                        maximum_profile_violation_value = metric_value
                        maximum_profile_violation_lower = lower
                        maximum_profile_violation_upper = upper
                profile_violation_count += int(metric_violation)
                bounds = calibration.zone_models[zone_id]["ai_bounds"]
                ai_lower = float(bounds["p01"])
                ai_upper = float(bounds["p99"])
                ai_tolerance = 1e-10 * max(1.0, abs(ai_lower), abs(ai_upper))
                clipped = np.clip(values, ai_lower, ai_upper)
                clipping_count += int(
                    np.count_nonzero(
                        (values < ai_lower - ai_tolerance)
                        | (values > ai_upper + ai_tolerance)
                    )
                )
                sample_count += int(values.size)
                log_ai[lateral_index, indices] = clipped
                rgt[lateral_index, indices] = zone_index + zeta
                state_id[lateral_index, indices] = int(item["state_id"])
                object_id[lateral_index, indices] = global_object_id
                object_xi[lateral_index, indices] = xi
                zone_id_grid[lateral_index, indices] = zone_index
                if local_object_index == event_target:
                    geometry_event_mask[lateral_index, indices] = True
                boundary[lateral_index, indices[0]] = True
                mean_value = float(np.mean(clipped))
                mean_background = float(np.mean(background))
                if item["state"] == "high_impedance" and mean_value <= mean_background:
                    reversal_count += 1
                if item["state"] == "low_impedance" and mean_value >= mean_background:
                    reversal_count += 1
                valid_columns += 1
            object_reversal = reversal_count / max(valid_columns, 1)
            object_clipping = clipping_count / max(sample_count, 1)
            object_profile_violation = profile_violation_count / max(valid_columns, 1)
            total_clipping_count += clipping_count
            total_truth_sample_count += sample_count
            if item["state"] != "background":
                total_reversal_count += reversal_count
                total_reversal_column_count += valid_columns
            if profile_violation_count:
                rejection_details.append(
                    {
                        "reason": "invalid_object_profile",
                        "zone_id": zone_id,
                        "object_id": global_object_id,
                        "state": item["state"],
                        "event_target": bool(local_object_index == event_target),
                        "count": profile_violation_count,
                        "denominator": valid_columns,
                        "fraction": object_profile_violation,
                        "metric": maximum_profile_violation_metric,
                        "value": maximum_profile_violation_value,
                        "lower": maximum_profile_violation_lower,
                        "upper": maximum_profile_violation_upper,
                        "excess_ratio": maximum_profile_violation_ratio,
                    }
                )
            if object_reversal > max_object_reversal_fraction:
                rejection_details.append(
                    {
                        "reason": "excessive_object_reversal_fraction",
                        "zone_id": zone_id,
                        "object_id": global_object_id,
                        "state": item["state"],
                        "event_target": bool(local_object_index == event_target),
                        "count": reversal_count,
                        "denominator": valid_columns,
                        "fraction": object_reversal,
                        "threshold": max_object_reversal_fraction,
                    }
                )
            if object_clipping > max_object_clipping_fraction:
                rejection_details.append(
                    {
                        "reason": "excessive_object_clipping_fraction",
                        "zone_id": zone_id,
                        "object_id": global_object_id,
                        "state": item["state"],
                        "event_target": bool(local_object_index == event_target),
                        "count": clipping_count,
                        "denominator": sample_count,
                        "fraction": object_clipping,
                        "threshold": max_object_clipping_fraction,
                    }
                )
            object_catalog.append(
                {
                    "realization_id": realization_id,
                    "scenario_id": scenario.scenario_id,
                    "zone_id": zone_id,
                    "object_id": global_object_id,
                    "state": item["state"],
                    "state_id": item["state_id"],
                    "base_duration_fraction": item["base_duration"],
                    "event_target": bool(local_object_index == event_target),
                    "duration_fraction_start": float(thickness_weights[local_object_index, 0]),
                    "duration_fraction_end": float(thickness_weights[local_object_index, -1]),
                    "minimum_duration_fraction": float(
                        np.min(thickness_weights[local_object_index])
                    ),
                    "maximum_duration_fraction": float(
                        np.max(thickness_weights[local_object_index])
                    ),
                    "minimum_duration_s": float(
                        np.min(thickness_weights[local_object_index] * zone_durations)
                    ),
                    "maximum_duration_s": float(
                        np.max(thickness_weights[local_object_index] * zone_durations)
                    ),
                    "minimum_truth_samples": (
                        int(minimum_object_truth_samples)
                        if minimum_object_truth_samples < np.iinfo(np.int32).max
                        else 0
                    ),
                    "maximum_truth_samples": int(maximum_object_truth_samples),
                    "event_multiplier_start": (
                        float(event_multiplier[0])
                        if local_object_index == event_target and event_multiplier is not None
                        else float("nan")
                    ),
                    "event_multiplier_end": (
                        float(event_multiplier[-1])
                        if local_object_index == event_target and event_multiplier is not None
                        else float("nan")
                    ),
                    "minimum_event_multiplier": (
                        float(np.min(event_multiplier))
                        if local_object_index == event_target and event_multiplier is not None
                        else float("nan")
                    ),
                    "maximum_event_multiplier": (
                        float(np.max(event_multiplier))
                        if local_object_index == event_target and event_multiplier is not None
                        else float("nan")
                    ),
                    "reversal_fraction": object_reversal,
                    "clipping_fraction": object_clipping,
                    "profile_violation_fraction": object_profile_violation,
                    "profile_projection_fraction": (
                        profile_projection_count / max(valid_columns, 1)
                    ),
                    "mean_profile_projection_scale": (
                        profile_projection_sum / max(valid_columns, 1)
                    ),
                    "minimum_profile_projection_scale": minimum_profile_projection,
                    "c0_conditioning_fraction": (
                        c0_conditioning_count / max(valid_columns, 1)
                    ),
                    "mean_c0_conditioning_adjustment": (
                        c0_conditioning_adjustment_sum / max(valid_columns, 1)
                    ),
                    "maximum_c0_conditioning_adjustment": maximum_c0_conditioning_adjustment,
                }
            )

    for lateral_index in range(n_lateral):
        valid = np.isfinite(log_ai[lateral_index])
        if not np.any(valid):
            raise ValueError("invalid_impedance")
        first, last = np.flatnonzero(valid)[[0, -1]]
        log_ai[lateral_index, :first] = log_ai[lateral_index, first]
        log_ai[lateral_index, last + 1 :] = log_ai[lateral_index, last]
        rgt[lateral_index, :first] = rgt[lateral_index, first]
        rgt[lateral_index, last + 1 :] = rgt[lateral_index, last]
    global_reversal = total_reversal_count / max(total_reversal_column_count, 1)
    global_clipping = total_clipping_count / max(total_truth_sample_count, 1)
    if global_reversal > max_global_reversal_fraction:
        rejection_details.append(
            {
                "reason": "excessive_global_reversal_fraction",
                "zone_id": "",
                "object_id": "",
                "state": "",
                "event_target": False,
                "count": total_reversal_count,
                "denominator": total_reversal_column_count,
                "fraction": global_reversal,
                "threshold": max_global_reversal_fraction,
            }
        )
    if global_clipping > max_global_clipping_fraction:
        rejection_details.append(
            {
                "reason": "excessive_global_clipping_fraction",
                "zone_id": "",
                "object_id": "",
                "state": "",
                "event_target": False,
                "count": total_clipping_count,
                "denominator": total_truth_sample_count,
                "fraction": global_clipping,
                "threshold": max_global_clipping_fraction,
            }
        )
    if rejection_details:
        reasons = [str(item["reason"]) for item in rejection_details]
        diagnostics = {
            "global_reversal_fraction": global_reversal,
            "global_clipping_fraction": global_clipping,
            "maximum_object_reversal_fraction": max(
                (float(row["reversal_fraction"]) for row in object_catalog),
                default=0.0,
            ),
            "maximum_object_clipping_fraction": max(
                (float(row["clipping_fraction"]) for row in object_catalog),
                default=0.0,
            ),
            "maximum_object_profile_violation_fraction": max(
                (float(row["profile_violation_fraction"]) for row in object_catalog),
                default=0.0,
            ),
            "maximum_object_profile_projection_fraction": max(
                (float(row["profile_projection_fraction"]) for row in object_catalog),
                default=0.0,
            ),
            "minimum_profile_projection_scale": min(
                (float(row["minimum_profile_projection_scale"]) for row in object_catalog),
                default=1.0,
            ),
            "maximum_object_c0_conditioning_fraction": max(
                (float(row["c0_conditioning_fraction"]) for row in object_catalog),
                default=0.0,
            ),
            "maximum_c0_conditioning_adjustment": max(
                (float(row["maximum_c0_conditioning_adjustment"]) for row in object_catalog),
                default=0.0,
            ),
            "minimum_non_event_truth_samples": min(
                (
                    int(row["minimum_truth_samples"])
                    for row in object_catalog
                    if not bool(row["event_target"])
                ),
                default=0,
            ),
            "rejected_object_count": len(
                {
                    (str(item.get("zone_id", "")), str(item.get("object_id", "")))
                    for item in rejection_details
                    if str(item.get("object_id", ""))
                }
            ),
        }
        raise GenerationRejected(reasons, diagnostics=diagnostics, details=rejection_details)

    taps = antialias_taps(factor)
    model_log_ai = downsample_continuous(log_ai, factor, taps)[..., : twt_model.size]
    rgt_model = downsample_continuous(rgt, factor, taps)[..., : twt_model.size]
    reflectivity_highres = np.stack(
        [acoustic_reflectivity_from_log_ai(trace) for trace in log_ai], axis=0
    )
    reflectivity_model = np.stack(
        [acoustic_reflectivity_from_log_ai(trace) for trace in model_log_ai], axis=0
    )
    seismic = np.stack(
        [forward_log_ai(trace, wavelet_values) for trace in model_log_ai], axis=0
    )
    state_fraction, dominant, zone_model, boundary_fraction, valid_model = categorical_model_grids(
        state_id, object_id, zone_id_grid, boundary, factor, twt_model.size
    )
    forward_valid_highres = np.isfinite(log_ai[:, :-1]) & np.isfinite(log_ai[:, 1:])
    forward_valid_model = np.isfinite(model_log_ai[:, :-1]) & np.isfinite(model_log_ai[:, 1:])
    correlation_warnings = [
        row
        for row in field_qc
        if np.isfinite(row["empirical_correlation_length_m"])
        and row["requested_lx_m"] < lateral[-1]
        and abs(row["empirical_correlation_length_m"] - row["effective_lx_m"])
        / row["effective_lx_m"]
        > 0.35
    ]
    return GeneratedSection(
        realization_id=realization_id,
        scenario=scenario,
        lateral_m=lateral,
        inline_float=np.asarray(inline_float, dtype=np.float64),
        xline_float=np.asarray(xline_float, dtype=np.float64),
        x_m=np.asarray(x_m, dtype=np.float64),
        y_m=np.asarray(y_m, dtype=np.float64),
        twt_highres_s=twt_highres,
        twt_model_s=twt_model,
        truth_log_ai_highres=log_ai,
        model_target_log_ai=model_log_ai,
        reflectivity_highres=reflectivity_highres,
        reflectivity_model=reflectivity_model,
        seismic_model_consistent=seismic,
        rgt_highres=rgt,
        rgt_model=rgt_model,
        state_id_highres=state_id,
        object_id_highres=object_id,
        object_xi_highres=object_xi,
        zone_id_highres=zone_id_grid,
        geometry_event_mask_highres=geometry_event_mask,
        boundary_mask_highres=boundary,
        boundary_fraction_model=boundary_fraction,
        boundary_mask_model=boundary_fraction > 0.0,
        state_fraction_model=state_fraction,
        dominant_object_id_model=dominant,
        zone_id_model=zone_model,
        valid_mask_model=valid_model,
        forward_valid_mask_highres=forward_valid_highres,
        forward_valid_mask_model=forward_valid_model,
        object_catalog=object_catalog,
        object_lateral_coefficients=object_lateral_coefficients,
        qc={
            "status": "ok",
            "global_reversal_fraction": global_reversal,
            "global_clipping_fraction": global_clipping,
            "maximum_object_reversal_fraction": max(
                (float(row["reversal_fraction"]) for row in object_catalog),
                default=0.0,
            ),
            "maximum_object_clipping_fraction": max(
                (float(row["clipping_fraction"]) for row in object_catalog),
                default=0.0,
            ),
            "maximum_object_profile_violation_fraction": max(
                (float(row["profile_violation_fraction"]) for row in object_catalog),
                default=0.0,
            ),
            "maximum_object_profile_projection_fraction": max(
                (float(row["profile_projection_fraction"]) for row in object_catalog),
                default=0.0,
            ),
            "minimum_profile_projection_scale": min(
                (float(row["minimum_profile_projection_scale"]) for row in object_catalog),
                default=1.0,
            ),
            "maximum_object_c0_conditioning_fraction": max(
                (float(row["c0_conditioning_fraction"]) for row in object_catalog),
                default=0.0,
            ),
            "maximum_c0_conditioning_adjustment": max(
                (float(row["maximum_c0_conditioning_adjustment"]) for row in object_catalog),
                default=0.0,
            ),
            "minimum_non_event_truth_samples": min(
                (
                    int(row["minimum_truth_samples"])
                    for row in object_catalog
                    if not bool(row["event_target"])
                ),
                default=0,
            ),
            "lateral_correlation_warning_count": len(correlation_warnings),
            "field_qc": field_qc,
        },
    )
