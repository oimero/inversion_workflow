"""Domain-neutral field-conditioned scientific truth generation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from cup.synthetic.core.calibration import (
    PROFILE_METRICS,
    ImpedanceCalibration,
    STATE_NAMES,
    object_profile_metrics,
)
from cup.synthetic.core.random import RandomNamespace, ar1_irregular, named_rng
from cup.synthetic.core.rejections import TruthGenerationRejected
from cup.synthetic.core.scenarios import GenerationScenario


@dataclass(frozen=True)
class TruthGenerationRequest:
    realization_id: str
    scenario: GenerationScenario
    global_seed: int
    random_namespace: RandomNamespace
    sample_domain: str
    axis_unit: str
    lateral_m: np.ndarray
    inline_float: np.ndarray
    xline_float: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    horizon_coordinates: np.ndarray
    model_sample_interval: float
    vertical_oversampling_factor: int
    minimum_highres_cells: int
    vertical_axis_origin: float | None
    context_extent: float
    sequence_minimum_duration_reference: str
    max_global_reversal_fraction: float
    max_object_reversal_fraction: float
    max_global_clipping_fraction: float
    max_object_clipping_fraction: float

    def __post_init__(self) -> None:
        domain = self.sample_domain.strip().lower()
        if (domain, self.axis_unit) not in {("time", "s"), ("depth", "m")}:
            raise ValueError("truth request domain/unit must be time/s or depth/m.")
        if not self.realization_id.strip():
            raise ValueError("truth realization_id must be non-empty.")
        if not np.isfinite(self.model_sample_interval) or self.model_sample_interval <= 0.0:
            raise ValueError("truth model sample interval must be positive.")
        if (
            self.vertical_oversampling_factor < 1
            or int(self.vertical_oversampling_factor) != self.vertical_oversampling_factor
            or self.minimum_highres_cells < 1
            or int(self.minimum_highres_cells) != self.minimum_highres_cells
        ):
            raise ValueError("truth sampling factors must be positive integers.")
        if self.sequence_minimum_duration_reference not in {"median", "minimum"}:
            raise ValueError(
                "sequence_minimum_duration_reference must be 'median' or 'minimum'."
            )
        if self.vertical_axis_origin is not None and not np.isfinite(self.vertical_axis_origin):
            raise ValueError("truth vertical axis origin must be finite.")
        if not np.isfinite(self.context_extent) or self.context_extent < 0.0:
            raise ValueError("truth context extent must be finite and non-negative.")
        for name in (
            "max_global_reversal_fraction",
            "max_object_reversal_fraction",
            "max_global_clipping_fraction",
            "max_object_clipping_fraction",
        ):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"truth QC threshold {name} must be finite.")
        lateral = np.asarray(self.lateral_m, dtype=np.float64).reshape(-1)
        if lateral.size < 2 or np.any(~np.isfinite(lateral)) or np.any(np.diff(lateral) <= 0.0):
            raise ValueError("truth lateral axis must be finite and strictly increasing.")
        one_dimensional = {
            "inline_float": self.inline_float,
            "xline_float": self.xline_float,
            "x_m": self.x_m,
            "y_m": self.y_m,
        }
        for name, values in one_dimensional.items():
            array = np.asarray(values, dtype=np.float64).reshape(-1)
            if array.shape != lateral.shape or np.any(~np.isfinite(array)):
                raise ValueError(f"truth {name} must be finite and match lateral_m.")
            object.__setattr__(self, name, array)
        horizons = np.asarray(self.horizon_coordinates, dtype=np.float64)
        if horizons.ndim != 2 or horizons.shape[0] != lateral.size:
            raise ValueError("truth horizons must have shape [lateral, horizon].")
        if np.any(~np.isfinite(horizons)) or np.any(np.diff(horizons, axis=1) <= 0.0):
            raise ValueError("crossing_horizons")
        object.__setattr__(self, "sample_domain", domain)
        object.__setattr__(self, "lateral_m", lateral)
        object.__setattr__(self, "horizon_coordinates", horizons)


@dataclass(frozen=True)
class SyntheticTruth:
    realization_id: str
    scenario: GenerationScenario
    sample_domain: str
    axis_unit: str
    highres_axis: np.ndarray
    highres_sample_interval: float
    lateral_m: np.ndarray
    inline_float: np.ndarray
    xline_float: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    log_ai_highres: np.ndarray
    rgt_highres: np.ndarray
    state_id_highres: np.ndarray
    object_id_highres: np.ndarray
    object_xi_highres: np.ndarray
    zone_id_highres: np.ndarray
    geometry_event_mask_highres: np.ndarray
    boundary_mask_highres: np.ndarray
    object_catalog: tuple[Mapping[str, Any], ...]
    object_lateral_coefficients: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "object_catalog",
            tuple(MappingProxyType(dict(row)) for row in self.object_catalog),
        )
        object.__setattr__(
            self,
            "object_lateral_coefficients",
            tuple(MappingProxyType(dict(row)) for row in self.object_lateral_coefficients),
        )
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def _rng_keys(
    calibration: ImpedanceCalibration,
    namespace: RandomNamespace,
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
        "benchmark_version": namespace.benchmark_version,
        "science_revision": namespace.science_revision,
        "random_stream_contract_version": namespace.random_stream_contract_version,
        "generator_family": namespace.generator_family,
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
    namespace: RandomNamespace,
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
            namespace,
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
                namespace,
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
    namespace: RandomNamespace,
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
                namespace,
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
    namespace: RandomNamespace,
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
                    namespace,
                    global_seed=global_seed,
                    stream_purpose=f"coefficient_{coefficient_name}",
                    realization_id=realization_id,
                    zone_id=zone_id,
                    object_id=item["object_id"],
                    coefficient_name=coefficient_name,
                    variant_id=scenario.geometry_variant_id,
                )
            )
            base = _truncated_normal(distribution, base_rng)
            field, qc = ar1_irregular(
                lateral_m,
                correlation_length_m=effective_lx,
                rng=named_rng(
                    **_rng_keys(
                        calibration,
                        namespace,
                        global_seed=global_seed,
                        stream_purpose="coefficient_lateral",
                        realization_id=realization_id,
                        zone_id=zone_id,
                        object_id=item["object_id"],
                        coefficient_name=coefficient_name,
                        variant_id=scenario.geometry_variant_id,
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
                    namespace,
                    global_seed=global_seed,
                    stream_purpose="thickness_lateral",
                    realization_id=realization_id,
                    zone_id=zone_id,
                    object_id=item["object_id"],
                    variant_id=scenario.geometry_variant_id,
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
        location = 0.35 if "035" in scenario.geometry_variant_id else 0.65
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
        raise TruthGenerationRejected(
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
            raise TruthGenerationRejected(
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


def generate_field_conditioned_truth(
    calibration: ImpedanceCalibration,
    request: TruthGenerationRequest,
) -> SyntheticTruth:
    """Generate only high-resolution underground truth and scientific diagnostics."""
    if request.random_namespace.generator_family != calibration.generator_family:
        raise ValueError("truth random namespace generator family does not match calibration.")
    realization_id = request.realization_id
    scenario = request.scenario
    global_seed = request.global_seed
    namespace = request.random_namespace
    lateral = request.lateral_m
    inline_float = request.inline_float
    xline_float = request.xline_float
    x_m = request.x_m
    y_m = request.y_m
    horizons = request.horizon_coordinates
    output_interval = float(request.model_sample_interval)
    factor = int(request.vertical_oversampling_factor)
    minimum_truth_samples = int(request.minimum_highres_cells)
    max_global_reversal_fraction = float(request.max_global_reversal_fraction)
    max_object_reversal_fraction = float(request.max_object_reversal_fraction)
    max_global_clipping_fraction = float(request.max_global_clipping_fraction)
    max_object_clipping_fraction = float(request.max_object_clipping_fraction)
    vertical_axis_origin = request.vertical_axis_origin
    context_extent = float(request.context_extent)
    sequence_minimum_duration_reference = request.sequence_minimum_duration_reference
    if horizons.shape != (lateral.size, len(calibration.ordered_horizons)):
        raise ValueError("horizon shape does not match lateral samples and calibrated horizons.")
    highres_interval = output_interval / factor
    if not np.isclose(highres_interval, calibration.truth_sample_interval, rtol=0.0, atol=1e-12):
        raise ValueError("impedance_calibration_source_mismatch:truth_dt")
    if vertical_axis_origin is None:
        start_coordinate = (
            np.floor(
                (float(np.min(horizons[:, 0])) - context_extent) / highres_interval
            )
            * highres_interval
        )
        end_coordinate = (
            np.ceil(
                (float(np.max(horizons[:, -1])) + context_extent) / highres_interval
            )
            * highres_interval
        )
    else:
        origin = float(vertical_axis_origin)
        start_coordinate = origin + np.floor(
            (float(np.min(horizons[:, 0])) - context_extent - origin) / output_interval
        ) * output_interval
        end_coordinate = origin + np.ceil(
            (float(np.max(horizons[:, -1])) + context_extent - origin) / output_interval
        ) * output_interval
    n_model_intervals = int(
        np.ceil((end_coordinate - start_coordinate) / output_interval)
    )
    n_highres = n_model_intervals * factor + 1
    highres_axis = (
        start_coordinate
        + np.arange(n_highres, dtype=np.float64) * highres_interval
    )
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
        if sequence_minimum_duration_reference == "median":
            reference_duration = float(np.median(zone_durations))
        elif sequence_minimum_duration_reference == "minimum":
            reference_duration = float(np.min(zone_durations))
        else:
            raise ValueError(
                "sequence_minimum_duration_reference must be 'median' or 'minimum'."
            )
        minimum_fraction = minimum_truth_samples * highres_interval / reference_duration
        objects = _sample_object_sequence(
            calibration,
            namespace,
            zone_id=zone_id,
            realization_id=realization_id,
            global_seed=global_seed,
            minimum_fraction=minimum_fraction,
            variant_id=scenario.geometry_variant_id,
        )
        coefficients, thickness_weights, qc_rows = _object_lateral_parameters(
            calibration,
            namespace,
            objects=objects,
            zone_id=zone_id,
            realization_id=realization_id,
            global_seed=global_seed,
            lateral_m=lateral,
            scenario=scenario,
        )
        field_qc.extend(qc_rows)
        minimum_fraction_by_lateral = (
            minimum_truth_samples * highres_interval / zone_durations
        )
        minimum_wedge_target_fraction = 2.0 * highres_interval / zone_durations
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
            namespace,
            zone_id=zone_id,
            realization_id=realization_id,
            global_seed=global_seed,
            variant_id=scenario.geometry_variant_id,
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
                mask = (highres_axis >= object_top) & (
                    highres_axis < object_bottom
                    if local_object_index + 1 < len(objects)
                    else highres_axis <= object_bottom
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
                xi = (highres_axis[indices] - object_top) / max(
                    object_bottom - object_top, highres_interval
                )
                zeta = (highres_axis[indices] - top) / max(
                    bottom - top, highres_interval
                )
                if indices.size >= 2:
                    qc_xi = xi
                    qc_zeta = zeta
                else:
                    qc_xi = np.linspace(0.0, 1.0, 5, dtype=np.float64)
                    qc_coordinate = object_top + qc_xi * (object_bottom - object_top)
                    qc_zeta = (qc_coordinate - top) / max(
                        bottom - top, highres_interval
                    )
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
                        "object_top_coordinate": float(object_top),
                        "object_bottom_coordinate": float(object_bottom),
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
                    "minimum_extent": float(
                        np.min(thickness_weights[local_object_index] * zone_durations)
                    ),
                    "maximum_extent": float(
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
        raise TruthGenerationRejected(
            reasons, diagnostics=diagnostics, details=rejection_details
        )

    correlation_warnings = [
        row
        for row in field_qc
        if np.isfinite(row["empirical_correlation_length_m"])
        and row["requested_lx_m"] < lateral[-1]
        and abs(row["empirical_correlation_length_m"] - row["effective_lx_m"])
        / row["effective_lx_m"]
        > 0.35
    ]
    return SyntheticTruth(
        realization_id=realization_id,
        scenario=scenario,
        sample_domain=request.sample_domain,
        axis_unit=request.axis_unit,
        highres_axis=highres_axis,
        highres_sample_interval=highres_interval,
        lateral_m=lateral,
        inline_float=np.asarray(inline_float, dtype=np.float64),
        xline_float=np.asarray(xline_float, dtype=np.float64),
        x_m=np.asarray(x_m, dtype=np.float64),
        y_m=np.asarray(y_m, dtype=np.float64),
        log_ai_highres=log_ai,
        rgt_highres=rgt,
        state_id_highres=state_id,
        object_id_highres=object_id,
        object_xi_highres=object_xi,
        zone_id_highres=zone_id_grid,
        geometry_event_mask_highres=geometry_event_mask,
        boundary_mask_highres=boundary,
        object_catalog=tuple(object_catalog),
        object_lateral_coefficients=tuple(object_lateral_coefficients),
        diagnostics={
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


__all__ = [
    "SyntheticTruth",
    "TruthGenerationRequest",
    "generate_field_conditioned_truth",
]
