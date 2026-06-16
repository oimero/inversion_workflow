"""Well-conditioned calibration for ``object_coefficients_v1``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


SCHEMA_VERSION = "synthoseis_lite_impedance_calibration_v1"
GENERATOR_FAMILY = "object_coefficients_v1"
STATE_NAMES = ("low_impedance", "background", "high_impedance")
PROFILE_METRICS = (
    "profile_mean",
    "endpoint_difference",
    "peak_to_peak",
    "internal_extreme_amplitude",
)


@dataclass(frozen=True)
class WellZoneCurves:
    well_name: str
    spatial_cluster_id: int
    zone_id: str
    top_horizon: str
    bottom_horizon: str
    twt_s: np.ndarray
    filtered_log_ai: np.ndarray
    full_log_ai: np.ndarray
    zone_top_s: float
    zone_bottom_s: float

    def __post_init__(self) -> None:
        arrays = [
            np.asarray(self.twt_s, dtype=np.float64).reshape(-1),
            np.asarray(self.filtered_log_ai, dtype=np.float64).reshape(-1),
            np.asarray(self.full_log_ai, dtype=np.float64).reshape(-1),
        ]
        if len({array.size for array in arrays}) != 1 or arrays[0].size < 2:
            raise ValueError("WellZoneCurves arrays must have matching lengths >= 2.")
        object.__setattr__(self, "twt_s", arrays[0])
        object.__setattr__(self, "filtered_log_ai", arrays[1])
        object.__setattr__(self, "full_log_ai", arrays[2])


@dataclass(frozen=True)
class ImpedanceCalibration:
    schema_version: str
    generator_family: str
    truth_dt_s: float
    state_threshold_sigma: float
    ordered_horizons: tuple[str, ...]
    zones: tuple[dict[str, Any], ...]
    parent: dict[str, Any]
    zone_models: dict[str, Any]
    source_runs: dict[str, str]
    source_hashes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ordered_horizons"] = list(self.ordered_horizons)
        payload["zones"] = list(self.zones)
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImpedanceCalibration":
        payload = dict(value)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported impedance calibration schema: {payload.get('schema_version')}")
        if payload.get("generator_family") != GENERATOR_FAMILY:
            raise ValueError(f"Unsupported generator family: {payload.get('generator_family')}")
        for zone_id, zone_model in dict(payload.get("zone_models") or {}).items():
            for state in STATE_NAMES:
                coefficients = (
                    dict(zone_model.get("states") or {})
                    .get(state, {})
                    .get("coefficients", {})
                )
                missing = [
                    name
                    for name in (*("c0", "c1", "c2"), *PROFILE_METRICS)
                    if name not in coefficients
                ]
                if missing:
                    raise ValueError(
                        "Unsupported impedance calibration content: "
                        f"{zone_id}/{state} lacks {missing}"
                    )
        return cls(
            schema_version=str(payload["schema_version"]),
            generator_family=str(payload["generator_family"]),
            truth_dt_s=float(payload["truth_dt_s"]),
            state_threshold_sigma=float(payload["state_threshold_sigma"]),
            ordered_horizons=tuple(payload["ordered_horizons"]),
            zones=tuple(payload["zones"]),
            parent=dict(payload["parent"]),
            zone_models=dict(payload["zone_models"]),
            source_runs=dict(payload["source_runs"]),
            source_hashes=dict(payload["source_hashes"]),
        )


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(data) & np.isfinite(mass) & (mass > 0.0)
    data = data[valid]
    mass = mass[valid]
    if data.size == 0:
        return float("nan")
    order = np.argsort(data, kind="mergesort")
    data = data[order]
    mass = mass[order]
    cumulative = np.cumsum(mass)
    threshold = float(np.clip(probability, 0.0, 1.0)) * cumulative[-1]
    return float(data[min(int(np.searchsorted(cumulative, threshold, side="left")), data.size - 1)])


def robust_location_scale(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    center = weighted_quantile(values, weights, 0.5)
    sigma = 1.4826 * weighted_quantile(np.abs(np.asarray(values) - center), weights, 0.5)
    return center, float(sigma)


def hierarchical_weights(frame: pd.DataFrame, *, unit_column: str) -> np.ndarray:
    """Cluster, well and within-well units each receive equal mass."""
    if frame.empty:
        return np.empty(0, dtype=np.float64)
    result = np.zeros(len(frame), dtype=np.float64)
    clusters = frame["spatial_cluster_id"].drop_duplicates().tolist()
    for cluster in clusters:
        cluster_mask = frame["spatial_cluster_id"].eq(cluster).to_numpy()
        wells = frame.loc[cluster_mask, "well_name"].drop_duplicates().tolist()
        for well in wells:
            mask = cluster_mask & frame["well_name"].eq(well).to_numpy()
            units = frame.loc[mask, unit_column].drop_duplicates().tolist()
            for unit in units:
                unit_mask = mask & frame[unit_column].eq(unit).to_numpy()
                count = int(np.count_nonzero(unit_mask))
                result[unit_mask] = 1.0 / (len(clusters) * len(wells) * len(units) * count)
    return result


def _fit_background(zeta: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(zeta) & np.isfinite(values)
    if np.count_nonzero(valid) < 3:
        raise ValueError("insufficient_filtered_background_samples")
    design = np.column_stack((np.ones(np.count_nonzero(valid)), 2.0 * zeta[valid] - 1.0))
    coefficients, *_ = np.linalg.lstsq(design, values[valid], rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def _merge_short_states(states: np.ndarray, residual: np.ndarray, centers: np.ndarray) -> np.ndarray:
    output = np.asarray(states, dtype=np.int8).copy()
    while output.size:
        starts = np.r_[0, np.flatnonzero(np.diff(output) != 0) + 1]
        ends = np.r_[starts[1:], output.size]
        short = np.flatnonzero((ends - starts) < 2)
        if short.size == 0:
            break
        index = int(short[0])
        start, end = int(starts[index]), int(ends[index])
        candidates: list[int] = []
        if index > 0:
            candidates.append(int(output[starts[index - 1]]))
        if index + 1 < starts.size:
            candidates.append(int(output[starts[index + 1]]))
        if not candidates:
            break
        errors = [
            float(np.sum((residual[start:end] - centers[candidate]) ** 2))
            for candidate in candidates
        ]
        output[start:end] = candidates[int(np.argmin(errors))]
    return output


def _fit_object_profile(xi: np.ndarray, residual: np.ndarray, huber_delta: float) -> np.ndarray:
    design = np.column_stack((np.ones(xi.size), 2.0 * xi - 1.0, np.sin(np.pi * xi)))
    if xi.size == 2:
        coefficients, *_ = np.linalg.lstsq(design[:, :2], residual, rcond=None)
        return np.array([coefficients[0], coefficients[1], 0.0], dtype=np.float64)
    initial, *_ = np.linalg.lstsq(design, residual, rcond=None)
    if xi.size == 3:
        return initial.astype(np.float64)
    fit = least_squares(
        lambda coefficients: design @ coefficients - residual,
        initial,
        loss="huber",
        f_scale=max(float(huber_delta), np.finfo(np.float64).eps),
    )
    return np.asarray(fit.x, dtype=np.float64)


def object_profile_metrics(xi: np.ndarray, values: np.ndarray) -> dict[str, float]:
    coordinate = np.asarray(xi, dtype=np.float64).reshape(-1)
    profile = np.asarray(values, dtype=np.float64).reshape(-1)
    if coordinate.size != profile.size or profile.size < 2:
        raise ValueError("Object profile metrics require matching arrays with at least two samples.")
    endpoint_line = np.interp(coordinate, coordinate[[0, -1]], profile[[0, -1]])
    interior = np.abs(profile - endpoint_line)
    return {
        "profile_mean": float(np.mean(profile)),
        "endpoint_difference": float(profile[-1] - profile[0]),
        "peak_to_peak": float(np.ptp(profile)),
        "internal_extreme_amplitude": float(np.max(interior)),
    }


def _distribution(frame: pd.DataFrame, column: str, weights: np.ndarray) -> dict[str, float]:
    values = frame[column].to_numpy(dtype=np.float64)
    center, sigma = robust_location_scale(values, weights)
    return {
        "median": center,
        "robust_sigma": sigma,
        "p01": weighted_quantile(values, weights, 0.01),
        "p99": weighted_quantile(values, weights, 0.99),
    }


def _blend(raw: float, parent: float, weight: float) -> float:
    if not np.isfinite(raw):
        return float(parent)
    return float(weight * raw + (1.0 - weight) * parent)


def _final_distribution(
    raw: Mapping[str, float],
    parent: Mapping[str, float],
    weight: float,
    *,
    scale_floor: float,
    scale_cap: float,
) -> dict[str, Any]:
    parent_sigma = float(parent["robust_sigma"])
    if not np.isfinite(parent_sigma) or parent_sigma <= 0.0:
        raise ValueError("invalid_impedance_calibration:nonpositive_parent_sigma")
    final_sigma = np.clip(
        _blend(float(raw["robust_sigma"]), parent_sigma, weight),
        scale_floor * parent_sigma,
        scale_cap * parent_sigma,
    )
    lower = _blend(float(raw["p01"]), float(parent["p01"]), weight)
    upper = _blend(float(raw["p99"]), float(parent["p99"]), weight)
    if not lower < upper:
        lower, upper = float(parent["p01"]), float(parent["p99"])
    return {
        "raw": dict(raw),
        "parent": dict(parent),
        "weight": float(weight),
        "median": _blend(float(raw["median"]), float(parent["median"]), weight),
        "robust_sigma": float(final_sigma),
        "lower": float(lower),
        "upper": float(upper),
    }


def calibrate_impedance(
    curves: Sequence[WellZoneCurves],
    *,
    truth_dt_s: float,
    ordered_horizons: Sequence[str],
    source_runs: Mapping[str, str],
    source_hashes: Mapping[str, str],
    state_threshold_sigma: float = 1.0,
    huber_delta_parent_sigma_floor: float = 0.05,
    coefficient_sigma_parent_floor: float = 0.05,
    coefficient_sigma_parent_cap: float = 3.0,
) -> tuple[ImpedanceCalibration, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calibrate zone backgrounds, Semi-Markov objects and profile coefficients."""
    if not curves:
        raise ValueError("No well-zone curves were provided.")
    sample_records: list[dict[str, Any]] = []
    background_records: list[dict[str, Any]] = []
    for item in curves:
        duration = float(item.zone_bottom_s - item.zone_top_s)
        if duration <= 0.0:
            continue
        zeta = (item.twt_s - item.zone_top_s) / duration
        valid = (
            (zeta >= 0.0)
            & (zeta <= 1.0)
            & np.isfinite(item.filtered_log_ai)
            & np.isfinite(item.full_log_ai)
        )
        if np.count_nonzero(valid) < 3:
            continue
        a, b = _fit_background(zeta[valid], item.filtered_log_ai[valid])
        background_records.append(
            {
                "well_name": item.well_name,
                "spatial_cluster_id": item.spatial_cluster_id,
                "zone_id": item.zone_id,
                "background_a": a,
                "background_b": b,
                "zone_duration_s": duration,
            }
        )
        background = a + b * (2.0 * zeta[valid] - 1.0)
        residual_values = item.full_log_ai[valid] - background
        for local_index, (time, coordinate, filtered, full, bg, residual) in enumerate(
            zip(
                item.twt_s[valid],
                zeta[valid],
                item.filtered_log_ai[valid],
                item.full_log_ai[valid],
                background,
                residual_values,
            )
        ):
            sample_records.append(
                {
                    "well_name": item.well_name,
                    "spatial_cluster_id": item.spatial_cluster_id,
                    "zone_id": item.zone_id,
                    "top_horizon": item.top_horizon,
                    "bottom_horizon": item.bottom_horizon,
                    "sample_id": local_index,
                    "twt_s": float(time),
                    "zeta": float(coordinate),
                    "filtered_log_ai": float(filtered),
                    "full_log_ai": float(full),
                    "background_log_ai": float(bg),
                    "residual": float(residual),
                    "zone_duration_s": duration,
                }
            )
    samples = pd.DataFrame.from_records(sample_records)
    backgrounds = pd.DataFrame.from_records(background_records)
    if samples.empty or backgrounds.empty:
        raise ValueError("No valid calibration samples were produced.")

    object_records: list[dict[str, Any]] = []
    profile_sample_records: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    zone_centers: dict[str, tuple[float, float]] = {}
    for zone_id, group in samples.groupby("zone_id", sort=False):
        weights = hierarchical_weights(group.reset_index(drop=True), unit_column="sample_id")
        center, sigma = robust_location_scale(group["residual"].to_numpy(), weights)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"invalid_impedance_calibration:nonpositive_state_sigma:{zone_id}")
        zone_centers[str(zone_id)] = (center, sigma)
        residual_values = group["residual"].to_numpy()
        sample_states = np.where(
            residual_values < center - state_threshold_sigma * sigma,
            0,
            np.where(residual_values > center + state_threshold_sigma * sigma, 2, 1),
        )
        samples.loc[group.index, "state_id_initial"] = sample_states.astype(np.int8)
        samples.loc[group.index, "state_initial"] = [
            STATE_NAMES[int(state)] for state in sample_states
        ]
        samples.loc[group.index, "state_center"] = center
        samples.loc[group.index, "state_sigma"] = sigma
        for threshold in (0.75, state_threshold_sigma, 1.25):
            values = residual_values
            states = np.where(
                values < center - threshold * sigma,
                0,
                np.where(values > center + threshold * sigma, 2, 1),
            )
            threshold_rows.append(
                {
                    "zone_id": zone_id,
                    "threshold_sigma": float(threshold),
                    "n_samples": int(values.size),
                    "low_fraction": float(np.mean(states == 0)),
                    "background_fraction": float(np.mean(states == 1)),
                    "high_fraction": float(np.mean(states == 2)),
                }
            )

    parent_residual_weights = hierarchical_weights(samples.reset_index(drop=True), unit_column="sample_id")
    _, parent_residual_sigma = robust_location_scale(samples["residual"].to_numpy(), parent_residual_weights)
    huber_floor = huber_delta_parent_sigma_floor * parent_residual_sigma
    for (well_name, zone_id), group in samples.groupby(["well_name", "zone_id"], sort=False):
        group = group.sort_values("zeta").reset_index(drop=True)
        center, sigma = zone_centers[str(zone_id)]
        residual = group["residual"].to_numpy()
        states = np.where(
            residual < center - state_threshold_sigma * sigma,
            0,
            np.where(residual > center + state_threshold_sigma * sigma, 2, 1),
        ).astype(np.int8)
        states = _merge_short_states(states, residual, np.array([center - sigma, center, center + sigma]))
        starts = np.r_[0, np.flatnonzero(np.diff(states) != 0) + 1]
        ends = np.r_[starts[1:], states.size]
        for object_index, (start, end) in enumerate(zip(starts, ends)):
            if end - start < 2:
                continue
            zeta = group["zeta"].to_numpy()[start:end]
            xi = (zeta - zeta[0]) / max(zeta[-1] - zeta[0], np.finfo(np.float64).eps)
            coefficients = _fit_object_profile(xi, residual[start:end], max(1.345 * sigma, huber_floor))
            fitted = (
                coefficients[0]
                + coefficients[1] * (2.0 * xi - 1.0)
                + coefficients[2] * np.sin(np.pi * xi)
            )
            profile_metrics = object_profile_metrics(xi, fitted)
            object_records.append(
                {
                    "well_name": well_name,
                    "spatial_cluster_id": int(group["spatial_cluster_id"].iloc[0]),
                    "zone_id": zone_id,
                    "object_id": f"{well_name}:{zone_id}:{object_index}",
                    "object_order": object_index,
                    "state_id": int(states[start]),
                    "state": STATE_NAMES[int(states[start])],
                    "zeta_top": float(zeta[0]),
                    "zeta_bottom": float(zeta[-1]),
                    "duration_fraction": float(zeta[-1] - zeta[0]),
                    "duration_s": float(group["twt_s"].iloc[end - 1] - group["twt_s"].iloc[start]),
                    "n_truth_samples": int(end - start),
                    "c0": float(coefficients[0]),
                    "c1": float(coefficients[1]),
                    "c2": float(coefficients[2]),
                    **profile_metrics,
                }
            )
            for point_index, (time, coordinate, observed, predicted) in enumerate(
                zip(
                    group["twt_s"].to_numpy()[start:end],
                    xi,
                    residual[start:end],
                    fitted,
                )
            ):
                profile_sample_records.append(
                    {
                        "well_name": well_name,
                        "spatial_cluster_id": int(group["spatial_cluster_id"].iloc[0]),
                        "zone_id": zone_id,
                        "object_id": f"{well_name}:{zone_id}:{object_index}",
                        "object_order": object_index,
                        "state_id": int(states[start]),
                        "state": STATE_NAMES[int(states[start])],
                        "point_index": int(point_index),
                        "twt_s": float(time),
                        "xi": float(coordinate),
                        "residual": float(observed),
                        "fitted_residual": float(predicted),
                        "fit_residual": float(observed - predicted),
                    }
                )
    objects = pd.DataFrame.from_records(object_records)
    if objects.empty:
        raise ValueError("No calibration objects were produced.")

    coefficient_columns = ("c0", "c1", "c2", *PROFILE_METRICS)
    parent_states: dict[str, Any] = {}
    for state in STATE_NAMES:
        group = objects.loc[objects["state"].eq(state)].reset_index(drop=True)
        if group.empty:
            raise ValueError(f"invalid_impedance_calibration:parent_state_missing:{state}")
        weights = hierarchical_weights(group, unit_column="object_id")
        parent_states[state] = {
            column: _distribution(group, column, weights) for column in coefficient_columns
        }
        duration_values = np.log(np.maximum(group["duration_fraction"].to_numpy(), 1e-9))
        duration_center = weighted_quantile(duration_values, weights, 0.5)
        duration_sigma = (
            weighted_quantile(duration_values, weights, 0.75)
            - weighted_quantile(duration_values, weights, 0.25)
        ) / 1.349
        parent_states[state]["log_duration"] = {
            "median": duration_center,
            "robust_sigma": max(float(duration_sigma), 1e-6),
            "p01": weighted_quantile(duration_values, weights, 0.01),
            "p99": weighted_quantile(duration_values, weights, 0.99),
        }

    zone_models: dict[str, Any] = {}
    qc_records: list[dict[str, Any]] = threshold_rows
    for zone_id, zone_samples in samples.groupby("zone_id", sort=False):
        zone_objects = objects.loc[objects["zone_id"].eq(zone_id)].reset_index(drop=True)
        zone_backgrounds = backgrounds.loc[backgrounds["zone_id"].eq(zone_id)].reset_index(drop=True)
        background_weights = hierarchical_weights(
            zone_backgrounds.assign(background_unit=zone_backgrounds["well_name"]),
            unit_column="background_unit",
        )
        background_model = {
            column: _distribution(zone_backgrounds, column, background_weights)
            for column in ("background_a", "background_b", "zone_duration_s")
        }
        sample_weights = hierarchical_weights(zone_samples.reset_index(drop=True), unit_column="sample_id")
        raw_ai_bounds = {
            "p01": weighted_quantile(zone_samples["full_log_ai"].to_numpy(), sample_weights, 0.01),
            "p99": weighted_quantile(zone_samples["full_log_ai"].to_numpy(), sample_weights, 0.99),
        }
        parent_ai_bounds = {
            "p01": weighted_quantile(samples["full_log_ai"].to_numpy(), parent_residual_weights, 0.01),
            "p99": weighted_quantile(samples["full_log_ai"].to_numpy(), parent_residual_weights, 0.99),
        }
        states_model: dict[str, Any] = {}
        for state in STATE_NAMES:
            group = zone_objects.loc[zone_objects["state"].eq(state)].reset_index(drop=True)
            n_wells = int(group["well_name"].nunique())
            n_clusters = int(group["spatial_cluster_id"].nunique())
            n_objects = int(len(group))
            weight = min(1.0, n_wells / 5.0, n_clusters / 3.0, n_objects / 20.0)
            if n_wells >= 5 and n_clusters >= 3 and n_objects >= 20:
                evidence = "field_calibrated"
            elif n_wells >= 3 and n_clusters >= 2 and n_objects > 0:
                evidence = "shrunk"
            else:
                evidence = "generic_prior"
            raw_distributions: dict[str, Any] = {}
            if group.empty:
                raw_distributions = {column: dict(parent_states[state][column]) for column in coefficient_columns}
                raw_duration = dict(parent_states[state]["log_duration"])
            else:
                object_weights = hierarchical_weights(group, unit_column="object_id")
                raw_distributions = {
                    column: _distribution(group, column, object_weights) for column in coefficient_columns
                }
                log_duration = np.log(np.maximum(group["duration_fraction"].to_numpy(), 1e-9))
                raw_duration = {
                    "median": weighted_quantile(log_duration, object_weights, 0.5),
                    "robust_sigma": (
                        weighted_quantile(log_duration, object_weights, 0.75)
                        - weighted_quantile(log_duration, object_weights, 0.25)
                    ) / 1.349,
                    "p01": weighted_quantile(log_duration, object_weights, 0.01),
                    "p99": weighted_quantile(log_duration, object_weights, 0.99),
                }
            states_model[state] = {
                "evidence": evidence,
                "n_wells": n_wells,
                "n_clusters": n_clusters,
                "n_objects": n_objects,
                "weight": weight,
                "coefficients": {
                    column: _final_distribution(
                        raw_distributions[column],
                        parent_states[state][column],
                        weight,
                        scale_floor=coefficient_sigma_parent_floor,
                        scale_cap=coefficient_sigma_parent_cap,
                    )
                    for column in coefficient_columns
                },
                "log_duration": _final_distribution(
                    raw_duration,
                    parent_states[state]["log_duration"],
                    weight,
                    scale_floor=coefficient_sigma_parent_floor,
                    scale_cap=coefficient_sigma_parent_cap,
                ),
            }
            qc_records.append(
                {
                    "zone_id": zone_id,
                    "state": state,
                    "threshold_sigma": state_threshold_sigma,
                    "n_wells": n_wells,
                    "n_clusters": n_clusters,
                    "n_objects": n_objects,
                    "evidence": evidence,
                    "shrink_weight": weight,
                }
            )

        transition = np.zeros((3, 3), dtype=np.float64)
        support: dict[str, Any] = {}
        initial = np.zeros(3, dtype=np.float64)
        for _, group in zone_objects.groupby("well_name", sort=False):
            ordered = group.sort_values("object_order")["state_id"].to_numpy(dtype=np.int64)
            if ordered.size:
                initial[ordered[0]] += 1.0
            for left, right in zip(ordered[:-1], ordered[1:]):
                if left != right:
                    transition[left, right] += 1.0
        for left in range(3):
            for right in range(3):
                if left == right:
                    continue
                edge = zone_objects.merge(
                    zone_objects,
                    on=["well_name", "zone_id"],
                    suffixes=("_a", "_b"),
                )
                edge = edge[
                    (edge["object_order_b"] == edge["object_order_a"] + 1)
                    & (edge["state_id_a"] == left)
                    & (edge["state_id_b"] == right)
                ]
                count = int(len(edge))
                wells_count = int(edge["well_name"].nunique())
                clusters_count = int(edge["spatial_cluster_id_a"].nunique()) if count else 0
                support[f"{left}->{right}"] = {
                    "count": count,
                    "n_wells": wells_count,
                    "n_clusters": clusters_count,
                    "source": (
                        "zone_supported"
                        if count >= 2 and wells_count >= 2 and clusters_count >= 2
                        else "parent_prior_only"
                    ),
                }
        transition = transition + 0.5
        np.fill_diagonal(transition, 0.0)
        transition /= transition.sum(axis=1, keepdims=True)
        initial = (initial + 0.5) / float(np.sum(initial + 0.5))
        zone_weight = min(
            1.0,
            zone_objects["well_name"].nunique() / 5.0,
            zone_objects["spatial_cluster_id"].nunique() / 3.0,
            len(zone_objects) / 60.0,
        )
        zone_models[str(zone_id)] = {
            "background": background_model,
            "states": states_model,
            "initial_probabilities": initial.tolist(),
            "transition_matrix": transition.tolist(),
            "transition_support": support,
            "ai_bounds": {
                "raw": raw_ai_bounds,
                "parent": parent_ai_bounds,
                "weight": zone_weight,
                "p01": _blend(raw_ai_bounds["p01"], parent_ai_bounds["p01"], zone_weight),
                "p99": _blend(raw_ai_bounds["p99"], parent_ai_bounds["p99"], zone_weight),
            },
        }

    zones = []
    for top, bottom in zip(ordered_horizons[:-1], ordered_horizons[1:]):
        zone_id = f"{top}__to__{bottom}"
        if zone_id not in zone_models:
            raise ValueError(f"Missing calibrated zone: {zone_id}")
        zones.append({"zone_id": zone_id, "top_horizon": top, "bottom_horizon": bottom})
    calibration = ImpedanceCalibration(
        schema_version=SCHEMA_VERSION,
        generator_family=GENERATOR_FAMILY,
        truth_dt_s=float(truth_dt_s),
        state_threshold_sigma=float(state_threshold_sigma),
        ordered_horizons=tuple(ordered_horizons),
        zones=tuple(zones),
        parent={"states": parent_states},
        zone_models=zone_models,
        source_runs=dict(source_runs),
        source_hashes=dict(source_hashes),
    )
    return (
        calibration,
        objects,
        pd.DataFrame.from_records(qc_records),
        samples,
        backgrounds,
        pd.DataFrame.from_records(profile_sample_records),
    )
