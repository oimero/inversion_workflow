"""Domain-neutral empirical RGT-amplitude calibration contracts.

Domain adapters prepare real and pilot sections on their native vertical axes.
Everything after the aligned ``seismic/rgt/valid_mask`` seam is shared by the
time- and depth-domain workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
import h5py

from cup.synthetic.schemas import (
    SCIENCE_CONTRACT,
    SEISMIC_AMPLITUDE_PRIOR_SCHEMA_VERSION,
    require_science_contract,
)
from cup.synthetic.core.v5_artifacts import REALIZATION_INDEX_COLUMNS
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    is_consumable_contract_status,
    repo_relative_path,
    require_contract_fingerprint,
    resolve_relative_path,
    sha256_file,
    write_json,
)


SCHEMA_VERSION = SEISMIC_AMPLITUDE_PRIOR_SCHEMA_VERSION


def parse_amplitude_calibration_controls(
    value: Any, *, required: bool
) -> dict[str, Any] | None:
    if value is None:
        if required:
            raise ValueError("calibrated_rgt_gain requires amplitude_calibration controls")
        return None
    if not isinstance(value, Mapping):
        raise ValueError("amplitude_calibration must be a mapping")
    allowed = {
        "pilot_attempts_per_scenario", "rgt_node_count", "lateral_node_count",
        "minimum_samples_per_cell", "smoothing_sigma_nodes",
        "mean_shrinkage", "max_abs_mean_log_gain", "max_abs_total_log_gain",
        "interaction_explained_variance", "maximum_interaction_rank",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"amplitude_calibration contains unknown keys: {unknown}")
    missing = sorted(allowed - set(value))
    if missing:
        raise ValueError(f"amplitude_calibration lacks keys: {missing}")
    integers = {
        key: int(value[key])
        for key in (
            "pilot_attempts_per_scenario", "rgt_node_count", "lateral_node_count",
            "minimum_samples_per_cell", "maximum_interaction_rank",
        )
    }
    if any(isinstance(value[key], bool) or integers[key] != value[key] or integers[key] <= 0 for key in integers):
        raise ValueError("amplitude_calibration count fields must be positive integers")
    if integers["rgt_node_count"] < 5:
        raise ValueError("amplitude_calibration.rgt_node_count must be at least 5")
    if integers["lateral_node_count"] < 5:
        raise ValueError("amplitude_calibration.lateral_node_count must be at least 5")
    floats = {
        key: float(value[key])
        for key in (
            "smoothing_sigma_nodes", "mean_shrinkage", "max_abs_mean_log_gain",
            "max_abs_total_log_gain",
            "interaction_explained_variance",
        )
    }
    if any(not np.isfinite(item) or item <= 0.0 for item in floats.values()):
        raise ValueError("amplitude_calibration numeric controls must be positive and finite")
    if floats["mean_shrinkage"] > 1.0:
        raise ValueError("amplitude_calibration.mean_shrinkage must be at most 1")
    if floats["interaction_explained_variance"] > 1.0:
        raise ValueError("amplitude_calibration.interaction_explained_variance must be at most 1")
    if floats["max_abs_total_log_gain"] < floats["max_abs_mean_log_gain"]:
        raise ValueError(
            "amplitude_calibration.max_abs_total_log_gain must be at least "
            "max_abs_mean_log_gain"
        )
    return {**integers, **floats}


@dataclass(frozen=True)
class AmplitudeCalibrationSection:
    field_id: str
    section_id: str
    seismic: np.ndarray
    rgt: np.ndarray
    valid_mask: np.ndarray
    lateral_m: np.ndarray
    scenario_id: str = "real"

    def __post_init__(self) -> None:
        seismic = np.asarray(self.seismic, dtype=np.float64)
        rgt = np.asarray(self.rgt, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        lateral = np.asarray(self.lateral_m, dtype=np.float64).reshape(-1)
        if seismic.ndim != 2 or rgt.shape != seismic.shape or valid.shape != seismic.shape:
            raise ValueError("amplitude calibration section arrays must be aligned 2D fields")
        if np.any(valid & (~np.isfinite(seismic) | ~np.isfinite(rgt))):
            raise ValueError("amplitude calibration section has non-finite public samples")
        if lateral.size < 2 or lateral.size != seismic.shape[0] or np.any(~np.isfinite(lateral)) or np.any(np.diff(lateral) <= 0.0):
            raise ValueError("amplitude calibration lateral_m must align and increase")
        object.__setattr__(self, "seismic", seismic)
        object.__setattr__(self, "rgt", rgt)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "lateral_m", lateral)


def canonical_sha256(value: Any) -> str:
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_pilot_compatibility_contract(
    *,
    sample_domain: str,
    sample_unit: str,
    axis_basis: str,
    script_cfg: Mapping[str, Any],
    input_contracts: Mapping[str, Any],
    horizon_inputs: Sequence[Mapping[str, str]],
    base_seismic_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the science inputs that must match between pilot and calibration."""
    inputs = {
        str(role): str(dict(value).get("contract_fingerprint_sha256") or "")
        for role, value in input_contracts.items()
    }
    payload = {
        "sample_domain": str(sample_domain),
        "sample_unit": str(sample_unit),
        "axis_basis": str(axis_basis),
        "ordered_horizons": [dict(value) for value in horizon_inputs],
        "sections": list(script_cfg["sections"]),
        "sampling": dict(script_cfg["sampling"]),
        "generation_scenarios": {
            key: list(script_cfg["generation"][key])
            for key in ("duration_modes", "geometry_families", "geometry_directions")
        },
        "impedance_family": str(script_cfg["impedance"]["family"]),
        "input_contract_fingerprints": inputs,
        "base_seismic_contract": dict(base_seismic_contract),
    }
    return {"contract": payload, "sha256": canonical_sha256(payload)}


def _validated_compatibility_sha(value: Mapping[str, Any], *, label: str) -> str:
    contract = value.get("contract")
    recorded = str(value.get("sha256") or "")
    if not isinstance(contract, Mapping) or not recorded:
        raise ValueError(f"{label} compatibility contract is incomplete")
    actual = canonical_sha256(contract)
    if recorded != actual:
        raise ValueError(f"{label} compatibility SHA-256 is stale")
    return recorded


def build_amplitude_pilot_config(script_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the same formal, base-only stratified pilot in either domain."""
    controls = script_cfg.get("amplitude_calibration")
    if not isinstance(controls, Mapping):
        raise ValueError("generate-amplitude-pilot requires amplitude_calibration controls")
    pilot = deepcopy(dict(script_cfg))
    attempts = int(controls["pilot_attempts_per_scenario"])
    pilot["global_seed"] = int(pilot["global_seed"]) + 104729
    pilot["generation"] = deepcopy(dict(pilot["generation"]))
    pilot["generation"]["attempts_per_scenario"] = attempts
    pilot["generation"]["acceptance_qc"] = deepcopy(
        dict(pilot["generation"]["acceptance_qc"])
    )
    pilot["generation"]["acceptance_qc"]["minimum_attempts_per_scenario"] = attempts
    pilot["seismic_views"] = {"operators": {}, "views": []}
    pilot["benchmark_purpose"] = "seismic_amplitude_prior_pilot"
    return pilot


def load_pilot_sections(pilot_dir: Path) -> list[AmplitudeCalibrationSection]:
    """Load the domain-neutral seismic/RGT/mask seam from a v5 pilot."""
    index = pd.read_csv(
        pilot_dir / "realization_index.csv", dtype=str, keep_default_na=False
    )
    required = set(REALIZATION_INDEX_COLUMNS)
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"amplitude pilot realization index lacks columns: {missing}")
    if index.empty:
        raise ValueError("amplitude pilot realization index contains no successful parents")
    if index["realization_id"].duplicated().any():
        raise ValueError("amplitude pilot realization index contains duplicate parents")
    result: list[AmplitudeCalibrationSection] = []
    with h5py.File(pilot_dir / "synthetic_benchmark.h5", "r") as h5:
        for row in index.to_dict(orient="records"):
            rid = str(row["realization_id"])
            root = f"/realizations/{rid}"
            expected_mask_path = f"{root}/masks/valid_mask"
            if str(row["valid_mask_dataset"]) != expected_mask_path:
                raise ValueError(
                    f"amplitude pilot parent {rid!r} has a non-v5 mask path"
                )
            required_datasets = {
                "seismic": f"{root}/seismic/seismic_observed",
                "rgt": f"{root}/truth/rgt_model",
                "valid_mask": expected_mask_path,
                "lateral_axis": f"{root}/axes/lateral_m",
            }
            missing_datasets = sorted(
                label for label, path in required_datasets.items() if path not in h5
            )
            if missing_datasets:
                raise ValueError(
                    f"amplitude pilot parent {rid!r} lacks required datasets: "
                    f"{missing_datasets}"
                )
            result.append(AmplitudeCalibrationSection(
                field_id=rid,
                section_id=str(row["section_id"]),
                scenario_id=str(row["scenario_id"]),
                seismic=np.asarray(h5[required_datasets["seismic"]][()]),
                rgt=np.asarray(h5[required_datasets["rgt"]][()]),
                valid_mask=np.asarray(
                    h5[required_datasets["valid_mask"]][()], dtype=bool
                ),
                lateral_m=np.asarray(
                    h5[required_datasets["lateral_axis"]][()], dtype=np.float64
                ),
            ))
    return result


def validate_amplitude_pilot(
    pilot_dir: Path,
    *,
    sample_domain: str,
    expected_compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    summary = json.loads((pilot_dir / "run_summary.json").read_text(encoding="utf-8"))
    require_science_contract(summary, label="amplitude pilot benchmark")
    status = str(summary.get("status") or "")
    if status not in {"success", "completed_with_warnings"}:
        raise ValueError(
            "amplitude pilot must be a completed formal benchmark; "
            f"got status={status!r}"
        )
    if summary.get("sample_domain") != sample_domain:
        raise ValueError(f"amplitude pilot sample domain must be {sample_domain!r}")
    if int(summary.get("seismic_view_count", -1)) != 0:
        raise ValueError("amplitude pilot must contain no views")
    if dict(summary.get("amplitude_pilot_compatibility") or {}) != dict(expected_compatibility):
        raise ValueError("amplitude pilot compatibility contract differs from current configuration")
    return summary


def rgt_from_horizons(axis: np.ndarray, horizons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map either TWT or TVDSS samples to the same dimensionless zone RGT."""
    vertical = np.asarray(axis, dtype=np.float64)[None, :]
    surfaces = np.asarray(horizons, dtype=np.float64)
    rgt = np.full((surfaces.shape[0], vertical.shape[1]), np.nan, dtype=np.float64)
    for zone in range(surfaces.shape[1] - 1):
        top = surfaces[:, zone, None]
        bottom = surfaces[:, zone + 1, None]
        inside = (vertical >= top) & (
            (vertical <= bottom) if zone == surfaces.shape[1] - 2 else (vertical < bottom)
        )
        fraction = (vertical - top) / (bottom - top)
        rgt[inside] = (zone + fraction)[inside]
    return rgt, np.isfinite(rgt)


def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(np.pad(values, radius, mode="edge"), kernel, mode="valid")


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return float("nan")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * float(np.sum(weights))
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _weighted_robust_sigma(values: np.ndarray, weights: np.ndarray) -> float:
    center = _weighted_median(values, weights)
    if not np.isfinite(center):
        return float("nan")
    return float(1.4826 * _weighted_median(np.abs(values - center), weights))


def _fill_vector(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    finite = np.flatnonzero(np.isfinite(output))
    if finite.size < 2:
        raise ValueError("amplitude statistic has disconnected support")
    return np.interp(np.arange(output.size), finite, output[finite])


def _fill_coarse_field(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    original_support = np.isfinite(output)
    for _ in range(2):
        for row in range(output.shape[0]):
            finite = np.flatnonzero(np.isfinite(output[row]))
            if finite.size >= 2:
                output[row] = np.interp(
                    np.arange(output.shape[1]), finite, output[row, finite]
                )
        for column in range(output.shape[1]):
            finite = np.flatnonzero(np.isfinite(output[:, column]))
            if finite.size >= 2:
                output[:, column] = np.interp(
                    np.arange(output.shape[0]), finite, output[finite, column]
                )
    if np.any(~np.isfinite(output)):
        raise ValueError("coarse amplitude field has disconnected support")
    output -= float(np.median(output[original_support]))
    return output


def _coarse_log_rms_field(
    section: AmplitudeCalibrationSection,
    rgt_knots: np.ndarray,
    *,
    lateral_node_count: int,
    minimum_samples: int,
) -> dict[str, Any]:
    mask = section.valid_mask
    if not np.any(mask):
        raise ValueError(f"amplitude field {section.field_id!r} has no valid samples")
    global_rms = float(np.sqrt(np.mean(section.seismic[mask] ** 2)))
    if not np.isfinite(global_rms) or global_rms <= 0.0:
        raise ValueError(f"amplitude field {section.field_id!r} has invalid RMS")
    lateral_nodes = np.linspace(
        float(section.lateral_m[0]),
        float(section.lateral_m[-1]),
        int(lateral_node_count),
        dtype=np.float64,
    )
    public_rgt = section.rgt[mask]
    if (
        float(np.min(public_rgt)) < float(rgt_knots[0]) - 1e-9
        or float(np.max(public_rgt)) > float(rgt_knots[-1]) + 1e-9
    ):
        raise ValueError(
            f"amplitude field {section.field_id!r} lies outside the RGT contract"
        )
    # Assign every valid sample to its nearest coarse-grid node.  This is the
    # Voronoi equivalent of non-overlapping local RMS windows and avoids one
    # full-section boolean allocation per cell during large pilot calibration.
    lateral_index = np.rint(
        (section.lateral_m - lateral_nodes[0])
        / (lateral_nodes[-1] - lateral_nodes[0])
        * (lateral_nodes.size - 1)
    ).astype(np.int64)
    selected_rgt = np.rint(
        (public_rgt - rgt_knots[0])
        / (rgt_knots[-1] - rgt_knots[0])
        * (rgt_knots.size - 1)
    ).astype(np.int64)
    lateral_grid = np.broadcast_to(lateral_index[:, None], section.rgt.shape)
    selected_lateral = lateral_grid[mask]
    selected_rgt = np.clip(selected_rgt, 0, rgt_knots.size - 1)
    flat_index = selected_lateral * rgt_knots.size + selected_rgt
    cell_count = lateral_nodes.size * rgt_knots.size
    counts = np.bincount(flat_index, minlength=cell_count).reshape(
        lateral_nodes.size, rgt_knots.size
    )
    energy = np.bincount(
        flat_index,
        weights=np.square(section.seismic[mask]),
        minlength=cell_count,
    ).reshape(lateral_nodes.size, rgt_knots.size)
    values = np.full(counts.shape, np.nan, dtype=np.float64)
    supported_cells = counts >= minimum_samples
    values[supported_cells] = np.log(np.maximum(
        np.sqrt(energy[supported_cells] / counts[supported_cells]),
        global_rms * 1e-12,
    ))
    supported = int(np.count_nonzero(np.isfinite(values)))
    if supported < max(9, values.size // 3):
        raise ValueError(
            f"amplitude field {section.field_id!r} has insufficient coarse support"
        )
    return {
        "field_id": section.field_id,
        "section_id": section.section_id,
        "scenario_id": section.scenario_id,
        "lateral_m": lateral_nodes,
        "rgt_knots": rgt_knots,
        "values": _fill_coarse_field(values),
        "counts": counts,
        "support_mask": supported_cells,
    }


def _coarse_fields(
    sections: Sequence[AmplitudeCalibrationSection],
    rgt_knots: np.ndarray,
    *,
    lateral_node_count: int,
    minimum_samples: int,
    source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fields: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    qc: list[dict[str, Any]] = []
    for section in sections:
        try:
            field = _coarse_log_rms_field(
                section,
                rgt_knots,
                lateral_node_count=lateral_node_count,
                minimum_samples=minimum_samples,
            )
        except ValueError as exc:
            qc.append({
                "source": source,
                "field_id": section.field_id,
                "section_id": section.section_id,
                "scenario_id": section.scenario_id,
                "status": "warning_excluded_from_fit",
                "warning": str(exc),
            })
            continue
        fields.append(field)
        qc.append({
            "source": source,
            "field_id": section.field_id,
            "section_id": section.section_id,
            "scenario_id": section.scenario_id,
            "status": "used",
            "warning": "",
            "supported_cell_count": int(np.count_nonzero(field["support_mask"])),
            "total_cell_count": int(np.asarray(field["support_mask"]).size),
            "supported_cell_fraction": float(np.mean(field["support_mask"])),
        })
        rows.extend(
            {
                "source": source,
                "field_id": section.field_id,
                "section_id": section.section_id,
                "scenario_id": section.scenario_id,
                "lateral_m": float(lateral),
                "rgt": float(rgt),
                "centered_log_rms": float(value),
                "sample_count": int(count),
                "supported": bool(supported),
            }
            for lateral, value_row, count_row, support_row in zip(
                field["lateral_m"], field["values"], field["counts"],
                field["support_mask"],
            )
            for rgt, value, count, supported in zip(
                rgt_knots, value_row, count_row, support_row
            )
        )
    return fields, rows, qc


def _field_curve(field: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(field["values"], dtype=np.float64)
    support = np.asarray(field["support_mask"], dtype=bool)
    return np.asarray([
        np.median(values[support[:, index], index])
        if np.any(support[:, index]) else np.nan
        for index in range(values.shape[1])
    ])


def _stratified_field_weights(
    fields: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    by_section: dict[str, dict[str, list[int]]] = {}
    for index, field in enumerate(fields):
        section = str(field["section_id"])
        scenario = str(field["scenario_id"])
        by_section.setdefault(section, {}).setdefault(scenario, []).append(index)
    weights = np.zeros(len(fields), dtype=np.float64)
    qc: list[dict[str, Any]] = []
    section_weight = 1.0 / len(by_section)
    for section, strata in sorted(by_section.items()):
        stratum_weight = 1.0 / len(strata)
        for scenario, indices in sorted(strata.items()):
            field_weight = section_weight * stratum_weight / len(indices)
            weights[indices] = field_weight
            qc.append({
                "section_id": section,
                "scenario_id": scenario,
                "accepted_realizations": len(indices),
                "stratum_weight_within_section": stratum_weight,
                "section_weight": section_weight,
                "field_weight": field_weight,
            })
    return weights, qc


def _weighted_location(
    fields: Sequence[Mapping[str, Any]], weights: np.ndarray
) -> np.ndarray:
    curves = np.vstack([_field_curve(field) for field in fields])
    return _fill_vector(np.asarray([
        _weighted_median(curves[:, index], weights)
        for index in range(curves.shape[1])
    ]))


def _decompose_fields(
    fields: Sequence[Mapping[str, Any]], source_mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgt_components: list[np.ndarray] = []
    lateral_components: list[np.ndarray] = []
    interactions: list[np.ndarray] = []
    for field in fields:
        values = np.asarray(field["values"], dtype=np.float64)
        support = np.asarray(field["support_mask"], dtype=bool)
        residual = np.where(support, values - source_mean[None, :], np.nan)
        rgt = _fill_vector(np.asarray([
            np.median(residual[np.isfinite(residual[:, index]), index])
            if np.any(np.isfinite(residual[:, index])) else np.nan
            for index in range(residual.shape[1])
        ]))
        rgt -= float(np.median(rgt))
        remainder = residual - rgt[None, :]
        lateral = _fill_vector(np.asarray([
            np.median(row[np.isfinite(row)]) if np.any(np.isfinite(row)) else np.nan
            for row in remainder
        ]))
        lateral -= float(np.median(lateral))
        interaction = np.where(support, remainder - lateral[:, None], np.nan)
        interaction -= float(np.median(interaction[np.isfinite(interaction)]))
        rgt_components.append(rgt)
        lateral_components.append(lateral)
        interactions.append(interaction)
    return (
        np.vstack(rgt_components),
        np.vstack(lateral_components),
        np.stack(interactions),
    )


def _correlation_length(
    values: np.ndarray, coordinates: Sequence[np.ndarray], field_weights: np.ndarray
) -> float:
    estimates: list[float] = []
    estimate_weights: list[float] = []
    threshold = float(np.exp(-1.0))
    for vector, axis, weight in zip(
        np.asarray(values, dtype=np.float64), coordinates, field_weights
    ):
        finite = np.isfinite(vector)
        vector = vector[finite]
        coordinate = np.asarray(axis, dtype=np.float64)[finite]
        centered = vector - float(np.mean(vector))
        variance = float(np.mean(centered * centered))
        if vector.size < 4 or variance <= np.finfo(np.float64).eps:
            continue
        for lag in range(1, vector.size):
            correlation = float(np.mean(centered[:-lag] * centered[lag:]) / variance)
            if correlation <= threshold:
                estimates.append(float(np.median(coordinate[lag:] - coordinate[:-lag])))
                estimate_weights.append(float(weight))
                break
    return (
        _weighted_median(np.asarray(estimates), np.asarray(estimate_weights))
        if estimates else float("nan")
    )


def _extra_sigma(
    real: np.ndarray,
    pilot: np.ndarray,
    real_weights: np.ndarray,
    pilot_weights: np.ndarray,
) -> tuple[float, float, float]:
    real_sample_weights = np.broadcast_to(
        real_weights.reshape((-1,) + (1,) * (real.ndim - 1)), real.shape
    ).copy()
    pilot_sample_weights = np.broadcast_to(
        pilot_weights.reshape((-1,) + (1,) * (pilot.ndim - 1)), pilot.shape
    ).copy()
    for index in range(real.shape[0]):
        finite_count = np.count_nonzero(np.isfinite(real[index]))
        if finite_count:
            real_sample_weights[index] /= finite_count
    for index in range(pilot.shape[0]):
        finite_count = np.count_nonzero(np.isfinite(pilot[index]))
        if finite_count:
            pilot_sample_weights[index] /= finite_count
    real_sigma = _weighted_robust_sigma(real, real_sample_weights)
    pilot_sigma = _weighted_robust_sigma(pilot, pilot_sample_weights)
    extra = float(np.sqrt(max(0.0, real_sigma * real_sigma - pilot_sigma * pilot_sigma)))
    return real_sigma, pilot_sigma, extra


def _weighted_pairwise_covariance(
    rows: np.ndarray, field_weights: np.ndarray, *, rows_per_field: int
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    field_weights = np.asarray(field_weights, dtype=np.float64)
    if rows.shape[0] != field_weights.size * rows_per_field:
        raise ValueError("interaction rows do not align with field weights")
    field_index = np.repeat(np.arange(field_weights.size), rows_per_field)
    covariance = np.zeros((rows.shape[1], rows.shape[1]), dtype=np.float64)
    for left in range(rows.shape[1]):
        for right in range(left, rows.shape[1]):
            valid = (
                np.isfinite(rows[:, left]) & np.isfinite(rows[:, right])
            )
            if np.count_nonzero(valid) < 2:
                continue
            valid_field_index = field_index[valid]
            valid_counts = np.bincount(
                valid_field_index, minlength=field_weights.size
            )
            w = field_weights[valid_field_index] / valid_counts[valid_field_index]
            w /= np.sum(w)
            x = rows[valid, left]
            y = rows[valid, right]
            value = float(np.sum(w * (x - np.sum(w * x)) * (y - np.sum(w * y))))
            covariance[left, right] = covariance[right, left] = value
    return covariance


def _interaction_prior(
    real: np.ndarray,
    pilot: np.ndarray,
    fields: Sequence[Mapping[str, Any]],
    real_weights: np.ndarray,
    pilot_weights: np.ndarray,
    rgt_knots: np.ndarray,
    *,
    explained_variance: float,
    maximum_rank: int,
) -> dict[str, Any]:
    real_rows = real.reshape(-1, real.shape[-1])
    pilot_rows = pilot.reshape(-1, pilot.shape[-1])
    real_cov = _weighted_pairwise_covariance(
        real_rows, real_weights, rows_per_field=real.shape[1]
    )
    pilot_cov = _weighted_pairwise_covariance(
        pilot_rows, pilot_weights, rows_per_field=pilot.shape[1]
    )
    covariance = 0.5 * ((real_cov - pilot_cov) + (real_cov - pilot_cov).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    positive = np.maximum(eigenvalues, 0.0)
    order = np.argsort(positive)[::-1]
    positive = positive[order]
    eigenvectors = eigenvectors[:, order]
    total = float(np.sum(positive))
    if total <= np.finfo(np.float64).eps:
        return {
            "enabled": False,
            "rank": 0,
            "rgt_modes": [],
            "eigenvalues": [],
            "lateral_correlation_lengths_m": [],
            "log_sigma": 0.0,
            "disabled_reason": "pilot_variance_explains_real_variance",
        }
    cumulative = np.cumsum(positive) / total
    rank = min(int(np.searchsorted(cumulative, explained_variance) + 1), maximum_rank)
    positive = positive[:rank]
    modes = eigenvectors[:, :rank].T
    lateral_lengths: list[float] = []
    for mode in modes:
        scores = np.stack([
            np.asarray([
                float(np.sum(row[finite] * mode[finite]) / np.sum(mode[finite] ** 2))
                if np.any(finite := np.isfinite(row)) and np.sum(mode[finite] ** 2) > 0.0
                else np.nan
                for row in matrix
            ])
            for matrix in real
        ])
        length = _correlation_length(
            scores,
            [np.asarray(field["lateral_m"]) for field in fields],
            real_weights,
        )
        lateral_lengths.append(length)
    finite = np.isfinite(lateral_lengths) & (np.asarray(lateral_lengths) > 0.0)
    if not np.all(finite):
        return {
            "enabled": False,
            "rank": 0,
            "rgt_modes": [],
            "eigenvalues": [],
            "lateral_correlation_lengths_m": [],
            "log_sigma": 0.0,
            "estimated_log_sigma_before_disable": float(
                np.sqrt(total / rgt_knots.size)
            ),
            "disabled_reason": "unstable_lateral_correlation_estimate",
        }
    return {
        "enabled": True,
        "rank": rank,
        "rgt_modes": modes.tolist(),
        "eigenvalues": positive.tolist(),
        "lateral_correlation_lengths_m": [float(value) for value in lateral_lengths],
        "log_sigma": float(np.sqrt(np.sum(positive) / rgt_knots.size)),
        "explained_extra_variance_fraction": float(np.sum(positive) / total),
        "disabled_reason": "",
    }


def fit_amplitude_prior(
    *,
    real_sections: Sequence[AmplitudeCalibrationSection],
    pilot_sections: Sequence[AmplitudeCalibrationSection],
    n_zones: int,
    controls: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not real_sections or not pilot_sections:
        raise ValueError("amplitude calibration requires real and pilot sections")
    knots = np.linspace(0.0, float(n_zones), int(controls["rgt_node_count"]))
    field_options = {
        "lateral_node_count": int(controls["lateral_node_count"]),
        "minimum_samples": int(controls["minimum_samples_per_cell"]),
    }
    real_fields, real_rows, real_qc = _coarse_fields(
        real_sections, knots, source="real", **field_options
    )
    pilot_fields, pilot_rows, pilot_qc = _coarse_fields(
        pilot_sections, knots, source="pilot", **field_options
    )
    if not real_fields or not pilot_fields:
        raise ValueError("amplitude calibration has no usable real or pilot fields")
    real_weights, _ = _stratified_field_weights(real_fields)
    pilot_weights, stratum_qc = _stratified_field_weights(pilot_fields)
    real_location = _weighted_location(real_fields, real_weights)
    pilot_location = _weighted_location(pilot_fields, pilot_weights)
    for row in stratum_qc:
        row["planned_realizations"] = int(controls["pilot_attempts_per_scenario"])
        row["acceptance_fraction"] = (
            float(row["accepted_realizations"]) / float(row["planned_realizations"])
        )
    raw = real_location - pilot_location
    smoothed = _smooth(raw, float(controls["smoothing_sigma_nodes"]))
    shrunk = float(controls["mean_shrinkage"]) * smoothed
    limit = float(controls["max_abs_mean_log_gain"])
    template = np.clip(shrunk, -limit, limit)
    template -= float(np.median(template))
    maximum = float(np.max(np.abs(template)))
    if maximum > limit:
        template *= limit / maximum

    real_rgt, real_lateral, real_interaction = _decompose_fields(
        real_fields, real_location
    )
    pilot_rgt, pilot_lateral, pilot_interaction = _decompose_fields(
        pilot_fields, pilot_location
    )
    rgt_real_sigma, rgt_pilot_sigma, rgt_extra_sigma = _extra_sigma(
        real_rgt, pilot_rgt, real_weights, pilot_weights
    )
    rgt_length = _correlation_length(
        real_rgt, [knots for _ in range(real_rgt.shape[0])], real_weights
    )
    rgt_enabled = bool(rgt_extra_sigma > 0.0 and np.isfinite(rgt_length) and rgt_length > 0.0)
    lateral_real_sigma, lateral_pilot_sigma, lateral_extra_sigma = _extra_sigma(
        real_lateral, pilot_lateral, real_weights, pilot_weights
    )
    lateral_length = _correlation_length(
        real_lateral,
        [np.asarray(field["lateral_m"]) for field in real_fields],
        real_weights,
    )
    lateral_enabled = bool(
        lateral_extra_sigma > 0.0
        and np.isfinite(lateral_length)
        and lateral_length > 0.0
    )
    interaction = _interaction_prior(
        real_interaction,
        pilot_interaction,
        real_fields,
        real_weights,
        pilot_weights,
        knots,
        explained_variance=float(controls["interaction_explained_variance"]),
        maximum_rank=int(controls["maximum_interaction_rank"]),
    )
    residual_model = {
        "rgt_component": {
            "enabled": rgt_enabled,
            "log_sigma": rgt_extra_sigma if rgt_enabled else 0.0,
            "correlation_length_rgt": rgt_length if rgt_enabled else None,
            "real_log_sigma": rgt_real_sigma,
            "pilot_log_sigma": rgt_pilot_sigma,
            "disabled_reason": "" if rgt_enabled else "pilot_variance_explains_real_variance_or_unstable_correlation",
        },
        "lateral_component": {
            "enabled": lateral_enabled,
            "log_sigma": lateral_extra_sigma if lateral_enabled else 0.0,
            "correlation_length_m": lateral_length if lateral_enabled else None,
            "real_log_sigma": lateral_real_sigma,
            "pilot_log_sigma": lateral_pilot_sigma,
            "disabled_reason": "" if lateral_enabled else "pilot_variance_explains_real_variance_or_unstable_correlation",
        },
        "interaction_component": interaction,
    }
    uncapped_total_extra_sigma = float(np.sqrt(
        (rgt_extra_sigma if rgt_enabled else 0.0) ** 2
        + (lateral_extra_sigma if lateral_enabled else 0.0) ** 2
        + float(interaction["log_sigma"]) ** 2
    ))
    mean_maximum = float(np.max(np.abs(template)))
    uncapped_maximum = mean_maximum + 3.0 * uncapped_total_extra_sigma
    total_limit = float(controls["max_abs_total_log_gain"])
    residual_shrinkage = 1.0
    if uncapped_maximum > total_limit and uncapped_total_extra_sigma > 0.0:
        residual_shrinkage = float(np.clip(
            (total_limit - mean_maximum) / (3.0 * uncapped_total_extra_sigma),
            0.0,
            1.0,
        ))
        for component_name in ("rgt_component", "lateral_component"):
            component = residual_model[component_name]
            component["log_sigma"] = float(component["log_sigma"]) * residual_shrinkage
        interaction["log_sigma"] = float(interaction["log_sigma"]) * residual_shrinkage
        interaction["eigenvalues"] = (
            np.asarray(interaction["eigenvalues"], dtype=np.float64)
            * residual_shrinkage ** 2
        ).tolist()
    total_extra_sigma = uncapped_total_extra_sigma * residual_shrinkage
    max_abs_log_gain = float(max(
        np.finfo(np.float64).eps,
        min(total_limit, mean_maximum + 3.0 * total_extra_sigma),
    ))
    mean_model = {
        "rgt_knots": knots.tolist(),
        "real_centered_log_rms": real_location.tolist(),
        "pilot_centered_log_rms": pilot_location.tolist(),
        "raw_mean_log_gain_rgt": raw.tolist(),
        "smoothed_mean_log_gain_rgt": smoothed.tolist(),
        "mean_log_gain_rgt": template.tolist(),
        "max_abs_log_gain": limit,
    }
    prior_sha = canonical_sha256({
        "mean_model": mean_model,
        "residual_model": residual_model,
        "max_abs_log_gain": max_abs_log_gain,
    })
    result = {
        "mean_model": mean_model,
        "residual_model": residual_model,
        "max_abs_log_gain": max_abs_log_gain,
        "prior_sha256": prior_sha,
        "estimation_diagnostics": {
            "n_real_fields": len(real_fields),
            "n_pilot_fields": len(pilot_fields),
            "total_extra_log_sigma": total_extra_sigma,
            "uncapped_total_extra_log_sigma": uncapped_total_extra_sigma,
            "uncapped_max_abs_log_gain": uncapped_maximum,
            "max_abs_total_log_gain": total_limit,
            "residual_shrinkage": residual_shrinkage,
            "real_weighted_supported_cell_fraction": float(np.sum(
                real_weights * np.asarray([
                    np.mean(field["support_mask"]) for field in real_fields
                ])
            )),
            "pilot_weighted_supported_cell_fraction": float(np.sum(
                pilot_weights * np.asarray([
                    np.mean(field["support_mask"]) for field in pilot_fields
                ])
            )),
            "common_supported_cell_fraction": float(np.mean(
                np.any(np.stack([
                    np.asarray(field["support_mask"], dtype=bool)
                    for field in real_fields
                ]), axis=0)
                & np.any(np.stack([
                    np.asarray(field["support_mask"], dtype=bool)
                    for field in pilot_fields
                ]), axis=0)
            )),
        },
    }
    summary = pd.DataFrame({
        "rgt": knots,
        "real_centered_log_rms": real_location,
        "pilot_centered_log_rms": pilot_location,
        "raw_mean_log_gain": raw,
        "smoothed_mean_log_gain": smoothed,
        "mean_log_gain": template,
    })
    return (
        result,
        pd.DataFrame(real_rows + pilot_rows),
        summary,
        pd.DataFrame(stratum_qc),
        pd.DataFrame(real_qc + pilot_qc),
    )


def publish_amplitude_prior(
    *,
    output_dir: Path,
    repo_root: Path,
    sample_domain: str,
    sample_unit: str,
    axis_basis: str,
    ordered_horizons: Sequence[str],
    real_sections: Sequence[AmplitudeCalibrationSection],
    pilot_sections: Sequence[AmplitudeCalibrationSection],
    controls: Mapping[str, Any],
    input_contracts: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    source_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit and atomically publish the shared seismic-amplitude prior."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    staging = output_dir.with_name(f".{output_dir.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        fit, fields, summary_frame, strata, field_qc = fit_amplitude_prior(
            real_sections=real_sections,
            pilot_sections=pilot_sections,
            n_zones=len(ordered_horizons) - 1,
            controls=controls,
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            **SCIENCE_CONTRACT,
            "sample_domain": sample_domain,
            "sample_unit": sample_unit,
            "axis_basis": axis_basis,
            "depth_basis": "tvdss" if sample_domain == "depth" else None,
            "semantics": "real_minus_pilot_coarse_2d_log_rms_amplitude_prior",
            "gauge": "median_log_gain_zero",
            "endpoint_extension_policy": "calibrated_full_horizon_contract",
            "ordered_horizons": list(ordered_horizons),
            **fit,
            "estimator": dict(controls),
            "pilot_compatibility": dict(compatibility),
            "inputs": dict(source_inputs),
        }
        artifact = staging / "seismic_amplitude_prior.json"
        fields_path = staging / "coarse_amplitude_fields.csv"
        summary_path = staging / "rgt_amplitude_summary.csv"
        strata_path = staging / "pilot_stratum_weights.csv"
        field_qc_path = staging / "amplitude_field_qc.csv"
        write_json(artifact, payload)
        fields.to_csv(fields_path, index=False)
        summary_frame.to_csv(summary_path, index=False)
        strata.to_csv(strata_path, index=False)
        field_qc.to_csv(field_qc_path, index=False)
        fingerprint = contract_fingerprint_sha256(
            contract_schema_version=SCHEMA_VERSION,
            semantics={
                **SCIENCE_CONTRACT,
                "sample_domain": sample_domain,
                "sample_unit": sample_unit,
                "axis_basis": axis_basis,
                "semantics": payload["semantics"],
                "prior_sha256": fit["prior_sha256"],
                "pilot_compatibility_sha256": compatibility["sha256"],
            },
            business_config=dict(controls),
            input_contracts=input_contracts,
            primary_artifacts={
                "seismic_amplitude_prior": artifact,
                "coarse_amplitude_fields": fields_path,
                "rgt_amplitude_summary": summary_path,
                "pilot_stratum_weights": strata_path,
                "amplitude_field_qc": field_qc_path,
            },
        )
        published_artifact = output_dir / artifact.name
        excluded_field_count = int((field_qc["status"] != "used").sum())
        run_summary = {
            "schema_version": SCHEMA_VERSION,
            **SCIENCE_CONTRACT,
            "status": "completed_with_warnings" if excluded_field_count else "success",
            "usable": True,
            "quality_warnings": [] if not excluded_field_count else ["amplitude_fields_excluded_for_insufficient_support"],
            "excluded_field_count": excluded_field_count,
            "sample_domain": sample_domain,
            "sample_unit": sample_unit,
            "axis_basis": axis_basis,
            "depth_basis": "tvdss" if sample_domain == "depth" else None,
            "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
            "contract_fingerprint_sha256": fingerprint,
            "input_contracts": dict(input_contracts),
            "pilot_compatibility": dict(compatibility),
            "outputs": {
                "seismic_amplitude_prior": repo_relative_path(published_artifact, root=repo_root),
                "coarse_amplitude_fields": repo_relative_path(output_dir / fields_path.name, root=repo_root),
                "rgt_amplitude_summary": repo_relative_path(output_dir / summary_path.name, root=repo_root),
                "pilot_stratum_weights": repo_relative_path(output_dir / strata_path.name, root=repo_root),
                "amplitude_field_qc": repo_relative_path(output_dir / field_qc_path.name, root=repo_root),
            },
        }
        write_json(staging / "run_summary.json", run_summary)
        _validate_calibration_payload(payload, expected_domain=sample_domain)
        staging.replace(output_dir)
        return run_summary
    except Exception as exc:
        write_json(staging / "run_failure.json", {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "staging_preserved": True,
        })
        raise


def _validate_calibration_payload(payload: Mapping[str, Any], *, expected_domain: str | None = None) -> None:
    require_science_contract(payload, label="seismic amplitude prior")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("seismic amplitude prior has unsupported schema")
    domain = str(payload.get("sample_domain") or "")
    if domain not in {"time", "depth"} or (expected_domain and domain != expected_domain):
        raise ValueError("seismic amplitude prior sample domain differs")
    expected = {"time": ("s", "twt"), "depth": ("m", "tvdss")}[domain]
    if (str(payload.get("sample_unit")), str(payload.get("axis_basis"))) != expected:
        raise ValueError("seismic amplitude prior axis contract is invalid")
    mean_model = dict(payload.get("mean_model") or {})
    residual_model = dict(payload.get("residual_model") or {})
    knots = np.asarray(mean_model.get("rgt_knots"), dtype=np.float64)
    template = np.asarray(mean_model.get("mean_log_gain_rgt"), dtype=np.float64)
    if knots.ndim != 1 or template.shape != knots.shape or knots.size < 2:
        raise ValueError("amplitude prior mean arrays are invalid")
    if np.any(~np.isfinite(knots)) or np.any(~np.isfinite(template)) or np.any(np.diff(knots) <= 0.0):
        raise ValueError("amplitude prior mean is not finite and monotonic")
    if abs(float(np.median(template))) > 1e-10:
        raise ValueError("amplitude prior violates its zero-median gauge")
    for name in ("rgt_component", "lateral_component", "interaction_component"):
        if not isinstance(residual_model.get(name), Mapping):
            raise ValueError(f"amplitude prior lacks residual component {name!r}")
    for name in ("rgt_component", "lateral_component"):
        component = dict(residual_model[name])
        sigma = float(component.get("log_sigma", np.nan))
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError(f"amplitude prior {name} log_sigma is invalid")
        if not bool(component.get("enabled")) and sigma != 0.0:
            raise ValueError(f"disabled amplitude prior {name} must have zero log_sigma")
    maximum = float(payload.get("max_abs_log_gain"))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("amplitude prior max_abs_log_gain must be positive and finite")
    interaction = dict(residual_model["interaction_component"])
    rank = int(interaction.get("rank", -1))
    modes = np.asarray(interaction.get("rgt_modes"), dtype=np.float64)
    eigenvalues = np.asarray(interaction.get("eigenvalues"), dtype=np.float64)
    lengths = np.asarray(
        interaction.get("lateral_correlation_lengths_m"), dtype=np.float64
    )
    if bool(interaction.get("enabled")) and (
        rank < 1
        or modes.shape != (rank, knots.size)
        or eigenvalues.shape != (rank,)
        or lengths.shape != (rank,)
        or np.any(~np.isfinite(modes))
        or np.any(~np.isfinite(eigenvalues))
        or np.any(eigenvalues <= 0.0)
        or np.any(~np.isfinite(lengths))
        or np.any(lengths <= 0.0)
    ):
        raise ValueError("amplitude prior interaction component is invalid")
    if not bool(interaction.get("enabled")) and (
        rank != 0
        or modes.size != 0
        or eigenvalues.size != 0
        or lengths.size != 0
        or float(interaction.get("log_sigma", np.nan)) != 0.0
    ):
        raise ValueError("disabled amplitude prior interaction component is invalid")
    estimator = dict(payload.get("estimator") or {})
    total_limit = float(estimator.get("max_abs_total_log_gain", np.nan))
    if not np.isfinite(total_limit) or total_limit <= 0.0 or maximum > total_limit + 1e-12:
        raise ValueError("amplitude prior exceeds its total log-gain limit")
    _validated_compatibility_sha(
        dict(payload.get("pilot_compatibility") or {}), label="amplitude prior"
    )
    expected_sha = canonical_sha256({
        "mean_model": mean_model,
        "residual_model": residual_model,
        "max_abs_log_gain": maximum,
    })
    if str(payload.get("prior_sha256") or "") != expected_sha:
        raise ValueError("amplitude prior SHA-256 is stale")


def load_seismic_amplitude_prior(
    path: Path,
    *,
    repo_root: Path,
    expected_domain: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    artifact = Path(path).resolve()
    summary_path = artifact.parent / "run_summary.json"
    if not artifact.is_file() or not summary_path.is_file():
        raise FileNotFoundError(artifact if not artifact.is_file() else summary_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _validate_calibration_payload(payload, expected_domain=expected_domain)
    require_science_contract(summary, label="seismic amplitude prior summary")
    if summary.get("schema_version") != SCHEMA_VERSION or not is_consumable_contract_status(summary.get("status")):
        raise ValueError("seismic amplitude prior summary is not consumable")
    recorded = resolve_relative_path(
        str(dict(summary.get("outputs") or {}).get("seismic_amplitude_prior") or ""),
        root=repo_root,
    )
    if recorded.resolve() != artifact:
        raise ValueError("amplitude prior summary points to a different artifact")
    return payload, {
        "artifact_sha256": sha256_file(artifact),
        "contract_fingerprint_sha256": require_contract_fingerprint(
            summary, label="seismic amplitude prior"
        ),
        "path": repo_relative_path(artifact, root=repo_root),
    }


def resolve_calibrated_seismic_views(
    view_config: Mapping[str, Any],
    *,
    prior_path: Path | None,
    repo_root: Path,
    sample_domain: str,
    ordered_horizons: Sequence[str],
    expected_pilot_compatibility: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    resolved = json.loads(json.dumps(dict(view_config)))
    operators = dict(resolved.get("operators") or {})
    calibrated = [
        key for key, value in operators.items()
        if isinstance(value, Mapping) and value.get("kind") == "calibrated_rgt_gain"
    ]
    if not calibrated:
        if prior_path is not None:
            raise ValueError("amplitude prior supplied without calibrated_rgt_gain")
        return resolved, None
    if prior_path is None:
        raise ValueError("calibrated_rgt_gain requires --seismic-amplitude-prior")
    payload, provenance = load_seismic_amplitude_prior(
        prior_path, repo_root=repo_root, expected_domain=sample_domain
    )
    expected_compatibility_sha = _validated_compatibility_sha(
        expected_pilot_compatibility, label="generation"
    )
    prior_compatibility_sha = _validated_compatibility_sha(
        dict(payload.get("pilot_compatibility") or {}), label="amplitude prior"
    )
    if prior_compatibility_sha != expected_compatibility_sha:
        raise ValueError(
            "amplitude prior Pilot compatibility differs from current generation"
        )
    if list(payload.get("ordered_horizons") or []) != list(ordered_horizons):
        raise ValueError("amplitude prior horizon contract differs from generation")
    mean_model = dict(payload["mean_model"])
    knots = np.asarray(mean_model["rgt_knots"], dtype=np.float64)
    if not np.isclose(knots[0], 0.0, atol=1e-12) or not np.isclose(
        knots[-1], len(ordered_horizons) - 1, atol=1e-12
    ):
        raise ValueError("amplitude prior does not cover the full RGT contract")
    for operator_id in calibrated:
        operator = dict(operators[operator_id])
        operator.update({
            "prior_schema_version": SCHEMA_VERSION,
            "prior_artifact_sha256": provenance["artifact_sha256"],
            "prior_contract_fingerprint_sha256": provenance["contract_fingerprint_sha256"],
            "prior_sha256": payload["prior_sha256"],
            "mean_model": payload["mean_model"],
            "residual_model": payload["residual_model"],
            "max_abs_log_gain": payload["max_abs_log_gain"],
        })
        operators[operator_id] = operator
    resolved["operators"] = operators
    return resolved, {
        **provenance,
        "schema_version": SCHEMA_VERSION,
        "prior_sha256": payload["prior_sha256"],
        "pilot_compatibility_sha256": prior_compatibility_sha,
    }


__all__ = [
    "AmplitudeCalibrationSection",
    "SCHEMA_VERSION",
    "build_pilot_compatibility_contract",
    "build_amplitude_pilot_config",
    "fit_amplitude_prior",
    "load_seismic_amplitude_prior",
    "load_pilot_sections",
    "parse_amplitude_calibration_controls",
    "publish_amplitude_prior",
    "resolve_calibrated_seismic_views",
    "rgt_from_horizons",
    "validate_amplitude_pilot",
]
