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
        "mean_shrinkage", "max_abs_mean_log_gain",
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
            "interaction_explained_variance",
        )
    }
    if any(not np.isfinite(item) or item <= 0.0 for item in floats.values()):
        raise ValueError("amplitude_calibration numeric controls must be positive and finite")
    if floats["mean_shrinkage"] > 1.0:
        raise ValueError("amplitude_calibration.mean_shrinkage must be at most 1")
    if floats["interaction_explained_variance"] > 1.0:
        raise ValueError("amplitude_calibration.interaction_explained_variance must be at most 1")
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
    index = pd.read_csv(pilot_dir / "realization_index.csv")
    required = {"realization_id", "section_id", "scenario_id", "status", "valid_mask_dataset"}
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"amplitude pilot realization index lacks columns: {missing}")
    result: list[AmplitudeCalibrationSection] = []
    with h5py.File(pilot_dir / "synthetic_benchmark.h5", "r") as h5:
        for row in index[index["status"].eq("ok")].to_dict(orient="records"):
            rid = str(row["realization_id"])
            root = f"/realizations/{rid}"
            result.append(AmplitudeCalibrationSection(
                field_id=rid,
                section_id=str(row["section_id"]),
                scenario_id=str(row["scenario_id"]),
                seismic=np.asarray(h5[f"{root}/seismic/seismic_observed"][()]),
                rgt=np.asarray(h5[f"{root}/truth/rgt_model"][()]),
                valid_mask=np.asarray(h5[str(row["valid_mask_dataset"])][()], dtype=bool),
                lateral_m=np.asarray(h5[f"{root}/axes/lateral_m"][()], dtype=np.float64),
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


def _robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    center = float(np.median(finite))
    return float(1.4826 * np.median(np.abs(finite - center)))


def _fill_coarse_field(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
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
    output -= float(np.median(output))
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
            }
            for lateral, value_row, count_row in zip(
                field["lateral_m"], field["values"], field["counts"]
            )
            for rgt, value, count in zip(rgt_knots, value_row, count_row)
        )
    return fields, rows, qc


def _field_curve(field: Mapping[str, Any]) -> np.ndarray:
    return np.median(np.asarray(field["values"], dtype=np.float64), axis=0)


def _real_location(fields: Sequence[Mapping[str, Any]]) -> np.ndarray:
    by_section: dict[str, list[np.ndarray]] = {}
    for field in fields:
        by_section.setdefault(str(field["section_id"]), []).append(_field_curve(field))
    return np.median(
        np.vstack([
            np.median(np.vstack(values), axis=0)
            for _, values in sorted(by_section.items())
        ]),
        axis=0,
    )


def _pilot_location(
    fields: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    by_stratum: dict[tuple[str, str], list[np.ndarray]] = {}
    for field in fields:
        key = (str(field["section_id"]), str(field["scenario_id"]))
        by_stratum.setdefault(key, []).append(_field_curve(field))
    by_section: dict[str, list[np.ndarray]] = {}
    qc: list[dict[str, Any]] = []
    for (section_id, scenario_id), values in sorted(by_stratum.items()):
        location = np.median(np.vstack(values), axis=0)
        by_section.setdefault(section_id, []).append(location)
        qc.append({
            "section_id": section_id,
            "scenario_id": scenario_id,
            "accepted_realizations": len(values),
            "stratum_weight_within_section": 0.0,
            "section_weight": 0.0,
        })
    for section_id, values in by_section.items():
        for row in qc:
            if row["section_id"] == section_id:
                row["stratum_weight_within_section"] = 1.0 / len(values)
                row["section_weight"] = 1.0 / len(by_section)
    survey = np.median(
        np.vstack([
            np.median(np.vstack(values), axis=0)
            for _, values in sorted(by_section.items())
        ]),
        axis=0,
    )
    return survey, qc


def _decompose_fields(
    fields: Sequence[Mapping[str, Any]], source_mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgt_components: list[np.ndarray] = []
    lateral_components: list[np.ndarray] = []
    interactions: list[np.ndarray] = []
    for field in fields:
        residual = np.asarray(field["values"], dtype=np.float64) - source_mean[None, :]
        rgt = np.median(residual, axis=0)
        rgt -= float(np.median(rgt))
        remainder = residual - rgt[None, :]
        lateral = np.median(remainder, axis=1)
        lateral -= float(np.median(lateral))
        interaction = remainder - lateral[:, None]
        interaction -= float(np.median(interaction))
        rgt_components.append(rgt)
        lateral_components.append(lateral)
        interactions.append(interaction)
    return (
        np.vstack(rgt_components),
        np.vstack(lateral_components),
        np.stack(interactions),
    )


def _correlation_length(values: np.ndarray, coordinates: Sequence[np.ndarray]) -> float:
    estimates: list[float] = []
    threshold = float(np.exp(-1.0))
    for vector, axis in zip(np.asarray(values, dtype=np.float64), coordinates):
        centered = vector - float(np.mean(vector))
        variance = float(np.mean(centered * centered))
        coordinate = np.asarray(axis, dtype=np.float64)
        if vector.size < 4 or variance <= np.finfo(np.float64).eps:
            continue
        for lag in range(1, vector.size):
            correlation = float(np.mean(centered[:-lag] * centered[lag:]) / variance)
            if correlation <= threshold:
                estimates.append(float(np.median(coordinate[lag:] - coordinate[:-lag])))
                break
    return float(np.median(estimates)) if estimates else float("nan")


def _extra_sigma(real: np.ndarray, pilot: np.ndarray) -> tuple[float, float, float]:
    real_sigma = _robust_sigma(real)
    pilot_sigma = _robust_sigma(pilot)
    extra = float(np.sqrt(max(0.0, real_sigma * real_sigma - pilot_sigma * pilot_sigma)))
    return real_sigma, pilot_sigma, extra


def _interaction_prior(
    real: np.ndarray,
    pilot: np.ndarray,
    fields: Sequence[Mapping[str, Any]],
    rgt_knots: np.ndarray,
    *,
    explained_variance: float,
    maximum_rank: int,
) -> dict[str, Any]:
    real_rows = real.reshape(-1, real.shape[-1])
    pilot_rows = pilot.reshape(-1, pilot.shape[-1])
    real_cov = np.cov(real_rows, rowvar=False)
    pilot_cov = np.cov(pilot_rows, rowvar=False)
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
        scores = np.stack([matrix @ mode for matrix in real])
        length = _correlation_length(
            scores,
            [np.asarray(field["lateral_m"]) for field in fields],
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
            "log_sigma": float(np.sqrt(total / rgt_knots.size)),
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
    real_location = _real_location(real_fields)
    pilot_location, stratum_qc = _pilot_location(pilot_fields)
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
        real_rgt, pilot_rgt
    )
    rgt_length = _correlation_length(
        real_rgt, [knots for _ in range(real_rgt.shape[0])]
    )
    rgt_enabled = bool(rgt_extra_sigma > 0.0 and np.isfinite(rgt_length) and rgt_length > 0.0)
    lateral_real_sigma, lateral_pilot_sigma, lateral_extra_sigma = _extra_sigma(
        real_lateral, pilot_lateral
    )
    lateral_length = _correlation_length(
        real_lateral,
        [np.asarray(field["lateral_m"]) for field in real_fields],
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
    total_extra_sigma = float(np.sqrt(
        (rgt_extra_sigma if rgt_enabled else 0.0) ** 2
        + (lateral_extra_sigma if lateral_enabled else 0.0) ** 2
        + float(interaction["log_sigma"]) ** 2
    ))
    max_abs_log_gain = float(max(
        np.finfo(np.float64).eps,
        np.max(np.abs(template)) + 3.0 * total_extra_sigma,
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
