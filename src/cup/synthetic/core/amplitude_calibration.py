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
import shutil
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
import h5py

from cup.synthetic.schemas import (
    SCIENCE_CONTRACT,
    SEISMIC_AMPLITUDE_CALIBRATION_SCHEMA_VERSION,
    require_science_contract,
)
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    repo_relative_path,
    require_contract_fingerprint,
    resolve_relative_path,
    sha256_file,
    write_json,
)


SCHEMA_VERSION = SEISMIC_AMPLITUDE_CALIBRATION_SCHEMA_VERSION


def parse_amplitude_calibration_controls(
    value: Any, *, required: bool
) -> dict[str, Any] | None:
    if value is None:
        if required:
            raise ValueError("empirical_rgt_gain requires amplitude_calibration controls")
        return None
    if not isinstance(value, Mapping):
        raise ValueError("amplitude_calibration must be a mapping")
    allowed = {
        "pilot_attempts_per_scenario", "rgt_node_count",
        "minimum_samples_per_node", "smoothing_sigma_nodes",
        "shrinkage", "max_abs_log_gain",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"amplitude_calibration contains unknown keys: {unknown}")
    missing = sorted(allowed - set(value))
    if missing:
        raise ValueError(f"amplitude_calibration lacks keys: {missing}")
    integers = {
        key: int(value[key])
        for key in ("pilot_attempts_per_scenario", "rgt_node_count", "minimum_samples_per_node")
    }
    if any(isinstance(value[key], bool) or integers[key] != value[key] or integers[key] <= 0 for key in integers):
        raise ValueError("amplitude_calibration count fields must be positive integers")
    if integers["rgt_node_count"] < 5:
        raise ValueError("amplitude_calibration.rgt_node_count must be at least 5")
    floats = {
        key: float(value[key])
        for key in ("smoothing_sigma_nodes", "shrinkage", "max_abs_log_gain")
    }
    if any(not np.isfinite(item) or item <= 0.0 for item in floats.values()):
        raise ValueError("amplitude_calibration numeric controls must be positive and finite")
    if floats["shrinkage"] > 1.0:
        raise ValueError("amplitude_calibration.shrinkage must be at most 1")
    return {**integers, **floats}


@dataclass(frozen=True)
class AmplitudeCalibrationSection:
    curve_id: str
    section_id: str
    seismic: np.ndarray
    rgt: np.ndarray
    valid_mask: np.ndarray
    scenario_id: str = "real"

    def __post_init__(self) -> None:
        seismic = np.asarray(self.seismic, dtype=np.float64)
        rgt = np.asarray(self.rgt, dtype=np.float64)
        valid = np.asarray(self.valid_mask, dtype=bool)
        if seismic.ndim != 2 or rgt.shape != seismic.shape or valid.shape != seismic.shape:
            raise ValueError("amplitude calibration section arrays must be aligned 2D fields")
        if np.any(valid & (~np.isfinite(seismic) | ~np.isfinite(rgt))):
            raise ValueError("amplitude calibration section has non-finite public samples")
        object.__setattr__(self, "seismic", seismic)
        object.__setattr__(self, "rgt", rgt)
        object.__setattr__(self, "valid_mask", valid)


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
    pilot["benchmark_purpose"] = "seismic_amplitude_calibration_pilot"
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
                curve_id=rid,
                section_id=str(row["section_id"]),
                scenario_id=str(row["scenario_id"]),
                seismic=np.asarray(h5[f"{root}/seismic/seismic_observed"][()]),
                rgt=np.asarray(h5[f"{root}/truth/rgt_model"][()]),
                valid_mask=np.asarray(h5[str(row["valid_mask_dataset"])][()], dtype=bool),
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
    if summary.get("status") != "success" or summary.get("sample_domain") != sample_domain:
        raise ValueError(f"amplitude pilot must be a successful formal {sample_domain} benchmark")
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


def _coarse_log_rms_curve(
    section: AmplitudeCalibrationSection,
    knots: np.ndarray,
    *,
    minimum_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = section.valid_mask
    if not np.any(mask):
        raise ValueError(f"amplitude curve {section.curve_id!r} has no valid samples")
    half_width = 0.5 * float(np.min(np.diff(knots))) + 1e-12
    global_rms = float(np.sqrt(np.mean(section.seismic[mask] ** 2)))
    if not np.isfinite(global_rms) or global_rms <= 0.0:
        raise ValueError(f"amplitude curve {section.curve_id!r} has invalid RMS")
    values = np.full(knots.shape, np.nan, dtype=np.float64)
    counts = np.zeros(knots.shape, dtype=np.int64)
    for index, knot in enumerate(knots):
        selected = mask & (np.abs(section.rgt - knot) <= half_width)
        counts[index] = int(np.count_nonzero(selected))
        if counts[index] >= minimum_samples:
            rms = float(np.sqrt(np.mean(section.seismic[selected] ** 2)))
            values[index] = np.log(max(rms, global_rms * 1e-12))
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size < max(3, knots.size // 3):
        raise ValueError(f"amplitude curve {section.curve_id!r} has insufficient RGT support")
    values = np.interp(np.arange(knots.size), finite, values[finite])
    values -= float(np.median(values))
    return values, counts


def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(np.pad(values, radius, mode="edge"), kernel, mode="valid")


def _curves(
    sections: Sequence[AmplitudeCalibrationSection],
    knots: np.ndarray,
    minimum_samples: int,
    source: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    curves: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for section in sections:
        curve, counts = _coarse_log_rms_curve(
            section, knots, minimum_samples=minimum_samples
        )
        curves[section.curve_id] = curve
        rows.extend(
            {
                "source": source,
                "curve_id": section.curve_id,
                "section_id": section.section_id,
                "scenario_id": section.scenario_id,
                "rgt": float(knot),
                "centered_log_rms": float(value),
                "sample_count": int(count),
            }
            for knot, value, count in zip(knots, curve, counts)
        )
    return curves, rows


def _stratified_pilot_location(
    sections: Sequence[AmplitudeCalibrationSection],
    curves: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Aggregate realization -> section/scenario -> section -> survey, all equally."""
    by_stratum: dict[tuple[str, str], list[np.ndarray]] = {}
    for section in sections:
        by_stratum.setdefault((section.section_id, section.scenario_id), []).append(
            curves[section.curve_id]
        )
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
    section_locations: list[np.ndarray] = []
    for section_id, values in sorted(by_section.items()):
        section_locations.append(np.median(np.vstack(values), axis=0))
        count = len(values)
        for row in qc:
            if row["section_id"] == section_id:
                row["stratum_weight_within_section"] = 1.0 / count
                row["section_weight"] = 1.0 / len(by_section)
    return np.median(np.vstack(section_locations), axis=0), qc


def fit_amplitude_template(
    *,
    real_sections: Sequence[AmplitudeCalibrationSection],
    pilot_sections: Sequence[AmplitudeCalibrationSection],
    n_zones: int,
    controls: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not real_sections or not pilot_sections:
        raise ValueError("amplitude calibration requires real and pilot sections")
    knots = np.linspace(0.0, float(n_zones), int(controls["rgt_node_count"]))
    minimum = int(controls["minimum_samples_per_node"])
    real_curves, real_rows = _curves(real_sections, knots, minimum, "real")
    pilot_curves, pilot_rows = _curves(pilot_sections, knots, minimum, "pilot")
    real_by_section: dict[str, list[np.ndarray]] = {}
    for section in real_sections:
        real_by_section.setdefault(section.section_id, []).append(real_curves[section.curve_id])
    real_location = np.median(
        np.vstack([np.median(np.vstack(values), axis=0) for _, values in sorted(real_by_section.items())]),
        axis=0,
    )
    pilot_location, stratum_qc = _stratified_pilot_location(pilot_sections, pilot_curves)
    for row in stratum_qc:
        row["planned_realizations"] = int(controls["pilot_attempts_per_scenario"])
        row["acceptance_fraction"] = (
            float(row["accepted_realizations"]) / float(row["planned_realizations"])
        )
    raw = real_location - pilot_location
    smoothed = _smooth(raw, float(controls["smoothing_sigma_nodes"]))
    shrunk = float(controls["shrinkage"]) * smoothed
    limit = float(controls["max_abs_log_gain"])
    template = np.clip(shrunk, -limit, limit)
    template -= float(np.median(template))
    maximum = float(np.max(np.abs(template)))
    if maximum > limit:
        template *= limit / maximum
    real_matrix = np.vstack(list(real_curves.values()))
    residual = real_matrix - real_location[None, :]
    result = {
        "rgt_knots": knots.tolist(),
        "real_centered_log_rms": real_location.tolist(),
        "pilot_centered_log_rms": pilot_location.tolist(),
        "raw_mean_log_gain_rgt": raw.tolist(),
        "smoothed_mean_log_gain_rgt": smoothed.tolist(),
        "mean_log_gain_rgt": template.tolist(),
        "template_sha256": canonical_sha256({
            "rgt_knots": knots.tolist(), "mean_log_gain_rgt": template.tolist()
        }),
        "random_prior_diagnostics": {
            "real_section_residual_log_sigma": float(
                1.4826 * np.median(np.abs(residual - np.median(residual)))
            ),
            "n_real_sections": len(real_by_section),
            "n_pilot_realizations": len(pilot_sections),
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
    return result, pd.DataFrame(real_rows + pilot_rows), summary, pd.DataFrame(stratum_qc)


def publish_amplitude_calibration(
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
    """Fit and atomically publish the shared amplitude-calibration contract."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    staging = output_dir.with_name(f".{output_dir.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        fit, curves, summary_frame, strata = fit_amplitude_template(
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
            "semantics": "real_minus_pilot_synthetic_coarse_rgt_log_rms_signature",
            "gauge": "median_log_gain_zero",
            "endpoint_extension_policy": "calibrated_full_horizon_contract",
            "ordered_horizons": list(ordered_horizons),
            **fit,
            "estimator": dict(controls),
            "pilot_compatibility": dict(compatibility),
            "inputs": dict(source_inputs),
        }
        artifact = staging / "seismic_amplitude_calibration.json"
        curves_path = staging / "amplitude_curves_by_source.csv"
        summary_path = staging / "rgt_amplitude_summary.csv"
        strata_path = staging / "pilot_stratum_weights.csv"
        write_json(artifact, payload)
        curves.to_csv(curves_path, index=False)
        summary_frame.to_csv(summary_path, index=False)
        strata.to_csv(strata_path, index=False)
        fingerprint = contract_fingerprint_sha256(
            contract_schema_version=SCHEMA_VERSION,
            semantics={
                **SCIENCE_CONTRACT,
                "sample_domain": sample_domain,
                "sample_unit": sample_unit,
                "axis_basis": axis_basis,
                "semantics": payload["semantics"],
                "template_sha256": fit["template_sha256"],
                "pilot_compatibility_sha256": compatibility["sha256"],
            },
            business_config=dict(controls),
            input_contracts=input_contracts,
            primary_artifacts={
                "seismic_amplitude_calibration": artifact,
                "amplitude_curves_by_source": curves_path,
                "rgt_amplitude_summary": summary_path,
                "pilot_stratum_weights": strata_path,
            },
        )
        published_artifact = output_dir / artifact.name
        run_summary = {
            "schema_version": SCHEMA_VERSION,
            **SCIENCE_CONTRACT,
            "status": "success",
            "sample_domain": sample_domain,
            "sample_unit": sample_unit,
            "axis_basis": axis_basis,
            "depth_basis": "tvdss" if sample_domain == "depth" else None,
            "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
            "contract_fingerprint_sha256": fingerprint,
            "input_contracts": dict(input_contracts),
            "pilot_compatibility": dict(compatibility),
            "outputs": {
                "seismic_amplitude_calibration": repo_relative_path(published_artifact, root=repo_root),
                "amplitude_curves_by_source": repo_relative_path(output_dir / curves_path.name, root=repo_root),
                "rgt_amplitude_summary": repo_relative_path(output_dir / summary_path.name, root=repo_root),
                "pilot_stratum_weights": repo_relative_path(output_dir / strata_path.name, root=repo_root),
            },
        }
        write_json(staging / "run_summary.json", run_summary)
        _validate_calibration_payload(payload, expected_domain=sample_domain)
        staging.replace(output_dir)
        return run_summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_calibration_payload(payload: Mapping[str, Any], *, expected_domain: str | None = None) -> None:
    require_science_contract(payload, label="seismic amplitude calibration")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("seismic amplitude calibration has unsupported schema")
    domain = str(payload.get("sample_domain") or "")
    if domain not in {"time", "depth"} or (expected_domain and domain != expected_domain):
        raise ValueError("seismic amplitude calibration sample domain differs")
    expected = {"time": ("s", "twt"), "depth": ("m", "tvdss")}[domain]
    if (str(payload.get("sample_unit")), str(payload.get("axis_basis"))) != expected:
        raise ValueError("seismic amplitude calibration axis contract is invalid")
    knots = np.asarray(payload.get("rgt_knots"), dtype=np.float64)
    template = np.asarray(payload.get("mean_log_gain_rgt"), dtype=np.float64)
    if knots.ndim != 1 or template.shape != knots.shape or knots.size < 2:
        raise ValueError("amplitude calibration template arrays are invalid")
    if np.any(~np.isfinite(knots)) or np.any(~np.isfinite(template)) or np.any(np.diff(knots) <= 0.0):
        raise ValueError("amplitude calibration template is not finite and monotonic")
    if abs(float(np.median(template))) > 1e-10:
        raise ValueError("amplitude calibration violates its zero-median gauge")
    expected_sha = canonical_sha256({
        "rgt_knots": knots.tolist(), "mean_log_gain_rgt": template.tolist()
    })
    if str(payload.get("template_sha256") or "") != expected_sha:
        raise ValueError("amplitude calibration template SHA-256 is stale")


def load_seismic_amplitude_calibration(
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
    require_science_contract(summary, label="seismic amplitude calibration summary")
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("status") != "success":
        raise ValueError("seismic amplitude calibration summary is not successful")
    recorded = resolve_relative_path(
        str(dict(summary.get("outputs") or {}).get("seismic_amplitude_calibration") or ""),
        root=repo_root,
    )
    if recorded.resolve() != artifact:
        raise ValueError("amplitude calibration summary points to a different artifact")
    return payload, {
        "artifact_sha256": sha256_file(artifact),
        "contract_fingerprint_sha256": require_contract_fingerprint(
            summary, label="seismic amplitude calibration"
        ),
        "path": repo_relative_path(artifact, root=repo_root),
    }


def resolve_empirical_seismic_views(
    view_config: Mapping[str, Any],
    *,
    calibration_path: Path | None,
    repo_root: Path,
    sample_domain: str,
    ordered_horizons: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    resolved = json.loads(json.dumps(dict(view_config)))
    operators = dict(resolved.get("operators") or {})
    empirical = [
        key for key, value in operators.items()
        if isinstance(value, Mapping) and value.get("kind") == "empirical_rgt_gain"
    ]
    if not empirical:
        if calibration_path is not None:
            raise ValueError("amplitude calibration supplied without empirical_rgt_gain")
        return resolved, None
    if calibration_path is None:
        raise ValueError("empirical_rgt_gain requires --seismic-amplitude-calibration")
    payload, provenance = load_seismic_amplitude_calibration(
        calibration_path, repo_root=repo_root, expected_domain=sample_domain
    )
    if list(payload.get("ordered_horizons") or []) != list(ordered_horizons):
        raise ValueError("amplitude calibration horizon contract differs from generation")
    knots = np.asarray(payload["rgt_knots"], dtype=np.float64)
    if not np.isclose(knots[0], 0.0, atol=1e-12) or not np.isclose(
        knots[-1], len(ordered_horizons) - 1, atol=1e-12
    ):
        raise ValueError("amplitude calibration does not cover the full RGT contract")
    for operator_id in empirical:
        operator = dict(operators[operator_id])
        operator.update({
            "calibration_schema_version": SCHEMA_VERSION,
            "calibration_artifact_sha256": provenance["artifact_sha256"],
            "calibration_contract_fingerprint_sha256": provenance["contract_fingerprint_sha256"],
            "template_sha256": payload["template_sha256"],
            "rgt_knots": payload["rgt_knots"],
            "mean_log_gain_rgt": payload["mean_log_gain_rgt"],
        })
        operators[operator_id] = operator
    resolved["operators"] = operators
    return resolved, {
        **provenance,
        "schema_version": SCHEMA_VERSION,
        "template_sha256": payload["template_sha256"],
    }


__all__ = [
    "AmplitudeCalibrationSection",
    "SCHEMA_VERSION",
    "build_pilot_compatibility_contract",
    "build_amplitude_pilot_config",
    "fit_amplitude_template",
    "load_seismic_amplitude_calibration",
    "load_pilot_sections",
    "parse_amplitude_calibration_controls",
    "publish_amplitude_calibration",
    "resolve_empirical_seismic_views",
    "rgt_from_horizons",
    "validate_amplitude_pilot",
]
