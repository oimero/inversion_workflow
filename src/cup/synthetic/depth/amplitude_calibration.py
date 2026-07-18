"""Empirical RGT-amplitude calibration for depth Synthoseis-lite.

The calibration freezes a shared, zero-gauge log-gain template from the
difference between real field sections and a stratified base-only synthetic
pilot.  It is deliberately separate from impedance calibration and final
benchmark generation.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import pandas as pd

from cup.synthetic.depth.generation import build_depth_sections, run_depth_generation
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


def _canonical_sha(value: Any) -> str:
    text = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bulk_bilinear_section(survey: Any, inline: np.ndarray, xline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read a section with one batched neighbor read and bilinear interpolation."""
    coordinates = [
        survey.line_geometry.line_to_index(float(il), float(xl))
        for il, xl in zip(inline, xline)
    ]
    neighbors: list[tuple[int, int]] = []
    weights: list[tuple[tuple[int, int, float], ...]] = []
    for i, j in coordinates:
        i0, j0 = int(np.floor(i)), int(np.floor(j))
        i1, j1 = int(np.ceil(i)), int(np.ceil(j))
        wi, wj = float(i - i0), float(j - j0)
        entries = (
            (i0, j0, (1.0 - wi) * (1.0 - wj)),
            (i0, j1, (1.0 - wi) * wj),
            (i1, j0, wi * (1.0 - wj)),
            (i1, j1, wi * wj),
        )
        weights.append(entries)
        neighbors.extend((ii, jj) for ii, jj, _ in entries)
    traces = survey.read_traces_at_indices(neighbors, domain="depth")
    axis = np.asarray(survey.sample_axis(domain="depth").values, dtype=np.float64)
    values = np.empty((len(weights), axis.size), dtype=np.float64)
    for row, entries in enumerate(weights):
        combined = np.zeros(axis.size, dtype=np.float64)
        for i, j, weight in entries:
            trace = np.asarray(traces[(i, j)].values, dtype=np.float64).reshape(-1)
            if trace.size != axis.size:
                raise ValueError("real section trace axis differs from survey depth axis")
            combined += weight * trace
        values[row] = combined
    return axis, values


def _section_rgt(axis: np.ndarray, horizons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(axis, dtype=np.float64)[None, :]
    surfaces = np.asarray(horizons, dtype=np.float64)
    rgt = np.full((surfaces.shape[0], z.shape[1]), np.nan, dtype=np.float64)
    for zone in range(surfaces.shape[1] - 1):
        top = surfaces[:, zone, None]
        bottom = surfaces[:, zone + 1, None]
        inside = (z >= top) & ((z <= bottom) if zone == surfaces.shape[1] - 2 else (z < bottom))
        fraction = (z - top) / (bottom - top)
        rgt[inside] = (zone + fraction)[inside]
    return rgt, np.isfinite(rgt)


def _coarse_log_rms_curve(
    seismic: np.ndarray,
    rgt: np.ndarray,
    valid: np.ndarray,
    knots: np.ndarray,
    *,
    minimum_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    amplitude = np.asarray(seismic, dtype=np.float64)
    coordinate = np.asarray(rgt, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(amplitude) & np.isfinite(coordinate)
    if not np.any(mask):
        raise ValueError("amplitude curve has no valid target-zone samples")
    half_width = 0.5 * float(np.min(np.diff(knots))) + 1e-12
    global_rms = float(np.sqrt(np.mean(amplitude[mask] ** 2)))
    if not np.isfinite(global_rms) or global_rms <= 0.0:
        raise ValueError("amplitude curve has zero or invalid RMS")
    values = np.full(knots.shape, np.nan, dtype=np.float64)
    counts = np.zeros(knots.shape, dtype=np.int64)
    for index, knot in enumerate(knots):
        selected = mask & (np.abs(coordinate - knot) <= half_width)
        counts[index] = int(np.count_nonzero(selected))
        if counts[index] >= minimum_samples:
            values[index] = np.log(max(float(np.sqrt(np.mean(amplitude[selected] ** 2))), global_rms * 1e-12))
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size < max(3, knots.size // 3):
        raise ValueError("amplitude curve has insufficient supported RGT nodes")
    values = np.interp(np.arange(knots.size), finite, values[finite])
    values -= float(np.median(values))
    return values, counts


def _smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(np.asarray(values, dtype=np.float64), radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _curve_rows(source: str, curve_id: str, section_id: str, knots: np.ndarray, values: np.ndarray, counts: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "source": source,
            "curve_id": curve_id,
            "section_id": section_id,
            "rgt": float(knot),
            "centered_log_rms": float(value),
            "sample_count": int(count),
        }
        for knot, value, count in zip(knots, values, counts)
    ]


def run_depth_amplitude_pilot(
    *,
    workflow: Any,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    forward_inputs: Mapping[str, Any],
    config_provenance: Mapping[str, str],
    calibration_path: Path,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate a formal, stratified base-only pilot with the production science path."""
    pilot = deepcopy(dict(script_cfg))
    attempts = int(pilot["amplitude_calibration"]["pilot_attempts_per_scenario"])
    pilot["global_seed"] = int(pilot["global_seed"]) + 104729
    pilot["generation"] = deepcopy(dict(pilot["generation"]))
    pilot["generation"]["attempts_per_scenario"] = attempts
    pilot["generation"]["acceptance_qc"] = deepcopy(dict(pilot["generation"]["acceptance_qc"]))
    pilot["generation"]["acceptance_qc"]["minimum_attempts_per_scenario"] = attempts
    pilot["seismic_views"] = {"operators": {}, "views": []}
    pilot["benchmark_purpose"] = "seismic_amplitude_calibration_pilot"
    return run_depth_generation(
        workflow=workflow,
        script_cfg=pilot,
        sources=sources,
        forward_inputs=forward_inputs,
        config_provenance=config_provenance,
        calibration_path=calibration_path,
        amplitude_calibration_path=None,
        repo_root=repo_root,
        output_dir=output_dir,
    )


def _validate_pilot(pilot_dir: Path, impedance_path: Path, *, repo_root: Path) -> tuple[dict[str, Any], str]:
    summary_path = pilot_dir / "run_summary.json"
    manifest_path = pilot_dir / "benchmark_manifest.json"
    h5_path = pilot_dir / "synthetic_benchmark.h5"
    for path in (summary_path, manifest_path, h5_path, pilot_dir / "realization_index.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require_science_contract(summary, label="amplitude pilot benchmark")
    if summary.get("status") != "success" or summary.get("sample_domain") != "depth":
        raise ValueError("amplitude pilot must be a successful formal depth benchmark")
    if int(summary.get("seismic_view_count", -1)) != 0:
        raise ValueError("amplitude pilot must contain no materialized seismic views")
    impedance_summary = json.loads((impedance_path.parent / "run_summary.json").read_text(encoding="utf-8"))
    expected = require_contract_fingerprint(impedance_summary, label="impedance calibration")
    actual = str(dict(summary.get("input_contracts") or {}).get("calibration", {}).get("contract_fingerprint_sha256") or "")
    if actual != expected:
        raise ValueError("amplitude pilot and selected impedance calibration differ")
    return summary, require_contract_fingerprint(summary, label="amplitude pilot benchmark")


def run_depth_amplitude_calibration(
    *,
    workflow: Any,
    script_cfg: Mapping[str, Any],
    calibration_path: Path,
    pilot_benchmark_dir: Path,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Fit and publish the real-minus-pilot shared RGT log-amplitude pattern."""
    output_dir.mkdir(parents=True, exist_ok=False)
    pilot_summary, pilot_fingerprint = _validate_pilot(
        pilot_benchmark_dir, calibration_path, repo_root=repo_root
    )
    sections, survey = build_depth_sections(
        workflow=workflow, script_cfg=script_cfg, repo_root=repo_root
    )
    n_zones = len(script_cfg["horizons"]) - 1
    controls = dict(script_cfg["amplitude_calibration"])
    knots = np.linspace(0.0, float(n_zones), int(controls["rgt_node_count"]))
    minimum_samples = int(controls["minimum_samples_per_node"])

    real_curves: list[np.ndarray] = []
    curve_rows: list[dict[str, Any]] = []
    for section in sections:
        axis, seismic = _bulk_bilinear_section(
            survey, section.inline_float, section.xline_float
        )
        rgt, valid = _section_rgt(axis, section.horizon_tvdss_m)
        curve, counts = _coarse_log_rms_curve(
            seismic, rgt, valid, knots, minimum_samples=minimum_samples
        )
        real_curves.append(curve)
        curve_rows.extend(_curve_rows("real", section.section_id, section.section_id, knots, curve, counts))

    index = pd.read_csv(pilot_benchmark_dir / "realization_index.csv")
    required = {"realization_id", "section_id", "status", "valid_mask_dataset"}
    missing = sorted(required - set(index))
    if missing:
        raise ValueError(f"amplitude pilot realization index lacks columns: {missing}")
    index = index[index["status"].eq("ok")]
    pilot_curves: list[np.ndarray] = []
    h5_path = pilot_benchmark_dir / "synthetic_benchmark.h5"
    with h5py.File(h5_path, "r") as h5:
        for row in index.to_dict(orient="records"):
            rid = str(row["realization_id"])
            root = f"/realizations/{rid}"
            seismic = np.asarray(h5[f"{root}/seismic/seismic_observed"][()], dtype=np.float64)
            rgt = np.asarray(h5[f"{root}/truth/rgt_model"][()], dtype=np.float64)
            valid = np.asarray(h5[str(row["valid_mask_dataset"])][()], dtype=bool)
            curve, counts = _coarse_log_rms_curve(
                seismic, rgt, valid, knots, minimum_samples=minimum_samples
            )
            pilot_curves.append(curve)
            curve_rows.extend(_curve_rows("pilot", rid, str(row["section_id"]), knots, curve, counts))
    if not real_curves or not pilot_curves:
        raise ValueError("amplitude calibration requires both real and pilot curves")

    real_matrix = np.vstack(real_curves)
    pilot_matrix = np.vstack(pilot_curves)
    real_location = np.median(real_matrix, axis=0)
    pilot_location = np.median(pilot_matrix, axis=0)
    raw = real_location - pilot_location
    smoothed = _smooth(raw, float(controls["smoothing_sigma_nodes"]))
    shrunk = float(controls["shrinkage"]) * smoothed
    limit = float(controls["max_abs_log_gain"])
    clipped = np.clip(shrunk, -limit, limit)
    template = clipped - float(np.median(clipped))
    maximum = float(np.max(np.abs(template)))
    if maximum > limit:
        template *= limit / maximum
    template_sha = _canonical_sha({"rgt_knots": knots.tolist(), "mean_log_gain_rgt": template.tolist()})

    residual = real_matrix - real_location[None, :]
    section_residual_sigma = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    data_root = resolve_relative_path(workflow.data_root, root=repo_root)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    horizon_inputs = [
        {
            "name": str(item["name"]),
            "path": repo_relative_path(resolve_relative_path(item["file"], root=data_root), root=repo_root),
            "sha256": sha256_file(resolve_relative_path(item["file"], root=data_root)),
        }
        for item in script_cfg["horizons"]
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        **SCIENCE_CONTRACT,
        "sample_domain": "depth",
        "sample_unit": "m",
        "depth_basis": "tvdss",
        "semantics": "real_minus_pilot_synthetic_coarse_rgt_log_rms_signature",
        "gauge": "median_log_gain_zero",
        "endpoint_extension_policy": "calibrated_full_horizon_contract",
        "ordered_horizons": [str(item["name"]) for item in script_cfg["horizons"]],
        "rgt_knots": knots.tolist(),
        "real_centered_log_rms": real_location.tolist(),
        "pilot_centered_log_rms": pilot_location.tolist(),
        "raw_mean_log_gain_rgt": raw.tolist(),
        "smoothed_mean_log_gain_rgt": smoothed.tolist(),
        "mean_log_gain_rgt": template.tolist(),
        "template_sha256": template_sha,
        "random_prior_diagnostics": {
            "real_section_residual_log_sigma": section_residual_sigma,
            "n_real_sections": len(real_curves),
            "n_pilot_realizations": len(pilot_curves),
        },
        "estimator": controls,
        "inputs": {
            "real_seismic": {
                "path": repo_relative_path(seismic_path, root=repo_root),
                "size_bytes": seismic_path.stat().st_size,
                "mtime_ns": seismic_path.stat().st_mtime_ns,
            },
            "horizons": horizon_inputs,
            "pilot_benchmark": repo_relative_path(pilot_benchmark_dir, root=repo_root),
            "pilot_contract_fingerprint_sha256": pilot_fingerprint,
        },
    }
    artifact_path = output_dir / "seismic_amplitude_calibration.json"
    write_json(artifact_path, payload)
    curves_path = output_dir / "amplitude_curves_by_source.csv"
    pd.DataFrame(curve_rows).to_csv(curves_path, index=False)
    summary_frame = pd.DataFrame({
        "rgt": knots,
        "real_centered_log_rms": real_location,
        "pilot_centered_log_rms": pilot_location,
        "raw_mean_log_gain": raw,
        "smoothed_mean_log_gain": smoothed,
        "mean_log_gain": template,
    })
    rgt_summary_path = output_dir / "rgt_amplitude_summary.csv"
    summary_frame.to_csv(rgt_summary_path, index=False)

    impedance_summary = json.loads((calibration_path.parent / "run_summary.json").read_text(encoding="utf-8"))
    input_contracts = {
        "impedance_calibration": {
            "path": repo_relative_path(calibration_path.parent / "run_summary.json", root=repo_root),
            "contract_fingerprint_sha256": require_contract_fingerprint(impedance_summary, label="impedance calibration"),
        },
        "pilot_benchmark": {
            "path": repo_relative_path(pilot_benchmark_dir / "run_summary.json", root=repo_root),
            "contract_fingerprint_sha256": pilot_fingerprint,
        },
    }
    fingerprint = contract_fingerprint_sha256(
        contract_schema_version=SCHEMA_VERSION,
        semantics={
            **SCIENCE_CONTRACT,
            "sample_domain": "depth",
            "depth_basis": "tvdss",
            "semantics": payload["semantics"],
            "gauge": payload["gauge"],
            "template_sha256": template_sha,
        },
        business_config=controls,
        input_contracts=input_contracts,
        primary_artifacts={
            "seismic_amplitude_calibration": artifact_path,
            "amplitude_curves_by_source": curves_path,
            "rgt_amplitude_summary": rgt_summary_path,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        **SCIENCE_CONTRACT,
        "status": "success",
        "sample_domain": "depth",
        "depth_basis": "tvdss",
        "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
        "contract_fingerprint_sha256": fingerprint,
        "input_contracts": input_contracts,
        "pilot_status": pilot_summary["status"],
        "outputs": {
            "seismic_amplitude_calibration": repo_relative_path(artifact_path, root=repo_root),
            "amplitude_curves_by_source": repo_relative_path(curves_path, root=repo_root),
            "rgt_amplitude_summary": repo_relative_path(rgt_summary_path, root=repo_root),
        },
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary


def load_seismic_amplitude_calibration(path: Path, *, repo_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Load one published artifact and fail closed on its sibling run contract."""
    artifact_path = Path(path).resolve()
    summary_path = artifact_path.parent / "run_summary.json"
    if not artifact_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(artifact_path if not artifact_path.is_file() else summary_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for value, label in ((payload, "seismic amplitude calibration"), (summary, "seismic amplitude calibration summary")):
        require_science_contract(value, label=label)
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{label} has unsupported schema")
    if summary.get("status") != "success" or payload.get("sample_domain") != "depth":
        raise ValueError("seismic amplitude calibration is not a successful depth contract")
    recorded = resolve_relative_path(
        str(dict(summary.get("outputs") or {}).get("seismic_amplitude_calibration") or ""),
        root=repo_root,
    )
    if recorded.resolve() != artifact_path:
        raise ValueError("amplitude calibration summary points to a different artifact")
    knots = np.asarray(payload.get("rgt_knots"), dtype=np.float64)
    template = np.asarray(payload.get("mean_log_gain_rgt"), dtype=np.float64)
    if knots.ndim != 1 or template.shape != knots.shape or knots.size < 2 or np.any(np.diff(knots) <= 0.0):
        raise ValueError("amplitude calibration template arrays are invalid")
    if np.any(~np.isfinite(knots)) or np.any(~np.isfinite(template)) or abs(float(np.median(template))) > 1e-10:
        raise ValueError("amplitude calibration violates finite zero-median template contract")
    expected_template_sha = _canonical_sha({"rgt_knots": knots.tolist(), "mean_log_gain_rgt": template.tolist()})
    if str(payload.get("template_sha256") or "") != expected_template_sha:
        raise ValueError("amplitude calibration template SHA-256 is stale")
    return payload, {
        "artifact_sha256": sha256_file(artifact_path),
        "contract_fingerprint_sha256": require_contract_fingerprint(summary, label="seismic amplitude calibration"),
        "path": repo_relative_path(artifact_path, root=repo_root),
    }


def resolve_empirical_seismic_views(
    view_config: Mapping[str, Any],
    *,
    calibration_path: Path | None,
    repo_root: Path,
    ordered_horizons: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Resolve external empirical operator parameters before view fingerprinting."""
    resolved = deepcopy(dict(view_config))
    operators = dict(resolved.get("operators") or {})
    empirical_ids = [
        key for key, value in operators.items()
        if isinstance(value, Mapping) and str(value.get("kind") or "") == "empirical_rgt_gain"
    ]
    if not empirical_ids:
        if calibration_path is not None:
            raise ValueError("seismic amplitude calibration was supplied but no empirical_rgt_gain operator is configured")
        return resolved, None
    if calibration_path is None:
        raise ValueError("empirical_rgt_gain requires --seismic-amplitude-calibration")
    payload, provenance = load_seismic_amplitude_calibration(calibration_path, repo_root=repo_root)
    if ordered_horizons is not None and list(payload.get("ordered_horizons") or []) != list(ordered_horizons):
        raise ValueError("seismic amplitude calibration horizon contract differs from generation")
    if ordered_horizons is not None:
        knots = np.asarray(payload["rgt_knots"], dtype=np.float64)
        expected_end = float(len(ordered_horizons) - 1)
        if not np.isclose(knots[0], 0.0, rtol=0.0, atol=1e-12) or not np.isclose(
            knots[-1], expected_end, rtol=0.0, atol=1e-12
        ):
            raise ValueError("seismic amplitude calibration does not cover the full horizon RGT contract")
    for operator_id in empirical_ids:
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
    return resolved, {**provenance, "schema_version": SCHEMA_VERSION, "template_sha256": payload["template_sha256"]}


__all__ = [
    "SCHEMA_VERSION",
    "load_seismic_amplitude_calibration",
    "resolve_empirical_seismic_views",
    "run_depth_amplitude_calibration",
    "run_depth_amplitude_pilot",
]
