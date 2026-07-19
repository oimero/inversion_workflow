"""Depth adapter for the shared empirical RGT-amplitude calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cup.synthetic.core import build_seismic_input_contract
from cup.synthetic.core.amplitude_calibration import (
    AmplitudeCalibrationSection,
    build_amplitude_pilot_config,
    build_pilot_compatibility_contract,
    load_pilot_sections,
    publish_amplitude_prior,
    rgt_from_horizons,
    validate_amplitude_pilot,
)
from cup.utils.io import (
    repo_relative_path,
    require_contract_fingerprint,
    resolve_relative_path,
    sha256_file,
)


def _bulk_bilinear_section(
    survey: Any, inline: np.ndarray, xline: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = [
        survey.line_geometry.line_to_index(float(il), float(xl))
        for il, xl in zip(inline, xline)
    ]
    neighbors: list[tuple[int, int]] = []
    weights: list[tuple[tuple[int, int, float], ...]] = []
    for i, j in coordinates:
        i0, j0, i1, j1 = int(np.floor(i)), int(np.floor(j)), int(np.ceil(i)), int(np.ceil(j))
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
            combined += weight * np.asarray(traces[(i, j)].values, dtype=np.float64).reshape(-1)
        values[row] = combined
    return axis, values


def depth_amplitude_input_contracts(
    *, calibration_path: Path, forward_inputs: Mapping[str, Any], repo_root: Path
) -> dict[str, dict[str, str]]:
    calibration_summary = json.loads(
        (calibration_path.parent / "run_summary.json").read_text(encoding="utf-8")
    )
    return {
        "calibration": {
            "path": repo_relative_path(calibration_path.parent / "run_summary.json", root=repo_root),
            "contract_fingerprint_sha256": require_contract_fingerprint(
                calibration_summary, label="depth impedance calibration"
            ),
        },
        "rock_physics_analysis": {
            "path": str(forward_inputs.get("_rock_physics_run_summary_path") or ""),
            "contract_fingerprint_sha256": str(
                forward_inputs["_rock_physics_contract_fingerprint_sha256"]
            ),
        },
        "depth_forward_model_inputs": {
            "path": str(forward_inputs.get("_path") or ""),
            "contract_fingerprint_sha256": str(forward_inputs["_contract_fingerprint_sha256"]),
        },
    }


def _horizon_inputs(
    *, workflow: Any, script_cfg: Mapping[str, Any], repo_root: Path
) -> list[dict[str, str]]:
    data_root = resolve_relative_path(workflow.data_root, root=repo_root)
    return [
        {
            "name": str(item["name"]),
            "path": repo_relative_path(
                resolve_relative_path(item["file"], root=data_root), root=repo_root
            ),
            "sha256": sha256_file(resolve_relative_path(item["file"], root=data_root)),
        }
        for item in script_cfg["horizons"]
    ]


def depth_pilot_compatibility(
    *,
    workflow: Any,
    script_cfg: Mapping[str, Any],
    calibration_path: Path,
    forward_inputs: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    data_root = resolve_relative_path(workflow.data_root, root=repo_root)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    seismic_contract = build_seismic_input_contract(
        "depth", operator="depth_ai_vp_highres_forward_antialias"
    )
    seismic_contract["survey"] = {
        "configuration": workflow.seismic.as_dict(),
        "path": repo_relative_path(seismic_path, root=repo_root),
        "size_bytes": seismic_path.stat().st_size,
        "mtime_ns": seismic_path.stat().st_mtime_ns,
    }
    return build_pilot_compatibility_contract(
        sample_domain="depth",
        sample_unit="m",
        axis_basis="tvdss",
        script_cfg=script_cfg,
        input_contracts=depth_amplitude_input_contracts(
            calibration_path=calibration_path,
            forward_inputs=forward_inputs,
            repo_root=repo_root,
        ),
        horizon_inputs=_horizon_inputs(
            workflow=workflow, script_cfg=script_cfg, repo_root=repo_root
        ),
        base_seismic_contract=seismic_contract,
    )


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
    from cup.synthetic.depth.generation import run_depth_generation

    pilot = build_amplitude_pilot_config(script_cfg)
    return run_depth_generation(
        workflow=workflow,
        script_cfg=pilot,
        sources=sources,
        forward_inputs=forward_inputs,
        config_provenance=config_provenance,
        calibration_path=calibration_path,
        repo_root=repo_root,
        output_dir=output_dir,
    )


def _real_sections(
    *, workflow: Any, script_cfg: Mapping[str, Any], repo_root: Path
) -> list[AmplitudeCalibrationSection]:
    from cup.synthetic.depth.generation import build_depth_sections

    geometries, survey = build_depth_sections(
        workflow=workflow, script_cfg=script_cfg, repo_root=repo_root
    )
    result: list[AmplitudeCalibrationSection] = []
    for geometry in geometries:
        axis, seismic = _bulk_bilinear_section(
            survey, geometry.inline_float, geometry.xline_float
        )
        rgt, valid = rgt_from_horizons(axis, geometry.horizon_tvdss_m)
        result.append(AmplitudeCalibrationSection(
            field_id=geometry.section_id,
            section_id=geometry.section_id,
            seismic=seismic,
            rgt=rgt,
            valid_mask=valid,
            lateral_m=geometry.lateral_m,
        ))
    return result


def run_depth_amplitude_calibration(
    *,
    workflow: Any,
    script_cfg: Mapping[str, Any],
    forward_inputs: Mapping[str, Any],
    calibration_path: Path,
    pilot_benchmark_dir: Path,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    controls = script_cfg.get("amplitude_calibration")
    if not isinstance(controls, Mapping):
        raise ValueError("calibrate-amplitude requires amplitude_calibration controls")
    compatibility = depth_pilot_compatibility(
        workflow=workflow,
        script_cfg=script_cfg,
        calibration_path=calibration_path,
        forward_inputs=forward_inputs,
        repo_root=repo_root,
    )
    pilot_summary = validate_amplitude_pilot(
        pilot_benchmark_dir,
        sample_domain="depth",
        expected_compatibility=compatibility,
    )
    pilot_fingerprint = require_contract_fingerprint(
        pilot_summary, label="depth amplitude pilot"
    )
    impedance_summary = json.loads(
        (calibration_path.parent / "run_summary.json").read_text(encoding="utf-8")
    )
    input_contracts = {
        "impedance_calibration": {
            "path": repo_relative_path(calibration_path.parent / "run_summary.json", root=repo_root),
            "contract_fingerprint_sha256": require_contract_fingerprint(
                impedance_summary, label="depth impedance calibration"
            ),
        },
        "pilot_benchmark": {
            "path": repo_relative_path(pilot_benchmark_dir / "run_summary.json", root=repo_root),
            "contract_fingerprint_sha256": pilot_fingerprint,
        },
    }
    data_root = resolve_relative_path(workflow.data_root, root=repo_root)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    return publish_amplitude_prior(
        output_dir=output_dir,
        repo_root=repo_root,
        sample_domain="depth",
        sample_unit="m",
        axis_basis="tvdss",
        ordered_horizons=[str(item["name"]) for item in script_cfg["horizons"]],
        real_sections=_real_sections(
            workflow=workflow, script_cfg=script_cfg, repo_root=repo_root
        ),
        pilot_sections=load_pilot_sections(pilot_benchmark_dir),
        controls=controls,
        input_contracts=input_contracts,
        compatibility=compatibility,
        source_inputs={
            "real_seismic": {
                "path": repo_relative_path(seismic_path, root=repo_root),
                "size_bytes": seismic_path.stat().st_size,
                "mtime_ns": seismic_path.stat().st_mtime_ns,
            },
            "horizons": _horizon_inputs(
                workflow=workflow, script_cfg=script_cfg, repo_root=repo_root
            ),
            "pilot_benchmark": repo_relative_path(pilot_benchmark_dir, root=repo_root),
        },
    )


__all__ = [
    "depth_amplitude_input_contracts",
    "depth_pilot_compatibility",
    "run_depth_amplitude_calibration",
    "run_depth_amplitude_pilot",
]
