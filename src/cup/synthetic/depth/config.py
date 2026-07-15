"""Strict configuration and source contracts for Synthoseis-lite v4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cup.config.sources import resolve_source_run
from cup.config.workflow import WorkflowConfig
from cup.synthetic.core.calibration import SCHEMA_VERSION as CALIBRATION_SCHEMA
from cup.synthetic.schemas import (
    BENCHMARK_SCHEMA_VERSION,
    DEPTH_FORWARD_MODEL_INPUTS_RUN_SCHEMA_VERSION,
    FORWARD_MODEL_INPUTS_SCHEMA_VERSION,
    ROCK_PHYSICS_ANALYSIS_SCHEMA_VERSION,
)
from cup.synthetic.core.config import parse_object_core_controls
from cup.utils.io import (
    load_yaml_config,
    require_contract_fingerprint,
    resolve_artifact_path,
    resolve_relative_path,
)


SCHEMA_VERSION = BENCHMARK_SCHEMA_VERSION
GENERATOR_FAMILY = "object_coefficients_v2"

_WORKFLOW_KEYS = {
    "data_root",
    "output_root",
    "assets",
    "seismic",
    "target_interval",
    "well_curves",
    "spatial_debias",
}


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], *, path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown keys: {unknown}.")


def _required_text(value: Mapping[str, Any], key: str, *, path: str) -> str:
    text = "" if value.get(key) is None else str(value[key]).strip()
    if not text:
        raise ValueError(f"{path}.{key} must be a non-empty string.")
    return text


def _positive_float(value: Any, *, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a positive number.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path} must be a positive number.")
    return result


def _positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a positive integer.") from exc
    if result <= 0 or float(result) != float(value):
        raise ValueError(f"{path} must be a positive integer.")
    return result


def load_composed_config(
    experiment_file: str | Path,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], WorkflowConfig, dict[str, str]]:
    """Load an experiment overlay without allowing workflow-asset overrides."""
    experiment_path = resolve_relative_path(experiment_file, root=repo_root)
    experiment = _mapping(load_yaml_config(experiment_path), path=str(experiment_path))
    _reject_unknown(experiment, {"workflow_config", "synthoseis_lite"}, path="experiment")
    common_path = resolve_relative_path(
        _required_text(experiment, "workflow_config", path="experiment"),
        root=repo_root,
    )
    common = _mapping(load_yaml_config(common_path), path=str(common_path))
    duplicated = sorted(_WORKFLOW_KEYS.intersection(experiment))
    if duplicated:
        raise ValueError(f"Synthoseis experiment must not override workflow fields: {duplicated}.")
    composed = dict(common)
    composed["synthoseis_lite"] = _mapping(
        experiment.get("synthoseis_lite"), path="synthoseis_lite"
    )
    workflow = WorkflowConfig.from_mapping(composed)
    return composed, workflow, {
        "experiment_file": str(experiment_path),
        "workflow_config": str(common_path),
    }


def parse_depth_config(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(config.get("synthoseis_lite"), path="synthoseis_lite")
    allowed_root = {
        "sample_domain", "benchmark_schema", "global_seed", "source_runs", "sampling", "geometry", "sections",
        "calibration", "impedance_attribute_generator", "generation", "splits",
        "seismic_input", "seismic_forward", "lfm", "seismic_mismatch", "canonical", "figures",
    }
    if "probe_selection" in root:
        raise ValueError(
            "synthoseis_lite.probe_selection is not part of the v4 canonical benchmark."
        )
    _reject_unknown(root, allowed_root, path="synthoseis_lite")
    if str(root.get("sample_domain") or "").casefold() != "depth" or str(root.get("benchmark_schema") or "") != SCHEMA_VERSION:
        raise ValueError(
            "Depth Synthoseis-lite v4 requires "
            f"synthoseis_lite.sample_domain='depth' and benchmark_schema={SCHEMA_VERSION!r}."
        )
    seismic_input = _mapping(
        root.get("seismic_input"), path="synthoseis_lite.seismic_input"
    )
    _reject_unknown(
        seismic_input, {"policy"}, path="synthoseis_lite.seismic_input"
    )
    if set(seismic_input) != {"policy"} or str(seismic_input.get("policy")) != "observed_highres_forward":
        raise ValueError(
            "Depth Synthoseis v4 requires "
            "seismic_input.policy='observed_highres_forward'."
        )
    seismic_forward = _mapping(
        root.get("seismic_forward"), path="synthoseis_lite.seismic_forward"
    )
    _reject_unknown(
        seismic_forward,
        {"backend", "dtype"},
        path="synthoseis_lite.seismic_forward",
    )
    backend = str(seismic_forward.get("backend") or "").strip().casefold()
    if backend not in {"auto", "numpy", "torch_cuda"}:
        raise ValueError(
            "synthoseis_lite.seismic_forward.backend must be auto, numpy, or torch_cuda."
        )
    dtype = str(seismic_forward.get("dtype") or "").strip().casefold()
    if dtype != "float64":
        raise ValueError("Depth Synthoseis forward dtype is fixed to float64.")
    seismic = _mapping(config.get("seismic"), path="seismic")
    if str(seismic.get("domain", "")).casefold() != "depth":
        raise ValueError("Depth Synthoseis v4 requires seismic.domain='depth'.")
    if str(seismic.get("depth_basis", "")).casefold() != "tvdss":
        raise ValueError("Depth Synthoseis v4 requires seismic.depth_basis='tvdss'.")

    target = _mapping(config.get("target_interval"), path="target_interval")
    raw_horizons = target.get("horizons")
    if not isinstance(raw_horizons, list) or len(raw_horizons) < 2:
        raise ValueError("target_interval.horizons must contain at least two entries.")
    horizons: list[dict[str, str]] = []
    for index, raw in enumerate(raw_horizons):
        item = _mapping(raw, path=f"target_interval.horizons[{index}]")
        _reject_unknown(
            item,
            {"name", "well_top", "file"},
            path=f"target_interval.horizons[{index}]",
        )
        horizons.append({
            "name": _required_text(item, "name", path=f"target_interval.horizons[{index}]"),
            "well_top": _required_text(item, "well_top", path=f"target_interval.horizons[{index}]"),
            "file": _required_text(item, "file", path=f"target_interval.horizons[{index}]"),
        })
    if len({item["name"] for item in horizons}) != len(horizons):
        raise ValueError("target_interval horizon names must be unique.")

    sources = _mapping(root.get("source_runs") or {}, path="synthoseis_lite.source_runs")
    _reject_unknown(
        sources,
        {
            "well_inventory_dir",
            "rock_physics_analysis_dir",
            "depth_forward_model_inputs_dir",
            "wavelet_batch_synthetic_depth_dir",
        },
        path="synthoseis_lite.source_runs",
    )

    sampling = _mapping(root.get("sampling"), path="synthoseis_lite.sampling")
    _reject_unknown(sampling, {"expected_model_dz_m", "vertical_oversampling_factor", "antialias"}, path="synthoseis_lite.sampling")
    antialias = _mapping(sampling.get("antialias"), path="synthoseis_lite.sampling.antialias")
    _reject_unknown(antialias, {"family", "taps_per_factor", "cutoff_output_nyquist_fraction", "kaiser_beta"}, path="synthoseis_lite.sampling.antialias")
    family = _required_text(antialias, "family", path="synthoseis_lite.sampling.antialias")
    if family != "zero_phase_fir_kaiser":
        raise ValueError("sampling.antialias.family must be 'zero_phase_fir_kaiser'.")
    cutoff = float(antialias.get("cutoff_output_nyquist_fraction"))
    if not 0.0 < cutoff <= 1.0:
        raise ValueError("sampling.antialias.cutoff_output_nyquist_fraction must be in (0, 1].")
    model_dz = _positive_float(sampling.get("expected_model_dz_m"), path="sampling.expected_model_dz_m")
    oversampling = _positive_int(sampling.get("vertical_oversampling_factor"), path="sampling.vertical_oversampling_factor")
    if model_dz != 5.0 or oversampling != 8:
        raise ValueError("Depth v4 is frozen to 5 m model sampling and 8x oversampling.")

    geometry = _mapping(root.get("geometry"), path="synthoseis_lite.geometry")
    _reject_unknown(geometry, {"lateral_sample_interval_m", "field_conditioned"}, path="synthoseis_lite.geometry")
    field = _mapping(geometry.get("field_conditioned"), path="synthoseis_lite.geometry.field_conditioned")
    _reject_unknown(field, {"enabled", "target_zone"}, path="synthoseis_lite.geometry.field_conditioned")
    if field.get("enabled") is not True:
        raise ValueError("Depth v4 requires geometry.field_conditioned.enabled=true.")
    target_zone = _mapping(field.get("target_zone"), path="synthoseis_lite.geometry.field_conditioned.target_zone")
    if target_zone != {"mode": "filled_target_zone"}:
        raise ValueError("Depth v4 target_zone must be exactly mode=filled_target_zone.")
    lateral_interval = _positive_float(geometry.get("lateral_sample_interval_m"), path="geometry.lateral_sample_interval_m")
    if lateral_interval != 25.0:
        raise ValueError("Depth v4 uses a frozen 25 m lateral sample interval.")

    raw_sections = root.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ValueError("synthoseis_lite.sections must be non-empty.")
    sections: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_sections):
        item = _mapping(raw, path=f"synthoseis_lite.sections[{index}]")
        _reject_unknown(item, {"section_id", "path"}, path=f"synthoseis_lite.sections[{index}]")
        points = item.get("path")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"synthoseis_lite.sections[{index}].path needs at least two points.")
        parsed_points = []
        for point_index, raw_point in enumerate(points):
            point = _mapping(raw_point, path=f"sections[{index}].path[{point_index}]")
            _reject_unknown(point, {"inline", "xline"}, path=f"sections[{index}].path[{point_index}]")
            parsed_points.append({"inline": float(point["inline"]), "xline": float(point["xline"])})
        sections.append({"section_id": _required_text(item, "section_id", path=f"sections[{index}]"), "path": parsed_points})

    calibration = _mapping(root.get("calibration"), path="synthoseis_lite.calibration")
    _reject_unknown(calibration, {"background_estimator", "huber_delta_sigma", "minimum_valid_cells_per_well_zone"}, path="synthoseis_lite.calibration")
    if calibration.get("background_estimator") != "per_well_zone_huber":
        raise ValueError("calibration.background_estimator must be per_well_zone_huber.")

    impedance = _mapping(root.get("impedance_attribute_generator"), path="synthoseis_lite.impedance_attribute_generator")
    _reject_unknown(
        impedance,
        {
            "family",
            "state_threshold_sigma",
            "lateral",
            "qc",
            "robust_scale",
            "duration_modes",
        },
        path="synthoseis_lite.impedance_attribute_generator",
    )
    if impedance.get("family") != GENERATOR_FAMILY:
        raise ValueError(f"impedance_attribute_generator.family must be {GENERATOR_FAMILY}.")
    object_core_controls = parse_object_core_controls(impedance)
    duration_models = _mapping(impedance.get("duration_modes"), path="impedance_attribute_generator.duration_modes")
    if set(duration_models) != {"standard"}:
        raise ValueError("Depth v4 supports only the standard duration mode.")
    standard_mode = _mapping(duration_models["standard"], path="duration_modes.standard")
    _reject_unknown(standard_mode, {"minimum_highres_cells"}, path="duration_modes.standard")

    generation = _mapping(root.get("generation"), path="synthoseis_lite.generation")
    _reject_unknown(generation, {"attempts_per_scenario", "duration_modes", "geometry_families", "geometry_directions", "acceptance_qc"}, path="synthoseis_lite.generation")
    if list(generation.get("duration_modes") or []) != ["standard"]:
        raise ValueError("generation.duration_modes must be [standard].")
    geometry_families = [str(value) for value in generation.get("geometry_families") or []]
    if not geometry_families or not set(geometry_families) <= {"none", "wedge", "pinchout"}:
        raise ValueError("generation.geometry_families contains unsupported values.")
    directions = [str(value) for value in generation.get("geometry_directions") or []]
    if not directions or not set(directions) <= {"left_to_right", "right_to_left"}:
        raise ValueError("generation.geometry_directions contains unsupported values.")
    acceptance = _mapping(generation.get("acceptance_qc"), path="generation.acceptance_qc")
    _reject_unknown(acceptance, {"minimum_attempts_per_scenario", "warning_fraction", "failure_fraction", "enforcement"}, path="generation.acceptance_qc")
    warning = float(acceptance.get("warning_fraction"))
    failure = float(acceptance.get("failure_fraction"))
    if not 0.0 <= failure < warning <= 1.0:
        raise ValueError("generation acceptance fractions must satisfy 0 <= failure < warning <= 1.")
    enforcement = str(acceptance.get("enforcement", "warn"))
    if enforcement not in {"warn", "fail_fast"}:
        raise ValueError("generation.acceptance_qc.enforcement must be warn or fail_fast.")

    splits = _mapping(root.get("splits"), path="synthoseis_lite.splits")
    _reject_unknown(splits, {"assignment_unit", "held_out_geometry_family"}, path="synthoseis_lite.splits")
    if splits.get("assignment_unit") != "parent_realization":
        raise ValueError("splits.assignment_unit must be parent_realization.")
    held_out = str(splits.get("held_out_geometry_family") or "")
    if held_out not in geometry_families:
        raise ValueError("held_out_geometry_family must be one configured geometry family.")

    lfm = _mapping(root.get("lfm"), path="synthoseis_lite.lfm")
    _reject_unknown(lfm, {"enabled", "ideal", "controlled_degraded"}, path="synthoseis_lite.lfm")
    if lfm.get("enabled") is not True:
        raise ValueError("Depth v4 requires lfm.enabled=true.")
    parsed_lfm: dict[str, Any] = {"enabled": True}
    for name, expected in (("ideal", 400.0), ("controlled_degraded", 400.0)):
        item = _mapping(lfm.get(name), path=f"lfm.{name}")
        allowed_lfm = {"minimum_wavelength_m", "numtaps", "kaiser_beta"}
        if name == "controlled_degraded":
            allowed_lfm |= {
                "constant_bias_sigma_log_ai",
                "linear_vertical_trend_sigma_log_ai",
                "zonewise_bias_sigma_log_ai",
                "lateral_smooth_bias_sigma_log_ai",
                "lateral_correlation_fraction",
                "amplitude_scale_sigma",
                "local_missing_control_bias",
                "over_smoothing",
            }
        _reject_unknown(item, allowed_lfm, path=f"lfm.{name}")
        wavelength = _positive_float(item.get("minimum_wavelength_m"), path=f"lfm.{name}.minimum_wavelength_m")
        if wavelength != expected:
            raise ValueError(f"lfm.{name}.minimum_wavelength_m must be {expected:g} in v4.")
        parsed_lfm[name] = {"minimum_wavelength_m": wavelength, "numtaps": _positive_int(item.get("numtaps"), path=f"lfm.{name}.numtaps"), "kaiser_beta": float(item.get("kaiser_beta"))}
        if name == "controlled_degraded":
            for key in ("constant_bias_sigma_log_ai", "linear_vertical_trend_sigma_log_ai", "zonewise_bias_sigma_log_ai", "lateral_smooth_bias_sigma_log_ai", "lateral_correlation_fraction", "amplitude_scale_sigma"):
                parsed_lfm[name][key] = _positive_float(item.get(key), path=f"lfm.{name}.{key}")
            local = _mapping(item.get("local_missing_control_bias"), path="lfm.controlled_degraded.local_missing_control_bias")
            _reject_unknown(local, {"enabled", "max_abs_log_ai", "lateral_width_fraction", "vertical_width_fraction"}, path="lfm.controlled_degraded.local_missing_control_bias")
            if not isinstance(local.get("enabled"), bool):
                raise ValueError("local_missing_control_bias.enabled must be boolean.")
            parsed_lfm[name]["local_missing_control_bias"] = {"enabled": local["enabled"], "max_abs_log_ai": _positive_float(local.get("max_abs_log_ai"), path="local_missing_control_bias.max_abs_log_ai"), "lateral_width_fraction": _positive_float(local.get("lateral_width_fraction"), path="local_missing_control_bias.lateral_width_fraction"), "vertical_width_fraction": _positive_float(local.get("vertical_width_fraction"), path="local_missing_control_bias.vertical_width_fraction")}
            smoothing = dict(item.get("over_smoothing") or {"enabled": False})
            _reject_unknown(
                smoothing,
                {"enabled", "minimum_wavelength_m", "numtaps", "kaiser_beta", "blend"},
                path="lfm.controlled_degraded.over_smoothing",
            )
            if not isinstance(smoothing.get("enabled"), bool):
                raise ValueError("lfm.controlled_degraded.over_smoothing.enabled must be boolean.")
            parsed_lfm[name]["over_smoothing"] = {"enabled": bool(smoothing["enabled"])}
            if bool(smoothing["enabled"]):
                blend = float(smoothing.get("blend"))
                if not 0.0 <= blend <= 1.0:
                    raise ValueError("lfm.controlled_degraded.over_smoothing.blend must be within [0, 1].")
                parsed_lfm[name]["over_smoothing"].update({
                    "minimum_wavelength_m": _positive_float(
                        smoothing.get("minimum_wavelength_m"),
                        path="lfm.controlled_degraded.over_smoothing.minimum_wavelength_m",
                    ),
                    "numtaps": _positive_int(
                        smoothing.get("numtaps"),
                        path="lfm.controlled_degraded.over_smoothing.numtaps",
                    ),
                    "kaiser_beta": float(smoothing.get("kaiser_beta")),
                    "blend": blend,
                })

    for disabled_name in ("canonical",):
        disabled = _mapping(root.get(disabled_name), path=f"synthoseis_lite.{disabled_name}")
        if disabled != {"enabled": False}:
            raise ValueError(f"Depth v4 requires {disabled_name}.enabled=false with no extra fields.")

    mismatch = _mapping(root.get("seismic_mismatch"), path="synthoseis_lite.seismic_mismatch")
    _reject_unknown(mismatch, {"enabled", "wavelet", "depth_static", "noise", "gain", "combined"}, path="synthoseis_lite.seismic_mismatch")
    if mismatch.get("enabled") is not True:
        raise ValueError("Depth v4 requires seismic_mismatch.enabled=true.")
    wavelet_mismatch = _mapping(mismatch.get("wavelet"), path="seismic_mismatch.wavelet")
    _reject_unknown(wavelet_mismatch, {"phase_rotation_degrees", "time_shift_s"}, path="seismic_mismatch.wavelet")
    depth_static = _mapping(mismatch.get("depth_static"), path="seismic_mismatch.depth_static")
    _reject_unknown(depth_static, {"shift_m"}, path="seismic_mismatch.depth_static")
    noise = _mapping(mismatch.get("noise"), path="seismic_mismatch.noise")
    _reject_unknown(noise, {"white_noise_rms_fraction", "colored_noise_rms_fraction", "colored_vertical_correlation_m"}, path="seismic_mismatch.noise")
    gain = _mapping(mismatch.get("gain"), path="seismic_mismatch.gain")
    _reject_unknown(
        gain,
        {
            "global_log_sigma",
            "tracewise_log_sigma",
            "vertical_lateral_log_sigma",
            "lateral_correlation_fraction",
            "vertical_correlation_fraction",
        },
        path="seismic_mismatch.gain",
    )
    combined = dict(mismatch.get("combined") or {"enabled": False})
    _reject_unknown(
        combined,
        {
            "enabled",
            "phase_rotation_degrees",
            "time_shift_s",
            "depth_static_m",
            "gain_log_sigma",
            "noise_rms_fraction",
        },
        path="seismic_mismatch.combined",
    )
    if not isinstance(combined.get("enabled"), bool):
        raise ValueError("seismic_mismatch.combined.enabled must be boolean.")
    parsed_mismatch = {
        "enabled": True,
        "wavelet": {
            "phase_rotation_degrees": [float(value) for value in wavelet_mismatch.get("phase_rotation_degrees", [])],
            "time_shift_s": [float(value) for value in wavelet_mismatch.get("time_shift_s", [])],
        },
        "depth_static": {"shift_m": [float(value) for value in depth_static.get("shift_m", [])]},
        "noise": {key: _positive_float(noise.get(key), path=f"seismic_mismatch.noise.{key}") for key in ("white_noise_rms_fraction", "colored_noise_rms_fraction", "colored_vertical_correlation_m")},
        "gain": {
            key: _positive_float(gain.get(key), path=f"seismic_mismatch.gain.{key}")
            for key in ("global_log_sigma", "tracewise_log_sigma")
        },
        "combined": {"enabled": bool(combined["enabled"])},
    }
    for key in ("vertical_lateral_log_sigma", "lateral_correlation_fraction", "vertical_correlation_fraction"):
        if key in gain and gain.get(key) is not None:
            parsed_mismatch["gain"][key] = _positive_float(gain.get(key), path=f"seismic_mismatch.gain.{key}")
    if bool(combined["enabled"]):
        parsed_mismatch["combined"].update({
            "phase_rotation_degrees": float(combined.get("phase_rotation_degrees")),
            "time_shift_s": float(combined.get("time_shift_s")),
            "depth_static_m": float(combined.get("depth_static_m", 0.0)),
            "gain_log_sigma": _positive_float(combined.get("gain_log_sigma"), path="seismic_mismatch.combined.gain_log_sigma"),
            "noise_rms_fraction": _positive_float(combined.get("noise_rms_fraction"), path="seismic_mismatch.combined.noise_rms_fraction"),
        })
    if parsed_mismatch["wavelet"]["phase_rotation_degrees"] != [-10.0, 10.0]:
        raise ValueError("Depth v4 wavelet phase rotations are frozen to [-10, 10] degrees.")
    if parsed_mismatch["wavelet"]["time_shift_s"] != [-0.001, 0.001]:
        raise ValueError("Depth v4 wavelet time shifts are frozen to [-0.001, 0.001] seconds.")
    if parsed_mismatch["depth_static"]["shift_m"] != [-2.5, 2.5]:
        raise ValueError("Depth v4 depth statics are frozen to [-2.5, 2.5] metres.")

    figures = _mapping(root.get("figures"), path="synthoseis_lite.figures")
    _reject_unknown(
        figures,
        {"enabled", "max_example_objects_per_zone_state", "report_examples"},
        path="synthoseis_lite.figures",
    )
    if not isinstance(figures.get("enabled"), bool):
        raise ValueError("figures.enabled must be boolean.")
    max_examples = _positive_int(
        figures.get("max_example_objects_per_zone_state"),
        path="figures.max_example_objects_per_zone_state",
    )
    report_examples = _mapping(
        figures.get("report_examples"), path="figures.report_examples"
    )

    return {
        "schema": SCHEMA_VERSION,
        "sample_domain": "depth",
        "benchmark_schema": SCHEMA_VERSION,
        "global_seed": int(root.get("global_seed")),
        "source_runs": {
            "well_inventory_dir": str(sources.get("well_inventory_dir") or "").strip(),
            "rock_physics_analysis_dir": str(sources.get("rock_physics_analysis_dir") or "").strip(),
            "depth_forward_model_inputs_dir": str(
                sources.get("depth_forward_model_inputs_dir") or ""
            ).strip(),
            "wavelet_batch_synthetic_depth_dir": str(sources.get("wavelet_batch_synthetic_depth_dir") or "").strip(),
        },
        "horizons": horizons,
        "sections": sections,
        "sampling": {
            "expected_model_dz_m": model_dz,
            "vertical_oversampling_factor": oversampling,
            "antialias": {"family": family, "taps_per_factor": _positive_int(antialias.get("taps_per_factor"), path="sampling.antialias.taps_per_factor"), "cutoff_output_nyquist_fraction": cutoff, "kaiser_beta": float(antialias.get("kaiser_beta"))},
        },
        "lateral_sample_interval_m": lateral_interval,
        "calibration": {"background_estimator": "per_well_zone_huber", "huber_delta_sigma": _positive_float(calibration.get("huber_delta_sigma"), path="calibration.huber_delta_sigma"), "minimum_valid_cells_per_well_zone": _positive_int(calibration.get("minimum_valid_cells_per_well_zone"), path="calibration.minimum_valid_cells_per_well_zone")},
        "impedance": {
            "family": GENERATOR_FAMILY,
            "state_threshold_sigma": _positive_float(
                impedance.get("state_threshold_sigma"),
                path="impedance.state_threshold_sigma",
            ),
            "minimum_highres_cells": _positive_int(
                standard_mode.get("minimum_highres_cells"),
                path="duration_modes.standard.minimum_highres_cells",
            ),
            **object_core_controls,
        },
        "generation": {"attempts_per_scenario": _positive_int(generation.get("attempts_per_scenario"), path="generation.attempts_per_scenario"), "duration_modes": ["standard"], "geometry_families": geometry_families, "geometry_directions": directions, "acceptance_qc": {"minimum_attempts_per_scenario": _positive_int(acceptance.get("minimum_attempts_per_scenario"), path="generation.acceptance_qc.minimum_attempts_per_scenario"), "warning_fraction": warning, "failure_fraction": failure, "enforcement": enforcement}},
        "splits": {"assignment_unit": "parent_realization", "held_out_geometry_family": held_out},
        "seismic_input": {"policy": "observed_highres_forward"},
        "seismic_forward": {"backend": backend, "dtype": "float64"},
        "lfm": parsed_lfm,
        "seismic_mismatch": parsed_mismatch,
        "figures": {
            "enabled": bool(figures["enabled"]),
            "max_example_objects_per_zone_state": max_examples,
            "report_examples": {
                key: str(value)
                for key, value in report_examples.items()
                if value is not None and str(value).strip()
            },
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def resolve_depth_sources(
    script_cfg: Mapping[str, Any],
    *,
    workflow: WorkflowConfig,
    repo_root: Path,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    output_root = resolve_relative_path(workflow.output_root, root=repo_root)
    resolved: dict[str, Path] = {}
    provenance: dict[str, Any] = {}
    definitions = {
        "well_inventory_dir": ("well_inventory", ["well_inventory.csv", "run_summary.json"]),
        "rock_physics_analysis_dir": (
            "rock_physics_analysis",
            [
                "run_summary.json",
                "well_input_inventory.csv",
                "modules/ai_vp_linear/rock_physics_relation.json",
            ],
        ),
        "depth_forward_model_inputs_dir": (
            "depth_forward_model_inputs",
            ["forward_model_inputs.json", "run_summary.json"],
        ),
        "wavelet_batch_synthetic_depth_dir": ("wavelet_batch_synthetic_depth", ["run_summary.json", "wavelet_batch_metrics.csv"]),
    }
    for key, (prefix, files) in definitions.items():
        explicit = str(script_cfg["source_runs"].get(key) or "").strip()
        path = resolve_source_run(
            explicit or None,
            output_root=output_root,
            prefix=prefix,
            required_files=files,
            root=repo_root,
            label=key,
        )
        resolved[key] = path
        source_summary = _load_json(path / "run_summary.json")
        provenance[key] = {
            "resolution_mode": "explicit" if explicit else "auto",
            "path": str(path),
            "contract_fingerprint_sha256": require_contract_fingerprint(
                source_summary, label=f"{key} {path}"
            ),
            "required_files": list(files),
        }

    rock_summary = _load_json(resolved["rock_physics_analysis_dir"] / "run_summary.json")
    if (
        rock_summary.get("schema") != ROCK_PHYSICS_ANALYSIS_SCHEMA_VERSION
        or rock_summary.get("status") != "success"
        or rock_summary.get("sample_domain") != "depth"
        or rock_summary.get("depth_basis") != "tvdss"
    ):
        raise ValueError(
            "rock_physics_analysis source is not a successful contracted run."
        )
    wavelet_batch_summary = _load_json(resolved["wavelet_batch_synthetic_depth_dir"] / "run_summary.json")
    if wavelet_batch_summary.get("status") != "success":
        raise ValueError("wavelet_batch_synthetic_depth run is not successful.")
    forward_run_summary = _load_json(
        resolved["depth_forward_model_inputs_dir"] / "run_summary.json"
    )
    if (
        forward_run_summary.get("schema")
        != DEPTH_FORWARD_MODEL_INPUTS_RUN_SCHEMA_VERSION
        or forward_run_summary.get("status") != "success"
        or forward_run_summary.get("sample_domain") != "depth"
        or forward_run_summary.get("depth_basis") != "tvdss"
    ):
        raise ValueError(
            "depth_forward_model_inputs source is not a successful depth/TVDSS run."
        )
    forward_path = (
        resolved["depth_forward_model_inputs_dir"] / "forward_model_inputs.json"
    )
    recorded_forward_path = resolve_artifact_path(
        dict(
            dict(forward_run_summary.get("artifacts") or {}).get(
                "forward_model_inputs"
            )
            or {}
        ).get("path"),
        root=repo_root,
        run_dir=resolved["depth_forward_model_inputs_dir"],
    )
    if (
        recorded_forward_path is None
        or recorded_forward_path.resolve() != forward_path.resolve()
    ):
        raise ValueError(
            "depth_forward_model_inputs summary does not record its selected "
            "forward_model_inputs artifact."
        )
    forward = _load_json(forward_path)
    if forward.get("schema") != FORWARD_MODEL_INPUTS_SCHEMA_VERSION:
        raise ValueError(f"forward_model_inputs schema must be {FORWARD_MODEL_INPUTS_SCHEMA_VERSION}.")
    if forward.get("sample_domain") != "depth" or forward.get("depth_basis") != "tvdss":
        raise ValueError("forward_model_inputs must declare depth/TVDSS.")
    rock_fingerprint = require_contract_fingerprint(
        rock_summary, label=f"rock physics run {resolved['rock_physics_analysis_dir']}"
    )
    recorded_rock_contract = dict(
        dict(forward.get("input_contracts") or {}).get("rock_physics_analysis")
        or {}
    )
    if str(recorded_rock_contract.get("contract_fingerprint_sha256") or "") != str(
        rock_fingerprint
    ):
        raise ValueError(
            "forward_model_inputs and selected rock_physics_analysis have different contracts."
        )
    relation_path = resolve_relative_path(
        str(dict(forward.get("ai_velocity_relation") or {}).get("path") or ""),
        root=repo_root,
    )
    expected_relation_path = (
        resolved["rock_physics_analysis_dir"]
        / "modules"
        / "ai_vp_linear"
        / "rock_physics_relation.json"
    )
    if relation_path.resolve() != expected_relation_path.resolve():
        raise ValueError(
            "forward_model_inputs relation does not belong to the selected rock-physics run."
        )
    inventory_summary = _load_json(resolved["well_inventory_dir"] / "run_summary.json")
    inventory_inputs = dict(inventory_summary.get("inputs") or {})
    inventory_geometry = dict(inventory_summary.get("geometry") or {})
    if inventory_inputs.get("seismic_domain") != "depth" or inventory_geometry.get("sample_domain") != "depth" or inventory_geometry.get("sample_unit") != "m":
        raise ValueError("well_inventory run is not a depth-domain metre survey inventory.")
    expected_seismic = resolve_relative_path(
        workflow.seismic.file,
        root=resolve_relative_path(workflow.data_root, root=repo_root),
    )
    recorded_seismic = resolve_relative_path(str(inventory_inputs.get("seismic_file") or ""), root=repo_root)
    if recorded_seismic.resolve() != expected_seismic.resolve():
        raise ValueError("well_inventory seismic source does not match the current workflow config.")
    forward["_path"] = str(forward_path)
    forward["_contract_fingerprint_sha256"] = require_contract_fingerprint(
        forward_run_summary,
        label=f"depth forward-model inputs run {resolved['depth_forward_model_inputs_dir']}",
    )
    forward["_rock_physics_contract_fingerprint_sha256"] = rock_fingerprint
    return resolved, provenance, forward


__all__ = [
    "CALIBRATION_SCHEMA",
    "GENERATOR_FAMILY",
    "SCHEMA_VERSION",
    "load_composed_config",
    "parse_depth_config",
    "resolve_depth_sources",
]
