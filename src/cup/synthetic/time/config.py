"""Strict time-domain Synthoseis-lite v5 configuration and source contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cup.config.sources import assert_recorded_source_matches, require_source_files
from cup.synthetic.core.calibration import GENERATOR_FAMILY
from cup.synthetic.core.config import parse_object_core_controls
from cup.synthetic.core.amplitude_calibration import parse_amplitude_calibration_controls
from cup.synthetic.core.views import resolve_view_specs, validate_view_units
from cup.synthetic.schemas import BENCHMARK_SCHEMA_VERSION, SCIENCE_REVISION
from cup.utils.io import resolve_relative_path


DATA_SCHEMA = BENCHMARK_SCHEMA_VERSION
IMPLEMENTATION_SCOPE = "impedance_truth_forward_qc_seismic_views"


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], *, path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown keys: {unknown}")


def _required_text(value: Mapping[str, Any], key: str, *, path: str) -> str:
    text = str(value.get(key) or "").strip()
    if not text:
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return text


def _positive(value: Any, *, path: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path} must be positive")
    return result


def _nonnegative(value: Any, *, path: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{path} must be non-negative")
    return result


def _parse_horizons(config: Mapping[str, Any]) -> list[dict[str, str]]:
    target = _mapping(config.get("target_interval"), path="target_interval")
    raw = target.get("horizons")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("target_interval.horizons must contain at least two entries")
    result = []
    for index, value in enumerate(raw):
        item = _mapping(value, path=f"target_interval.horizons[{index}]")
        _reject_unknown(item, {"name", "well_top", "file"}, path=f"target_interval.horizons[{index}]")
        result.append({
            "name": _required_text(item, "name", path=f"target_interval.horizons[{index}]"),
            "well_top": _required_text(item, "well_top", path=f"target_interval.horizons[{index}]"),
            "file": _required_text(item, "file", path=f"target_interval.horizons[{index}]"),
        })
    if len({item["name"] for item in result}) != len(result):
        raise ValueError("target_interval.horizons names must be unique")
    return result


def _parse_sections(root: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = root.get("sections")
    if not isinstance(raw, list) or not raw:
        raise ValueError("synthoseis_lite.sections must be non-empty")
    result = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        item = _mapping(value, path=f"synthoseis_lite.sections[{index}]")
        _reject_unknown(item, {"section_id", "path"}, path=f"synthoseis_lite.sections[{index}]")
        section_id = _required_text(item, "section_id", path=f"sections[{index}]")
        if section_id in seen:
            raise ValueError(f"duplicate section_id: {section_id}")
        seen.add(section_id)
        points = item.get("path")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"sections[{index}].path must contain at least two points")
        parsed_points = []
        for point_index, value in enumerate(points):
            point = _mapping(value, path=f"sections[{index}].path[{point_index}]")
            _reject_unknown(point, {"inline", "xline"}, path=f"sections[{index}].path[{point_index}]")
            parsed_points.append({"inline": float(point["inline"]), "xline": float(point["xline"])})
        result.append({"section_id": section_id, "path": parsed_points})
    return result


def _parse_sources(root: Mapping[str, Any]) -> dict[str, str]:
    raw = _mapping(root.get("source_runs"), path="synthoseis_lite.source_runs")
    keys = {"well_preprocess_dir", "well_auto_tie_dir", "wavelet_generation_dir"}
    _reject_unknown(raw, keys, path="synthoseis_lite.source_runs")
    return {key: _required_text(raw, key, path="synthoseis_lite.source_runs") for key in keys}


def _parse_impedance(root: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(root.get("impedance_attribute_generator"), path="synthoseis_lite.impedance_attribute_generator")
    _reject_unknown(raw, {"family", "state_threshold_sigma", "lateral", "qc", "robust_scale", "duration_modes"}, path="impedance_attribute_generator")
    if str(raw.get("family")) != GENERATOR_FAMILY:
        raise ValueError(f"impedance_attribute_generator.family must be {GENERATOR_FAMILY}")
    controls = parse_object_core_controls(raw)
    duration = _mapping(raw.get("duration_modes"), path="impedance_attribute_generator.duration_modes")
    _reject_unknown(duration, {"standard"}, path="duration_modes")
    standard = _mapping(duration.get("standard"), path="duration_modes.standard")
    _reject_unknown(standard, {"minimum_highres_cells"}, path="duration_modes.standard")
    return {
        "family": GENERATOR_FAMILY,
        "state_threshold_sigma": _positive(raw.get("state_threshold_sigma"), path="state_threshold_sigma"),
        "minimum_highres_cells": int(_positive(standard.get("minimum_highres_cells"), path="minimum_highres_cells")),
        **controls,
    }


def parse_synthoseis_config(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(config.get("synthoseis_lite"), path="synthoseis_lite")
    allowed = {
        "sample_domain", "benchmark_schema", "science_revision", "global_seed",
        "source_runs", "sampling", "geometry", "sections", "impedance_attribute_generator",
        "generation", "splits", "seismic_input", "forward_qc", "seismic_views", "amplitude_calibration", "figures",
    }
    _reject_unknown(root, allowed, path="synthoseis_lite")
    if str(root.get("sample_domain") or "").casefold() != "time":
        raise ValueError("Time Synthoseis-lite requires sample_domain='time'")
    if str(root.get("benchmark_schema") or "") != DATA_SCHEMA:
        raise ValueError(f"Time Synthoseis-lite requires benchmark_schema={DATA_SCHEMA!r}")
    if str(root.get("science_revision") or "") != SCIENCE_REVISION:
        raise ValueError(f"Time Synthoseis-lite requires science_revision={SCIENCE_REVISION!r}")

    sampling = _mapping(root.get("sampling"), path="sampling")
    _reject_unknown(sampling, {"expected_output_dt_s", "vertical_oversampling_factor"}, path="sampling")
    output_dt = _positive(sampling.get("expected_output_dt_s"), path="sampling.expected_output_dt_s")
    factor = int(_positive(sampling.get("vertical_oversampling_factor"), path="sampling.vertical_oversampling_factor"))
    geometry = _mapping(root.get("geometry"), path="geometry")
    _reject_unknown(geometry, {"lateral_sample_interval_m", "field_conditioned"}, path="geometry")
    field = _mapping(geometry.get("field_conditioned"), path="geometry.field_conditioned")
    _reject_unknown(field, {"enabled", "target_zone"}, path="geometry.field_conditioned")
    if field.get("enabled") is not True:
        raise ValueError("geometry.field_conditioned.enabled must be true")
    target_zone = _mapping(field.get("target_zone"), path="geometry.field_conditioned.target_zone")
    _reject_unknown(target_zone, {"mode", "nearest_distance_limit", "outlier_threshold", "outlier_min_neighbor_count", "min_thickness_s"}, path="target_zone")
    if str(target_zone.get("mode")) != "filled_target_zone":
        raise ValueError("time target_zone.mode must be filled_target_zone")

    generation = _mapping(root.get("generation"), path="generation")
    _reject_unknown(generation, {"attempts_per_scenario", "duration_modes", "geometry_families", "geometry_directions", "acceptance_qc"}, path="generation")
    duration_modes = [str(value) for value in generation.get("duration_modes") or []]
    geometry_families = [str(value) for value in generation.get("geometry_families") or []]
    geometry_directions = [str(value) for value in generation.get("geometry_directions") or []]
    if not duration_modes or not set(duration_modes) <= {"standard", "ultra_thin_stress"}:
        raise ValueError("generation.duration_modes contains unsupported values")
    if not geometry_families or not set(geometry_families) <= {"none", "wedge", "pinchout"}:
        raise ValueError("generation.geometry_families contains unsupported values")
    if not geometry_directions or not set(geometry_directions) <= {"left_to_right", "right_to_left"}:
        raise ValueError("generation.geometry_directions contains unsupported values")
    acceptance = _mapping(generation.get("acceptance_qc"), path="generation.acceptance_qc")
    _reject_unknown(acceptance, {"minimum_attempts_per_scenario", "warning_fraction", "severe_warning_fraction"}, path="generation.acceptance_qc")
    warning = float(acceptance["warning_fraction"]); severe_warning = float(acceptance["severe_warning_fraction"])
    if not 0.0 <= severe_warning < warning <= 1.0:
        raise ValueError("generation acceptance fractions must satisfy 0 <= severe_warning < warning <= 1")

    splits = _mapping(root.get("splits"), path="splits")
    _reject_unknown(splits, {"assignment_unit", "held_out_geometry_family"}, path="splits")
    if splits.get("assignment_unit") != "parent_realization" or str(splits.get("held_out_geometry_family")) not in geometry_families:
        raise ValueError("splits must assign parent_realization and name a configured held-out geometry family")
    seismic_input = _mapping(root.get("seismic_input"), path="seismic_input")
    _reject_unknown(seismic_input, {"policy"}, path="seismic_input")
    if seismic_input.get("policy") != "observed_highres_forward":
        raise ValueError("seismic_input.policy must be observed_highres_forward")
    forward_qc = _mapping(root.get("forward_qc"), path="forward_qc")
    highres = _mapping(forward_qc.get("highres_forward"), path="forward_qc.highres_forward")
    _reject_unknown(forward_qc, {"highres_forward"}, path="forward_qc")
    _reject_unknown(highres, {"enabled", "required"}, path="forward_qc.highres_forward")
    if highres.get("enabled") is not True or highres.get("required") is not True:
        raise ValueError("forward_qc.highres_forward.enabled and required must be true")

    views = _mapping(root.get("seismic_views"), path="seismic_views")
    resolve_view_specs(views)
    validate_view_units(views, axis_unit="s")
    has_empirical = any(
        isinstance(item, Mapping) and item.get("kind") == "empirical_rgt_gain"
        for item in dict(views.get("operators") or {}).values()
    )
    amplitude = parse_amplitude_calibration_controls(
        root.get("amplitude_calibration"), required=has_empirical
    )
    figures = _mapping(root.get("figures"), path="figures")
    _reject_unknown(figures, {"enabled", "max_example_objects_per_zone_state", "report_examples"}, path="figures")
    return {
        "sample_domain": "time", "benchmark_schema": DATA_SCHEMA, "science_revision": SCIENCE_REVISION,
        "global_seed": int(root.get("global_seed")), "source_runs": _parse_sources(root),
        "sampling": {"expected_output_dt_s": output_dt, "vertical_oversampling_factor": factor},
        "horizons": _parse_horizons(config), "sections": _parse_sections(root),
        "lateral_sample_interval_m": _positive(geometry.get("lateral_sample_interval_m"), path="geometry.lateral_sample_interval_m"),
        "target_zone": {key: target_zone.get(key) for key in ("mode", "nearest_distance_limit", "outlier_threshold", "outlier_min_neighbor_count", "min_thickness_s")},
        "impedance": _parse_impedance(root),
        "generation": {"attempts_per_scenario": int(_positive(generation.get("attempts_per_scenario"), path="generation.attempts_per_scenario")), "duration_modes": duration_modes, "geometry_families": geometry_families, "geometry_directions": geometry_directions, "acceptance_qc": {"minimum_attempts_per_scenario": int(_positive(acceptance.get("minimum_attempts_per_scenario"), path="acceptance.minimum_attempts_per_scenario")), "warning_fraction": warning, "severe_warning_fraction": severe_warning}},
        "splits": {"assignment_unit": "parent_realization", "held_out_geometry_family": str(splits["held_out_geometry_family"])},
        "seismic_input": {"policy": "observed_highres_forward"},
        "forward_qc": {"highres_forward_enabled": True, "highres_forward_required": True},
        "amplitude_calibration": amplitude,
        "seismic_views": views,
        "figures": {"enabled": bool(figures.get("enabled", True)), "max_example_objects_per_zone_state": int(figures.get("max_example_objects_per_zone_state", 1)), "report_examples": dict(figures.get("report_examples") or {})},
    }


def resolve_sources(script_cfg: Mapping[str, Any], *, repo_root: Path) -> dict[str, Path]:
    sources = {key: resolve_relative_path(value, root=repo_root) for key, value in script_cfg["source_runs"].items()}
    required = {
        "well_preprocess_dir": ["run_summary.json", "well_preprocess_status.csv"],
        "well_auto_tie_dir": ["run_summary.json", "well_tie_metrics.csv"],
        "wavelet_generation_dir": ["run_summary.json", "selected_wavelet.csv", "selected_wavelet_summary.json", "evaluation_well_spatial_clusters.csv"],
    }
    for key, files in required.items():
        require_source_files(sources[key], files, label=key)
    with (sources["wavelet_generation_dir"] / "selected_wavelet_summary.json").open("r", encoding="utf-8") as handle:
        wavelet_summary = json.load(handle)
    assert_recorded_source_matches(wavelet_summary, "source_auto_tie_dir", sources["well_auto_tie_dir"], root=repo_root)
    with (sources["well_auto_tie_dir"] / "run_summary.json").open("r", encoding="utf-8") as handle:
        auto_tie_summary = json.load(handle)
    recorded = str(dict(auto_tie_summary.get("inputs") or {}).get("preprocess_status_file") or "")
    if not recorded:
        raise ValueError("well_auto_tie run summary lacks preprocess_status_file")
    if resolve_relative_path(recorded, root=repo_root).parent.resolve() != sources["well_preprocess_dir"].resolve():
        raise ValueError("source_run_mismatch:well_preprocess_dir")
    return sources


__all__ = ["DATA_SCHEMA", "IMPLEMENTATION_SCOPE", "parse_synthoseis_config", "resolve_sources"]
