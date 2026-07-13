"""Configuration parsing and validation for synthoseis-lite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cup.synthetic.core.calibration import GENERATOR_FAMILY
from cup.synthetic.schemas import BENCHMARK_SCHEMA_VERSION
from cup.synthetic.core.config import parse_object_core_controls
from cup.config.sources import assert_recorded_source_matches, require_source_files
from cup.utils.io import resolve_relative_path


DATA_SCHEMA = BENCHMARK_SCHEMA_VERSION
IMPLEMENTATION_SCOPE = (
    "impedance_truth_forward_qc_lfm_and_seismic_mismatch"
)


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], *, path: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ValueError(f"{path} contains unknown keys: {unknown}")


def _require_keys(value: Mapping[str, Any], required: set[str], *, path: str) -> None:
    missing = sorted(key for key in required if key not in value)
    if missing:
        raise ValueError(f"{path} lacks required keys: {missing}")


def _required_text(value: Mapping[str, Any], key: str, *, path: str) -> str:
    text = "" if value.get(key) is None else str(value[key]).strip()
    if not text:
        raise ValueError(f"{path}.{key} must be a non-empty string.")
    return text


def _optional_float(value: Mapping[str, Any], key: str) -> float | None:
    raw = value.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.casefold() in {"none", "null", "nan"}:
        return None
    return float(raw)


def parse_synthoseis_config(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(config.get("synthoseis_lite"), path="synthoseis_lite")
    if "probe_selection" in root:
        raise ValueError(
            "synthoseis_lite.probe_selection is not part of the v4 canonical benchmark."
        )
    _reject_unknown(
        root,
        {
            "sample_domain",
            "benchmark_schema",
            "global_seed",
            "source_runs",
            "sampling",
            "geometry",
            "sections",
            "impedance_attribute_generator",
            "generation",
            "splits",
            "seismic_input",
            "forward_qc",
            "lfm",
            "seismic_mismatch",
            "figures",
        },
        path="synthoseis_lite",
    )
    _require_keys(
        root,
        {
            "sample_domain",
            "benchmark_schema",
            "global_seed",
            "source_runs",
            "sampling",
            "geometry",
            "sections",
            "impedance_attribute_generator",
            "generation",
            "splits",
            "seismic_input",
            "forward_qc",
            "lfm",
            "seismic_mismatch",
            "figures",
        },
        path="synthoseis_lite",
    )
    sample_domain = str(root.get("sample_domain") or "").casefold()
    benchmark_schema = str(root.get("benchmark_schema") or "")
    if sample_domain != "time" or benchmark_schema != DATA_SCHEMA:
        raise ValueError(
            "Time-domain Synthoseis-lite requires "
            f"synthoseis_lite.sample_domain='time' and benchmark_schema={DATA_SCHEMA!r}."
        )
    sources = _mapping(root.get("source_runs"), path="synthoseis_lite.source_runs")
    source_keys = [
        "well_preprocess_dir",
        "well_auto_tie_dir",
        "wavelet_generation_dir",
    ]
    _reject_unknown(sources, set(source_keys), path="synthoseis_lite.source_runs")
    source_runs = {
        key: _required_text(sources, key, path="synthoseis_lite.source_runs")
        for key in source_keys
    }
    sampling = _mapping(root.get("sampling"), path="synthoseis_lite.sampling")
    sampling_keys = {"expected_output_dt_s", "vertical_oversampling_factor"}
    _reject_unknown(
        sampling,
        sampling_keys,
        path="synthoseis_lite.sampling",
    )
    _require_keys(sampling, sampling_keys, path="synthoseis_lite.sampling")
    geometry = _mapping(root.get("geometry"), path="synthoseis_lite.geometry")
    geometry_keys = {"lateral_sample_interval_m", "field_conditioned", "canonical"}
    _reject_unknown(
        geometry,
        geometry_keys,
        path="synthoseis_lite.geometry",
    )
    _require_keys(geometry, geometry_keys, path="synthoseis_lite.geometry")
    field = _mapping(
        geometry.get("field_conditioned"),
        path="synthoseis_lite.geometry.field_conditioned",
    )
    field_keys = {"enabled", "target_zone"}
    _reject_unknown(
        field,
        field_keys,
        path="synthoseis_lite.geometry.field_conditioned",
    )
    _require_keys(field, field_keys, path="synthoseis_lite.geometry.field_conditioned")
    if field.get("enabled") is not True:
        raise ValueError(
            "Time Synthoseis-lite v4 requires geometry.field_conditioned.enabled=true."
        )
    canonical = _mapping(
        geometry.get("canonical"), path="synthoseis_lite.geometry.canonical"
    )
    canonical_keys = {
        "enabled",
        "lateral_sample_interval_m",
        "lateral_samples",
        "center_twt_s",
        "vertical_extent_periods",
        "thin_bed_period_ratios",
        "wedge_transition_fraction",
        "pinchout_termination_fraction",
        "dip_drop_period_ratios",
        "lateral_contrast_multipliers",
    }
    _reject_unknown(
        canonical, canonical_keys, path="synthoseis_lite.geometry.canonical"
    )
    _require_keys(canonical, canonical_keys, path="synthoseis_lite.geometry.canonical")
    target_zone = _mapping(
        field.get("target_zone"),
        path="synthoseis_lite.geometry.field_conditioned.target_zone",
    )
    target_zone_keys = {
        "mode",
        "nearest_distance_limit",
        "outlier_threshold",
        "outlier_min_neighbor_count",
        "min_thickness_s",
    }
    _reject_unknown(
        target_zone,
        target_zone_keys,
        path="synthoseis_lite.geometry.field_conditioned.target_zone",
    )
    _require_keys(
        target_zone,
        target_zone_keys,
        path="synthoseis_lite.geometry.field_conditioned.target_zone",
    )
    if "horizons" in field:
        raise ValueError(
            "synthoseis_lite.geometry.field_conditioned.horizons is retired; use top-level target_interval.horizons."
        )
    if "sections" in field:
        raise ValueError(
            "synthoseis_lite.geometry.field_conditioned.sections is retired; use synthoseis_lite.sections."
        )
    target_interval = _mapping(config.get("target_interval"), path="target_interval")
    horizons = target_interval.get("horizons")
    if root.get("sections_file") is not None:
        raise ValueError(
            "synthoseis_lite.sections_file is retired; put sections in synthoseis_lite.sections."
        )
    sections = root.get("sections")
    if not isinstance(horizons, list) or len(horizons) < 2:
        raise ValueError("target_interval.horizons needs at least two entries.")
    if not isinstance(sections, list) or not sections:
        raise ValueError("synthoseis_lite.sections must be non-empty.")
    horizon_items = []
    for index, item in enumerate(horizons):
        item = _mapping(item, path=f"target_interval.horizons[{index}]")
        horizon_keys = {"name", "well_top", "file"}
        _reject_unknown(item, horizon_keys, path=f"target_interval.horizons[{index}]")
        _require_keys(item, horizon_keys, path=f"target_interval.horizons[{index}]")
        horizon_items.append(
            {
                "name": _required_text(item, "name", path=f"horizons[{index}]"),
                "well_top": _required_text(item, "well_top", path=f"horizons[{index}]"),
                "file": _required_text(item, "file", path=f"horizons[{index}]"),
            }
        )
    section_items = []
    for index, item in enumerate(sections):
        item = _mapping(item, path=f"synthoseis_lite.sections[{index}]")
        section_keys = {"section_id", "path"}
        _reject_unknown(item, section_keys, path=f"synthoseis_lite.sections[{index}]")
        _require_keys(item, section_keys, path=f"synthoseis_lite.sections[{index}]")
        path_points = item.get("path")
        if not isinstance(path_points, list) or len(path_points) < 2:
            raise ValueError(f"sections[{index}].path needs at least two points.")
        parsed_points = []
        for point_index, point in enumerate(path_points):
            point_path = f"synthoseis_lite.sections[{index}].path[{point_index}]"
            point_item = _mapping(point, path=point_path)
            point_keys = {"inline", "xline"}
            _reject_unknown(point_item, point_keys, path=point_path)
            _require_keys(point_item, point_keys, path=point_path)
            parsed_points.append(
                {
                    "inline": float(point_item["inline"]),
                    "xline": float(point_item["xline"]),
                }
            )
        section_items.append(
            {
                "section_id": _required_text(
                    item, "section_id", path=f"sections[{index}]"
                ),
                "path": parsed_points,
            }
        )
    impedance = _mapping(
        root.get("impedance_attribute_generator"),
        path="synthoseis_lite.impedance_attribute_generator",
    )
    impedance_keys = {
        "family",
        "state_threshold_sigma",
        "lateral",
        "qc",
        "robust_scale",
        "duration_modes",
    }
    _reject_unknown(
        impedance,
        impedance_keys,
        path="synthoseis_lite.impedance_attribute_generator",
    )
    _require_keys(
        impedance, impedance_keys, path="synthoseis_lite.impedance_attribute_generator"
    )
    lateral = _mapping(
        impedance.get("lateral"),
        path="synthoseis_lite.impedance_attribute_generator.lateral",
    )
    lateral_keys = {
        "correlation_length_section_fractions",
        "coefficient_sigma_multipliers",
        "thickness_log_sigma_values",
    }
    _reject_unknown(
        lateral,
        lateral_keys,
        path="synthoseis_lite.impedance_attribute_generator.lateral",
    )
    _require_keys(
        lateral,
        lateral_keys,
        path="synthoseis_lite.impedance_attribute_generator.lateral",
    )
    qc = _mapping(
        impedance.get("qc"), path="synthoseis_lite.impedance_attribute_generator.qc"
    )
    qc_keys = {
        "max_global_reversal_fraction",
        "max_object_reversal_fraction",
        "max_global_clipping_fraction",
        "max_object_clipping_fraction",
    }
    _reject_unknown(
        qc,
        qc_keys,
        path="synthoseis_lite.impedance_attribute_generator.qc",
    )
    _require_keys(qc, qc_keys, path="synthoseis_lite.impedance_attribute_generator.qc")
    robust_scale = _mapping(
        impedance.get("robust_scale"),
        path="synthoseis_lite.impedance_attribute_generator.robust_scale",
    )
    robust_scale_keys = {
        "huber_delta_parent_sigma_floor",
        "coefficient_sigma_parent_floor",
        "coefficient_sigma_parent_cap",
    }
    _reject_unknown(
        robust_scale,
        robust_scale_keys,
        path="synthoseis_lite.impedance_attribute_generator.robust_scale",
    )
    _require_keys(
        robust_scale,
        robust_scale_keys,
        path="synthoseis_lite.impedance_attribute_generator.robust_scale",
    )
    object_core_controls = parse_object_core_controls(impedance)
    state_threshold_sigma = float(impedance.get("state_threshold_sigma"))
    if not np.isfinite(state_threshold_sigma) or state_threshold_sigma <= 0.0:
        raise ValueError("impedance_attribute_generator.state_threshold_sigma must be positive.")
    duration_modes = _mapping(
        impedance.get("duration_modes"),
        path="synthoseis_lite.impedance_attribute_generator.duration_modes",
    )
    _reject_unknown(
        duration_modes,
        {"standard"},
        path="synthoseis_lite.impedance_attribute_generator.duration_modes",
    )
    _require_keys(
        duration_modes,
        {"standard"},
        path="synthoseis_lite.impedance_attribute_generator.duration_modes",
    )
    standard_duration = _mapping(
        duration_modes.get("standard"),
        path="synthoseis_lite.impedance_attribute_generator.duration_modes.standard",
    )
    _reject_unknown(
        standard_duration,
        {"minimum_highres_cells"},
        path="synthoseis_lite.impedance_attribute_generator.duration_modes.standard",
    )
    _require_keys(
        standard_duration,
        {"minimum_highres_cells"},
        path="synthoseis_lite.impedance_attribute_generator.duration_modes.standard",
    )
    generation = _mapping(root.get("generation"), path="synthoseis_lite.generation")
    generation_keys = {
        "attempts_per_scenario",
        "duration_modes",
        "geometry_families",
        "geometry_directions",
        "acceptance_qc",
    }
    _reject_unknown(
        generation,
        generation_keys,
        path="synthoseis_lite.generation",
    )
    _require_keys(generation, generation_keys, path="synthoseis_lite.generation")
    attempts_per_scenario = int(generation.get("attempts_per_scenario"))
    if attempts_per_scenario <= 0:
        raise ValueError("generation.attempts_per_scenario must be a positive integer.")
    configured_duration_modes = [str(value) for value in generation["duration_modes"]]
    if not configured_duration_modes or not set(configured_duration_modes) <= {
        "standard",
        "ultra_thin_stress",
    }:
        raise ValueError("generation.duration_modes contains unsupported values.")
    configured_geometry_families = [
        str(value) for value in generation["geometry_families"]
    ]
    if not configured_geometry_families or not set(configured_geometry_families) <= {
        "none",
        "wedge",
        "pinchout",
    }:
        raise ValueError("generation.geometry_families contains unsupported values.")
    configured_geometry_directions = [
        str(value) for value in generation["geometry_directions"]
    ]
    if not configured_geometry_directions or not set(
        configured_geometry_directions
    ) <= {"left_to_right", "right_to_left"}:
        raise ValueError("generation.geometry_directions contains unsupported values.")
    acceptance = _mapping(
        generation.get("acceptance_qc"), path="synthoseis_lite.generation.acceptance_qc"
    )
    acceptance_keys = {
        "minimum_attempts_per_scenario",
        "warning_fraction",
        "failure_fraction",
        "enforcement",
    }
    _reject_unknown(
        acceptance,
        acceptance_keys,
        path="synthoseis_lite.generation.acceptance_qc",
    )
    _require_keys(
        acceptance,
        {
            "minimum_attempts_per_scenario",
            "warning_fraction",
            "failure_fraction",
        },
        path="synthoseis_lite.generation.acceptance_qc",
    )
    acceptance_failure = float(acceptance.get("failure_fraction"))
    acceptance_warning = float(acceptance.get("warning_fraction"))
    if not 0.0 <= acceptance_failure < acceptance_warning <= 1.0:
        raise ValueError(
            "generation acceptance fractions must satisfy "
            "0 <= failure < warning <= 1."
        )
    acceptance_enforcement = str(acceptance.get("enforcement", "warn"))
    if acceptance_enforcement not in {"warn", "fail_fast"}:
        raise ValueError(
            "generation.acceptance_qc.enforcement must be warn or fail_fast."
        )
    splits = _mapping(root.get("splits"), path="synthoseis_lite.splits")
    split_keys = {"assignment_unit", "held_out_geometry_family"}
    _reject_unknown(
        splits,
        split_keys,
        path="synthoseis_lite.splits",
    )
    _require_keys(splits, split_keys, path="synthoseis_lite.splits")
    if splits.get("assignment_unit") != "parent_realization":
        raise ValueError("splits.assignment_unit must be parent_realization.")
    seismic_input = _mapping(
        root.get("seismic_input"), path="synthoseis_lite.seismic_input"
    )
    _reject_unknown(
        seismic_input, {"policy"}, path="synthoseis_lite.seismic_input"
    )
    _require_keys(seismic_input, {"policy"}, path="synthoseis_lite.seismic_input")
    if str(seismic_input.get("policy") or "") != "observed_highres_forward":
        raise ValueError(
            "Time Synthoseis v4 requires "
            "seismic_input.policy='observed_highres_forward'."
        )
    forward_qc = _mapping(root.get("forward_qc"), path="synthoseis_lite.forward_qc")
    _reject_unknown(forward_qc, {"highres_forward"}, path="synthoseis_lite.forward_qc")
    _require_keys(forward_qc, {"highres_forward"}, path="synthoseis_lite.forward_qc")
    highres_forward = _mapping(
        forward_qc.get("highres_forward"),
        path="synthoseis_lite.forward_qc.highres_forward",
    )
    highres_forward_keys = {"enabled", "required"}
    _reject_unknown(
        highres_forward,
        highres_forward_keys,
        path="synthoseis_lite.forward_qc.highres_forward",
    )
    _require_keys(
        highres_forward,
        highres_forward_keys,
        path="synthoseis_lite.forward_qc.highres_forward",
    )
    if highres_forward.get("enabled") is not True or highres_forward.get("required") is not True:
        raise ValueError(
            "Time Synthoseis v4 requires forward_qc.highres_forward.enabled=true "
            "and required=true."
        )
    figures = _mapping(root.get("figures"), path="synthoseis_lite.figures")
    figure_keys = {"enabled", "max_example_objects_per_zone_state", "report_examples"}
    _reject_unknown(figures, figure_keys, path="synthoseis_lite.figures")
    _require_keys(figures, figure_keys, path="synthoseis_lite.figures")
    report_examples = dict(figures.get("report_examples") or {})
    lfm = _mapping(root.get("lfm"), path="synthoseis_lite.lfm")
    lfm_keys = {"enabled", "ideal", "controlled_degraded"}
    _reject_unknown(lfm, lfm_keys, path="synthoseis_lite.lfm")
    _require_keys(lfm, lfm_keys, path="synthoseis_lite.lfm")
    lfm_ideal = _mapping(lfm.get("ideal"), path="synthoseis_lite.lfm.ideal")
    lfm_ideal_keys = {"cutoff_hz", "numtaps", "kaiser_beta"}
    _reject_unknown(lfm_ideal, lfm_ideal_keys, path="synthoseis_lite.lfm.ideal")
    _require_keys(lfm_ideal, lfm_ideal_keys, path="synthoseis_lite.lfm.ideal")
    lfm_degraded = _mapping(
        lfm.get("controlled_degraded"), path="synthoseis_lite.lfm.controlled_degraded"
    )
    lfm_degraded_keys = {
        "constant_bias_sigma_log_ai",
        "linear_twt_trend_sigma_log_ai",
        "zonewise_bias_sigma_log_ai",
        "lateral_smooth_bias_sigma_log_ai",
        "lateral_correlation_fraction",
        "amplitude_scale_sigma",
        "over_smoothing",
        "local_missing_control_bias",
    }
    _reject_unknown(
        lfm_degraded,
        lfm_degraded_keys,
        path="synthoseis_lite.lfm.controlled_degraded",
    )
    _require_keys(
        lfm_degraded,
        lfm_degraded_keys,
        path="synthoseis_lite.lfm.controlled_degraded",
    )
    lfm_over_smoothing = _mapping(
        lfm_degraded.get("over_smoothing"),
        path="synthoseis_lite.lfm.controlled_degraded.over_smoothing",
    )
    lfm_over_smoothing_keys = {
        "enabled",
        "cutoff_hz",
        "numtaps",
        "kaiser_beta",
        "blend",
    }
    _reject_unknown(
        lfm_over_smoothing,
        lfm_over_smoothing_keys,
        path="synthoseis_lite.lfm.controlled_degraded.over_smoothing",
    )
    _require_keys(
        lfm_over_smoothing,
        {"enabled"},
        path="synthoseis_lite.lfm.controlled_degraded.over_smoothing",
    )
    if not isinstance(lfm_over_smoothing.get("enabled"), bool):
        raise ValueError(
            "synthoseis_lite.lfm.controlled_degraded.over_smoothing.enabled must be boolean."
        )
    if bool(lfm_over_smoothing["enabled"]):
        _require_keys(
            lfm_over_smoothing,
            lfm_over_smoothing_keys,
            path="synthoseis_lite.lfm.controlled_degraded.over_smoothing",
        )
    lfm_missing = _mapping(
        lfm_degraded.get("local_missing_control_bias"),
        path="synthoseis_lite.lfm.controlled_degraded.local_missing_control_bias",
    )
    lfm_missing_keys = {
        "enabled",
        "max_abs_log_ai",
        "lateral_width_fraction",
        "twt_width_fraction",
    }
    _reject_unknown(
        lfm_missing,
        lfm_missing_keys,
        path="synthoseis_lite.lfm.controlled_degraded.local_missing_control_bias",
    )
    _require_keys(
        lfm_missing,
        lfm_missing_keys,
        path="synthoseis_lite.lfm.controlled_degraded.local_missing_control_bias",
    )
    mismatch = _mapping(
        root.get("seismic_mismatch"), path="synthoseis_lite.seismic_mismatch"
    )
    mismatch_keys = {"enabled", "noise", "gain", "wavelet", "combined"}
    _reject_unknown(mismatch, mismatch_keys, path="synthoseis_lite.seismic_mismatch")
    _require_keys(mismatch, mismatch_keys, path="synthoseis_lite.seismic_mismatch")
    mismatch_noise = _mapping(
        mismatch.get("noise"), path="synthoseis_lite.seismic_mismatch.noise"
    )
    mismatch_noise_keys = {
        "white_noise_rms_fraction",
        "colored_noise_rms_fraction",
        "absolute_noise_rms_floor",
        "colored_time_correlation_samples",
    }
    _reject_unknown(
        mismatch_noise,
        mismatch_noise_keys,
        path="synthoseis_lite.seismic_mismatch.noise",
    )
    _require_keys(
        mismatch_noise,
        mismatch_noise_keys,
        path="synthoseis_lite.seismic_mismatch.noise",
    )
    mismatch_gain = _mapping(
        mismatch.get("gain"), path="synthoseis_lite.seismic_mismatch.gain"
    )
    mismatch_gain_keys = {
        "global_log_sigma",
        "tracewise_log_sigma",
        "time_lateral_log_sigma",
        "lateral_correlation_fraction",
        "time_correlation_fraction",
    }
    _reject_unknown(
        mismatch_gain,
        mismatch_gain_keys,
        path="synthoseis_lite.seismic_mismatch.gain",
    )
    _require_keys(
        mismatch_gain,
        mismatch_gain_keys,
        path="synthoseis_lite.seismic_mismatch.gain",
    )
    mismatch_wavelet = _mapping(
        mismatch.get("wavelet"), path="synthoseis_lite.seismic_mismatch.wavelet"
    )
    mismatch_wavelet_keys = {"phase_rotation_degrees", "time_shift_s"}
    _reject_unknown(
        mismatch_wavelet,
        mismatch_wavelet_keys,
        path="synthoseis_lite.seismic_mismatch.wavelet",
    )
    _require_keys(
        mismatch_wavelet,
        mismatch_wavelet_keys,
        path="synthoseis_lite.seismic_mismatch.wavelet",
    )
    mismatch_combined = _mapping(
        mismatch.get("combined"), path="synthoseis_lite.seismic_mismatch.combined"
    )
    mismatch_combined_keys = {
        "enabled",
        "phase_rotation_degrees",
        "time_shift_s",
        "gain_log_sigma",
        "noise_rms_fraction",
    }
    _reject_unknown(
        mismatch_combined,
        mismatch_combined_keys,
        path="synthoseis_lite.seismic_mismatch.combined",
    )
    _require_keys(
        mismatch_combined,
        mismatch_combined_keys,
        path="synthoseis_lite.seismic_mismatch.combined",
    )
    parsed = {
        "sample_domain": "time",
        "benchmark_schema": DATA_SCHEMA,
        "global_seed": int(root.get("global_seed", 20260615)),
        "source_runs": source_runs,
        "sampling": {
            "expected_output_dt_s": float(sampling.get("expected_output_dt_s", 0.002)),
            "vertical_oversampling_factor": int(
                sampling.get("vertical_oversampling_factor", 8)
            ),
        },
        "horizons": horizon_items,
        "sections": section_items,
        "lateral_sample_interval_m": float(
            geometry.get("lateral_sample_interval_m", 25.0)
        ),
        "target_zone": {
            "mode": str(target_zone.get("mode", "filled_target_zone")),
            "nearest_distance_limit": _optional_float(
                target_zone, "nearest_distance_limit"
            ),
            "outlier_threshold": _optional_float(target_zone, "outlier_threshold"),
            "outlier_min_neighbor_count": int(
                target_zone.get("outlier_min_neighbor_count", 2)
            ),
            "min_thickness_s": _optional_float(target_zone, "min_thickness_s"),
        },
        "canonical": {
            "enabled": bool(canonical.get("enabled", True)),
            "lateral_sample_interval_m": float(
                canonical.get(
                    "lateral_sample_interval_m",
                    geometry.get("lateral_sample_interval_m", 25.0),
                )
            ),
            "lateral_samples": int(canonical.get("lateral_samples", 128)),
            "center_twt_s": float(canonical.get("center_twt_s", 1.5)),
            "vertical_extent_periods": float(
                canonical.get("vertical_extent_periods", 6.0)
            ),
            "thin_bed_period_ratios": [
                float(value)
                for value in canonical.get(
                    "thin_bed_period_ratios",
                    [1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0],
                )
            ],
            "wedge_transition_fraction": float(
                canonical.get("wedge_transition_fraction", 0.80)
            ),
            "pinchout_termination_fraction": float(
                canonical.get("pinchout_termination_fraction", 0.75)
            ),
            "dip_drop_period_ratios": [
                float(value)
                for value in canonical.get(
                    "dip_drop_period_ratios",
                    [0.25, 0.5, 1.0],
                )
            ],
            "lateral_contrast_multipliers": [
                float(value)
                for value in canonical.get(
                    "lateral_contrast_multipliers",
                    [0.25, 0.5, 1.0, 2.0],
                )
            ],
        },
        "impedance": {
            "family": str(impedance.get("family", GENERATOR_FAMILY)),
            "state_threshold_sigma": state_threshold_sigma,
            **object_core_controls,
            "minimum_highres_cells": int(
                standard_duration.get("minimum_highres_cells")
            ),
            "minimum_attempts_per_scenario": int(
                acceptance.get("minimum_attempts_per_scenario")
            ),
            "scenario_acceptance_warning_fraction": float(
                acceptance.get("warning_fraction")
            ),
            "scenario_acceptance_failure_fraction": float(
                acceptance.get("failure_fraction")
            ),
            "scenario_acceptance_enforcement": acceptance_enforcement,
        },
        "generation": {
            "attempts_per_scenario": attempts_per_scenario,
            "duration_modes": configured_duration_modes,
            "geometry_families": configured_geometry_families,
            "geometry_directions": configured_geometry_directions,
            "acceptance_qc": {
                "minimum_attempts_per_scenario": int(
                    acceptance.get("minimum_attempts_per_scenario")
                ),
                "warning_fraction": float(acceptance.get("warning_fraction")),
                "failure_fraction": float(acceptance.get("failure_fraction")),
                "enforcement": acceptance_enforcement,
            },
        },
        "splits": {
            "assignment_unit": "parent_realization",
            "held_out_geometry_family": str(
                splits.get("held_out_geometry_family") or ""
            ),
        },
        "forward_qc": {
            "highres_forward_enabled": bool(highres_forward.get("enabled", True)),
            "highres_forward_required": bool(highres_forward.get("required", True)),
        },
        "seismic_input": {"policy": "observed_highres_forward"},
        "lfm": {
            "enabled": bool(lfm.get("enabled", True)),
            "ideal": {
                "cutoff_hz": float(lfm_ideal.get("cutoff_hz", 15.0)),
                "numtaps": int(lfm_ideal.get("numtaps", 129)),
                "kaiser_beta": float(lfm_ideal.get("kaiser_beta", 8.6)),
            },
            "controlled_degraded": {
                "constant_bias_sigma_log_ai": float(
                    lfm_degraded.get("constant_bias_sigma_log_ai", 0.02)
                ),
                "linear_twt_trend_sigma_log_ai": float(
                    lfm_degraded.get("linear_twt_trend_sigma_log_ai", 0.02)
                ),
                "zonewise_bias_sigma_log_ai": float(
                    lfm_degraded.get("zonewise_bias_sigma_log_ai", 0.03)
                ),
                "lateral_smooth_bias_sigma_log_ai": float(
                    lfm_degraded.get("lateral_smooth_bias_sigma_log_ai", 0.02)
                ),
                "lateral_correlation_fraction": float(
                    lfm_degraded.get("lateral_correlation_fraction", 0.30)
                ),
                "amplitude_scale_sigma": float(
                    lfm_degraded.get("amplitude_scale_sigma", 0.05)
                ),
                "over_smoothing": (
                    {
                        "enabled": True,
                        "cutoff_hz": float(lfm_over_smoothing.get("cutoff_hz", 6.0)),
                        "numtaps": int(lfm_over_smoothing.get("numtaps", 129)),
                        "kaiser_beta": float(
                            lfm_over_smoothing.get("kaiser_beta", 8.6)
                        ),
                        "blend": float(lfm_over_smoothing.get("blend", 1.0)),
                    }
                    if bool(lfm_over_smoothing["enabled"])
                    else {"enabled": False}
                ),
                "local_missing_control_bias": {
                    "enabled": bool(lfm_missing.get("enabled", True)),
                    "max_abs_log_ai": float(lfm_missing.get("max_abs_log_ai", 0.04)),
                    "lateral_width_fraction": float(
                        lfm_missing.get("lateral_width_fraction", 0.30)
                    ),
                    "twt_width_fraction": float(
                        lfm_missing.get("twt_width_fraction", 0.30)
                    ),
                },
            },
        },
        "seismic_mismatch": {
            "enabled": bool(mismatch.get("enabled", True)),
            "noise": {
                "white_noise_rms_fraction": float(
                    mismatch_noise.get("white_noise_rms_fraction", 0.05)
                ),
                "colored_noise_rms_fraction": float(
                    mismatch_noise.get("colored_noise_rms_fraction", 0.05)
                ),
                "absolute_noise_rms_floor": float(
                    mismatch_noise.get("absolute_noise_rms_floor", 0.01)
                ),
                "colored_time_correlation_samples": float(
                    mismatch_noise.get("colored_time_correlation_samples", 5.0)
                ),
            },
            "gain": {
                "global_log_sigma": float(mismatch_gain.get("global_log_sigma", 0.15)),
                "tracewise_log_sigma": float(
                    mismatch_gain.get("tracewise_log_sigma", 0.15)
                ),
                "time_lateral_log_sigma": float(
                    mismatch_gain.get("time_lateral_log_sigma", 0.15)
                ),
                "lateral_correlation_fraction": float(
                    mismatch_gain.get("lateral_correlation_fraction", 0.30)
                ),
                "time_correlation_fraction": float(
                    mismatch_gain.get("time_correlation_fraction", 0.25)
                ),
            },
            "wavelet": {
                "phase_rotation_degrees": [
                    float(value)
                    for value in mismatch_wavelet.get(
                        "phase_rotation_degrees",
                        [-10.0, 10.0],
                    )
                ],
                "time_shift_s": [
                    float(value)
                    for value in mismatch_wavelet.get(
                        "time_shift_s",
                        [-0.001, 0.001],
                    )
                ],
            },
            "combined": {
                "enabled": bool(mismatch_combined.get("enabled", True)),
                "phase_rotation_degrees": float(
                    mismatch_combined.get("phase_rotation_degrees", 10.0)
                ),
                "time_shift_s": float(
                    mismatch_combined.get("time_shift_s", 0.001)
                ),
                "gain_log_sigma": float(mismatch_combined.get("gain_log_sigma", 0.10)),
                "noise_rms_fraction": float(
                    mismatch_combined.get("noise_rms_fraction", 0.05)
                ),
            },
        },
        "figures": {
            "enabled": bool(figures.get("enabled", True)),
            "max_example_objects_per_zone_state": int(
                figures.get("max_example_objects_per_zone_state", 1)
            ),
            "report_examples": {
                key: str(value)
                for key, value in report_examples.items()
                if value is not None and str(value).strip()
            },
        },
    }
    _validate_canonical_config(parsed["canonical"])
    _validate_lfm_config(
        parsed["lfm"], output_dt_s=parsed["sampling"]["expected_output_dt_s"]
    )
    _validate_seismic_mismatch_config(parsed["seismic_mismatch"])
    if (
        parsed["splits"]["held_out_geometry_family"]
        not in parsed["generation"]["geometry_families"]
    ):
        raise ValueError(
            "splits.held_out_geometry_family must be one configured generation geometry family."
        )
    return parsed


def resolve_sources(
    script_cfg: Mapping[str, Any], *, repo_root: Path
) -> dict[str, Path]:
    sources = {
        key: resolve_relative_path(value, root=repo_root)
        for key, value in script_cfg["source_runs"].items()
    }
    required = {
        "well_preprocess_dir": ["run_summary.json", "well_preprocess_status.csv"],
        "well_auto_tie_dir": ["run_summary.json", "well_tie_metrics.csv"],
        "wavelet_generation_dir": [
            "run_summary.json",
            "selected_wavelet.csv",
            "selected_wavelet_summary.json",
            "evaluation_well_spatial_clusters.csv",
        ],
    }
    for key, names in required.items():
        directory = sources[key]
        require_source_files(directory, names, label=key)
    with (sources["wavelet_generation_dir"] / "selected_wavelet_summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        wavelet_summary = json.load(handle)
    assert_recorded_source_matches(
        wavelet_summary,
        "source_auto_tie_dir",
        sources["well_auto_tie_dir"],
        root=repo_root,
    )
    with (sources["well_auto_tie_dir"] / "run_summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        auto_tie_summary = json.load(handle)
    preprocess_status_text = str(
        dict(auto_tie_summary.get("inputs") or {}).get("preprocess_status_file") or ""
    )
    if not preprocess_status_text:
        raise ValueError("source_run_mismatch: well_auto_tie summary lacks preprocess_status_file")
    recorded_preprocess_dir = resolve_relative_path(preprocess_status_text, root=repo_root).parent
    if recorded_preprocess_dir.resolve() != sources["well_preprocess_dir"].resolve():
        raise ValueError("source_run_mismatch:well_preprocess_dir")
    return sources


def _validate_canonical_config(config: Mapping[str, Any]) -> None:
    if int(config["lateral_samples"]) < 8:
        raise ValueError(
            "synthoseis_lite.geometry.canonical.lateral_samples must be >= 8."
        )
    if float(config["lateral_sample_interval_m"]) <= 0.0:
        raise ValueError("canonical lateral_sample_interval_m must be positive.")
    if float(config["center_twt_s"]) <= 0.0:
        raise ValueError("canonical center_twt_s must be positive.")
    if float(config["vertical_extent_periods"]) < 2.0:
        raise ValueError("canonical vertical_extent_periods must be >= 2.")
    for key in (
        "thin_bed_period_ratios",
        "dip_drop_period_ratios",
        "lateral_contrast_multipliers",
    ):
        values = list(config[key])
        if not values or any(
            not np.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError(f"canonical {key} must contain positive finite values.")
    for key in ("wedge_transition_fraction", "pinchout_termination_fraction"):
        value = float(config[key])
        if not 0.0 < value < 1.0:
            raise ValueError(f"canonical {key} must be within (0, 1).")


def _validate_lfm_config(config: Mapping[str, Any], *, output_dt_s: float) -> None:
    if not bool(config["enabled"]):
        return
    nyquist = 0.5 / float(output_dt_s)
    ideal = config["ideal"]
    degraded = config["controlled_degraded"]
    cutoff = float(ideal["cutoff_hz"])
    if not 0.0 < cutoff < nyquist:
        raise ValueError("lfm.ideal.cutoff_hz must be within (0, Nyquist).")
    if not np.isclose(cutoff, 15.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            "Time Synthoseis-lite v4 canonical LFM requires lfm.ideal.cutoff_hz=15.0."
        )
    if int(ideal["numtaps"]) < 3:
        raise ValueError("lfm.ideal.numtaps must be >= 3.")
    if float(ideal["kaiser_beta"]) < 0.0:
        raise ValueError("lfm.ideal.kaiser_beta must be nonnegative.")
    for key in (
        "constant_bias_sigma_log_ai",
        "linear_twt_trend_sigma_log_ai",
        "zonewise_bias_sigma_log_ai",
        "lateral_smooth_bias_sigma_log_ai",
        "amplitude_scale_sigma",
    ):
        value = float(degraded[key])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"lfm.controlled_degraded.{key} must be nonnegative.")
    fraction = float(degraded["lateral_correlation_fraction"])
    if not np.isfinite(fraction) or fraction <= 0.0:
        raise ValueError(
            "lfm.controlled_degraded.lateral_correlation_fraction must be positive."
        )
    smoothing = degraded["over_smoothing"]
    if not isinstance(smoothing["enabled"], bool):
        raise ValueError("lfm over_smoothing.enabled must be boolean.")
    if bool(smoothing["enabled"]):
        over_cutoff = float(smoothing["cutoff_hz"])
        if not 0.0 < over_cutoff <= cutoff:
            raise ValueError(
                "lfm over_smoothing.cutoff_hz must be within (0, ideal cutoff]."
            )
        if int(smoothing["numtaps"]) < 3:
            raise ValueError("lfm over_smoothing.numtaps must be >= 3.")
        if not 0.0 <= float(smoothing["blend"]) <= 1.0:
            raise ValueError("lfm over_smoothing.blend must be within [0, 1].")
    missing = degraded["local_missing_control_bias"]
    if float(missing["max_abs_log_ai"]) < 0.0:
        raise ValueError(
            "lfm local_missing_control_bias.max_abs_log_ai must be nonnegative."
        )
    for key in ("lateral_width_fraction", "twt_width_fraction"):
        if not 0.0 < float(missing[key]) <= 1.0:
            raise ValueError(
                f"lfm local_missing_control_bias.{key} must be within (0, 1]."
            )


def _validate_seismic_mismatch_config(config: Mapping[str, Any]) -> None:
    if not bool(config["enabled"]):
        return
    noise = config["noise"]
    for key in (
        "white_noise_rms_fraction",
        "colored_noise_rms_fraction",
        "absolute_noise_rms_floor",
    ):
        value = float(noise[key])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"seismic_mismatch.noise.{key} must be nonnegative.")
    if float(noise["colored_time_correlation_samples"]) <= 0.0:
        raise ValueError(
            "seismic_mismatch.noise.colored_time_correlation_samples must be positive."
        )
    gain = config["gain"]
    for key in (
        "global_log_sigma",
        "tracewise_log_sigma",
        "time_lateral_log_sigma",
    ):
        value = float(gain[key])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"seismic_mismatch.gain.{key} must be nonnegative.")
    for key in ("lateral_correlation_fraction", "time_correlation_fraction"):
        value = float(gain[key])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"seismic_mismatch.gain.{key} must be positive.")
    wavelet = config["wavelet"]
    for key in ("phase_rotation_degrees", "time_shift_s"):
        values = list(wavelet[key])
        if not values or any(not np.isfinite(float(value)) for value in values):
            raise ValueError(
                f"seismic_mismatch.wavelet.{key} must contain finite values."
            )
    combined = config["combined"]
    for key in (
        "phase_rotation_degrees",
        "time_shift_s",
        "gain_log_sigma",
        "noise_rms_fraction",
    ):
        value = float(combined[key])
        if not np.isfinite(value):
            raise ValueError(f"seismic_mismatch.combined.{key} must be finite.")
    if (
        float(combined["gain_log_sigma"]) < 0.0
        or float(combined["noise_rms_fraction"]) < 0.0
    ):
        raise ValueError(
            "seismic_mismatch.combined gain/noise magnitudes must be nonnegative."
        )


