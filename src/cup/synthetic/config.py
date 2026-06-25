"""Configuration parsing and validation for synthoseis-lite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cup.synthetic.calibration import GENERATOR_FAMILY
from cup.utils.io import resolve_relative_path


DATA_SCHEMA = "synthoseis_lite_v1"
IMPLEMENTATION_SCOPE = "impedance_truth_frequency_probes_forward_qc_lfm_and_seismic_mismatch"


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)

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
    sources = _mapping(root.get("source_runs"), path="synthoseis_lite.source_runs")
    source_runs = {
        key: _required_text(sources, key, path="synthoseis_lite.source_runs")
        for key in (
            "forward_observability_dir",
            "well_preprocess_dir",
            "well_auto_tie_dir",
            "wavelet_generation_dir",
        )
    }
    sampling = _mapping(root.get("sampling") or {}, path="synthoseis_lite.sampling")
    geometry = _mapping(root.get("geometry") or {}, path="synthoseis_lite.geometry")
    field = _mapping(geometry.get("field_conditioned") or {}, path="synthoseis_lite.geometry.field_conditioned")
    canonical = _mapping(geometry.get("canonical") or {}, path="synthoseis_lite.geometry.canonical")
    target_zone = _mapping(field.get("target_zone") or {}, path="synthoseis_lite.geometry.field_conditioned.target_zone")
    if "horizons" in field:
        raise ValueError("synthoseis_lite.geometry.field_conditioned.horizons is retired; use top-level target_interval.horizons.")
    if "sections" in field:
        raise ValueError("synthoseis_lite.geometry.field_conditioned.sections is retired; use synthoseis_lite.sections.")
    target_interval = _mapping(config.get("target_interval"), path="target_interval")
    horizons = target_interval.get("horizons")
    if root.get("sections_file") is not None:
        raise ValueError("synthoseis_lite.sections_file is retired; put sections in synthoseis_lite.sections.")
    sections = root.get("sections")
    if not isinstance(horizons, list) or len(horizons) < 2:
        raise ValueError("target_interval.horizons needs at least two entries.")
    if not isinstance(sections, list) or not sections:
        raise ValueError("synthoseis_lite.sections must be non-empty.")
    horizon_items = []
    for index, item in enumerate(horizons):
        item = _mapping(item, path=f"synthoseis_lite.geometry.field_conditioned.horizons[{index}]")
        horizon_items.append(
            {
                "name": _required_text(item, "name", path=f"horizons[{index}]"),
                "file": _required_text(item, "file", path=f"horizons[{index}]"),
            }
        )
    section_items = []
    for index, item in enumerate(sections):
        item = _mapping(item, path=f"synthoseis_lite.sections[{index}]")
        path_points = item.get("path")
        if not isinstance(path_points, list) or len(path_points) < 2:
            raise ValueError(f"sections[{index}].path needs at least two points.")
        section_items.append(
            {
                "section_id": _required_text(item, "section_id", path=f"sections[{index}]"),
                "path": [
                    {
                        "inline": float(_mapping(point, path="section.path point")["inline"]),
                        "xline": float(_mapping(point, path="section.path point")["xline"]),
                    }
                    for point in path_points
                ],
            }
        )
    impedance = dict(root.get("impedance_attribute_generator") or {})
    lateral = dict(impedance.get("lateral") or {})
    qc = dict(impedance.get("qc") or {})
    robust_scale = dict(impedance.get("robust_scale") or {})
    generation = dict(root.get("generation") or {})
    forward_qc = dict(root.get("forward_qc") or {})
    highres_mismatch = dict(forward_qc.get("highres_mismatch") or {})
    probe_selection = dict(root.get("probe_selection") or {})
    figures = dict(root.get("figures") or {})
    report_examples = dict(figures.get("report_examples") or {})
    lfm = dict(root.get("lfm") or {})
    lfm_ideal = dict(lfm.get("ideal") or {})
    lfm_degraded = dict(lfm.get("controlled_degraded") or {})
    lfm_over_smoothing = dict(lfm_degraded.get("over_smoothing") or {})
    lfm_missing = dict(lfm_degraded.get("local_missing_control_bias") or {})
    mismatch = dict(root.get("seismic_mismatch") or {})
    mismatch_noise = dict(mismatch.get("noise") or {})
    mismatch_gain = dict(mismatch.get("gain") or {})
    mismatch_wavelet = dict(mismatch.get("wavelet") or {})
    mismatch_combined = dict(mismatch.get("combined") or {})
    raw_lateral_shapes = probe_selection.get(
        "lateral_shapes",
        [
            "section_coherent",
            {
                "name": "localized_tukey",
                "centered_fraction": 0.40,
                "alpha": 0.5,
            },
        ],
    )
    lateral_shapes = []
    for index, item in enumerate(raw_lateral_shapes):
        if isinstance(item, str):
            lateral_shapes.append({"name": item})
        else:
            lateral_shapes.append(
                _mapping(item, path=f"synthoseis_lite.probe_selection.lateral_shapes[{index}]")
            )
    parsed = {
        "global_seed": int(root.get("global_seed", 20260615)),
        "source_runs": source_runs,
        "sampling": {
            "expected_output_dt_s": float(sampling.get("expected_output_dt_s", 0.002)),
            "vertical_oversampling_factor": int(sampling.get("vertical_oversampling_factor", 8)),
        },
        "horizons": horizon_items,
        "sections": section_items,
        "lateral_sample_interval_m": float(geometry.get("lateral_sample_interval_m", 25.0)),
        "target_zone": {
            "mode": str(target_zone.get("mode", "filled_target_zone")),
            "nearest_distance_limit": _optional_float(target_zone, "nearest_distance_limit"),
            "outlier_threshold": _optional_float(target_zone, "outlier_threshold"),
            "outlier_min_neighbor_count": int(target_zone.get("outlier_min_neighbor_count", 2)),
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
            "vertical_extent_periods": float(canonical.get("vertical_extent_periods", 6.0)),
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
            "state_threshold_sigma": float(impedance.get("state_threshold_sigma", 1.0)),
            "huber_delta_parent_sigma_floor": float(
                robust_scale.get("huber_delta_parent_sigma_floor", 0.05)
            ),
            "coefficient_sigma_parent_floor": float(
                robust_scale.get("coefficient_sigma_parent_floor", 0.05)
            ),
            "coefficient_sigma_parent_cap": float(
                robust_scale.get("coefficient_sigma_parent_cap", 3.0)
            ),
            "correlation_length_section_fractions": [
                float(value) for value in lateral.get("correlation_length_section_fractions", [0.1, 0.3, 1.0])
            ],
            "coefficient_sigma_multipliers": [
                float(value) for value in lateral.get("coefficient_sigma_multipliers", [0.25, 0.5])
            ],
            "thickness_log_sigma_values": [
                float(value) for value in lateral.get("thickness_log_sigma_values", [0.10, 0.25])
            ],
            "max_global_reversal_fraction": float(qc.get("max_global_reversal_fraction", 0.10)),
            "max_object_reversal_fraction": float(qc.get("max_object_reversal_fraction", 0.25)),
            "max_global_clipping_fraction": float(qc.get("max_global_clipping_fraction", 0.005)),
            "max_object_clipping_fraction": float(qc.get("max_object_clipping_fraction", 0.02)),
            "minimum_attempts_per_scenario": int(qc.get("minimum_attempts_per_scenario", 20)),
            "scenario_acceptance_warning_fraction": float(
                qc.get("scenario_acceptance_warning_fraction", 0.80)
            ),
            "scenario_acceptance_failure_fraction": float(
                qc.get("scenario_acceptance_failure_fraction", 0.50)
            ),
        },
        "generation": {
            "attempts_per_scenario": int(generation.get("attempts_per_scenario", 20)),
            "duration_modes": list(generation.get("duration_modes", ["standard"])),
            "geometry_families": list(generation.get("geometry_families", ["none", "wedge", "pinchout"])),
            "geometry_directions": list(
                generation.get("geometry_directions", ["left_to_right", "right_to_left"])
            ),
        },
        "forward_qc": {
            "highres_mismatch_enabled": bool(
                highres_mismatch.get("enabled", True)
            ),
            "highres_mismatch_required": bool(
                highres_mismatch.get("required", False)
            ),
        },
        "lfm": {
            "enabled": bool(lfm.get("enabled", True)),
            "ideal": {
                "cutoff_hz": float(lfm_ideal.get("cutoff_hz", 10.0)),
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
                "over_smoothing": {
                    "cutoff_hz": float(lfm_over_smoothing.get("cutoff_hz", 6.0)),
                    "numtaps": int(lfm_over_smoothing.get("numtaps", 129)),
                    "kaiser_beta": float(
                        lfm_over_smoothing.get("kaiser_beta", 8.6)
                    ),
                    "blend": float(lfm_over_smoothing.get("blend", 1.0)),
                },
                "local_missing_control_bias": {
                    "enabled": bool(lfm_missing.get("enabled", True)),
                    "max_abs_log_ai": float(
                        lfm_missing.get("max_abs_log_ai", 0.04)
                    ),
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
                "time_shift_samples": [
                    float(value)
                    for value in mismatch_wavelet.get(
                        "time_shift_samples",
                        [-0.5, 0.5],
                    )
                ],
            },
            "combined": {
                "enabled": bool(mismatch_combined.get("enabled", True)),
                "phase_rotation_degrees": float(
                    mismatch_combined.get("phase_rotation_degrees", 10.0)
                ),
                "time_shift_samples": float(
                    mismatch_combined.get("time_shift_samples", 0.5)
                ),
                "gain_log_sigma": float(mismatch_combined.get("gain_log_sigma", 0.10)),
                "noise_rms_fraction": float(
                    mismatch_combined.get("noise_rms_fraction", 0.05)
                ),
            },
        },
        "probe_selection": {
            "enabled": bool(probe_selection.get("enabled", True)),
            "weak_representatives_per_band": int(
                probe_selection.get("weak_representatives_per_band", 3)
            ),
            "unsupported_representatives_per_band": int(
                probe_selection.get("unsupported_representatives_per_band", 3)
            ),
            "minimum_noise_equivalent_clusters": int(
                probe_selection.get("minimum_noise_equivalent_clusters", 3)
            ),
            "low_probe_energy_warning_fraction": float(
                probe_selection.get("low_probe_energy_warning_fraction", 0.01)
            ),
            "conservative_to_nominal_warning_ratio": float(
                probe_selection.get(
                    "conservative_to_nominal_warning_ratio",
                    1.5,
                )
            ),
            "vertical_tukey_alpha": float(
                probe_selection.get("vertical_tukey_alpha", 0.5)
            ),
            "amplitude_multipliers": [
                float(value)
                for value in probe_selection.get(
                    "amplitude_multipliers",
                    [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
                )
            ],
            "phases": [
                str(value)
                for value in probe_selection.get("phases", ["sin", "cos"])
            ],
            "lateral_shapes": lateral_shapes,
            "field_parent_geometry_family": str(
                probe_selection.get("field_parent_geometry_family", "none")
            ),
            "field_parents_per_section": int(
                probe_selection.get("field_parents_per_section", 1)
            ),
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
    _validate_lfm_config(parsed["lfm"], output_dt_s=parsed["sampling"]["expected_output_dt_s"])
    _validate_seismic_mismatch_config(parsed["seismic_mismatch"])
    _validate_probe_config(parsed["probe_selection"])
    return parsed


def resolve_sources(script_cfg: Mapping[str, Any], *, repo_root: Path) -> dict[str, Path]:
    sources = {
        key: resolve_relative_path(value, root=repo_root)
        for key, value in script_cfg["source_runs"].items()
    }
    required = {
        "forward_observability_dir": [
            "run_summary.json",
            "frequency_evidence_bands.csv",
            "well_frequency_sensitivity.csv",
        ],
        "well_preprocess_dir": ["well_preprocess_status.csv"],
        "well_auto_tie_dir": ["well_tie_metrics.csv"],
        "wavelet_generation_dir": [
            "selected_wavelet.csv",
            "selected_wavelet_summary.json",
            "evaluation_well_spatial_clusters.csv",
        ],
    }
    for key, names in required.items():
        directory = sources[key]
        if not directory.is_dir():
            raise FileNotFoundError(f"Source run does not exist: {directory}")
        missing = [name for name in names if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{key} is missing {missing}: {directory}")
    with (sources["forward_observability_dir"] / "run_summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        summary = json.load(handle)
    recorded = summary.get("source_runs") or {}
    for key in ("well_preprocess_dir", "well_auto_tie_dir", "wavelet_generation_dir"):
        if key not in recorded:
            raise ValueError(f"source_run_mismatch: observability summary lacks {key}")
        if resolve_relative_path(recorded[key], root=repo_root).resolve() != sources[key].resolve():
            raise ValueError(f"source_run_mismatch:{key}")
    return sources

def _validate_canonical_config(config: Mapping[str, Any]) -> None:
    if int(config["lateral_samples"]) < 8:
        raise ValueError("synthoseis_lite.geometry.canonical.lateral_samples must be >= 8.")
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
        if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
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
        raise ValueError("lfm.controlled_degraded.lateral_correlation_fraction must be positive.")
    smoothing = degraded["over_smoothing"]
    over_cutoff = float(smoothing["cutoff_hz"])
    if not 0.0 < over_cutoff <= cutoff:
        raise ValueError("lfm over_smoothing.cutoff_hz must be within (0, ideal cutoff].")
    if int(smoothing["numtaps"]) < 3:
        raise ValueError("lfm over_smoothing.numtaps must be >= 3.")
    if not 0.0 <= float(smoothing["blend"]) <= 1.0:
        raise ValueError("lfm over_smoothing.blend must be within [0, 1].")
    missing = degraded["local_missing_control_bias"]
    if float(missing["max_abs_log_ai"]) < 0.0:
        raise ValueError("lfm local_missing_control_bias.max_abs_log_ai must be nonnegative.")
    for key in ("lateral_width_fraction", "twt_width_fraction"):
        if not 0.0 < float(missing[key]) <= 1.0:
            raise ValueError(f"lfm local_missing_control_bias.{key} must be within (0, 1].")

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
        raise ValueError("seismic_mismatch.noise.colored_time_correlation_samples must be positive.")
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
    for key in ("phase_rotation_degrees", "time_shift_samples"):
        values = list(wavelet[key])
        if not values or any(not np.isfinite(float(value)) for value in values):
            raise ValueError(f"seismic_mismatch.wavelet.{key} must contain finite values.")
    combined = config["combined"]
    for key in (
        "phase_rotation_degrees",
        "time_shift_samples",
        "gain_log_sigma",
        "noise_rms_fraction",
    ):
        value = float(combined[key])
        if not np.isfinite(value):
            raise ValueError(f"seismic_mismatch.combined.{key} must be finite.")
    if float(combined["gain_log_sigma"]) < 0.0 or float(combined["noise_rms_fraction"]) < 0.0:
        raise ValueError("seismic_mismatch.combined gain/noise magnitudes must be nonnegative.")

def _validate_probe_config(config: Mapping[str, Any]) -> None:
    if int(config["weak_representatives_per_band"]) < 1:
        raise ValueError("weak_representatives_per_band must be positive.")
    if int(config["unsupported_representatives_per_band"]) < 1:
        raise ValueError("unsupported_representatives_per_band must be positive.")
    if int(config["minimum_noise_equivalent_clusters"]) < 1:
        raise ValueError("minimum_noise_equivalent_clusters must be positive.")
    if int(config["field_parents_per_section"]) < 1:
        raise ValueError("field_parents_per_section must be positive.")
    if str(config["field_parent_geometry_family"]) not in {
        "none",
        "wedge",
        "pinchout",
    }:
        raise ValueError(
            "field_parent_geometry_family must be none, wedge, or pinchout."
        )
    if not 0.0 <= float(config["vertical_tukey_alpha"]) <= 1.0:
        raise ValueError("vertical_tukey_alpha must be within [0, 1].")
    multipliers = list(config["amplitude_multipliers"])
    if (
        not multipliers
        or 0.0 not in multipliers
        or any(not np.isfinite(value) or value < 0.0 for value in multipliers)
    ):
        raise ValueError("probe amplitude_multipliers must be finite, nonnegative, and include 0.")
    if set(config["phases"]) != {"sin", "cos"}:
        raise ValueError("probe phases must contain exactly sin and cos.")
    names = [str(item.get("name", "")) for item in config["lateral_shapes"]]
    if set(names) != {"section_coherent", "localized_tukey"}:
        raise ValueError(
            "probe lateral_shapes must contain section_coherent and localized_tukey."
        )
    localized = next(
        item
        for item in config["lateral_shapes"]
        if str(item["name"]) == "localized_tukey"
    )
    if not 0.0 < float(localized.get("centered_fraction", 0.0)) <= 1.0:
        raise ValueError("localized_tukey centered_fraction must be within (0, 1].")
    if not 0.0 <= float(localized.get("alpha", -1.0)) <= 1.0:
        raise ValueError("localized_tukey alpha must be within [0, 1].")
