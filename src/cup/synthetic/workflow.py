"""Project-data adapters for the synthoseis-lite impedance-truth slice."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import scipy

from cup.petrel.load import (
    import_interpretation_petrel,
    import_well_tops_petrel,
)
from cup.seismic.survey import open_survey
from cup.seismic.target_zone import TargetZone
from cup.seismic.wavelet import (
    infer_wavelet_dt,
    load_wavelet_csv,
    validate_wavelet_normalization,
)
from cup.synthetic.calibration import (
    GENERATOR_FAMILY,
    SCHEMA_VERSION as CALIBRATION_SCHEMA,
    ImpedanceCalibration,
    WellZoneCurves,
    calibrate_impedance,
)
from cup.synthetic.canonical import (
    CANONICAL_FAMILIES,
    canonical_reference_impedance,
    canonical_scenarios,
    generate_canonical_section,
)
from cup.synthetic.generation import (
    GenerationRejected,
    GenerationScenario,
    antialias_taps,
    generate_field_section,
)
from cup.synthetic.forward import (
    HighresForwardResult,
    HighresWavelet,
    highres_forward_to_model_grid,
    model_grid_closure_qc,
    resample_wavelet_to_highres,
)
from cup.synthetic.figures import (
    write_calibration_figures,
    write_generation_figures,
)
from cup.synthetic.io import (
    sha256_file,
    write_generated_section,
    write_highres_forward_result,
    write_lfm_result,
    write_probe_result,
    write_seismic_variant_result,
)
from cup.synthetic.lfm import LfmResult, derive_lfm_priors
from cup.synthetic.probes import (
    ProbeFrequency,
    build_probe_frequency_catalog,
    frequency_catalog_rows,
    generate_probe,
    probe_variants,
)
from cup.synthetic.seismic_variants import (
    SeismicVariantResult,
    generate_seismic_variants,
)
from cup.time_config import TimeWorkflowConfig
from cup.utils.io import (
    repo_relative_path,
    resolve_artifact_path,
    resolve_relative_path,
    write_json,
)
from cup.well.assets import normalize_well_name
from cup.well.las import read_las_curve
from cup.well.td import find_well_top_md, load_workflow_time_depth_table_csv


DATA_SCHEMA = "synthoseis_lite_v1"
IMPLEMENTATION_SCOPE = "impedance_truth_frequency_probes_forward_qc_lfm_and_seismic_mismatch"


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest()


@dataclass(frozen=True)
class SectionGeometry:
    section_id: str
    lateral_m: np.ndarray
    inline_float: np.ndarray
    xline_float: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    horizon_twt_s: np.ndarray
    qc_rows: tuple[dict[str, Any], ...] = ()


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
    sampling = _mapping(root.get("sampling"), path="synthoseis_lite.sampling")
    geometry = _mapping(root.get("geometry"), path="synthoseis_lite.geometry")
    field = _mapping(geometry.get("field_conditioned"), path="synthoseis_lite.geometry.field_conditioned")
    canonical = _mapping(geometry.get("canonical") or {}, path="synthoseis_lite.geometry.canonical")
    target_zone = _mapping(field.get("target_zone") or {}, path="synthoseis_lite.geometry.field_conditioned.target_zone")
    horizons = field.get("horizons")
    sections = field.get("sections")
    if not isinstance(horizons, list) or len(horizons) < 2:
        raise ValueError("synthoseis_lite.geometry.field_conditioned.horizons needs at least two entries.")
    if not isinstance(sections, list) or not sections:
        raise ValueError("synthoseis_lite.geometry.field_conditioned.sections must be non-empty.")
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
        item = _mapping(item, path=f"synthoseis_lite.geometry.field_conditioned.sections[{index}]")
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


def _artifact(value: Any, *, run_dir: Path, repo_root: Path, label: str) -> Path:
    path = resolve_artifact_path(value, root=repo_root, run_dir=run_dir)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _piecewise_cell_average(
    md_m: np.ndarray,
    ai: np.ndarray,
    *,
    table: Any,
    centers_s: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    md = np.asarray(md_m, dtype=np.float64).reshape(-1)
    values = np.asarray(ai, dtype=np.float64).reshape(-1)
    centers = np.asarray(centers_s, dtype=np.float64).reshape(-1)
    result = np.full(centers.shape, np.nan, dtype=np.float64)
    finite = (
        np.isfinite(md)
        & np.isfinite(values)
        & (values > 0.0)
        & (md >= float(table.md[0]))
        & (md <= float(table.md[-1]))
    )
    changes = np.diff(np.r_[False, finite, False].astype(np.int8))
    runs = zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
    half = 0.5 * float(dt_s)
    for start, end in runs:
        if end - start < 2:
            continue
        run_md = md[start:end]
        run_twt = np.interp(run_md, table.md, table.twt)
        run_values = np.log(values[start:end])
        if np.any(np.diff(run_twt) <= 0.0):
            continue
        candidates = np.flatnonzero(
            (centers - half >= run_twt[0]) & (centers + half <= run_twt[-1])
        )
        for index in candidates:
            left = centers[index] - half
            right = centers[index] + half
            interior = (run_twt > left) & (run_twt < right)
            x = np.r_[left, run_twt[interior], right]
            y = np.r_[
                np.interp(left, run_twt, run_values),
                run_values[interior],
                np.interp(right, run_twt, run_values),
            ]
            result[index] = float(np.trapezoid(y, x) / dt_s)
    return result


def build_calibration_inputs(
    *,
    workflow: TimeWorkflowConfig,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    repo_root: Path,
) -> tuple[list[WellZoneCurves], dict[str, Any]]:
    clusters = pd.read_csv(sources["wavelet_generation_dir"] / "evaluation_well_spatial_clusters.csv")
    metrics = pd.read_csv(sources["well_auto_tie_dir"] / "well_tie_metrics.csv")
    preprocess = pd.read_csv(sources["well_preprocess_dir"] / "well_preprocess_status.csv")
    for frame, column in ((clusters, "well_name"), (metrics, "well_name"), (preprocess, "well_name")):
        frame["_well_key"] = frame[column].map(normalize_well_name)
        if frame["_well_key"].duplicated().any():
            raise ValueError(f"Duplicate normalized well name in {column} table.")
    wells = clusters.merge(metrics, on=["well_name", "_well_key"], validate="one_to_one").merge(
        preprocess[["_well_key", "preprocess_status", "preprocessed_las"]],
        on="_well_key",
        validate="one_to_one",
    )
    well_tops = import_well_tops_petrel(
        resolve_relative_path(workflow.assets.well_tops_file, root=resolve_relative_path(workflow.data_root, root=repo_root))
    )
    wavelet_time, _ = load_wavelet_csv(sources["wavelet_generation_dir"] / "selected_wavelet.csv")
    output_dt = infer_wavelet_dt(wavelet_time)
    expected = float(script_cfg["sampling"]["expected_output_dt_s"])
    if not np.isclose(output_dt, expected, rtol=0.0, atol=1e-9):
        raise ValueError(f"sampling_mismatch:wavelet_dt={output_dt}:expected={expected}")
    truth_dt = output_dt / int(script_cfg["sampling"]["vertical_oversampling_factor"])
    ordered_horizons = [item["name"] for item in script_cfg["horizons"]]
    records: list[WellZoneCurves] = []
    well_status: list[dict[str, Any]] = []
    for row in wells.to_dict(orient="records"):
        well_name = str(row["well_name"])
        try:
            if str(row.get("tie_status", "")).casefold() != "success":
                raise ValueError("tie_status_not_success")
            if str(row.get("preprocess_status", "")).casefold() != "passed":
                raise ValueError("preprocess_status_not_passed")
            filtered_path = _artifact(
                row.get("filtered_las_file"),
                run_dir=sources["well_auto_tie_dir"],
                repo_root=repo_root,
                label=f"{well_name} filtered LAS",
            )
            full_path = _artifact(
                row.get("preprocessed_las"),
                run_dir=sources["well_preprocess_dir"],
                repo_root=repo_root,
                label=f"{well_name} preprocessed LAS",
            )
            table_path = _artifact(
                row.get("optimized_tdt_file"),
                run_dir=sources["well_auto_tie_dir"],
                repo_root=repo_root,
                label=f"{well_name} optimized TDT",
            )
            table = load_workflow_time_depth_table_csv(table_path)
            if not table.is_md_domain:
                raise ValueError("optimized_tdt_not_md_domain")
            filtered = read_las_curve(filtered_path, "AI", match_policy="exact", allow_all_nan=True)
            full = read_las_curve(full_path, "AI", match_policy="exact", allow_all_nan=True)
            horizon_times = []
            for horizon in ordered_horizons:
                md = find_well_top_md(well_tops, well_name=well_name, surface=horizon)
                if not float(table.md[0]) <= md <= float(table.md[-1]):
                    raise ValueError(f"outside_tdt_support:{horizon}")
                horizon_times.append(float(np.interp(md, table.md, table.twt)))
            if np.any(np.diff(horizon_times) <= 0.0):
                raise ValueError("misordered_horizons")
            zone_count = 0
            for zone_index, (top, bottom) in enumerate(zip(horizon_times[:-1], horizon_times[1:])):
                centers = np.arange(top + 0.5 * truth_dt, bottom, truth_dt, dtype=np.float64)
                filtered_values = _piecewise_cell_average(
                    filtered.basis,
                    filtered.values,
                    table=table,
                    centers_s=centers,
                    dt_s=truth_dt,
                )
                full_values = _piecewise_cell_average(
                    full.basis,
                    full.values,
                    table=table,
                    centers_s=centers,
                    dt_s=truth_dt,
                )
                valid = np.isfinite(filtered_values) & np.isfinite(full_values)
                if np.count_nonzero(valid) < 3:
                    continue
                top_name, bottom_name = ordered_horizons[zone_index : zone_index + 2]
                records.append(
                    WellZoneCurves(
                        well_name=well_name,
                        spatial_cluster_id=int(row["spatial_cluster_id"]),
                        zone_id=f"{top_name}__to__{bottom_name}",
                        top_horizon=top_name,
                        bottom_horizon=bottom_name,
                        twt_s=centers[valid],
                        filtered_log_ai=filtered_values[valid],
                        full_log_ai=full_values[valid],
                        zone_top_s=top,
                        zone_bottom_s=bottom,
                    )
                )
                zone_count += 1
            if zone_count == 0:
                raise ValueError("no_valid_zones")
            well_status.append({"well_name": well_name, "status": "ok", "n_zones": zone_count, "reasons": ""})
        except Exception as exc:
            well_status.append(
                {
                    "well_name": well_name,
                    "status": "rejected",
                    "n_zones": 0,
                    "reasons": f"{type(exc).__name__}:{exc}",
                }
            )
    return records, {"well_status": well_status, "output_dt_s": output_dt, "truth_dt_s": truth_dt}


def run_calibration(
    *,
    workflow: TimeWorkflowConfig,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    inputs, input_qc = build_calibration_inputs(
        workflow=workflow,
        script_cfg=script_cfg,
        sources=sources,
        repo_root=repo_root,
    )
    source_hashes = {}
    for directory_key, names in {
        "forward_observability_dir": ["run_summary.json", "frequency_evidence_bands.csv"],
        "well_preprocess_dir": ["well_preprocess_status.csv"],
        "well_auto_tie_dir": ["well_tie_metrics.csv"],
        "wavelet_generation_dir": ["selected_wavelet.csv", "evaluation_well_spatial_clusters.csv"],
    }.items():
        for name in names:
            source_hashes[f"{directory_key}/{name}"] = sha256_file(sources[directory_key] / name)
    calibration, objects, qc, samples, backgrounds, profile_samples = calibrate_impedance(
        inputs,
        truth_dt_s=float(input_qc["truth_dt_s"]),
        ordered_horizons=[item["name"] for item in script_cfg["horizons"]],
        source_runs={
            key: repo_relative_path(path, root=repo_root) for key, path in sources.items()
        },
        source_hashes=source_hashes,
        state_threshold_sigma=float(script_cfg["impedance"]["state_threshold_sigma"]),
        huber_delta_parent_sigma_floor=float(
            script_cfg["impedance"]["huber_delta_parent_sigma_floor"]
        ),
        coefficient_sigma_parent_floor=float(
            script_cfg["impedance"]["coefficient_sigma_parent_floor"]
        ),
        coefficient_sigma_parent_cap=float(
            script_cfg["impedance"]["coefficient_sigma_parent_cap"]
        ),
    )
    objects_path = output_dir / "well_object_catalog.csv"
    qc_path = output_dir / "calibration_qc.csv"
    status_path = output_dir / "well_status.csv"
    samples_path = output_dir / "well_calibration_samples.csv"
    backgrounds_path = output_dir / "well_background_fits.csv"
    profile_samples_path = output_dir / "well_object_profile_samples.csv"
    objects.to_csv(objects_path, index=False)
    qc.to_csv(qc_path, index=False)
    samples.to_csv(samples_path, index=False)
    backgrounds.to_csv(backgrounds_path, index=False)
    profile_samples.to_csv(profile_samples_path, index=False)
    pd.DataFrame.from_records(input_qc["well_status"]).to_csv(status_path, index=False)
    payload = calibration.to_dict()
    payload["artifact_hashes"] = {
        "well_object_catalog.csv": sha256_file(objects_path),
        "calibration_qc.csv": sha256_file(qc_path),
        "well_calibration_samples.csv": sha256_file(samples_path),
        "well_background_fits.csv": sha256_file(backgrounds_path),
        "well_object_profile_samples.csv": sha256_file(profile_samples_path),
    }
    write_json(output_dir / "impedance_calibration.json", payload)
    figure_summary = write_calibration_figures(
        output_dir,
        script_cfg.get("figures", {}),
    )
    summary = {
        "schema_version": CALIBRATION_SCHEMA,
        "generator_family": GENERATOR_FAMILY,
        "implementation_scope": IMPLEMENTATION_SCOPE,
        "source_runs": calibration.source_runs,
        "n_well_zone_inputs": len(inputs),
        "n_objects": int(len(objects)),
        "well_status_counts": pd.Series(
            [row["status"] for row in input_qc["well_status"]]
        ).value_counts().to_dict(),
        "outputs": {
            "impedance_calibration": repo_relative_path(
                output_dir / "impedance_calibration.json", root=repo_root
            ),
            "well_object_catalog": repo_relative_path(objects_path, root=repo_root),
            "calibration_qc": repo_relative_path(qc_path, root=repo_root),
            "well_calibration_samples": repo_relative_path(samples_path, root=repo_root),
            "well_background_fits": repo_relative_path(backgrounds_path, root=repo_root),
            "well_object_profile_samples": repo_relative_path(profile_samples_path, root=repo_root),
        },
        "figures": {
            "generated_count": int(figure_summary.get("generated_count", 0)),
            "skipped_count": int(figure_summary.get("skipped_count", 0)),
            "figure_manifest": repo_relative_path(
                Path(str(figure_summary.get("figure_manifest", output_dir / "figures" / "figure_manifest.json"))),
                root=repo_root,
            ),
        },
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary


def load_calibration(path: Path) -> ImpedanceCalibration:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ImpedanceCalibration.from_dict(json.load(handle))


def _resample_section_path(
    points: Sequence[Mapping[str, float]],
    *,
    geometry: Any,
    sample_interval_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertex_lines = np.asarray([[float(point["inline"]), float(point["xline"])] for point in points])
    vertex_xy = np.asarray(
        [geometry.line_to_coord(inline, xline) for inline, xline in vertex_lines],
        dtype=np.float64,
    )
    segment_lengths = np.linalg.norm(np.diff(vertex_xy, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    total = float(cumulative[-1])
    if total <= 0.0:
        raise ValueError("invalid_section_path")
    lateral = np.arange(0.0, total, float(sample_interval_m), dtype=np.float64)
    if lateral.size == 0 or not np.isclose(lateral[-1], total):
        lateral = np.r_[lateral, total]
    x = np.interp(lateral, cumulative, vertex_xy[:, 0])
    y = np.interp(lateral, cumulative, vertex_xy[:, 1])
    lines = np.asarray([geometry.coord_to_line(xi, yi) for xi, yi in zip(x, y)])
    return lateral, lines[:, 0], lines[:, 1], x, y


def _nearest_grid_index(value: float, *, minimum: float, step: float, size: int) -> int:
    index = int(round((float(value) - float(minimum)) / float(step)))
    return max(0, min(index, int(size) - 1))


def _section_target_zone_qc_rows(
    *,
    section_id: str,
    lateral_m: np.ndarray,
    inline_float: np.ndarray,
    xline_float: np.ndarray,
    horizon_values: np.ndarray,
    target_zone: TargetZone,
    horizon_names: Sequence[str],
    geometry_mode: str,
) -> list[dict[str, Any]]:
    geometry = target_zone.geometry
    inline_min = float(geometry["inline_min"])
    xline_min = float(geometry["xline_min"])
    inline_step = float(geometry["inline_step"])
    xline_step = float(geometry["xline_step"])
    inline_size, xline_size = target_zone.valid_control_mask.shape
    rows: list[dict[str, Any]] = []
    for sample_index, (distance, il, xl) in enumerate(zip(lateral_m, inline_float, xline_float)):
        i = _nearest_grid_index(il, minimum=inline_min, step=inline_step, size=inline_size)
        j = _nearest_grid_index(xl, minimum=xline_min, step=xline_step, size=xline_size)
        trace_valid_control = bool(target_zone.valid_control_mask[i, j])
        trace_filled_model = bool(target_zone.filled_model_mask[i, j])
        trace_filled_by_thickness = bool(trace_filled_model and not trace_valid_control)
        for horizon_index, horizon_name in enumerate(horizon_names):
            surface = target_zone.get_horizon_surface(str(horizon_name))
            sample = surface.sample_at_line(float(il), float(xl))
            rows.append(
                {
                    "section_id": section_id,
                    "geometry_mode": geometry_mode,
                    "sample_index": int(sample_index),
                    "lateral_m": float(distance),
                    "inline_float": float(il),
                    "xline_float": float(xl),
                    "nearest_grid_inline": float(inline_min + i * inline_step),
                    "nearest_grid_xline": float(xline_min + j * xline_step),
                    "nearest_grid_index_inline": int(i),
                    "nearest_grid_index_xline": int(j),
                    "horizon_name": str(horizon_name),
                    "horizon_twt_s": float(horizon_values[sample_index, horizon_index]),
                    "horizon_sample_method": str(sample.method),
                    "horizon_support_status": str(sample.support_status),
                    "raw_pick": bool(target_zone.raw_pick_masks[str(horizon_name)][i, j]),
                    "linear_support": bool(
                        target_zone.interpolation_support_masks[str(horizon_name)][i, j]
                    ),
                    "nearest_distance_grid": float(
                        target_zone.nearest_distance_grids[str(horizon_name)][i, j]
                    ),
                    "trace_valid_control": trace_valid_control,
                    "trace_filled_model": trace_filled_model,
                    "trace_filled_by_thickness_interpolation": trace_filled_by_thickness,
                    "trace_no_support": bool(target_zone.no_support_mask[i, j]),
                    "trace_crossing": bool(target_zone.crossing_mask[i, j]),
                    "trace_thin": bool(target_zone.thin_mask[i, j]),
                }
            )
    return rows


def build_section_geometries(
    *,
    workflow: TimeWorkflowConfig,
    script_cfg: Mapping[str, Any],
    repo_root: Path,
) -> list[SectionGeometry]:
    seismic_path = resolve_relative_path(
        workflow.seismic.file,
        root=resolve_relative_path(workflow.data_root, root=repo_root),
    )
    segy_options = {
        key: value
        for key, value in workflow.seismic.as_dict().items()
        if key in {"iline", "xline", "istep", "xstep"} and value is not None
    }
    survey = open_survey(
        seismic_path,
        workflow.seismic.type,
        segy_options=segy_options or None,
    )
    target_zone_cfg = dict(script_cfg.get("target_zone") or {})
    mode = str(target_zone_cfg.get("mode", "filled_target_zone"))
    if mode != "filled_target_zone":
        raise ValueError(
            "synthoseis_lite field-conditioned geometry only supports "
            f"filled_target_zone, got {mode!r}."
        )
    survey_geometry = survey.describe_geometry(domain="time")
    raw_horizon_dfs: dict[str, pd.DataFrame] = {}
    ordered_horizons = [str(item["name"]) for item in script_cfg["horizons"]]
    for horizon in script_cfg["horizons"]:
        path = resolve_relative_path(
            horizon["file"],
            root=resolve_relative_path(workflow.data_root, root=repo_root),
        )
        frame = import_interpretation_petrel(path)
        frame = frame.copy()
        frame["interpretation"] = np.abs(frame["interpretation"].to_numpy(dtype=np.float64))
        raw_horizon_dfs[str(horizon["name"])] = frame
    target_zone = TargetZone(
        raw_horizon_dfs,
        survey_geometry,
        ordered_horizons,
        nearest_distance_limit=target_zone_cfg.get("nearest_distance_limit"),
        outlier_threshold=target_zone_cfg.get("outlier_threshold"),
        outlier_min_neighbor_count=int(target_zone_cfg.get("outlier_min_neighbor_count", 2)),
        min_thickness=target_zone_cfg.get("min_thickness_s"),
    )
    surfaces = [target_zone.get_horizon_surface(name) for name in ordered_horizons]
    sections: list[SectionGeometry] = []
    for section in script_cfg["sections"]:
        lateral, inline, xline, x, y = _resample_section_path(
            section["path"],
            geometry=survey.line_geometry,
            sample_interval_m=float(script_cfg["lateral_sample_interval_m"]),
        )
        horizon_values = np.column_stack(
            [
                np.asarray(
                    [surface.value_at_line(il, xl) for il, xl in zip(inline, xline)],
                    dtype=np.float64,
                )
                for surface in surfaces
            ]
        )
        if np.any(np.diff(horizon_values, axis=1) <= 0.0):
            raise ValueError(f"crossing_horizons:{section['section_id']}")
        qc_rows = _section_target_zone_qc_rows(
            section_id=str(section["section_id"]),
            lateral_m=lateral,
            inline_float=inline,
            xline_float=xline,
            horizon_values=horizon_values,
            target_zone=target_zone,
            horizon_names=ordered_horizons,
            geometry_mode=mode,
        )
        sections.append(
            SectionGeometry(
                section_id=section["section_id"],
                lateral_m=lateral,
                inline_float=inline,
                xline_float=xline,
                x_m=x,
                y_m=y,
                horizon_twt_s=horizon_values,
                qc_rows=tuple(qc_rows),
            )
        )
    return sections


def generation_scenarios(script_cfg: Mapping[str, Any]) -> list[GenerationScenario]:
    generation = script_cfg["generation"]
    impedance = script_cfg["impedance"]
    scenarios: list[GenerationScenario] = []
    for duration_mode in generation["duration_modes"]:
        for correlation in impedance["correlation_length_section_fractions"]:
            pairs = zip(
                impedance["coefficient_sigma_multipliers"],
                impedance["thickness_log_sigma_values"],
            )
            for coefficient_sigma, thickness_sigma in pairs:
                for family in generation["geometry_families"]:
                    directions = generation["geometry_directions"] if family != "none" else ["none"]
                    variants = ["035", "065"] if family == "pinchout" else [""]
                    for direction in directions:
                        for variant in variants:
                            scenario_id = (
                                f"{duration_mode}__lx{correlation:g}__a{coefficient_sigma:g}"
                                f"__t{thickness_sigma:g}__{family}__{direction}"
                                + (f"__{variant}" if variant else "")
                            )
                            scenarios.append(
                                GenerationScenario(
                                    scenario_id=scenario_id,
                                    duration_mode=str(duration_mode),
                                    geometry_family=str(family),
                                    geometry_direction=str(direction),
                                    correlation_length_fraction=float(correlation),
                                    coefficient_sigma_multiplier=float(coefficient_sigma),
                                    thickness_log_sigma=float(thickness_sigma),
                                    variant_id=str(variant),
                                )
                            )
    return scenarios


def _load_probe_frequencies(
    *,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
) -> list[ProbeFrequency]:
    config = script_cfg["probe_selection"]
    if not bool(config["enabled"]):
        return []
    evidence = pd.read_csv(
        sources["forward_observability_dir"] / "frequency_evidence_bands.csv"
    )
    sensitivity = pd.read_csv(
        sources["forward_observability_dir"] / "well_frequency_sensitivity.csv"
    )
    return build_probe_frequency_catalog(
        evidence,
        sensitivity,
        weak_representatives_per_band=int(
            config["weak_representatives_per_band"]
        ),
        unsupported_representatives_per_band=int(
            config["unsupported_representatives_per_band"]
        ),
        minimum_clusters=int(config["minimum_noise_equivalent_clusters"]),
    )


def _section_forward_qc(
    section: Any,
    *,
    wavelet: np.ndarray,
    highres_wavelet: HighresWavelet | None,
    required: bool,
) -> tuple[HighresForwardResult | None, dict[str, Any]]:
    qc = model_grid_closure_qc(
        section.model_target_log_ai,
        section.seismic_model_consistent,
        wavelet,
    )
    if highres_wavelet is None:
        return None, {
            **qc,
            "highres_forward_status": "disabled",
            "highres_forward_reasons": "",
        }
    try:
        result = highres_forward_to_model_grid(
            section.truth_log_ai_highres,
            section.seismic_model_consistent,
            highres_wavelet=highres_wavelet,
            forward_valid_mask_model=section.forward_valid_mask_model,
        )
        return result, {
            **qc,
            **result.qc,
            "highres_forward_reasons": "",
        }
    except Exception as exc:
        if required:
            raise ValueError(f"highres_forward_qc_failed:{exc}") from exc
        return None, {
            **qc,
            "highres_forward_status": "failed",
            "highres_forward_reasons": f"{type(exc).__name__}:{exc}",
        }


def _lfm_records(result: LfmResult, *, base_path: str) -> dict[str, Any]:
    return {
        **result.qc,
        "lfm_versions": "ideal;controlled_degraded",
        "lfm_ideal_dataset": (
            "" if not base_path else f"{base_path}/priors/lfm_ideal"
        ),
        "lfm_controlled_degraded_dataset": (
            ""
            if not base_path
            else f"{base_path}/priors/lfm_controlled_degraded"
        ),
        "residual_vs_lfm_ideal_dataset": (
            ""
            if not base_path
            else f"{base_path}/residuals/residual_vs_lfm_ideal"
        ),
        "residual_vs_lfm_controlled_degraded_dataset": (
            ""
            if not base_path
            else f"{base_path}/residuals/residual_vs_lfm_controlled_degraded"
        ),
    }


def _seismic_variant_records_for_sample(
    *,
    h5: h5py.File,
    owner_path: str,
    source_index_record: Mapping[str, Any],
    seismic_model_consistent: np.ndarray,
    forward_valid_mask: np.ndarray,
    lateral_m: np.ndarray,
    script_cfg: Mapping[str, Any],
    qc_only: bool,
    source_variant_id: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = generate_seismic_variants(
        seismic_model_consistent=seismic_model_consistent,
        forward_valid_mask=forward_valid_mask,
        lateral_m=lateral_m,
        config=script_cfg["seismic_mismatch"],
        global_seed=int(script_cfg["global_seed"]),
        generator_family=GENERATOR_FAMILY,
        realization_id=str(source_index_record["parent_realization_id"]),
        source_variant_id=source_variant_id or str(source_index_record["sample_id"]),
    )
    index_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    source_sample_id = str(source_index_record["sample_id"])
    source_kind = str(source_index_record.get("sample_kind", "base"))
    for result in results:
        variant_group = (
            ""
            if qc_only
            else write_seismic_variant_result(
                h5,
                owner_path=owner_path,
                result=result,
            )
        )
        sample_id = f"{source_sample_id}__seismic__{result.variant_id}"
        sample_kind = (
            "frequency_probe_seismic_variant"
            if source_kind == "frequency_probe"
            else "seismic_variant"
        )
        record = {
            **dict(source_index_record),
            "sample_id": sample_id,
            "realization_id": sample_id,
            "source_sample_id": source_sample_id,
            "source_sample_kind": source_kind,
            "sample_kind": sample_kind,
            "hdf5_group": variant_group,
            "seismic_variant_id": result.variant_id,
            "seismic_mismatch_family": result.mismatch_family,
            "seismic_observed_dataset": (
                "" if not variant_group else f"{variant_group}/seismic_observed"
            ),
            "positive_gain_dataset": (
                "" if not variant_group else f"{variant_group}/positive_gain"
            ),
            "additive_noise_dataset": (
                "" if not variant_group else f"{variant_group}/additive_noise"
            ),
        }
        index_records.append(record)
        result_records.append({**record, **result.qc})
    return index_records, result_records


def _probe_records_for_parent(
    *,
    h5: h5py.File,
    parent_path: str,
    section: Any,
    suite: str,
    section_id: str,
    split: str,
    frequencies: Sequence[ProbeFrequency],
    script_cfg: Mapping[str, Any],
    wavelet: np.ndarray,
    highres_wavelet: HighresWavelet | None,
    base_highres_forward: HighresForwardResult | None,
    qc_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = script_cfg["probe_selection"]
    taps = antialias_taps(
        int(script_cfg["sampling"]["vertical_oversampling_factor"])
    )
    index_records: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    seismic_variant_records: list[dict[str, Any]] = []
    for frequency in frequencies:
        variants = probe_variants(
            frequency,
            amplitude_multipliers=config["amplitude_multipliers"],
            phases=config["phases"],
            lateral_shapes=config["lateral_shapes"],
        )
        for variant in variants:
            result = generate_probe(
                section,
                frequency,
                variant,
                wavelet=wavelet,
                antialias_filter_taps=taps,
                vertical_tukey_alpha=float(config["vertical_tukey_alpha"]),
                lateral_shapes=config["lateral_shapes"],
                low_probe_energy_warning_fraction=float(
                    config["low_probe_energy_warning_fraction"]
                ),
            )
            closure = model_grid_closure_qc(
                result.model_target_log_ai,
                result.seismic_model_consistent,
                wavelet,
            )
            lfm_result = derive_lfm_priors(
                section,
                config=script_cfg["lfm"],
                global_seed=int(script_cfg["global_seed"]),
                generator_family=GENERATOR_FAMILY,
                model_target_log_ai=result.model_target_log_ai,
                degradation_variant_id=section.realization_id,
            )
            if variant.amplitude_multiplier == 0.0:
                highres_result = base_highres_forward
                highres_qc = (
                    {}
                    if base_highres_forward is None
                    else base_highres_forward.qc
                )
            elif highres_wavelet is not None:
                try:
                    highres_result = highres_forward_to_model_grid(
                        section.truth_log_ai_highres
                        + result.probe_log_ai_highres,
                        result.seismic_model_consistent,
                        highres_wavelet=highres_wavelet,
                        forward_valid_mask_model=(
                            section.forward_valid_mask_model
                        ),
                    )
                    highres_qc = highres_result.qc
                except Exception as exc:
                    if bool(
                        script_cfg["forward_qc"][
                            "highres_mismatch_required"
                        ]
                    ):
                        raise ValueError(
                            f"highres_forward_qc_failed:"
                            f"{section.realization_id}:{variant.variant_id}:{exc}"
                        ) from exc
                    highres_result = None
                    highres_qc = {
                        "highres_forward_status": "failed",
                        "highres_forward_reasons": (
                            f"{type(exc).__name__}:{exc}"
                        ),
                    }
            else:
                highres_result = None
                highres_qc = {
                    "highres_forward_status": "disabled",
                    "highres_forward_reasons": "",
                }
            hdf5_group = (
                ""
                if qc_only
                else write_probe_result(
                    h5,
                    realization_path=parent_path,
                    frequency=frequency,
                    result=result,
                    highres_forward=highres_result,
                    lfm_result=lfm_result,
                )
            )
            sample_id = (
                f"{section.realization_id}__probe__{variant.variant_id}"
            )
            pair_id = (
                f"{section.realization_id}__probe__"
                f"{variant.paired_zero_variant_id}"
            )
            index_record = {
                "sample_id": sample_id,
                "realization_id": sample_id,
                "parent_realization_id": section.realization_id,
                "suite": suite,
                "section_id": section_id,
                "scenario_id": "frequency_probe",
                "geometry_family": section.scenario.geometry_family,
                "duration_mode": section.scenario.duration_mode,
                "split": split,
                "hdf5_group": hdf5_group,
                "attempt_id": "",
                "status": "ok",
                "reasons": "",
                "sample_kind": "frequency_probe",
                "probe_variant_id": variant.variant_id,
                "paired_zero_sample_id": pair_id,
                "probe_frequency_hz": frequency.frequency_hz,
                "probe_phase": variant.phase,
                "probe_lateral_shape": variant.lateral_shape,
                "probe_amplitude_multiplier": (
                    variant.amplitude_multiplier
                ),
                **_lfm_records(lfm_result, base_path=hdf5_group),
            }
            index_records.append(index_record)
            seismic_index, seismic_results = _seismic_variant_records_for_sample(
                h5=h5,
                owner_path=(
                    hdf5_group
                    if hdf5_group
                    else f"{parent_path}/probes/{variant.variant_id}"
                ),
                source_index_record=index_record,
                seismic_model_consistent=result.seismic_model_consistent,
                forward_valid_mask=section.forward_valid_mask_model,
                lateral_m=section.lateral_m,
                script_cfg=script_cfg,
                qc_only=qc_only,
                source_variant_id=variant.variant_id,
            )
            index_records.extend(seismic_index)
            seismic_variant_records.extend(seismic_results)
            result_records.append(
                {
                    **index_record,
                    "evidence_status": frequency.evidence_status,
                    "operator_support": frequency.operator_support,
                    "experiment_class": frequency.experiment_class,
                    "selection_reason": frequency.selection_reason,
                    "noise_equivalent_calibration_status": (
                        frequency.calibration_status
                    ),
                    "wavelet_uncertainty_warning": bool(
                        np.isfinite(
                            frequency.conservative_to_nominal_ratio
                        )
                        and frequency.conservative_to_nominal_ratio
                        > float(
                            config[
                                "conservative_to_nominal_warning_ratio"
                            ]
                        )
                    ),
                    "valid_nominal_cluster_count": (
                        frequency.valid_nominal_cluster_count
                    ),
                    "valid_conservative_cluster_count": (
                        frequency.valid_conservative_cluster_count
                    ),
                    **result.qc,
                    **closure,
                    **highres_qc,
                    **lfm_result.qc,
                }
            )
    return index_records, result_records, seismic_variant_records


def _run_canonical_generation(
    *,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    calibration: ImpedanceCalibration,
    calibration_path: Path,
    repo_root: Path,
    output_dir: Path,
    wavelet_time: np.ndarray,
    wavelet: np.ndarray,
    output_dt: float,
    qc_only: bool,
) -> dict[str, Any]:
    config = script_cfg["canonical"]
    if not bool(config["enabled"]):
        raise ValueError("synthoseis_lite.geometry.canonical.enabled is false.")
    scenarios = canonical_scenarios(config)
    index_records: list[dict[str, Any]] = []
    object_records: list[dict[str, Any]] = []
    object_lateral_records: list[dict[str, Any]] = []
    qc_records: list[dict[str, Any]] = []
    probe_records: list[dict[str, Any]] = []
    seismic_variant_records: list[dict[str, Any]] = []
    probe_frequencies = _load_probe_frequencies(
        script_cfg=script_cfg,
        sources=sources,
    )
    highres_wavelet = (
        resample_wavelet_to_highres(
            wavelet_time,
            wavelet,
            factor=int(
                script_cfg["sampling"]["vertical_oversampling_factor"]
            ),
        )
        if bool(
            script_cfg["forward_qc"]["highres_mismatch_enabled"]
        )
        else None
    )
    h5_path = output_dir / "synthetic_benchmark.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["schema_version"] = DATA_SCHEMA
        h5.attrs["generator_family"] = calibration.generator_family
        h5.attrs["implementation_scope"] = IMPLEMENTATION_SCOPE
        h5.attrs["suite"] = "canonical"
        for scenario in scenarios:
            generated = generate_canonical_section(
                calibration,
                scenario=scenario,
                config=config,
                output_dt_s=output_dt,
                wavelet_time_s=wavelet_time,
                wavelet=wavelet,
                vertical_oversampling_factor=int(
                    script_cfg["sampling"]["vertical_oversampling_factor"]
                ),
            )
            thickness_error = float(generated.qc["maximum_thickness_absolute_error_s"])
            if thickness_error > calibration.truth_dt_s + 1e-12:
                raise ValueError(
                    f"canonical_geometry_qc_failed:{scenario.scenario_id}:thickness_error"
                )
            if scenario.family == "pinchout":
                termination_error = float(generated.qc["termination_absolute_error_m"])
                termination_tolerance = float(
                    generated.qc["termination_grid_resolution_tolerance_m"]
                )
                if termination_error > termination_tolerance + 1e-12:
                    raise ValueError(
                        f"canonical_geometry_qc_failed:{scenario.scenario_id}:termination_error"
                    )
            highres_result, forward_qc = _section_forward_qc(
                generated,
                wavelet=wavelet,
                highres_wavelet=highres_wavelet,
                required=bool(
                    script_cfg["forward_qc"][
                        "highres_mismatch_required"
                    ]
                ),
            )
            generated.qc.update(forward_qc)
            lfm_result = derive_lfm_priors(
                generated,
                config=script_cfg["lfm"],
                global_seed=int(script_cfg["global_seed"]),
                generator_family=GENERATOR_FAMILY,
                degradation_variant_id=generated.realization_id,
            )
            generated.qc.update(lfm_result.qc)
            hdf5_group = "" if qc_only else write_generated_section(h5, generated)
            if not qc_only and highres_result is not None:
                write_highres_forward_result(
                    h5,
                    realization_path=hdf5_group,
                    result=highres_result,
                )
            if not qc_only:
                write_lfm_result(
                    h5,
                    realization_path=hdf5_group,
                    result=lfm_result,
                )
            object_records.extend(generated.object_catalog)
            record = {
                "sample_id": scenario.scenario_id,
                "realization_id": scenario.scenario_id,
                "parent_realization_id": scenario.scenario_id,
                "suite": "canonical",
                "section_id": "canonical",
                "scenario_id": scenario.scenario_id,
                "geometry_family": scenario.family,
                "duration_mode": "canonical",
                "split": "benchmark",
                "hdf5_group": hdf5_group,
                "attempt_id": 0,
                "status": "ok",
                "reasons": "",
                "sample_kind": "base",
                "canonical_parameter_name": scenario.parameter_name,
                "canonical_parameter_value": scenario.parameter_value,
                "canonical_parameter_unit": scenario.parameter_unit,
                **_lfm_records(lfm_result, base_path=hdf5_group),
            }
            index_records.append(record)
            seismic_index, seismic_results = _seismic_variant_records_for_sample(
                h5=h5,
                owner_path=(
                    hdf5_group
                    if hdf5_group
                    else f"/realizations/{generated.realization_id}"
                ),
                source_index_record=record,
                seismic_model_consistent=generated.seismic_model_consistent,
                forward_valid_mask=generated.forward_valid_mask_model,
                lateral_m=generated.lateral_m,
                script_cfg=script_cfg,
                qc_only=qc_only,
                source_variant_id="base",
            )
            index_records.extend(seismic_index)
            seismic_variant_records.extend(seismic_results)
            qc_records.append({**record, **generated.qc})
            if scenario.family == "frequency_probe" and probe_frequencies:
                (
                    probe_index,
                    parent_probe_records,
                    probe_seismic_records,
                ) = _probe_records_for_parent(
                    h5=h5,
                    parent_path=(
                        hdf5_group
                        if hdf5_group
                        else f"/realizations/{generated.realization_id}"
                    ),
                    section=generated,
                    suite="canonical",
                    section_id="canonical",
                    split="benchmark",
                    frequencies=probe_frequencies,
                    script_cfg=script_cfg,
                    wavelet=wavelet,
                    highres_wavelet=highres_wavelet,
                    base_highres_forward=highres_result,
                    qc_only=qc_only,
                )
                index_records.extend(probe_index)
                probe_records.extend(parent_probe_records)
                seismic_variant_records.extend(probe_seismic_records)

    index = pd.DataFrame.from_records(index_records)
    index.to_csv(output_dir / "sample_index.csv", index=False)
    object_columns = [
        "realization_id",
        "scenario_id",
        "zone_id",
        "object_id",
        "state",
        "state_id",
        "event_target",
        "minimum_duration_s",
        "maximum_duration_s",
        "minimum_truth_samples",
        "maximum_truth_samples",
        "canonical_family",
        "canonical_parameter_name",
        "canonical_parameter_value",
        "canonical_parameter_unit",
        "expected_termination_lateral_m",
        "expected_section_drop_s",
        "contrast_multiplier_start",
        "contrast_multiplier_end",
    ]
    pd.DataFrame.from_records(object_records, columns=object_columns).to_csv(
        output_dir / "object_catalog.csv",
        index=False,
    )
    qc_frame = pd.DataFrame.from_records(qc_records)
    qc_frame.to_csv(output_dir / "generation_qc.csv", index=False)
    qc_frame.to_csv(output_dir / "canonical_geometry_qc.csv", index=False)
    pd.DataFrame.from_records(probe_records).to_csv(
        output_dir / "frequency_probe_results.csv",
        index=False,
    )
    pd.DataFrame.from_records(seismic_variant_records).to_csv(
        output_dir / "seismic_variant_results.csv",
        index=False,
    )
    probe_frequency_frame = pd.DataFrame.from_records(
        frequency_catalog_rows(probe_frequencies)
    )
    if not probe_frequency_frame.empty:
        probe_frequency_frame["wavelet_uncertainty_warning"] = (
            probe_frequency_frame[
                "conservative_to_nominal_ratio"
            ]
            > float(
                script_cfg["probe_selection"][
                    "conservative_to_nominal_warning_ratio"
                ]
            )
        )
    probe_frequency_frame.to_csv(
        output_dir / "probe_frequency_catalog.csv",
        index=False,
    )
    rejection_columns = [
        "realization_id",
        "section_id",
        "scenario_id",
        "geometry_family",
        "attempt_id",
        "reason",
        "zone_id",
        "object_id",
        "state",
        "event_target",
        "count",
        "denominator",
        "fraction",
        "threshold",
        "metric",
        "value",
        "lower",
        "upper",
        "excess_ratio",
        "lateral_index",
    ]
    pd.DataFrame(columns=rejection_columns).to_csv(
        output_dir / "generation_rejection_details.csv",
        index=False,
    )
    catalog = index[index["sample_kind"].eq("base")][
        [
            "section_id",
            "scenario_id",
            "geometry_family",
            "canonical_parameter_name",
            "canonical_parameter_value",
            "canonical_parameter_unit",
            "status",
        ]
    ].copy()
    catalog["attempt_count"] = 1
    catalog["acceptance_fraction"] = 1.0
    catalog["acceptance_status"] = "fixed_public_benchmark"
    catalog.to_csv(output_dir / "scenario_catalog.csv", index=False)
    figure_summary = write_generation_figures(
        output_dir,
        script_cfg.get("figures", {}),
        suite="canonical",
        qc_only=qc_only,
    )
    reference = canonical_reference_impedance(calibration)
    manifest = {
        "schema_version": DATA_SCHEMA,
        "generator_family": calibration.generator_family,
        "implementation_scope": IMPLEMENTATION_SCOPE,
        "suite": "canonical",
        "development_limited": False,
        "qc_only": bool(qc_only),
        "source_runs": {
            key: repo_relative_path(path, root=repo_root) for key, path in sources.items()
        },
        "impedance_calibration": repo_relative_path(calibration_path, root=repo_root),
        "impedance_calibration_sha256": sha256_file(calibration_path),
        "global_seed": int(script_cfg["global_seed"]),
        "output_dt_s": output_dt,
        "truth_dt_s": calibration.truth_dt_s,
        "n_sections": 1,
        "n_scenarios": len(scenarios),
        "attempts_per_scenario": 1,
        "canonical_families": list(CANONICAL_FAMILIES),
        "canonical_config": dict(config),
        "canonical_reference_impedance": reference,
        "probe_selection": dict(script_cfg["probe_selection"]),
        "probe_frequency_count": len(probe_frequencies),
        "probe_variant_count": len(probe_records),
        "seismic_variant_count": len(seismic_variant_records),
        "forward_qc": dict(script_cfg["forward_qc"]),
        "lfm": dict(script_cfg["lfm"]),
        "seismic_mismatch": dict(script_cfg["seismic_mismatch"]),
        "highres_wavelet": (
            {}
            if highres_wavelet is None
            else {
                "dt_s": float(
                    highres_wavelet.time_s[1]
                    - highres_wavelet.time_s[0]
                ),
                "n_samples": int(highres_wavelet.amplitude.size),
                "l2_energy": float(
                    np.linalg.norm(highres_wavelet.amplitude)
                ),
                "sha256": _array_sha256(
                    highres_wavelet.amplitude
                ),
            }
        ),
        "antialias_filter": {
            "implementation": "scipy.signal.firwin/resample_poly",
            "scipy_version": scipy.__version__,
            "factor": int(
                script_cfg["sampling"][
                    "vertical_oversampling_factor"
                ]
            ),
            "numtaps": int(
                antialias_taps(
                    int(
                        script_cfg["sampling"][
                            "vertical_oversampling_factor"
                        ]
                    )
                ).size
            ),
            "cutoff_output_nyquist_fraction": 0.9,
            "kaiser_beta": 8.6,
            "taps_sha256": _array_sha256(
                antialias_taps(
                    int(
                        script_cfg["sampling"][
                            "vertical_oversampling_factor"
                        ]
                    )
                )
            ),
        },
        "probe_source_hashes": {
            "frequency_evidence_bands.csv": sha256_file(
                sources["forward_observability_dir"]
                / "frequency_evidence_bands.csv"
            ),
            "well_frequency_sensitivity.csv": sha256_file(
                sources["forward_observability_dir"]
                / "well_frequency_sensitivity.csv"
            ),
        },
        "not_yet_implemented": [],
        "figures": {
            "generated_count": int(figure_summary.get("generated_count", 0)),
            "skipped_count": int(figure_summary.get("skipped_count", 0)),
            "figure_manifest": repo_relative_path(
                Path(str(figure_summary.get("figure_manifest", output_dir / "figures" / "figure_manifest.json"))),
                root=repo_root,
            ),
        },
        "files": {
            "synthetic_benchmark.h5": sha256_file(h5_path),
            "sample_index.csv": sha256_file(output_dir / "sample_index.csv"),
            "object_catalog.csv": sha256_file(output_dir / "object_catalog.csv"),
            "generation_qc.csv": sha256_file(output_dir / "generation_qc.csv"),
            "canonical_geometry_qc.csv": sha256_file(
                output_dir / "canonical_geometry_qc.csv"
            ),
            "frequency_probe_results.csv": sha256_file(
                output_dir / "frequency_probe_results.csv"
            ),
            "probe_frequency_catalog.csv": sha256_file(
                output_dir / "probe_frequency_catalog.csv"
            ),
            "seismic_variant_results.csv": sha256_file(
                output_dir / "seismic_variant_results.csv"
            ),
            "generation_rejection_details.csv": sha256_file(
                output_dir / "generation_rejection_details.csv"
            ),
            "scenario_catalog.csv": sha256_file(output_dir / "scenario_catalog.csv"),
            "figures/figure_manifest.json": sha256_file(
                output_dir / "figures" / "figure_manifest.json"
            ),
        },
    }
    write_json(output_dir / "benchmark_manifest.json", manifest)
    summary = {
        **manifest,
        "status": "ok",
        "accepted_realizations": len(scenarios),
        "rejected_realizations": 0,
        "failed_scenario_count": 0,
        "probe_variant_count": len(probe_records),
        "seismic_variant_count": len(seismic_variant_records),
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary


def run_generation(
    *,
    workflow: TimeWorkflowConfig,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    calibration_path: Path,
    repo_root: Path,
    output_dir: Path,
    debug_attempt_limit: int | None = None,
    geometry_families: Sequence[str] | None = None,
    qc_only: bool = False,
    suite: str = "field_conditioned",
) -> dict[str, Any]:
    calibration = load_calibration(calibration_path)
    for key, recorded in calibration.source_runs.items():
        if key in sources and resolve_relative_path(recorded, root=repo_root).resolve() != sources[key].resolve():
            raise ValueError(f"impedance_calibration_source_mismatch:{key}")
    for relative_name, expected_hash in calibration.source_hashes.items():
        directory_key, filename = relative_name.split("/", maxsplit=1)
        if directory_key not in sources:
            raise ValueError(f"impedance_calibration_source_mismatch:{directory_key}")
        actual_hash = sha256_file(sources[directory_key] / filename)
        if actual_hash != expected_hash:
            raise ValueError(f"impedance_calibration_source_mismatch:sha256:{relative_name}")
    output_dir.mkdir(parents=True, exist_ok=False)
    wavelet_time, wavelet = load_wavelet_csv(sources["wavelet_generation_dir"] / "selected_wavelet.csv")
    wavelet, qc = validate_wavelet_normalization(
        wavelet_time,
        wavelet,
        expected_l2_energy=1.0,
        l2_energy_tolerance=1e-5,
        max_center_abs_time_s=1e-9,
        allow_small_renormalization=True,
    )
    if qc.status != "ok":
        raise ValueError(f"invalid_wavelet:{qc.reasons}")
    output_dt = infer_wavelet_dt(wavelet_time)
    if suite == "canonical":
        if geometry_families:
            raise ValueError("--geometry-family is only valid for the field_conditioned suite.")
        return _run_canonical_generation(
            script_cfg=script_cfg,
            sources=sources,
            calibration=calibration,
            calibration_path=calibration_path,
            repo_root=repo_root,
            output_dir=output_dir,
            wavelet_time=wavelet_time,
            wavelet=wavelet,
            output_dt=output_dt,
            qc_only=qc_only,
        )
    if suite != "field_conditioned":
        raise ValueError(f"Unsupported synthoseis-lite suite: {suite}")
    sections = build_section_geometries(
        workflow=workflow,
        script_cfg=script_cfg,
        repo_root=repo_root,
    )
    section_geometry_qc_path = output_dir / "section_geometry_qc.csv"
    section_geometry_qc = pd.DataFrame.from_records(
        [row for section in sections for row in section.qc_rows]
    )
    section_geometry_qc.to_csv(section_geometry_qc_path, index=False)
    scenarios = generation_scenarios(script_cfg)
    if geometry_families:
        selected = {str(value) for value in geometry_families}
        unknown = selected.difference({"none", "wedge", "pinchout"})
        if unknown:
            raise ValueError(f"Unsupported geometry filters: {sorted(unknown)}")
        scenarios = [scenario for scenario in scenarios if scenario.geometry_family in selected]
        if not scenarios:
            raise ValueError("No generation scenarios remain after geometry filtering.")
    attempts = int(script_cfg["generation"]["attempts_per_scenario"])
    development_limited = debug_attempt_limit is not None
    if debug_attempt_limit is not None:
        attempts = min(attempts, int(debug_attempt_limit))
    index_records: list[dict[str, Any]] = []
    object_records: list[dict[str, Any]] = []
    object_lateral_records: list[dict[str, Any]] = []
    qc_records: list[dict[str, Any]] = []
    rejection_records: list[dict[str, Any]] = []
    probe_records: list[dict[str, Any]] = []
    seismic_variant_records: list[dict[str, Any]] = []
    probe_frequencies = _load_probe_frequencies(
        script_cfg=script_cfg,
        sources=sources,
    )
    highres_wavelet = (
        resample_wavelet_to_highres(
            wavelet_time,
            wavelet,
            factor=int(
                script_cfg["sampling"]["vertical_oversampling_factor"]
            ),
        )
        if bool(
            script_cfg["forward_qc"]["highres_mismatch_enabled"]
        )
        else None
    )
    probe_parent_counts = {section.section_id: 0 for section in sections}
    h5_path = output_dir / "synthetic_benchmark.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["schema_version"] = DATA_SCHEMA
        h5.attrs["generator_family"] = calibration.generator_family
        h5.attrs["implementation_scope"] = IMPLEMENTATION_SCOPE
        h5.attrs["suite"] = "field_conditioned"
        for section in sections:
            for scenario in scenarios:
                for attempt_id in range(attempts):
                    realization_id = f"{section.section_id}__{scenario.scenario_id}__a{attempt_id:03d}"
                    hdf5_group = ""
                    probe_index_local: list[dict[str, Any]] = []
                    probe_records_local: list[dict[str, Any]] = []
                    seismic_variant_records_local: list[dict[str, Any]] = []
                    base_lfm_index: dict[str, Any] = {}
                    try:
                        minimum_truth_samples = 2 if scenario.duration_mode == "ultra_thin_stress" else 4
                        generated = generate_field_section(
                            calibration,
                            realization_id=realization_id,
                            scenario=scenario,
                            global_seed=int(script_cfg["global_seed"]),
                            lateral_m=section.lateral_m,
                            inline_float=section.inline_float,
                            xline_float=section.xline_float,
                            x_m=section.x_m,
                            y_m=section.y_m,
                            horizon_twt_s=section.horizon_twt_s,
                            output_dt_s=output_dt,
                            wavelet=wavelet,
                            vertical_oversampling_factor=int(
                                script_cfg["sampling"]["vertical_oversampling_factor"]
                            ),
                            minimum_truth_samples=minimum_truth_samples,
                            max_global_reversal_fraction=float(
                                script_cfg["impedance"]["max_global_reversal_fraction"]
                            ),
                            max_object_reversal_fraction=float(
                                script_cfg["impedance"]["max_object_reversal_fraction"]
                            ),
                            max_global_clipping_fraction=float(
                                script_cfg["impedance"]["max_global_clipping_fraction"]
                            ),
                            max_object_clipping_fraction=float(
                                script_cfg["impedance"]["max_object_clipping_fraction"]
                            ),
                        )
                        highres_result, forward_qc = _section_forward_qc(
                            generated,
                            wavelet=wavelet,
                            highres_wavelet=highres_wavelet,
                            required=bool(
                                script_cfg["forward_qc"][
                                    "highres_mismatch_required"
                                ]
                            ),
                        )
                        generated.qc.update(forward_qc)
                        lfm_result = derive_lfm_priors(
                            generated,
                            config=script_cfg["lfm"],
                            global_seed=int(script_cfg["global_seed"]),
                            generator_family=GENERATOR_FAMILY,
                            degradation_variant_id=generated.realization_id,
                        )
                        generated.qc.update(lfm_result.qc)
                        hdf5_group = "" if qc_only else write_generated_section(h5, generated)
                        if not qc_only and highres_result is not None:
                            write_highres_forward_result(
                                h5,
                                realization_path=hdf5_group,
                                result=highres_result,
                            )
                        if not qc_only:
                            write_lfm_result(
                                h5,
                                realization_path=hdf5_group,
                                result=lfm_result,
                            )
                        base_lfm_index = _lfm_records(
                            lfm_result,
                            base_path=hdf5_group,
                        )
                        base_record_for_variants = {
                            "sample_id": realization_id,
                            "realization_id": realization_id,
                            "parent_realization_id": realization_id,
                            "suite": "field_conditioned",
                            "section_id": section.section_id,
                            "scenario_id": scenario.scenario_id,
                            "geometry_family": scenario.geometry_family,
                            "duration_mode": scenario.duration_mode,
                            "split": "test" if scenario.geometry_family == "pinchout" else "unassigned",
                            "hdf5_group": hdf5_group,
                            "attempt_id": attempt_id,
                            "status": "ok",
                            "reasons": "",
                            "sample_kind": "base",
                            **base_lfm_index,
                        }
                        (
                            base_seismic_index,
                            base_seismic_results,
                        ) = _seismic_variant_records_for_sample(
                            h5=h5,
                            owner_path=(
                                hdf5_group
                                if hdf5_group
                                else f"/realizations/{generated.realization_id}"
                            ),
                            source_index_record=base_record_for_variants,
                            seismic_model_consistent=generated.seismic_model_consistent,
                            forward_valid_mask=generated.forward_valid_mask_model,
                            lateral_m=generated.lateral_m,
                            script_cfg=script_cfg,
                            qc_only=qc_only,
                            source_variant_id="base",
                        )
                        probe_index_local.extend(base_seismic_index)
                        seismic_variant_records_local.extend(base_seismic_results)
                        probe_config = script_cfg["probe_selection"]
                        is_probe_parent = (
                            probe_frequencies
                            and scenario.geometry_family
                            == str(
                                probe_config[
                                    "field_parent_geometry_family"
                                ]
                            )
                            and probe_parent_counts[section.section_id]
                            < int(
                                probe_config[
                                    "field_parents_per_section"
                                ]
                            )
                        )
                        if is_probe_parent:
                            try:
                                (
                                    probe_index_extra,
                                    probe_records_extra,
                                    probe_seismic_records_extra,
                                ) = _probe_records_for_parent(
                                    h5=h5,
                                    parent_path=(
                                        hdf5_group
                                        if hdf5_group
                                        else f"/realizations/{generated.realization_id}"
                                    ),
                                    section=generated,
                                    suite="field_conditioned",
                                    section_id=section.section_id,
                                    split="benchmark",
                                    frequencies=probe_frequencies,
                                    script_cfg=script_cfg,
                                    wavelet=wavelet,
                                    highres_wavelet=highres_wavelet,
                                    base_highres_forward=highres_result,
                                    qc_only=qc_only,
                                )
                                probe_index_local.extend(probe_index_extra)
                                probe_records_local.extend(probe_records_extra)
                                seismic_variant_records_local.extend(
                                    probe_seismic_records_extra
                                )
                            except Exception as exc:
                                raise RuntimeError(
                                    "frequency_probe_generation_failed:"
                                    f"{generated.realization_id}:{exc}"
                                ) from exc
                            probe_parent_counts[section.section_id] += 1
                        object_records.extend(generated.object_catalog)
                        object_lateral_records.extend(
                            generated.object_lateral_coefficients
                        )
                        status = "ok"
                        reasons = ""
                        qc_payload = generated.qc
                    except GenerationRejected as exc:
                        hdf5_group = ""
                        status = "rejected"
                        reasons = ";".join(exc.reasons)
                        qc_payload = exc.diagnostics
                        rejection_records.extend(
                            {
                                "realization_id": realization_id,
                                "section_id": section.section_id,
                                "scenario_id": scenario.scenario_id,
                                "geometry_family": scenario.geometry_family,
                                "attempt_id": attempt_id,
                                **detail,
                            }
                            for detail in exc.details
                        )
                    except Exception as exc:
                        if str(exc).startswith(
                            (
                                "highres_forward_qc_failed",
                                "invalid_model_grid_forward",
                                "frequency_probe_generation_failed",
                            )
                        ):
                            raise
                        if hdf5_group and hdf5_group in h5:
                            del h5[hdf5_group]
                        hdf5_group = ""
                        status = "rejected"
                        reasons = f"{type(exc).__name__}:{exc}"
                        qc_payload = {}
                    record = {
                        "sample_id": realization_id,
                        "realization_id": realization_id,
                        "parent_realization_id": realization_id,
                        "suite": "field_conditioned",
                        "section_id": section.section_id,
                        "scenario_id": scenario.scenario_id,
                        "geometry_family": scenario.geometry_family,
                        "duration_mode": scenario.duration_mode,
                        "split": "test" if scenario.geometry_family == "pinchout" else "unassigned",
                        "hdf5_group": hdf5_group,
                        "attempt_id": attempt_id,
                        "status": status,
                        "reasons": reasons,
                        "sample_kind": "base",
                        **base_lfm_index,
                    }
                    index_records.append(record)
                    if status == "ok":
                        index_records.extend(probe_index_local)
                        probe_records.extend(probe_records_local)
                        seismic_variant_records.extend(seismic_variant_records_local)
                    qc_records.append({**record, **{key: value for key, value in qc_payload.items() if key != "field_qc"}})
    index = pd.DataFrame.from_records(index_records)
    index.to_csv(output_dir / "sample_index.csv", index=False)
    object_columns = [
        "realization_id",
        "scenario_id",
        "zone_id",
        "object_id",
        "state",
        "state_id",
        "base_duration_fraction",
        "event_target",
        "duration_fraction_start",
        "duration_fraction_end",
        "minimum_duration_fraction",
        "maximum_duration_fraction",
        "minimum_duration_s",
        "maximum_duration_s",
        "minimum_truth_samples",
        "maximum_truth_samples",
        "event_multiplier_start",
        "event_multiplier_end",
        "minimum_event_multiplier",
        "maximum_event_multiplier",
        "reversal_fraction",
        "clipping_fraction",
        "profile_violation_fraction",
        "profile_projection_fraction",
        "mean_profile_projection_scale",
        "minimum_profile_projection_scale",
        "c0_conditioning_fraction",
        "mean_c0_conditioning_adjustment",
        "maximum_c0_conditioning_adjustment",
    ]
    pd.DataFrame.from_records(object_records, columns=object_columns).to_csv(
        output_dir / "object_catalog.csv",
        index=False,
    )
    object_lateral_columns = [
        "realization_id",
        "scenario_id",
        "zone_id",
        "local_object_index",
        "calibration_object_id",
        "object_id",
        "state",
        "state_id",
        "event_target",
        "lateral_index",
        "lateral_m",
        "c0",
        "c1",
        "c2",
        "thickness_fraction",
        "object_top_s",
        "object_bottom_s",
        "profile_projection_scale",
        "c0_conditioning_adjustment",
    ]
    pd.DataFrame.from_records(
        object_lateral_records,
        columns=object_lateral_columns,
    ).to_csv(output_dir / "object_lateral_coefficients.csv", index=False)
    pd.DataFrame.from_records(qc_records).to_csv(output_dir / "generation_qc.csv", index=False)
    pd.DataFrame.from_records(probe_records).to_csv(
        output_dir / "frequency_probe_results.csv",
        index=False,
    )
    pd.DataFrame.from_records(seismic_variant_records).to_csv(
        output_dir / "seismic_variant_results.csv",
        index=False,
    )
    probe_frequency_frame = pd.DataFrame.from_records(
        frequency_catalog_rows(probe_frequencies)
    )
    if not probe_frequency_frame.empty:
        probe_frequency_frame["wavelet_uncertainty_warning"] = (
            probe_frequency_frame[
                "conservative_to_nominal_ratio"
            ]
            > float(
                script_cfg["probe_selection"][
                    "conservative_to_nominal_warning_ratio"
                ]
            )
        )
    probe_frequency_frame.to_csv(
        output_dir / "probe_frequency_catalog.csv",
        index=False,
    )
    rejection_columns = [
        "realization_id",
        "section_id",
        "scenario_id",
        "geometry_family",
        "attempt_id",
        "reason",
        "zone_id",
        "object_id",
        "state",
        "event_target",
        "count",
        "denominator",
        "fraction",
        "threshold",
        "metric",
        "value",
        "lower",
        "upper",
        "excess_ratio",
        "lateral_index",
    ]
    pd.DataFrame.from_records(rejection_records, columns=rejection_columns).to_csv(
        output_dir / "generation_rejection_details.csv",
        index=False,
    )
    catalog = (
        index[index["sample_kind"].eq("base")]
        .groupby(["section_id", "scenario_id"], dropna=False)["status"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    catalog["attempt_count"] = catalog.get("ok", 0) + catalog.get("rejected", 0)
    catalog["acceptance_fraction"] = catalog.get("ok", 0) / catalog["attempt_count"]
    if development_limited:
        catalog["acceptance_status"] = "development_limit_no_verdict"
    else:
        minimum = int(script_cfg["impedance"]["minimum_attempts_per_scenario"])
        warning = float(script_cfg["impedance"]["scenario_acceptance_warning_fraction"])
        failure = float(script_cfg["impedance"]["scenario_acceptance_failure_fraction"])
        catalog["acceptance_status"] = np.where(
            catalog["attempt_count"] < minimum,
            "insufficient_attempts_for_acceptance_qc",
            np.where(
                catalog["acceptance_fraction"] < failure,
                "failed",
                np.where(catalog["acceptance_fraction"] < warning, "warning", "ok"),
            ),
        )
    catalog.to_csv(output_dir / "scenario_catalog.csv", index=False)
    figure_summary = write_generation_figures(
        output_dir,
        script_cfg.get("figures", {}),
        suite="field_conditioned",
        qc_only=qc_only,
    )
    manifest = {
        "schema_version": DATA_SCHEMA,
        "generator_family": calibration.generator_family,
        "implementation_scope": IMPLEMENTATION_SCOPE,
        "development_limited": development_limited,
        "qc_only": bool(qc_only),
        "suite": "field_conditioned",
        "source_runs": {
            key: repo_relative_path(path, root=repo_root) for key, path in sources.items()
        },
        "impedance_calibration": repo_relative_path(calibration_path, root=repo_root),
        "impedance_calibration_sha256": sha256_file(calibration_path),
        "global_seed": int(script_cfg["global_seed"]),
        "output_dt_s": output_dt,
        "truth_dt_s": calibration.truth_dt_s,
        "n_sections": len(sections),
        "n_scenarios": len(scenarios),
        "attempts_per_scenario": attempts,
        "geometry_filters": sorted({scenario.geometry_family for scenario in scenarios}),
        "field_geometry": {
            "mode": str(script_cfg.get("target_zone", {}).get("mode", "filled_target_zone")),
            "target_zone": dict(script_cfg.get("target_zone") or {}),
            "section_geometry_qc": repo_relative_path(section_geometry_qc_path, root=repo_root),
        },
        "probe_selection": dict(script_cfg["probe_selection"]),
        "probe_frequency_count": len(probe_frequencies),
        "probe_variant_count": len(probe_records),
        "probe_parent_counts": probe_parent_counts,
        "seismic_variant_count": len(seismic_variant_records),
        "forward_qc": dict(script_cfg["forward_qc"]),
        "lfm": dict(script_cfg["lfm"]),
        "seismic_mismatch": dict(script_cfg["seismic_mismatch"]),
        "highres_wavelet": (
            {}
            if highres_wavelet is None
            else {
                "dt_s": float(
                    highres_wavelet.time_s[1]
                    - highres_wavelet.time_s[0]
                ),
                "n_samples": int(highres_wavelet.amplitude.size),
                "l2_energy": float(
                    np.linalg.norm(highres_wavelet.amplitude)
                ),
                "sha256": _array_sha256(
                    highres_wavelet.amplitude
                ),
            }
        ),
        "antialias_filter": {
            "implementation": "scipy.signal.firwin/resample_poly",
            "scipy_version": scipy.__version__,
            "factor": int(
                script_cfg["sampling"][
                    "vertical_oversampling_factor"
                ]
            ),
            "numtaps": int(
                antialias_taps(
                    int(
                        script_cfg["sampling"][
                            "vertical_oversampling_factor"
                        ]
                    )
                ).size
            ),
            "cutoff_output_nyquist_fraction": 0.9,
            "kaiser_beta": 8.6,
            "taps_sha256": _array_sha256(
                antialias_taps(
                    int(
                        script_cfg["sampling"][
                            "vertical_oversampling_factor"
                        ]
                    )
                )
            ),
        },
        "probe_source_hashes": {
            "frequency_evidence_bands.csv": sha256_file(
                sources["forward_observability_dir"]
                / "frequency_evidence_bands.csv"
            ),
            "well_frequency_sensitivity.csv": sha256_file(
                sources["forward_observability_dir"]
                / "well_frequency_sensitivity.csv"
            ),
        },
        "not_yet_implemented": [],
        "figures": {
            "generated_count": int(figure_summary.get("generated_count", 0)),
            "skipped_count": int(figure_summary.get("skipped_count", 0)),
            "figure_manifest": repo_relative_path(
                Path(str(figure_summary.get("figure_manifest", output_dir / "figures" / "figure_manifest.json"))),
                root=repo_root,
            ),
        },
        "files": {
            "synthetic_benchmark.h5": sha256_file(h5_path),
            "sample_index.csv": sha256_file(output_dir / "sample_index.csv"),
            "object_catalog.csv": sha256_file(output_dir / "object_catalog.csv"),
            "object_lateral_coefficients.csv": sha256_file(
                output_dir / "object_lateral_coefficients.csv"
            ),
            "generation_qc.csv": sha256_file(output_dir / "generation_qc.csv"),
            "frequency_probe_results.csv": sha256_file(
                output_dir / "frequency_probe_results.csv"
            ),
            "probe_frequency_catalog.csv": sha256_file(
                output_dir / "probe_frequency_catalog.csv"
            ),
            "seismic_variant_results.csv": sha256_file(
                output_dir / "seismic_variant_results.csv"
            ),
            "generation_rejection_details.csv": sha256_file(
                output_dir / "generation_rejection_details.csv"
            ),
            "scenario_catalog.csv": sha256_file(output_dir / "scenario_catalog.csv"),
            "section_geometry_qc.csv": sha256_file(section_geometry_qc_path),
            "figures/figure_manifest.json": sha256_file(
                output_dir / "figures" / "figure_manifest.json"
            ),
        },
    }
    write_json(output_dir / "benchmark_manifest.json", manifest)
    failure_statuses = {"failed", "insufficient_attempts_for_acceptance_qc"}
    failed_scenarios = catalog["acceptance_status"].isin(failure_statuses)
    summary = {
        **manifest,
        "status": (
            "development_limited"
            if development_limited
            else ("failed_acceptance_qc" if bool(failed_scenarios.any()) else "ok")
        ),
        "accepted_realizations": int(
            (
                index["sample_kind"].eq("base")
                & index["status"].eq("ok")
            ).sum()
        ),
        "rejected_realizations": int(
            (
                index["sample_kind"].eq("base")
                & index["status"].eq("rejected")
            ).sum()
        ),
        "failed_scenario_count": int(failed_scenarios.sum()),
        "probe_variant_count": len(probe_records),
        "seismic_variant_count": len(seismic_variant_records),
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary
