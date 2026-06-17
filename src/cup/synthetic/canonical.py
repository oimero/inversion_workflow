"""Deterministic two-dimensional canonical impedance benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from cup.seismic.observability import acoustic_reflectivity_from_log_ai, forward_log_ai
from cup.seismic.wavelet import wavelet_spectrum_features
from cup.synthetic.calibration import ImpedanceCalibration
from cup.synthetic.dsp import antialias_taps, downsample_continuous
from cup.synthetic.generation import GeneratedSection, GenerationScenario
from cup.synthetic.grids import categorical_model_grids


CANONICAL_FAMILIES = (
    "horizontal_thin_beds",
    "wedge",
    "pinchout",
    "dipping_layers",
    "lateral_impedance_change",
    "frequency_probe",
)


@dataclass(frozen=True)
class CanonicalScenario:
    scenario_id: str
    family: str
    parameter_name: str
    parameter_value: float
    parameter_unit: str


def canonical_reference_impedance(
    calibration: ImpedanceCalibration,
) -> dict[str, float]:
    """Derive a conservative background and contrast from frozen well calibration."""
    backgrounds: list[float] = []
    contrasts: list[float] = []
    upper_margins: list[float] = []
    lower_margins: list[float] = []
    for model in calibration.zone_models.values():
        background = float(model["background"]["background_a"]["median"])
        backgrounds.append(background)
        high = float(model["states"]["high_impedance"]["coefficients"]["c0"]["median"])
        low = float(model["states"]["low_impedance"]["coefficients"]["c0"]["median"])
        contrasts.extend((abs(high), abs(low)))
        upper_margins.append(float(model["ai_bounds"]["p99"]) - background)
        lower_margins.append(background - float(model["ai_bounds"]["p01"]))
    background = float(np.median(backgrounds))
    raw_contrast = float(np.median(contrasts))
    safe_margin = 0.8 * min(float(np.median(upper_margins)), float(np.median(lower_margins)))
    contrast = min(raw_contrast, safe_margin)
    if not np.isfinite(background) or not np.isfinite(contrast) or contrast <= 0.0:
        raise ValueError("invalid_impedance_calibration:canonical_reference")
    return {
        "background_log_ai": background,
        "raw_reference_contrast_log_ai": raw_contrast,
        "reference_contrast_log_ai": contrast,
    }


def canonical_scenarios(config: Mapping[str, Any]) -> list[CanonicalScenario]:
    scenarios: list[CanonicalScenario] = []
    for ratio in config["thin_bed_period_ratios"]:
        scenarios.append(
            CanonicalScenario(
                scenario_id=f"horizontal_thin_beds__period_ratio_{float(ratio):g}",
                family="horizontal_thin_beds",
                parameter_name="thickness_period_ratio",
                parameter_value=float(ratio),
                parameter_unit="ratio",
            )
        )
    scenarios.append(
        CanonicalScenario(
            scenario_id="wedge__0_to_1_period",
            family="wedge",
            parameter_name="maximum_thickness_period_ratio",
            parameter_value=1.0,
            parameter_unit="ratio",
        )
    )
    scenarios.append(
        CanonicalScenario(
            scenario_id="pinchout__quarter_period__termination_0.75",
            family="pinchout",
            parameter_name="termination_section_fraction",
            parameter_value=float(config["pinchout_termination_fraction"]),
            parameter_unit="fraction",
        )
    )
    for ratio in config["dip_drop_period_ratios"]:
        scenarios.append(
            CanonicalScenario(
                scenario_id=f"dipping_layers__drop_period_ratio_{float(ratio):g}",
                family="dipping_layers",
                parameter_name="section_drop_period_ratio",
                parameter_value=float(ratio),
                parameter_unit="ratio",
            )
        )
    for multiplier in config["lateral_contrast_multipliers"]:
        scenarios.append(
            CanonicalScenario(
                scenario_id=f"lateral_impedance_change__right_multiplier_{float(multiplier):g}",
                family="lateral_impedance_change",
                parameter_name="right_contrast_multiplier",
                parameter_value=float(multiplier),
                parameter_unit="ratio",
            )
        )
    scenarios.append(
        CanonicalScenario(
            scenario_id="frequency_probe__smooth_background",
            family="frequency_probe",
            parameter_name="background_type",
            parameter_value=0.0,
            parameter_unit="smooth_background",
        )
    )
    return scenarios


def _target_geometry(
    scenario: CanonicalScenario,
    *,
    lateral_fraction: np.ndarray,
    center_twt_s: float,
    peak_period_s: float,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(lateral_fraction, dtype=np.float64)
    if scenario.family == "horizontal_thin_beds":
        thickness = np.full(x.shape, scenario.parameter_value * peak_period_s)
        top = np.full(x.shape, center_twt_s) - 0.5 * thickness
        contrast_multiplier = np.ones(x.shape)
    elif scenario.family == "wedge":
        margin = 0.5 * (1.0 - float(config["wedge_transition_fraction"]))
        ramp = np.clip((x - margin) / float(config["wedge_transition_fraction"]), 0.0, 1.0)
        thickness = peak_period_s * ramp
        top = np.full(x.shape, center_twt_s - 0.5 * peak_period_s)
        contrast_multiplier = np.ones(x.shape)
    elif scenario.family == "pinchout":
        termination = float(config["pinchout_termination_fraction"])
        thickness = 0.25 * peak_period_s * np.clip(1.0 - x / termination, 0.0, 1.0)
        top = np.full(x.shape, center_twt_s - 0.125 * peak_period_s)
        contrast_multiplier = np.ones(x.shape)
    elif scenario.family == "dipping_layers":
        drop = scenario.parameter_value * peak_period_s
        top = center_twt_s - 0.125 * peak_period_s + drop * (x - 0.5)
        thickness = np.full(x.shape, 0.25 * peak_period_s)
        contrast_multiplier = np.ones(x.shape)
    elif scenario.family == "lateral_impedance_change":
        thickness = np.full(x.shape, 0.25 * peak_period_s)
        top = np.full(x.shape, center_twt_s - 0.125 * peak_period_s)
        contrast_multiplier = np.where(x < 0.5, 1.0, scenario.parameter_value)
    elif scenario.family == "frequency_probe":
        thickness = np.zeros(x.shape)
        top = np.full(x.shape, center_twt_s)
        contrast_multiplier = np.zeros(x.shape)
    else:
        raise ValueError(f"Unsupported canonical family: {scenario.family}")
    return top, thickness, contrast_multiplier


def _canonical_qc(
    scenario: CanonicalScenario,
    *,
    lateral_m: np.ndarray,
    top_s: np.ndarray,
    thickness_s: np.ndarray,
    contrast_multiplier: np.ndarray,
    truth_dt_s: float,
    target_mask: np.ndarray,
    peak_frequency_hz: float,
    peak_period_s: float,
    reference_contrast: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    sample_thickness = np.sum(target_mask, axis=1) * truth_dt_s
    positive = sample_thickness > 0.0
    qc: dict[str, Any] = {
        "status": "ok",
        "suite": "canonical",
        "canonical_family": scenario.family,
        "canonical_parameter_name": scenario.parameter_name,
        "canonical_parameter_value": scenario.parameter_value,
        "canonical_parameter_unit": scenario.parameter_unit,
        "peak_frequency_hz": peak_frequency_hz,
        "peak_period_s": peak_period_s,
        "reference_contrast_log_ai": reference_contrast,
        "expected_minimum_thickness_s": float(np.min(thickness_s)),
        "expected_maximum_thickness_s": float(np.max(thickness_s)),
        "grid_minimum_positive_thickness_s": (
            float(np.min(sample_thickness[positive])) if np.any(positive) else 0.0
        ),
        "grid_maximum_thickness_s": float(np.max(sample_thickness)),
        "maximum_thickness_absolute_error_s": float(
            np.max(np.abs(sample_thickness - thickness_s))
        ),
        "expected_top_start_s": float(top_s[0]),
        "expected_top_end_s": float(top_s[-1]),
        "expected_contrast_multiplier_start": float(contrast_multiplier[0]),
        "expected_contrast_multiplier_end": float(contrast_multiplier[-1]),
    }
    if scenario.family == "pinchout":
        expected = float(config["pinchout_termination_fraction"] * lateral_m[-1])
        nonzero = np.flatnonzero(positive)
        inferred = float(lateral_m[nonzero[-1]]) if nonzero.size else float("nan")
        initial_thickness = float(thickness_s[0])
        thickness_slope = initial_thickness / max(expected, np.finfo(np.float64).eps)
        grid_tolerance = (
            truth_dt_s / max(thickness_slope, np.finfo(np.float64).eps)
            + float(np.median(np.diff(lateral_m)))
        )
        qc.update(
            {
                "expected_termination_lateral_m": expected,
                "analytic_termination_lateral_m": expected,
                "analytic_termination_absolute_error_m": 0.0,
                "grid_last_nonzero_lateral_m": inferred,
                "termination_absolute_error_m": (
                    abs(inferred - expected) if np.isfinite(inferred) else float("nan")
                ),
                "termination_grid_resolution_tolerance_m": grid_tolerance,
            }
        )
    if scenario.family == "dipping_layers":
        qc["expected_section_drop_s"] = float(top_s[-1] - top_s[0])
    return qc


def generate_canonical_section(
    calibration: ImpedanceCalibration,
    *,
    scenario: CanonicalScenario,
    config: Mapping[str, Any],
    output_dt_s: float,
    wavelet_time_s: np.ndarray,
    wavelet: np.ndarray,
    vertical_oversampling_factor: int = 8,
) -> GeneratedSection:
    """Generate one fixed canonical geometry and its nominal closed forward model."""
    wavelet_time = np.asarray(wavelet_time_s, dtype=np.float64).reshape(-1)
    wavelet_values = np.asarray(wavelet, dtype=np.float64).reshape(-1)
    features = wavelet_spectrum_features(wavelet_time, wavelet_values)
    peak_frequency = float(features.dominant_frequency_hz)
    if peak_frequency <= 0.0:
        raise ValueError("invalid_wavelet:nonpositive_peak_frequency")
    peak_period = 1.0 / peak_frequency
    factor = int(vertical_oversampling_factor)
    truth_dt = float(output_dt_s) / factor
    if not np.isclose(truth_dt, calibration.truth_dt_s, rtol=0.0, atol=1e-12):
        raise ValueError("impedance_calibration_source_mismatch:truth_dt")

    n_lateral = int(config["lateral_samples"])
    lateral = np.arange(n_lateral, dtype=np.float64) * float(config["lateral_sample_interval_m"])
    lateral_fraction = lateral / float(lateral[-1])
    center = float(config["center_twt_s"])
    top_s, thickness_s, contrast_multiplier = _target_geometry(
        scenario,
        lateral_fraction=lateral_fraction,
        center_twt_s=center,
        peak_period_s=peak_period,
        config=config,
    )
    bottom_s = top_s + thickness_s
    target_extent = float(config["vertical_extent_periods"]) * peak_period
    context = (wavelet_values.size // 2) * float(output_dt_s)
    geometry_start = min(center - 0.5 * target_extent, float(np.min(top_s)))
    geometry_end = max(center + 0.5 * target_extent, float(np.max(bottom_s)))
    start_s = np.floor((geometry_start - context) / truth_dt) * truth_dt
    end_s = np.ceil((geometry_end + context) / truth_dt) * truth_dt
    n_model_intervals = int(np.ceil((end_s - start_s) / output_dt_s))
    n_highres = n_model_intervals * factor + 1
    twt_highres = start_s + np.arange(n_highres, dtype=np.float64) * truth_dt
    twt_model = twt_highres[::factor]

    reference = canonical_reference_impedance(calibration)
    background = float(reference["background_log_ai"])
    contrast = float(reference["reference_contrast_log_ai"])
    log_ai = np.full((n_lateral, n_highres), background, dtype=np.float64)
    target_mask = np.zeros((n_lateral, n_highres), dtype=bool)
    rgt = np.broadcast_to(
        (twt_highres - geometry_start) / max(geometry_end - geometry_start, truth_dt),
        log_ai.shape,
    ).copy()
    valid_geometry = np.broadcast_to(
        (twt_highres >= geometry_start) & (twt_highres <= geometry_end),
        log_ai.shape,
    )
    state_id = np.where(valid_geometry, 1, -1).astype(np.int8)
    object_id = np.where(valid_geometry, 0, -1).astype(np.int32)
    object_xi = np.full(log_ai.shape, np.nan, dtype=np.float64)
    zone_id = np.where(valid_geometry, 0, -1).astype(np.int16)
    boundary = np.zeros(log_ai.shape, dtype=bool)
    for lateral_index in range(n_lateral):
        thickness = float(thickness_s[lateral_index])
        if thickness <= 0.0:
            continue
        mask = (twt_highres >= top_s[lateral_index]) & (
            twt_highres < bottom_s[lateral_index]
        )
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            continue
        target_mask[lateral_index, indices] = True
        log_ai[lateral_index, indices] = (
            background + contrast * contrast_multiplier[lateral_index]
        )
        state_id[lateral_index, indices] = 2
        object_id[lateral_index, indices] = 1
        object_xi[lateral_index, indices] = (
            twt_highres[indices] - top_s[lateral_index]
        ) / thickness
        boundary[lateral_index, indices[0]] = True
        if indices[-1] + 1 < n_highres:
            boundary[lateral_index, indices[-1] + 1] = True

    taps = antialias_taps(factor)
    model_log_ai = downsample_continuous(log_ai, factor, taps)[..., : twt_model.size]
    rgt_model = downsample_continuous(rgt, factor, taps)[..., : twt_model.size]
    reflectivity_highres = np.stack(
        [acoustic_reflectivity_from_log_ai(trace) for trace in log_ai],
        axis=0,
    )
    reflectivity_model = np.stack(
        [acoustic_reflectivity_from_log_ai(trace) for trace in model_log_ai],
        axis=0,
    )
    seismic = np.stack(
        [forward_log_ai(trace, wavelet_values) for trace in model_log_ai],
        axis=0,
    )
    state_fraction, dominant, zone_model, boundary_fraction, valid_model = (
        categorical_model_grids(
            state_id,
            object_id,
            zone_id,
            boundary,
            factor,
            twt_model.size,
        )
    )
    qc = _canonical_qc(
        scenario,
        lateral_m=lateral,
        top_s=top_s,
        thickness_s=thickness_s,
        contrast_multiplier=contrast_multiplier,
        truth_dt_s=truth_dt,
        target_mask=target_mask,
        peak_frequency_hz=peak_frequency,
        peak_period_s=peak_period,
        reference_contrast=contrast,
        config=config,
    )
    generation_scenario = GenerationScenario(
        scenario_id=scenario.scenario_id,
        duration_mode="canonical",
        geometry_family=scenario.family,
        geometry_direction="fixed",
        correlation_length_fraction=0.0,
        coefficient_sigma_multiplier=0.0,
        thickness_log_sigma=0.0,
        variant_id="",
    )
    object_catalog = (
        []
        if scenario.family == "frequency_probe"
        else [{
            "realization_id": scenario.scenario_id,
            "scenario_id": scenario.scenario_id,
            "zone_id": "canonical",
            "object_id": 1,
            "state": "high_impedance",
            "state_id": 2,
            "event_target": True,
            "minimum_duration_s": float(np.min(thickness_s)),
            "maximum_duration_s": float(np.max(thickness_s)),
            "minimum_truth_samples": int(np.min(np.sum(target_mask, axis=1))),
            "maximum_truth_samples": int(np.max(np.sum(target_mask, axis=1))),
            "canonical_family": scenario.family,
            "canonical_parameter_name": scenario.parameter_name,
            "canonical_parameter_value": scenario.parameter_value,
            "canonical_parameter_unit": scenario.parameter_unit,
            "expected_termination_lateral_m": qc.get(
                "expected_termination_lateral_m",
                float("nan"),
            ),
            "expected_section_drop_s": qc.get("expected_section_drop_s", float("nan")),
            "contrast_multiplier_start": float(contrast_multiplier[0]),
            "contrast_multiplier_end": float(contrast_multiplier[-1]),
        }]
    )
    return GeneratedSection(
        realization_id=scenario.scenario_id,
        scenario=generation_scenario,
        lateral_m=lateral,
        inline_float=np.full(n_lateral, np.nan),
        xline_float=np.full(n_lateral, np.nan),
        x_m=lateral.copy(),
        y_m=np.zeros(n_lateral),
        twt_highres_s=twt_highres,
        twt_model_s=twt_model,
        truth_log_ai_highres=log_ai,
        model_target_log_ai=model_log_ai,
        reflectivity_highres=reflectivity_highres,
        reflectivity_model=reflectivity_model,
        seismic_model_consistent=seismic,
        rgt_highres=rgt,
        rgt_model=rgt_model,
        state_id_highres=state_id,
        object_id_highres=object_id,
        object_xi_highres=object_xi,
        zone_id_highres=zone_id,
        geometry_event_mask_highres=target_mask,
        boundary_mask_highres=boundary,
        boundary_fraction_model=boundary_fraction,
        boundary_mask_model=boundary_fraction > 0.0,
        state_fraction_model=state_fraction,
        dominant_object_id_model=dominant,
        zone_id_model=zone_model,
        valid_mask_model=valid_model,
        forward_valid_mask_highres=(
            valid_geometry[:, :-1] & valid_geometry[:, 1:]
        ),
        forward_valid_mask_model=(
            valid_model[:, :-1] & valid_model[:, 1:]
        ),
        object_catalog=object_catalog,
        object_lateral_coefficients=[],
        qc=qc,
    )
