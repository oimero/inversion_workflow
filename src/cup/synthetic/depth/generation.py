"""Depth field-conditioned generation and v2 artifact writer."""

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
from scipy.signal import firwin, hilbert

from cup.petrel.load import import_interpretation_petrel
from cup.physics.numpy_backend import forward_depth, reflectivity_from_log_ai, velocity_from_ai
from cup.seismic.survey import open_survey
from cup.seismic.target_zone import TargetZone
from cup.seismic.wavelet import load_wavelet_csv
from cup.synthetic.calibration import ImpedanceCalibration
from cup.synthetic.figures import write_generation_figures
from cup.synthetic.generation import GenerationRejected, GenerationScenario
from cup.synthetic.generation_pipeline import generation_scenarios
from cup.synthetic.random import named_rng
from cup.synthetic.depth.config import GENERATOR_FAMILY, SCHEMA_VERSION
from cup.synthetic.depth.object_core_adapter import (
    generate_depth_object_core_section,
    load_depth_calibration_for_object_core,
)
from cup.utils.io import array_sha256, repo_relative_path, resolve_relative_path, sha256_file, write_json


@dataclass(frozen=True)
class DepthSectionGeometry:
    section_id: str
    lateral_m: np.ndarray
    inline_float: np.ndarray
    xline_float: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    horizon_tvdss_m: np.ndarray
    qc_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DepthGeneratedSection:
    realization_id: str
    scenario: GenerationScenario
    geometry: DepthSectionGeometry
    tvdss_highres_m: np.ndarray
    tvdss_model_m: np.ndarray
    log_ai_highres: np.ndarray
    vp_highres_mps: np.ndarray
    model_target_log_ai: np.ndarray
    vp_model_mps: np.ndarray
    seismic_observed: np.ndarray
    seismic_model_consistent: np.ndarray
    subgrid_forward_residual: np.ndarray
    lfm_ideal: np.ndarray
    lfm_controlled_degraded: np.ndarray
    valid_mask_model: np.ndarray
    observed_valid_mask: np.ndarray
    physics_valid_mask: np.ndarray
    categorical: dict[str, np.ndarray]
    object_catalog: list[dict[str, Any]]
    object_lateral_coefficients: list[dict[str, Any]]
    qc: dict[str, Any]


def _survey(workflow: Any, *, repo_root: Path) -> Any:
    data_root = resolve_relative_path(workflow.data_root, root=repo_root)
    path = resolve_relative_path(workflow.seismic.file, root=data_root)
    options = {key: value for key, value in workflow.seismic.as_dict().items() if key in {"iline", "xline", "istep", "xstep"}}
    survey = open_survey(path, workflow.seismic.type, segy_options=options or None)
    geometry = survey.line_geometry
    for axis in (geometry.inline_axis, geometry.xline_axis):
        if axis.count <= 0 or not np.isfinite(axis.step) or (axis.count > 1 and axis.step == 0.0):
            raise ValueError(f"Invalid survey line axis: {axis.name}")
    for inline in (geometry.inline_axis.minimum, geometry.inline_axis.maximum):
        for xline in (geometry.xline_axis.minimum, geometry.xline_axis.maximum):
            xy = geometry.line_to_coord(inline, xline)
            restored = geometry.coord_to_line(*xy)
            if not np.allclose(restored, (inline, xline), rtol=0.0, atol=1e-8 * max(abs(geometry.inline_axis.step), abs(geometry.xline_axis.step), 1.0)):
                raise ValueError("Survey line/XY corner round-trip failed.")
    return survey


def _resample_path(
    points: Sequence[Mapping[str, float]],
    *,
    geometry: Any,
    interval_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lines = np.asarray([[float(item["inline"]), float(item["xline"])] for item in points])
    xy = np.asarray([geometry.line_to_coord(il, xl) for il, xl in lines], dtype=np.float64)
    lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    if cumulative[-1] <= 0.0:
        raise ValueError("invalid_section_path")
    lateral = np.arange(0.0, cumulative[-1], interval_m, dtype=np.float64)
    if lateral.size == 0 or not np.isclose(lateral[-1], cumulative[-1]):
        lateral = np.r_[lateral, cumulative[-1]]
    x = np.interp(lateral, cumulative, xy[:, 0])
    y = np.interp(lateral, cumulative, xy[:, 1])
    line_samples = np.asarray([geometry.coord_to_line(xi, yi) for xi, yi in zip(x, y)])
    # Round-trip is a runtime contract, not a fixture tied to a particular step.
    roundtrip = np.asarray([geometry.line_to_coord(il, xl) for il, xl in line_samples])
    tolerance = max(1e-7, 1e-8 * max(float(np.ptp(x)), float(np.ptp(y)), 1.0))
    if not np.allclose(roundtrip, np.column_stack((x, y)), rtol=0.0, atol=tolerance):
        raise ValueError("section_line_xy_roundtrip_failed")
    return lateral, line_samples[:, 0], line_samples[:, 1], x, y


def build_depth_sections(
    *, workflow: Any, script_cfg: Mapping[str, Any], repo_root: Path
) -> tuple[list[DepthSectionGeometry], Any]:
    survey = _survey(workflow, repo_root=repo_root)
    geometry = survey.describe_geometry(domain="depth")
    data_root = resolve_relative_path(workflow.data_root, root=repo_root)
    raw = {
        str(item["name"]): import_interpretation_petrel(resolve_relative_path(item["file"], root=data_root))
        for item in script_cfg["horizons"]
    }
    zone = TargetZone(
        raw,
        geometry,
        [str(item["name"]) for item in script_cfg["horizons"]],
        min_thickness=float(script_cfg["sampling"]["expected_model_dz_m"]),
    )
    surfaces = [zone.get_horizon_surface(str(item["name"])) for item in script_cfg["horizons"]]
    sections: list[DepthSectionGeometry] = []
    for section in script_cfg["sections"]:
        lateral, il, xl, x, y = _resample_path(
            section["path"], geometry=survey.line_geometry,
            interval_m=float(script_cfg["lateral_sample_interval_m"]),
        )
        values = np.column_stack([
            np.asarray([surface.value_at_line(i, j) for i, j in zip(il, xl)], dtype=np.float64)
            for surface in surfaces
        ])
        if np.any(~np.isfinite(values)) or np.any(np.diff(values, axis=1) <= 0.0):
            raise ValueError(f"unsupported_or_crossing_horizons:{section['section_id']}")
        qc = []
        for sample_index, (distance, i, j) in enumerate(zip(lateral, il, xl)):
            index_i = survey.line_geometry.inline_axis.index_of_line(float(i))
            index_j = survey.line_geometry.xline_axis.index_of_line(float(j))
            nearest_i = int(np.clip(round(index_i), 0, zone.no_support_mask.shape[0] - 1))
            nearest_j = int(np.clip(round(index_j), 0, zone.no_support_mask.shape[1] - 1))
            if bool(zone.no_support_mask[nearest_i, nearest_j]):
                raise ValueError(f"section_has_no_interpretation_support:{section['section_id']}:{sample_index}")
            for horizon_index, surface in enumerate(surfaces):
                sample = surface.sample_at_line(float(i), float(j))
                qc.append({
                    "section_id": section["section_id"], "sample_index": sample_index,
                    "lateral_m": distance, "inline_float": i, "xline_float": j,
                    "inline_index_float": index_i, "xline_index_float": index_j,
                    "inline_step": survey.line_geometry.inline_axis.step,
                    "xline_step": survey.line_geometry.xline_axis.step,
                    "horizon_name": script_cfg["horizons"][horizon_index]["name"],
                    "horizon_tvdss_m": sample.value, "sample_method": sample.method,
                    "support_status": sample.support_status,
                })
        sections.append(DepthSectionGeometry(
            section_id=str(section["section_id"]), lateral_m=lateral,
            inline_float=il, xline_float=xl, x_m=x, y_m=y,
            horizon_tvdss_m=values, qc_rows=tuple(qc),
        ))
    return sections, survey


def _antialias_taps(config: Mapping[str, Any], factor: int) -> np.ndarray:
    count = int(config["taps_per_factor"]) * int(factor) + 1
    if count % 2 == 0:
        raise ValueError("Depth antialias FIR must have odd length.")
    return firwin(
        count,
        float(config["cutoff_output_nyquist_fraction"]) / int(factor),
        window=("kaiser", float(config["kaiser_beta"])),
        scale=True,
    ).astype(np.float64)


def _valid_filter_decimate(
    values: np.ndarray, *, factor: int, taps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=np.float64)
    n_model = (data.shape[-1] - 1) // factor + 1
    output = data[..., ::factor].copy()
    valid = np.zeros(n_model, dtype=bool)
    half = taps.size // 2
    model_high_indices = np.arange(n_model, dtype=np.int64) * factor
    supported = (model_high_indices >= half) & (model_high_indices < data.shape[-1] - half)
    for row_index, row in enumerate(data.reshape((-1, data.shape[-1]))):
        filtered = np.convolve(row, taps, mode="valid")
        centers = model_high_indices[supported] - half
        output.reshape((-1, n_model))[row_index, supported] = filtered[centers]
    valid[supported] = True
    return output, valid


def _spatial_lowpass(values: np.ndarray, *, dz_m: float, wavelength_m: float, numtaps: int, beta: float) -> np.ndarray:
    nyquist = 0.5 / dz_m
    cutoff = 1.0 / wavelength_m
    if not 0.0 < cutoff < nyquist:
        raise ValueError("LFM spatial cutoff is outside (0, Nyquist).")
    count = int(numtaps)
    if count < 3 or count % 2 == 0:
        raise ValueError("LFM numtaps must be odd and >= 3.")
    taps = firwin(count, cutoff / nyquist, window=("kaiser", float(beta)), scale=True)
    half = count // 2
    return np.stack([
        np.convolve(np.pad(row, (half, half), mode="edge"), taps, mode="valid")
        for row in np.asarray(values, dtype=np.float64)
    ])


def _phase_rotate(wavelet: np.ndarray, degrees: float) -> np.ndarray:
    analytic = hilbert(np.asarray(wavelet, dtype=np.float64))
    return np.real(analytic * np.exp(1j * np.deg2rad(float(degrees))))


def _shift_wavelet(time_s: np.ndarray, amplitude: np.ndarray, shift_s: float) -> np.ndarray:
    return np.interp(time_s - float(shift_s), time_s, amplitude, left=0.0, right=0.0)


def _target_mask(axis: np.ndarray, horizons: np.ndarray) -> np.ndarray:
    return (axis[None, :] >= horizons[:, :1]) & (axis[None, :] <= horizons[:, -1:])


def _controlled_degraded_lfm(
    base: np.ndarray,
    *,
    config: Mapping[str, Any],
    section: DepthSectionGeometry,
    axis_m: np.ndarray,
    global_seed: int,
    realization_id: str,
) -> np.ndarray:
    output = np.asarray(base, dtype=np.float64).copy()

    def rng(purpose: str) -> np.random.Generator:
        return named_rng(
            global_seed=global_seed, benchmark_version=SCHEMA_VERSION,
            generator_family=GENERATOR_FAMILY, stream_purpose=purpose,
            realization_id=realization_id,
        )

    output += rng("lfm_constant_bias").normal(0.0, float(config["constant_bias_sigma_log_ai"]))
    trend_amplitude = rng("lfm_vertical_trend").normal(0.0, float(config["linear_vertical_trend_sigma_log_ai"]))
    output += trend_amplitude * np.linspace(-1.0, 1.0, axis_m.size)[None, :]
    zone_rng = rng("lfm_zonewise_bias")
    for zone_index in range(section.horizon_tvdss_m.shape[1] - 1):
        bias = zone_rng.normal(0.0, float(config["zonewise_bias_sigma_log_ai"]))
        top = section.horizon_tvdss_m[:, zone_index]
        bottom = section.horizon_tvdss_m[:, zone_index + 1]
        mask = (axis_m[None, :] >= top[:, None]) & (axis_m[None, :] <= bottom[:, None])
        output[mask] += bias
    lateral_rng = rng("lfm_lateral_smooth_bias")
    field = lateral_rng.normal(size=section.lateral_m.size)
    requested = float(config["lateral_correlation_fraction"]) * max(float(section.lateral_m[-1]), 1.0)
    spacing = float(np.median(np.diff(section.lateral_m)))
    alpha = np.exp(-spacing / max(requested, spacing))
    for index in range(1, field.size):
        field[index] = alpha * field[index - 1] + np.sqrt(1.0 - alpha**2) * field[index]
    field *= float(config["lateral_smooth_bias_sigma_log_ai"]) / max(float(np.std(field)), np.finfo(np.float64).eps)
    output += field[:, None]
    center = float(np.mean(output))
    scale = np.exp(rng("lfm_amplitude_scale").normal(0.0, float(config["amplitude_scale_sigma"])))
    output = center + scale * (output - center)
    local = dict(config["local_missing_control_bias"])
    if bool(local["enabled"]):
        local_rng = rng("lfm_local_missing_control_bias")
        center_l = local_rng.uniform(0.2, 0.8) * max(float(section.lateral_m[-1]), 1.0)
        center_z = local_rng.uniform(float(axis_m[0]), float(axis_m[-1]))
        width_l = max(float(local["lateral_width_fraction"]) * max(float(section.lateral_m[-1]), 1.0), spacing)
        width_z = max(float(local["vertical_width_fraction"]) * float(axis_m[-1] - axis_m[0]), float(np.diff(axis_m[:2])[0]))
        amplitude = local_rng.uniform(-1.0, 1.0) * float(local["max_abs_log_ai"])
        blob = np.exp(-0.5 * ((section.lateral_m[:, None] - center_l) / width_l) ** 2 - 0.5 * ((axis_m[None, :] - center_z) / width_z) ** 2)
        output += amplitude * blob
    return output


def generate_depth_realization(
    calibration: ImpedanceCalibration,
    calibration_payload: Mapping[str, Any],
    *,
    section: DepthSectionGeometry,
    scenario: GenerationScenario,
    attempt_id: int,
    script_cfg: Mapping[str, Any],
    forward_inputs: Mapping[str, Any],
    survey: Any,
    repo_root: Path,
) -> DepthGeneratedSection:
    realization_id = f"{section.section_id}__{scenario.scenario_id}__a{attempt_id:03d}"
    wavelet_path = resolve_relative_path(forward_inputs["wavelet"]["path"], root=repo_root)
    wavelet_time, wavelet = load_wavelet_csv(wavelet_path)
    factor = int(script_cfg["sampling"]["vertical_oversampling_factor"])
    model_dz = float(script_cfg["sampling"]["expected_model_dz_m"])
    high_dz = model_dz / factor
    taps = _antialias_taps(script_cfg["sampling"]["antialias"], factor)
    antialias_half_m = (taps.size // 2) * high_dz
    wavelet_half_s = float(np.max(np.abs(wavelet_time)))
    maximum_vp = float(calibration_payload["maximum_allowed_vp_mps"])
    halo = np.ceil((0.5 * maximum_vp * wavelet_half_s) / model_dz) * model_dz
    context = float(halo + antialias_half_m)
    survey_axis = np.asarray(survey.sample_axis(domain="depth").values, dtype=np.float64)
    object_core = generate_depth_object_core_section(
        calibration,
        realization_id=realization_id,
        scenario=scenario,
        global_seed=int(script_cfg["global_seed"]),
        lateral_m=section.lateral_m,
        inline_float=section.inline_float,
        xline_float=section.xline_float,
        x_m=section.x_m,
        y_m=section.y_m,
        horizon_tvdss_m=section.horizon_tvdss_m,
        model_dz_m=model_dz,
        vertical_oversampling_factor=factor,
        minimum_highres_cells=int(script_cfg["impedance"]["minimum_highres_cells"]),
        max_global_reversal_fraction=float(script_cfg["impedance"]["max_global_reversal_fraction"]),
        max_object_reversal_fraction=float(script_cfg["impedance"]["max_object_reversal_fraction"]),
        vertical_axis_origin_m=float(survey_axis[0]),
        context_extent_m=context,
    )
    if float(object_core.qc.get("global_clipping_fraction", 0.0)) != 0.0:
        raise GenerationRejected(["nonzero_v2_clipping"], diagnostics=dict(object_core.qc), details=[])
    high_axis = np.asarray(object_core.tvdss_highres_m, dtype=np.float64)
    model_axis = np.asarray(object_core.tvdss_model_m, dtype=np.float64)
    if model_axis.size < 2 or not np.array_equal(high_axis[::factor], model_axis):
        raise ValueError("Depth highres/model axes are not strictly nested.")
    survey_indices = np.searchsorted(survey_axis, model_axis)
    if np.any(survey_indices >= survey_axis.size) or not np.allclose(survey_axis[survey_indices], model_axis, rtol=0.0, atol=1e-9):
        raise ValueError(f"section_context_outside_survey_axis:{section.section_id}")

    log_high = np.asarray(object_core.log_ai_highres, dtype=np.float64)
    model_log, antialias_valid_1d = _valid_filter_decimate(log_high, factor=factor, taps=taps)
    relation = forward_inputs["ai_velocity_relation"]
    ai_high = np.exp(log_high)
    ai_model = np.exp(model_log)
    vp_high = velocity_from_ai(ai_high, a=float(relation["a"]), b=float(relation["b"]))
    vp_model = velocity_from_ai(ai_model, a=float(relation["a"]), b=float(relation["b"]))
    seismic_high = forward_depth(log_high, vp_high, high_axis, wavelet_time, wavelet)
    seismic_observed, seismic_antialias_valid = _valid_filter_decimate(seismic_high, factor=factor, taps=taps)
    seismic_model = forward_depth(model_log, vp_model, model_axis, wavelet_time, wavelet)
    residual = seismic_observed - seismic_model
    target = _target_mask(model_axis, section.horizon_tvdss_m)
    valid = target & antialias_valid_1d[None, :] & seismic_antialias_valid[None, :]

    ideal_cfg = script_cfg["lfm"]["ideal"]
    degraded_cfg = script_cfg["lfm"]["controlled_degraded"]
    lfm_ideal = _spatial_lowpass(model_log, dz_m=model_dz, wavelength_m=float(ideal_cfg["minimum_wavelength_m"]), numtaps=int(ideal_cfg["numtaps"]), beta=float(ideal_cfg["kaiser_beta"]))
    lfm_degraded = _spatial_lowpass(model_log, dz_m=model_dz, wavelength_m=float(degraded_cfg["minimum_wavelength_m"]), numtaps=int(degraded_cfg["numtaps"]), beta=float(degraded_cfg["kaiser_beta"]))
    lfm_degraded = _controlled_degraded_lfm(
        lfm_degraded, config=degraded_cfg, section=section, axis_m=model_axis,
        global_seed=int(script_cfg["global_seed"]), realization_id=realization_id,
    )

    reconstructed = forward_depth(model_log, vp_model, model_axis, wavelet_time, wavelet)
    if not np.allclose(reconstructed, seismic_model, rtol=1e-12, atol=1e-12):
        raise ValueError("model_grid_forward_closure_failed")
    categorical = {
        "state_id_highres": object_core.state_id_highres,
        "object_id_highres": object_core.object_id_highres,
        "object_xi_highres": object_core.object_xi_highres,
        "zone_id_highres": object_core.zone_id_highres,
        "geometry_event_mask_highres": object_core.geometry_event_mask_highres,
        "boundary_mask_highres": object_core.boundary_mask_highres,
        "boundary_fraction_model": object_core.boundary_fraction_model,
        "boundary_mask_model": object_core.boundary_mask_model,
        "state_fraction_model": object_core.state_fraction_model,
        "dominant_object_id_model": object_core.dominant_object_id_model,
        "zone_id_model": object_core.zone_id_model,
    }
    observed_values = seismic_observed[valid]
    model_values = seismic_model[valid]
    residual_values = residual[valid]
    observed_rms = float(np.sqrt(np.mean(observed_values**2)))
    model_rms = float(np.sqrt(np.mean(model_values**2)))
    residual_rms = float(np.sqrt(np.mean(residual_values**2)))
    correlation = float(np.corrcoef(observed_values, model_values)[0, 1]) if observed_values.size >= 2 and np.std(observed_values) > 0.0 and np.std(model_values) > 0.0 else float("nan")
    return DepthGeneratedSection(
        realization_id=realization_id, scenario=scenario, geometry=section,
        tvdss_highres_m=high_axis, tvdss_model_m=model_axis,
        log_ai_highres=log_high, vp_highres_mps=vp_high,
        model_target_log_ai=model_log, vp_model_mps=vp_model,
        seismic_observed=seismic_observed, seismic_model_consistent=seismic_model,
        subgrid_forward_residual=residual, lfm_ideal=lfm_ideal,
        lfm_controlled_degraded=lfm_degraded, valid_mask_model=target,
        observed_valid_mask=valid, physics_valid_mask=valid.copy(),
        categorical=categorical,
        object_catalog=object_core.object_catalog,
        object_lateral_coefficients=object_core.object_lateral_coefficients,
        qc={
            **{key: value for key, value in object_core.qc.items() if key != "field_qc"},
            "physics_halo_m": float(halo), "antialias_filter_half_width_m": float(antialias_half_m),
            "physics_halo_samples": int(round(halo / model_dz)),
            "context_m": context, "maximum_allowed_vp_mps": maximum_vp,
            "highres_seismic_sha256": array_sha256(seismic_high),
            "antialias_taps_sha256": array_sha256(taps),
            "antialias_numtaps": int(taps.size), "antialias_scipy_version": scipy.__version__,
            "seismic_observed_rms": observed_rms,
            "seismic_model_consistent_rms": model_rms,
            "subgrid_residual_rms": residual_rms,
            "subgrid_residual_nrmse": residual_rms / max(observed_rms, np.finfo(np.float64).eps),
            "subgrid_observed_model_correlation": correlation,
            "subgrid_amplitude_scale_ratio": observed_rms / max(model_rms, np.finfo(np.float64).eps),
        },
    )


def build_attempt_plan(script_cfg: Mapping[str, Any], sections: Sequence[DepthSectionGeometry]) -> pd.DataFrame:
    scenarios = generation_scenarios(script_cfg)
    rows = []
    attempts = int(script_cfg["generation"]["attempts_per_scenario"])
    held_out = str(script_cfg["splits"]["held_out_geometry_family"])
    for section in sections:
        for scenario in scenarios:
            for attempt_id in range(attempts):
                parent = f"{section.section_id}__{scenario.scenario_id}__a{attempt_id:03d}"
                rows.append({
                    "parent_realization_id": parent, "section_id": section.section_id,
                    "scenario_id": scenario.scenario_id, "geometry_family": scenario.geometry_family,
                    "geometry_direction": scenario.geometry_direction, "duration_mode": scenario.duration_mode,
                    "attempt_id": attempt_id,
                    "evaluation_role": "geometry_holdout" if scenario.geometry_family == held_out else "development_pool",
                })
    return pd.DataFrame.from_records(rows)


def _dataset(group: h5py.Group, name: str, values: np.ndarray, *, unit: str, axis_path: str, axis_order: str) -> h5py.Dataset:
    data = np.asarray(values)
    dataset = group.create_dataset(name, data=data, compression="gzip", shuffle=True)
    dataset.attrs["unit"] = unit
    dataset.attrs["sample_domain"] = "depth"
    dataset.attrs["axis_path"] = axis_path
    dataset.attrs["axis_order"] = axis_order
    dataset.attrs["shape_json"] = json.dumps(list(data.shape))
    dataset.attrs["dtype"] = str(data.dtype)
    dataset.attrs["sha256"] = array_sha256(data)
    return dataset


def _write_base(h5: h5py.File, section: DepthGeneratedSection) -> str:
    path = f"/realizations/{section.realization_id}"
    root = h5.create_group(path)
    root.attrs["sample_domain"] = "depth"
    root.attrs["depth_basis"] = "tvdss"
    axes = root.create_group("axes")
    _dataset(axes, "lateral_m", section.geometry.lateral_m, unit="m", axis_path=f"{path}/axes/lateral_m", axis_order="lateral")
    _dataset(axes, "tvdss_highres_m", section.tvdss_highres_m, unit="m", axis_path=f"{path}/axes/tvdss_highres_m", axis_order="tvdss_highres")
    _dataset(axes, "tvdss_model_m", section.tvdss_model_m, unit="m", axis_path=f"{path}/axes/tvdss_model_m", axis_order="tvdss_model")
    for name, values in (("inline_float", section.geometry.inline_float), ("xline_float", section.geometry.xline_float), ("x_m", section.geometry.x_m), ("y_m", section.geometry.y_m)):
        _dataset(axes, name, values, unit="line" if "line" in name else "m", axis_path=f"{path}/axes/lateral_m", axis_order="lateral")
    truth = root.create_group("truth")
    for name, values, unit, axis in (
        ("log_ai_highres", section.log_ai_highres, "ln(m/s*g/cm3)", "tvdss_highres_m"),
        ("vp_highres_mps", section.vp_highres_mps, "m/s", "tvdss_highres_m"),
        ("model_target_log_ai", section.model_target_log_ai, "ln(m/s*g/cm3)", "tvdss_model_m"),
        ("vp_model_mps", section.vp_model_mps, "m/s", "tvdss_model_m"),
    ):
        _dataset(truth, name, values.astype(np.float32), unit=unit, axis_path=f"{path}/axes/{axis}", axis_order="lateral,tvdss")
    categorical = truth.create_group("categorical")
    for name, values in section.categorical.items():
        axis = "tvdss_highres_m" if values.shape[1] == section.tvdss_highres_m.size else "tvdss_model_m"
        order = "lateral,tvdss,state" if values.ndim == 3 else "lateral,tvdss"
        _dataset(categorical, name, values, unit="category", axis_path=f"{path}/axes/{axis}", axis_order=order)
    seismic = root.create_group("seismic")
    for name, values in (("seismic_observed", section.seismic_observed), ("seismic_model_consistent", section.seismic_model_consistent), ("subgrid_forward_residual", section.subgrid_forward_residual)):
        _dataset(seismic, name, values.astype(np.float32), unit="amplitude", axis_path=f"{path}/axes/tvdss_model_m", axis_order="lateral,tvdss")
    priors = root.create_group("priors")
    _dataset(priors, "lfm_ideal", section.lfm_ideal.astype(np.float32), unit="ln(m/s*g/cm3)", axis_path=f"{path}/axes/tvdss_model_m", axis_order="lateral,tvdss")
    _dataset(priors, "lfm_controlled_degraded", section.lfm_controlled_degraded.astype(np.float32), unit="ln(m/s*g/cm3)", axis_path=f"{path}/axes/tvdss_model_m", axis_order="lateral,tvdss")
    masks = root.create_group("masks")
    for name, values in (("valid_mask_model", section.valid_mask_model), ("observed_valid_mask", section.observed_valid_mask), ("physics_valid_mask", section.physics_valid_mask)):
        _dataset(masks, name, values, unit="bool", axis_path=f"{path}/axes/tvdss_model_m", axis_order="lateral,tvdss")
    qc = root.create_group("qc")
    for key, value in section.qc.items():
        if np.isscalar(value) and not isinstance(value, (dict, list, tuple)):
            qc.attrs[key] = value
    return path


def _static_shift(
    data: np.ndarray,
    axis: np.ndarray,
    shift_m: float,
    *,
    source_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros_like(data)
    valid = np.zeros_like(data, dtype=bool)
    source = axis - float(shift_m)
    inside = (source >= axis[0]) & (source <= axis[-1])
    for index, trace in enumerate(data):
        output[index, inside] = np.interp(source[inside], axis, trace)
        positions = np.searchsorted(axis, source[inside], side="left")
        positions = np.clip(positions, 0, axis.size - 1)
        exact = np.isclose(axis[positions], source[inside], rtol=0.0, atol=1e-12)
        lower = np.maximum(positions - 1, 0)
        upper = positions
        source_valid = np.asarray(source_valid_mask[index], dtype=bool)
        supported = np.where(
            exact,
            source_valid[upper],
            source_valid[lower] & source_valid[upper],
        )
        valid[index, np.flatnonzero(inside)] = supported
    return output, valid


def _write_variants(
    h5: h5py.File,
    section: DepthGeneratedSection,
    *,
    script_cfg: Mapping[str, Any],
    forward_inputs: Mapping[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    config = script_cfg["seismic_mismatch"]
    if not bool(config.get("enabled", True)):
        return []
    root_path = f"/realizations/{section.realization_id}"
    variants = h5[root_path].create_group("seismic_variants")
    rows = []
    base_mask = section.observed_valid_mask

    def persist(variant_id: str, family: str, values: np.ndarray, mask: np.ndarray, parameters: Mapping[str, Any]) -> None:
        group = variants.create_group(variant_id)
        _dataset(group, "seismic_observed", values.astype(np.float32), unit="amplitude", axis_path=f"{root_path}/axes/tvdss_model_m", axis_order="lateral,tvdss")
        _dataset(group, "observed_valid_mask", mask, unit="bool", axis_path=f"{root_path}/axes/tvdss_model_m", axis_order="lateral,tvdss")
        group.attrs["mismatch_family"] = family
        group.attrs["parameters_json"] = json.dumps(dict(parameters), sort_keys=True)
        rows.append({"variant_id": variant_id, "mismatch_family": family, **dict(parameters)})

    wavelet_time, wavelet = load_wavelet_csv(resolve_relative_path(forward_inputs["wavelet"]["path"], root=repo_root))
    factor = int(script_cfg["sampling"]["vertical_oversampling_factor"])
    taps = _antialias_taps(script_cfg["sampling"]["antialias"], factor)
    wave_cfg = dict(config.get("wavelet") or {})
    for degrees in wave_cfg.get("phase_rotation_degrees", []):
        perturbed = _phase_rotate(wavelet, float(degrees))
        high = forward_depth(section.log_ai_highres, section.vp_highres_mps, section.tvdss_highres_m, wavelet_time, perturbed)
        observed, valid_1d = _valid_filter_decimate(high, factor=factor, taps=taps)
        persist(f"wavelet_phase_{float(degrees):+g}deg", "wavelet_phase", observed, base_mask & valid_1d[None, :], {"phase_rotation_degrees": float(degrees)})
    for shift_s in wave_cfg.get("time_shift_s", []):
        perturbed = _shift_wavelet(wavelet_time, wavelet, float(shift_s))
        high = forward_depth(section.log_ai_highres, section.vp_highres_mps, section.tvdss_highres_m, wavelet_time, perturbed)
        observed, valid_1d = _valid_filter_decimate(high, factor=factor, taps=taps)
        persist(f"wavelet_shift_{float(shift_s)*1000:+g}ms", "wavelet_time_shift", observed, base_mask & valid_1d[None, :], {"wavelet_time_shift_s": float(shift_s)})
    for shift_m in dict(config.get("depth_static") or {}).get("shift_m", []):
        shifted, shifted_valid = _static_shift(
            section.seismic_observed,
            section.tvdss_model_m,
            float(shift_m),
            source_valid_mask=base_mask,
        )
        persist(
            f"depth_static_{float(shift_m):+g}m",
            "depth_static",
            shifted,
            shifted_valid,
            {"depth_static_m": float(shift_m)},
        )

    rms = float(np.sqrt(np.mean(section.seismic_observed[base_mask] ** 2)))
    noise_cfg = dict(config.get("noise") or {})
    for name, fraction in (("white_noise", noise_cfg.get("white_noise_rms_fraction")), ("colored_noise", noise_cfg.get("colored_noise_rms_fraction"))):
        if fraction is None:
            continue
        rng = named_rng(global_seed=int(script_cfg["global_seed"]), benchmark_version=SCHEMA_VERSION, generator_family=GENERATOR_FAMILY, stream_purpose=name, realization_id=section.realization_id)
        noise = rng.standard_normal(section.seismic_observed.shape)
        if name == "colored_noise":
            length = max(float(noise_cfg.get("colored_vertical_correlation_m", 25.0)) / float(np.diff(section.tvdss_model_m[:2])[0]), 1.0)
            alpha = np.exp(-1.0 / length)
            for index in range(1, noise.shape[-1]):
                noise[:, index] = alpha * noise[:, index - 1] + np.sqrt(1.0 - alpha**2) * noise[:, index]
        noise *= float(fraction) * rms / max(float(np.sqrt(np.mean(noise**2))), np.finfo(np.float64).eps)
        persist(name, name, section.seismic_observed + noise, base_mask, {"noise_rms_fraction": float(fraction)})
    gain_cfg = dict(config.get("gain") or {})
    for name, sigma in (("global_gain", gain_cfg.get("global_log_sigma")), ("tracewise_gain", gain_cfg.get("tracewise_log_sigma"))):
        if sigma is None:
            continue
        rng = named_rng(global_seed=int(script_cfg["global_seed"]), benchmark_version=SCHEMA_VERSION, generator_family=GENERATOR_FAMILY, stream_purpose=name, realization_id=section.realization_id)
        shape = (1, 1) if name == "global_gain" else (section.seismic_observed.shape[0], 1)
        gain = np.exp(rng.normal(0.0, float(sigma), size=shape))
        persist(name, name, section.seismic_observed * gain, base_mask, {"gain_log_sigma": float(sigma)})
    return rows


def run_depth_generation(
    *,
    workflow: Any,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    source_provenance: Mapping[str, Any],
    forward_inputs: Mapping[str, Any],
    config_provenance: Mapping[str, str],
    calibration_path: Path,
    repo_root: Path,
    output_dir: Path,
    debug_attempt_limit: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    calibration, calibration_payload = load_depth_calibration_for_object_core(calibration_path)
    if calibration_payload.get("forward_model_inputs_sha256") != forward_inputs["_sha256"]:
        raise ValueError("impedance calibration forward_model_inputs SHA-256 mismatch.")
    if list(calibration_payload.get("horizon_contract") or []) != list(script_cfg["horizons"]):
        raise ValueError("impedance calibration horizon contract differs from current common config.")
    expected_truth_dz = float(script_cfg["sampling"]["expected_model_dz_m"]) / int(script_cfg["sampling"]["vertical_oversampling_factor"])
    if not np.isclose(float(calibration_payload["truth_dz_m"]), expected_truth_dz, rtol=0.0, atol=1e-12):
        raise ValueError("impedance calibration truth_dz_m differs from current sampling config.")
    for key, path in sources.items():
        recorded = str(dict(calibration_payload.get("source_runs") or {}).get(key) or "")
        if not recorded or resolve_relative_path(recorded, root=repo_root).resolve() != path.resolve():
            raise ValueError(f"impedance calibration source run mismatch: {key}")
    sections, survey = build_depth_sections(workflow=workflow, script_cfg=script_cfg, repo_root=repo_root)
    pd.DataFrame([row for section in sections for row in section.qc_rows]).to_csv(output_dir / "section_geometry_qc.csv", index=False)
    plan = build_attempt_plan(script_cfg, sections)
    if debug_attempt_limit is not None:
        plan = plan.groupby(["section_id", "scenario_id"], sort=False).head(int(debug_attempt_limit)).reset_index(drop=True)
    plan.to_csv(output_dir / "attempt_plan.csv", index=False)
    scenarios = {item.scenario_id: item for item in generation_scenarios(script_cfg)}
    sections_by_id = {item.section_id: item for item in sections}
    index_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    highres_rows: list[dict[str, Any]] = []
    subgrid_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    h5_path = output_dir / "synthetic_benchmark.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["schema"] = SCHEMA_VERSION
        h5.attrs["sample_domain"] = "depth"
        h5.attrs["depth_basis"] = "tvdss"
        h5.attrs["axis_positive_direction"] = "down"
        h5.attrs["generator_family"] = GENERATOR_FAMILY
        h5.attrs["forward_model_inputs_sha256"] = forward_inputs["_sha256"]
        h5.attrs["impedance_calibration_sha256"] = sha256_file(calibration_path)
        for row in plan.to_dict(orient="records"):
            section = sections_by_id[str(row["section_id"])]
            scenario = scenarios[str(row["scenario_id"])]
            try:
                generated = generate_depth_realization(
                    calibration, calibration_payload, section=section, scenario=scenario,
                    attempt_id=int(row["attempt_id"]), script_cfg=script_cfg,
                    forward_inputs=forward_inputs, survey=survey, repo_root=repo_root,
                )
                group_path = _write_base(h5, generated)
                base_id = generated.realization_id
                common = {
                    "sample_domain": "depth", "depth_basis": "tvdss", "status": "ok",
                    "suite": "field_conditioned",
                    "parent_realization_id": base_id, "section_id": section.section_id,
                    "scenario_id": scenario.scenario_id, "geometry_family": scenario.geometry_family,
                    "geometry_direction": scenario.geometry_direction, "duration_mode": scenario.duration_mode,
                    "evaluation_role": row["evaluation_role"],
                    "held_out_geometry_family": script_cfg["splits"]["held_out_geometry_family"],
                    "model_sample_count": generated.tvdss_model_m.size,
                    "model_dz_m": float(np.diff(generated.tvdss_model_m[:2])[0]),
                    "physics_halo_m": generated.qc["physics_halo_m"],
                    "physics_halo_samples": generated.qc["physics_halo_samples"],
                    "forward_model_inputs_sha256": forward_inputs["_sha256"],
                }
                local_index_rows = [{
                    **common, "sample_id": base_id, "sample_kind": "base", "source_sample_id": "",
                    "hdf5_group": group_path, "seismic_input_dataset": f"{group_path}/seismic/seismic_observed",
                    "physics_target_dataset": f"{group_path}/seismic/seismic_model_consistent",
                }]
                local_variant_rows = []
                generated_variants = _write_variants(
                    h5,
                    generated,
                    script_cfg=script_cfg,
                    forward_inputs=forward_inputs,
                    repo_root=repo_root,
                )
                for variant in generated_variants:
                    variant_id = f"{base_id}__{variant['variant_id']}"
                    variant_path = f"{group_path}/seismic_variants/{variant['variant_id']}"
                    local_index_rows.append({
                        **common, "sample_id": variant_id, "sample_kind": "seismic_variant",
                        "source_sample_id": base_id, "hdf5_group": variant_path,
                        "seismic_input_dataset": f"{variant_path}/seismic_observed",
                        "physics_target_dataset": f"{group_path}/seismic/seismic_model_consistent",
                        "mismatch_family": variant["mismatch_family"],
                        "mismatch_parameters_json": json.dumps(
                            {
                                key: value
                                for key, value in variant.items()
                                if key not in {"variant_id", "mismatch_family"}
                            },
                            sort_keys=True,
                        ),
                    })
                    local_variant_rows.append({"parent_realization_id": base_id, **variant})
                local_highres_row = {
                    "parent_realization_id": base_id,
                    "highres_seismic_sha256": generated.qc["highres_seismic_sha256"],
                    "physics_halo_m": generated.qc["physics_halo_m"],
                    "antialias_filter_half_width_m": generated.qc["antialias_filter_half_width_m"],
                    "context_m": generated.qc["context_m"],
                    "vertical_oversampling_factor": int(script_cfg["sampling"]["vertical_oversampling_factor"]),
                    "highres_dz_m": float(np.diff(generated.tvdss_highres_m[:2])[0]),
                    "model_dz_m": float(np.diff(generated.tvdss_model_m[:2])[0]),
                    "antialias_numtaps": generated.qc["antialias_numtaps"],
                    "antialias_taps_sha256": generated.qc["antialias_taps_sha256"],
                }
                local_subgrid_row = {
                    "parent_realization_id": base_id,
                    "seismic_observed_rms": generated.qc["seismic_observed_rms"],
                    "seismic_model_consistent_rms": generated.qc["seismic_model_consistent_rms"],
                    "subgrid_residual_rms": generated.qc["subgrid_residual_rms"],
                    "subgrid_residual_nrmse": generated.qc["subgrid_residual_nrmse"],
                    "subgrid_observed_model_correlation": generated.qc["subgrid_observed_model_correlation"],
                    "subgrid_amplitude_scale_ratio": generated.qc["subgrid_amplitude_scale_ratio"],
                }
                # Commit tabular records only after the complete HDF5 parent,
                # including every configured variant and QC row, is ready.
                index_rows.extend(local_index_rows)
                variant_rows.extend(local_variant_rows)
                object_rows.extend(generated.object_catalog)
                coefficient_rows.extend(generated.object_lateral_coefficients)
                highres_rows.append(local_highres_row)
                subgrid_rows.append(local_subgrid_row)
            except (GenerationRejected, ValueError, FloatingPointError) as exc:
                failed_parent = str(row["parent_realization_id"])
                failed_group = f"/realizations/{failed_parent}"
                if failed_group in h5:
                    del h5[failed_group]
                rejection_rows.append({**row, "status": "rejected", "reason": f"{type(exc).__name__}:{exc}"})

    index = pd.DataFrame.from_records(index_rows)
    index.to_csv(output_dir / "sample_index.csv", index=False)
    pd.DataFrame.from_records(object_rows).to_csv(output_dir / "object_catalog.csv", index=False)
    pd.DataFrame.from_records(coefficient_rows).to_csv(output_dir / "object_lateral_coefficients.csv", index=False)
    pd.DataFrame.from_records(rejection_rows).to_csv(output_dir / "generation_rejection_details.csv", index=False)
    pd.DataFrame.from_records(highres_rows).to_csv(output_dir / "highres_forward_qc.csv", index=False)
    pd.DataFrame.from_records(subgrid_rows).to_csv(output_dir / "subgrid_forward_qc.csv", index=False)
    pd.DataFrame.from_records(variant_rows).to_csv(output_dir / "seismic_variant_results.csv", index=False)
    base = index[index.get("sample_kind", pd.Series(dtype=str)).eq("base")].copy() if not index.empty else index
    accepted = base.groupby(["section_id", "scenario_id"]).size().rename("accepted_count") if not base.empty else pd.Series(dtype=int)
    attempted = plan.groupby(["section_id", "scenario_id"]).size().rename("attempt_count")
    catalog = attempted.to_frame().join(accepted, how="left").fillna({"accepted_count": 0}).reset_index()
    catalog["accepted_count"] = catalog["accepted_count"].astype(int)
    catalog["acceptance_fraction"] = catalog["accepted_count"] / catalog["attempt_count"]
    qc_cfg = script_cfg["generation"]["acceptance_qc"]
    catalog["acceptance_status"] = np.where(
        catalog["attempt_count"] < int(qc_cfg["minimum_attempts_per_scenario"]), "insufficient_attempts",
        np.where(catalog["acceptance_fraction"] < float(qc_cfg["failure_fraction"]), "failed", np.where(catalog["acceptance_fraction"] < float(qc_cfg["warning_fraction"]), "warning", "ok")),
    )
    catalog.to_csv(output_dir / "scenario_catalog.csv", index=False)
    catalog.to_csv(output_dir / "generation_qc.csv", index=False)
    development = debug_attempt_limit is not None
    failure_reason = "depth_generation_no_accepted_realizations" if base.empty else ""
    if not development and not base.empty:
        if catalog["acceptance_status"].isin({"failed", "insufficient_attempts"}).any():
            failure_reason = "depth_generation_acceptance_qc_failed"

    figure_summary = write_generation_figures(
        output_dir,
        script_cfg.get("figures", {}),
        suite="field_conditioned",
        qc_only=False,
    )

    file_names = [
        "synthetic_benchmark.h5", "sample_index.csv", "attempt_plan.csv", "scenario_catalog.csv",
        "generation_qc.csv", "generation_rejection_details.csv", "object_catalog.csv",
        "object_lateral_coefficients.csv", "highres_forward_qc.csv", "subgrid_forward_qc.csv",
        "seismic_variant_results.csv", "section_geometry_qc.csv", "figures/figure_manifest.json",
    ]
    manifest = {
        "schema": SCHEMA_VERSION, "status": "development_limited" if development else ("failed" if failure_reason else "success"),
        "sample_domain": "depth", "depth_basis": "tvdss", "generator_family": GENERATOR_FAMILY,
        "forward_model_inputs_sha256": forward_inputs["_sha256"],
        "forward_model_inputs_path": str(forward_inputs["_path"]),
        "forward_inputs": {
            "wavelet_sha256": forward_inputs["wavelet"]["sha256"],
            "ai_velocity_relation_sha256": forward_inputs["ai_velocity_relation"]["sha256"],
            "well_input_inventory_sha256": forward_inputs["well_input_inventory_sha256"],
            "source_preprocessed_las": list(forward_inputs["source_preprocessed_las"]),
        },
        "impedance_calibration": repo_relative_path(calibration_path, root=repo_root),
        "impedance_calibration_sha256": sha256_file(calibration_path),
        "canonical_enabled": False, "probe_enabled": False,
        "sampling": dict(script_cfg["sampling"]),
        "lfm": dict(script_cfg["lfm"]),
        "seismic_mismatch": dict(script_cfg["seismic_mismatch"]),
        "source_runs": {key: repo_relative_path(path, root=repo_root) for key, path in sources.items()},
        "source_provenance": dict(source_provenance), "config_provenance": dict(config_provenance),
        "figures": {
            "generated_count": int(figure_summary.get("generated_count", 0)),
            "skipped_count": int(figure_summary.get("skipped_count", 0)),
            "figure_manifest": repo_relative_path(
                Path(str(figure_summary.get("figure_manifest", output_dir / "figures" / "figure_manifest.json"))),
                root=repo_root,
            ),
        },
        "split_policy": {
            "assignment_unit": "parent_realization",
            "held_out_geometry_family": script_cfg["splits"]["held_out_geometry_family"],
            "split_assignment_owner": "training",
        },
        "sample_counts": {
            "by_evaluation_role": {str(key): int(value) for key, value in base.groupby("evaluation_role").size().items()} if not base.empty else {},
            "seen_parent_realizations": int((~base["geometry_family"].eq(script_cfg["splits"]["held_out_geometry_family"])).sum()) if not base.empty else 0,
            "held_out_parent_realizations": int(base["geometry_family"].eq(script_cfg["splits"]["held_out_geometry_family"]).sum()) if not base.empty else 0,
        },
        "random_stream": {
            "algorithm": "SHA-256/PCG64DXSM",
            "benchmark_version": SCHEMA_VERSION,
            "stream_purpose_registry": [
                "state_sequence", "duration", "zone_background",
                "coefficient_<name>", "coefficient_lateral", "thickness_lateral",
                "lfm_constant_bias", "lfm_vertical_trend", "lfm_zone_bias",
                "lfm_lateral_bias", "lfm_amplitude_scale", "lfm_local_missing_control",
                "white_noise", "colored_noise", "global_gain", "tracewise_gain",
            ],
        },
        "files": {name: sha256_file(output_dir / name) for name in file_names},
    }
    write_json(output_dir / "benchmark_manifest.json", manifest)
    summary = {**manifest, "accepted_parent_realizations": int(len(base)), "rejected_parent_realizations": int(len(rejection_rows))}
    write_json(output_dir / "run_summary.json", summary)
    if failure_reason:
        raise RuntimeError(failure_reason)
    return summary


__all__ = ["build_attempt_plan", "build_depth_sections", "run_depth_generation"]
