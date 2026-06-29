"""Adapters between depth-domain v2 and the domain-neutral object core.

The first depth Synthoseis slice deliberately reuses the existing object-core
calibration and realization generator.  That core was originally written for
time-domain v1, so its Python interface still uses names such as ``twt_s`` and
``truth_dt_s``.  Keep that vocabulary confined to this Adapter; depth callers
should use TVDSS/metre names everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cup.synthetic.calibration import ImpedanceCalibration, WellZoneCurves, calibrate_impedance
from cup.synthetic.generation import GeneratedSection, GenerationScenario, generate_field_section
from cup.synthetic.depth.config import CALIBRATION_SCHEMA, GENERATOR_FAMILY


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


@dataclass(frozen=True)
class DepthObjectCoreSection:
    tvdss_highres_m: np.ndarray
    tvdss_model_m: np.ndarray
    log_ai_highres: np.ndarray
    model_target_log_ai: np.ndarray
    state_id_highres: np.ndarray
    object_id_highres: np.ndarray
    object_xi_highres: np.ndarray
    zone_id_highres: np.ndarray
    geometry_event_mask_highres: np.ndarray
    boundary_mask_highres: np.ndarray
    boundary_fraction_model: np.ndarray
    boundary_mask_model: np.ndarray
    state_fraction_model: np.ndarray
    dominant_object_id_model: np.ndarray
    zone_id_model: np.ndarray
    categorical: dict[str, np.ndarray]
    object_catalog: list[dict[str, Any]]
    object_lateral_coefficients: list[dict[str, Any]]
    qc: dict[str, Any]


def depth_well_zone_curves_for_object_core(
    *,
    well_name: str,
    spatial_cluster_id: str,
    zone_id: str,
    top_horizon: str,
    bottom_horizon: str,
    tvdss_m: np.ndarray,
    fitted_log_ai: np.ndarray,
    observed_log_ai: np.ndarray,
    zone_top_tvdss_m: float,
    zone_bottom_tvdss_m: float,
) -> WellZoneCurves:
    """Expose one depth well zone through the current object-core interface."""
    return WellZoneCurves(
        well_name=well_name,
        spatial_cluster_id=spatial_cluster_id,
        zone_id=zone_id,
        top_horizon=top_horizon,
        bottom_horizon=bottom_horizon,
        twt_s=np.asarray(tvdss_m, dtype=np.float64),
        filtered_log_ai=np.asarray(fitted_log_ai, dtype=np.float64),
        full_log_ai=np.asarray(observed_log_ai, dtype=np.float64),
        zone_top_s=float(zone_top_tvdss_m),
        zone_bottom_s=float(zone_bottom_tvdss_m),
    )


def depth_frame_from_object_core(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename object-core vertical columns back to depth-domain names."""
    return frame.rename(columns={
        "twt_s": "tvdss_m",
        "duration_s": "thickness_m",
        "zone_duration_s": "zone_thickness_m",
        "minimum_duration_s": "minimum_thickness_m",
        "maximum_duration_s": "maximum_thickness_m",
        "object_top_s": "object_top_tvdss_m",
        "object_bottom_s": "object_bottom_tvdss_m",
    })


def depth_catalog_from_object_core(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    replacements = {
        "object_top_s": "object_top_tvdss_m",
        "object_bottom_s": "object_bottom_tvdss_m",
        "minimum_duration_s": "minimum_thickness_m",
        "maximum_duration_s": "maximum_thickness_m",
    }
    return [{replacements.get(key, key): value for key, value in record.items()} for record in records]


def depth_payload_from_object_core_calibration(
    legacy: ImpedanceCalibration,
    *,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    payload = legacy.to_dict()
    payload["schema_version"] = CALIBRATION_SCHEMA
    payload["generator_family"] = GENERATOR_FAMILY
    payload["truth_dz_m"] = float(payload.pop("truth_dt_s"))
    payload["sample_domain"] = "depth"
    payload["depth_basis"] = "tvdss"
    payload["vertical_axis_unit"] = "m"
    for model in payload["zone_models"].values():
        background = model["background"]
        if "zone_duration_s" in background:
            background["zone_thickness_m"] = background.pop("zone_duration_s")
    per_zone = {
        zone_id: {
            "minimum": float(model["ai_bounds"]["p01"]),
            "maximum": float(model["ai_bounds"]["p99"]),
        }
        for zone_id, model in payload["zone_models"].items()
    }
    payload["generation_log_ai_bounds"] = {
        "per_zone": per_zone,
        "global": {
            "minimum": min(item["minimum"] for item in per_zone.values()),
            "maximum": max(item["maximum"] for item in per_zone.values()),
        },
    }
    payload.update(dict(extra))
    return payload


def load_depth_calibration_for_object_core(path: Path) -> tuple[ImpedanceCalibration, dict[str, Any]]:
    """Load depth v2 storage and adapt only the object-core seam."""
    payload = _json(path)
    if payload.get("schema_version") != CALIBRATION_SCHEMA:
        raise ValueError(f"Expected {CALIBRATION_SCHEMA}, got {payload.get('schema_version')}.")
    legacy = dict(payload)
    legacy["truth_dt_s"] = float(legacy.pop("truth_dz_m"))
    for model in legacy["zone_models"].values():
        background = model["background"]
        if "zone_thickness_m" in background:
            background["zone_duration_s"] = background.pop("zone_thickness_m")
    adapter = ImpedanceCalibration(
        schema_version=CALIBRATION_SCHEMA,
        generator_family=GENERATOR_FAMILY,
        truth_dt_s=float(legacy["truth_dt_s"]),
        state_threshold_sigma=float(legacy["state_threshold_sigma"]),
        ordered_horizons=tuple(legacy["ordered_horizons"]),
        zones=tuple(legacy["zones"]),
        parent=dict(legacy["parent"]),
        zone_models=dict(legacy["zone_models"]),
        source_runs=dict(legacy["source_runs"]),
        source_hashes=dict(legacy["source_hashes"]),
    )
    return adapter, payload


def calibrate_depth_object_core(
    inputs: Sequence[WellZoneCurves],
    *,
    truth_dz_m: float,
    ordered_horizons: Sequence[str],
    source_runs: Mapping[str, str],
    source_hashes: Mapping[str, str],
    state_threshold_sigma: float,
) -> tuple[ImpedanceCalibration, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the current object-core calibration with depth-domain naming."""
    return calibrate_impedance(
        inputs,
        truth_dt_s=float(truth_dz_m),
        ordered_horizons=list(ordered_horizons),
        source_runs=dict(source_runs),
        source_hashes=dict(source_hashes),
        state_threshold_sigma=float(state_threshold_sigma),
    )


def generate_depth_object_core_section(
    calibration: ImpedanceCalibration,
    *,
    realization_id: str,
    scenario: GenerationScenario,
    global_seed: int,
    lateral_m: np.ndarray,
    inline_float: np.ndarray,
    xline_float: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
    horizon_tvdss_m: np.ndarray,
    model_dz_m: float,
    vertical_oversampling_factor: int,
    minimum_highres_cells: int,
    max_global_reversal_fraction: float,
    max_object_reversal_fraction: float,
    vertical_axis_origin_m: float,
    context_extent_m: float,
) -> DepthObjectCoreSection:
    legacy: GeneratedSection = generate_field_section(
        calibration,
        realization_id=realization_id,
        scenario=scenario,
        global_seed=int(global_seed),
        lateral_m=lateral_m,
        inline_float=inline_float,
        xline_float=xline_float,
        x_m=x_m,
        y_m=y_m,
        horizon_twt_s=horizon_tvdss_m,
        output_dt_s=float(model_dz_m),
        wavelet=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        vertical_oversampling_factor=int(vertical_oversampling_factor),
        minimum_truth_samples=int(minimum_highres_cells),
        max_global_reversal_fraction=float(max_global_reversal_fraction),
        max_object_reversal_fraction=float(max_object_reversal_fraction),
        max_global_clipping_fraction=0.0,
        max_object_clipping_fraction=0.0,
        vertical_axis_origin=float(vertical_axis_origin_m),
        context_extent=float(context_extent_m),
    )
    return DepthObjectCoreSection(
        tvdss_highres_m=np.asarray(legacy.twt_highres_s, dtype=np.float64),
        tvdss_model_m=np.asarray(legacy.twt_model_s, dtype=np.float64),
        log_ai_highres=np.asarray(legacy.truth_log_ai_highres, dtype=np.float64),
        model_target_log_ai=np.asarray(legacy.target_log_ai, dtype=np.float64),
        state_id_highres=np.asarray(legacy.state_id_highres),
        object_id_highres=np.asarray(legacy.object_id_highres),
        object_xi_highres=np.asarray(legacy.object_xi_highres),
        zone_id_highres=np.asarray(legacy.zone_id_highres),
        geometry_event_mask_highres=np.asarray(legacy.geometry_event_mask_highres),
        boundary_mask_highres=np.asarray(legacy.boundary_mask_highres),
        boundary_fraction_model=np.asarray(legacy.boundary_fraction_model),
        boundary_mask_model=np.asarray(legacy.boundary_mask_model),
        state_fraction_model=np.asarray(legacy.state_fraction_model),
        dominant_object_id_model=np.asarray(legacy.dominant_object_id_model),
        zone_id_model=np.asarray(legacy.zone_id_model),
        categorical=dict(legacy.categorical),
        object_catalog=depth_catalog_from_object_core(legacy.object_catalog),
        object_lateral_coefficients=depth_catalog_from_object_core(legacy.object_lateral_coefficients),
        qc=dict(legacy.qc),
    )


__all__ = [
    "DepthObjectCoreSection",
    "calibrate_depth_object_core",
    "depth_catalog_from_object_core",
    "depth_frame_from_object_core",
    "depth_payload_from_object_core_calibration",
    "depth_well_zone_curves_for_object_core",
    "generate_depth_object_core_section",
    "load_depth_calibration_for_object_core",
]
