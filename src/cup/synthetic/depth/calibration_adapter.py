"""Translate depth calibration artifacts to the shared scientific calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cup.synthetic.core.calibration import ImpedanceCalibration, WellZoneCurves, calibrate_impedance
from cup.synthetic.depth.config import CALIBRATION_SCHEMA, GENERATOR_FAMILY


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def depth_well_zone_curves_for_object_core(
    *,
    well_name: str,
    spatial_cluster_id: str,
    zone_id: str,
    top_horizon: str,
    bottom_horizon: str,
    tvdss_m: np.ndarray,
    filtered_log_ai: np.ndarray,
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
        filtered_log_ai=np.asarray(filtered_log_ai, dtype=np.float64),
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


def depth_catalog_from_synthetic_truth(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    replacements = {
        "object_top_coordinate": "object_top_tvdss_m",
        "object_bottom_coordinate": "object_bottom_tvdss_m",
        "minimum_extent": "minimum_thickness_m",
        "maximum_extent": "maximum_thickness_m",
    }
    return [
        {replacements.get(key, key): value for key, value in record.items()}
        for record in records
    ]


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
    """Load depth storage and adapt only the object-core seam."""
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
        input_contracts={
            str(key): dict(value)
            for key, value in dict(legacy["input_contracts"]).items()
        },
    )
    return adapter, payload


def calibrate_depth_object_core(
    inputs: Sequence[WellZoneCurves],
    *,
    truth_dz_m: float,
    ordered_horizons: Sequence[str],
    source_runs: Mapping[str, str],
    input_contracts: Mapping[str, Mapping[str, str]],
    state_threshold_sigma: float,
    huber_delta_parent_sigma_floor: float,
    coefficient_sigma_parent_floor: float,
    coefficient_sigma_parent_cap: float,
) -> tuple[ImpedanceCalibration, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the current object-core calibration with depth-domain naming."""
    return calibrate_impedance(
        inputs,
        truth_dt_s=float(truth_dz_m),
        ordered_horizons=list(ordered_horizons),
        source_runs=dict(source_runs),
        input_contracts=input_contracts,
        state_threshold_sigma=float(state_threshold_sigma),
        huber_delta_parent_sigma_floor=float(huber_delta_parent_sigma_floor),
        coefficient_sigma_parent_floor=float(coefficient_sigma_parent_floor),
        coefficient_sigma_parent_cap=float(coefficient_sigma_parent_cap),
    )



__all__ = [
    "calibrate_depth_object_core",
    "depth_catalog_from_object_core",
    "depth_catalog_from_synthetic_truth",
    "depth_frame_from_object_core",
    "depth_payload_from_object_core_calibration",
    "depth_well_zone_curves_for_object_core",
    "load_depth_calibration_for_object_core",
]
