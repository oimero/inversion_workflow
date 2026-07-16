"""Depth-domain calibration adapters for Synthoseis-lite v4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cup.petrel.load import import_interpretation_petrel, import_well_tops_petrel
from cup.seismic.survey import open_survey
from cup.seismic.target_zone import TargetZone
from cup.synthetic.reporting.figures import write_calibration_figures
from cup.synthetic.schemas import SCIENCE_CONTRACT
from cup.synthetic.depth.config import CALIBRATION_SCHEMA, GENERATOR_FAMILY
from cup.synthetic.depth.calibration_adapter import (
    calibrate_depth_object_core,
    depth_frame_from_object_core,
    depth_payload_from_object_core_calibration,
    depth_well_zone_curves_for_object_core,
    load_depth_calibration_for_object_core,
)
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    repo_relative_path,
    resolve_relative_path,
    write_json,
)
from cup.utils.statistics import radius_connected_components
from cup.well.assets import normalize_well_name
from cup.well.las import read_las_curve
from cup.well.td import find_well_top_md


def _required_file(path_value: Any, *, repo_root: Path, label: str) -> Path:
    path = resolve_relative_path(str(path_value), root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _load_step5_shifted_las_sources(
    metrics_path: Path,
    *,
    repo_root: Path,
) -> list[dict[str, Any]]:
    metrics = pd.read_csv(metrics_path)
    required = {
        "well_name",
        "status",
        "shifted_preprocessed_las_path",
        "shifted_filtered_las_path",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"Step 5 wavelet_batch_metrics.csv lacks columns: {missing}")
    rows: list[dict[str, Any]] = []
    for row in metrics.to_dict(orient="records"):
        if str(row.get("status", "")).casefold() != "ok":
            continue
        well_name = str(row["well_name"])
        full_path = _required_file(
            row["shifted_preprocessed_las_path"],
            repo_root=repo_root,
            label=f"{well_name} shifted_preprocessed_las",
        )
        filtered_path = _required_file(
            row["shifted_filtered_las_path"],
            repo_root=repo_root,
            label=f"{well_name} shifted_filtered_las",
        )
        rows.append(
            {
                "well_id": well_name,
                "full_las_path": repo_relative_path(full_path, root=repo_root),
                "filtered_las_path": repo_relative_path(filtered_path, root=repo_root),
            }
        )
    if not rows:
        raise ValueError("Step 5 wavelet_batch_metrics.csv has no successful shifted LAS wells.")
    return rows


def _continuous_cell_average(
    axis_m: np.ndarray,
    values: np.ndarray,
    *,
    centers_m: np.ndarray,
    cell_size_m: float,
) -> np.ndarray:
    """Average a piecewise-linear curve without bridging non-finite runs."""
    axis = np.asarray(axis_m, dtype=np.float64).reshape(-1)
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    centers = np.asarray(centers_m, dtype=np.float64).reshape(-1)
    if axis.shape != data.shape or axis.size < 2 or np.any(np.diff(axis) <= 0.0):
        raise ValueError("LAS TVDSS axis must be strictly increasing and aligned with AI.")
    result = np.full(centers.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(axis) & np.isfinite(data) & (data > 0.0)
    changes = np.diff(np.r_[False, finite, False].astype(np.int8))
    half = 0.5 * float(cell_size_m)
    for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        if stop - start < 2:
            continue
        x_run = axis[start:stop]
        y_run = np.log(data[start:stop])
        candidates = np.flatnonzero(
            (centers - half >= x_run[0]) & (centers + half <= x_run[-1])
        )
        for index in candidates:
            left = centers[index] - half
            right = centers[index] + half
            interior = (x_run > left) & (x_run < right)
            x = np.r_[left, x_run[interior], right]
            y = np.r_[
                np.interp(left, x_run, y_run),
                y_run[interior],
                np.interp(right, x_run, y_run),
            ]
            result[index] = float(np.trapezoid(y, x) / cell_size_m)
    return result


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="mergesort")
    data = values[order]
    mass = weights[order]
    return float(data[min(np.searchsorted(np.cumsum(mass), 0.5 * np.sum(mass)), data.size - 1)])


def _huber_background(
    zeta: np.ndarray,
    log_ai: np.ndarray,
    *,
    delta_sigma: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    z = np.asarray(zeta, dtype=np.float64).reshape(-1)
    y = np.asarray(log_ai, dtype=np.float64).reshape(-1)
    design = np.column_stack((np.ones(z.size), 2.0 * z - 1.0))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    weights = np.ones(y.size, dtype=np.float64)
    scale = float("nan")
    for _ in range(100):
        residual = y - design @ coefficients
        center = _weighted_median(residual, weights)
        scale = 1.4826 * _weighted_median(np.abs(residual - center), weights)
        if not np.isfinite(scale):
            raise ValueError("nonfinite_huber_scale")
        if scale <= np.finfo(np.float64).eps:
            if float(np.max(np.abs(residual))) <= 1e-12:
                weights = np.ones_like(weights)
                break
            raise ValueError("zero_huber_scale_for_nonexact_fit")
        threshold = float(delta_sigma) * scale
        absolute = np.abs(residual - center)
        new_weights = np.ones_like(weights)
        outside = absolute > threshold
        new_weights[outside] = threshold / absolute[outside]
        root = np.sqrt(new_weights)
        updated, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
        if np.max(np.abs(updated - coefficients)) <= 1e-12 * max(1.0, np.max(np.abs(coefficients))):
            coefficients = updated
            weights = new_weights
            break
        coefficients = updated
        weights = new_weights
    fitted = design @ coefficients
    residual = y - fitted
    return fitted, weights, {
        "background_a": float(coefficients[0]),
        "background_b": float(coefficients[1]),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "mad_scale": float(scale),
        "max_abs_residual": float(np.max(np.abs(residual))),
    }


def _survey_and_target_zone(
    *,
    workflow: Any,
    script_cfg: Mapping[str, Any],
    repo_root: Path,
) -> tuple[Any, TargetZone, dict[str, Path]]:
    data_root = resolve_relative_path(workflow.data_root, root=repo_root)
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    options = {key: value for key, value in workflow.seismic.as_dict().items() if key in {"iline", "xline", "istep", "xstep"}}
    survey = open_survey(seismic_path, workflow.seismic.type, segy_options=options or None)
    geometry = survey.describe_geometry(domain="depth")
    raw: dict[str, pd.DataFrame] = {}
    paths: dict[str, Path] = {}
    for horizon in script_cfg["horizons"]:
        path = resolve_relative_path(horizon["file"], root=data_root)
        raw[str(horizon["name"])] = import_interpretation_petrel(path)
        paths[str(horizon["name"])] = path
    zone = TargetZone(
        raw,
        geometry,
        [str(item["name"]) for item in script_cfg["horizons"]],
        min_thickness=float(script_cfg["sampling"]["expected_model_dz_m"]),
    )
    return survey, zone, paths


load_depth_calibration_as_legacy = load_depth_calibration_for_object_core


def run_depth_calibration(
    *,
    workflow: Any,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    source_provenance: Mapping[str, Any],
    forward_inputs: Mapping[str, Any],
    config_provenance: Mapping[str, str],
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    inventory_path = sources["well_inventory_dir"] / "well_inventory.csv"
    inventory = pd.read_csv(inventory_path)
    required_inventory = {"well_name", "kb_m", "inline_float", "xline_float", "surface_x", "surface_y"}
    missing = sorted(required_inventory - set(inventory))
    if missing:
        raise ValueError(f"well_inventory.csv lacks columns: {missing}")
    inventory["_key"] = inventory["well_name"].map(normalize_well_name)
    if inventory["_key"].duplicated().any():
        raise ValueError("well_inventory.csv contains duplicate normalized well names.")
    inventory = inventory.set_index("_key", drop=False)

    forward_path = Path(str(forward_inputs["_path"]))
    _required_file(
        forward_inputs["wavelet"]["path"], repo_root=repo_root, label="forward wavelet"
    )
    _required_file(
        forward_inputs["ai_velocity_relation"]["path"], repo_root=repo_root, label="AI-Vp relation"
    )
    input_inventory_path = sources["rock_physics_analysis_dir"] / "well_input_inventory.csv"
    if not input_inventory_path.is_file():
        raise FileNotFoundError(input_inventory_path)
    step5_metrics_path = sources["wavelet_batch_synthetic_depth_dir"] / "wavelet_batch_metrics.csv"
    shifted_las_sources = _load_step5_shifted_las_sources(step5_metrics_path, repo_root=repo_root)

    survey, target_zone, horizon_paths = _survey_and_target_zone(
        workflow=workflow, script_cfg=script_cfg, repo_root=repo_root
    )
    geometry = survey.describe_geometry(domain="depth")
    model_dz = float(script_cfg["sampling"]["expected_model_dz_m"])
    if not np.isclose(float(geometry["sample_step"]), model_dz, rtol=0.0, atol=1e-9):
        raise ValueError(f"Survey depth sampling is {geometry['sample_step']}, expected {model_dz} m.")
    factor = int(script_cfg["sampling"]["vertical_oversampling_factor"])
    truth_dz = model_dz / factor
    origin = float(geometry["sample_min"])

    well_tops_path = resolve_relative_path(
        workflow.assets.well_tops_file,
        root=resolve_relative_path(workflow.data_root, root=repo_root),
    )
    well_tops = import_well_tops_petrel(well_tops_path)
    clusters_source = []
    for item in shifted_las_sources:
        key = normalize_well_name(str(item["well_id"]))
        if key not in inventory.index:
            raise ValueError(f"Step 5 successful well is absent from Step 1 inventory: {item['well_id']}")
        row = inventory.loc[key]
        x, y = float(row["surface_x"]), float(row["surface_y"])
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f"Well has non-finite surface coordinates: {item['well_id']}")
        clusters_source.append((str(item["well_id"]), x, y))
    cluster_labels = radius_connected_components(
        np.asarray([[x, y] for _, x, y in clusters_source]),
        float(workflow.spatial_debias.cluster_radius_m),
    )
    cluster_by_well = {name: int(label) for (name, _, _), label in zip(clusters_source, cluster_labels)}

    inputs: list[WellZoneCurves] = []
    well_status: list[dict[str, Any]] = []
    zone_status: list[dict[str, Any]] = []
    horizon_audit: list[dict[str, Any]] = []
    background_qc: list[dict[str, Any]] = []
    background_samples: list[dict[str, Any]] = []
    for source in shifted_las_sources:
        well_name = str(source["well_id"])
        key = normalize_well_name(well_name)
        row = inventory.loc[key]
        full_las_path = _required_file(
            source["full_las_path"],
            repo_root=repo_root,
            label=f"{well_name} shifted_preprocessed_las",
        )
        filtered_las_path = _required_file(
            source["filtered_las_path"],
            repo_root=repo_root,
            label=f"{well_name} shifted_filtered_las",
        )
        kb_m = float(row["kb_m"])
        il = float(row["inline_float"])
        xl = float(row["xline_float"])
        if not np.all(np.isfinite([kb_m, il, xl])):
            raise ValueError(f"Well lacks finite KB/inline/xline: {well_name}")
        il_index = survey.line_geometry.inline_axis.index_of_line(il)
        xl_index = survey.line_geometry.xline_axis.index_of_line(xl)
        nearest_i = int(np.clip(round(il_index), 0, target_zone.no_support_mask.shape[0] - 1))
        nearest_j = int(np.clip(round(xl_index), 0, target_zone.no_support_mask.shape[1] - 1))
        if bool(target_zone.no_support_mask[nearest_i, nearest_j]):
            raise ValueError(f"no_interpretation_support_at_well:{well_name}")
        full_ai = read_las_curve(full_las_path, "AI", match_policy="exact", allow_all_nan=True)
        filtered_ai = read_las_curve(filtered_las_path, "AI", match_policy="exact", allow_all_nan=True)
        if str(getattr(full_ai, "unit", "")) != "m/s*g/cm3":
            raise ValueError(
                f"Full AI unit mismatch for {well_name}: expected 'm/s*g/cm3', got {getattr(full_ai, 'unit', None)!r}"
            )
        if str(getattr(filtered_ai, "unit", "")) != "m/s*g/cm3":
            raise ValueError(
                f"Filtered AI unit mismatch for {well_name}: expected 'm/s*g/cm3', got {getattr(filtered_ai, 'unit', None)!r}"
            )
        full_md = np.asarray(full_ai.basis, dtype=np.float64)
        filtered_md = np.asarray(filtered_ai.basis, dtype=np.float64)
        full_tvdss = full_md - kb_m
        filtered_tvdss = filtered_md - kb_m
        full_values = np.asarray(full_ai.values, dtype=np.float64)
        filtered_values = np.asarray(filtered_ai.values, dtype=np.float64)
        tops: dict[str, tuple[float, float]] = {}
        for horizon in script_cfg["horizons"]:
            name = str(horizon["name"])
            try:
                top_md = find_well_top_md(well_tops, well_name=well_name, surface=str(horizon["well_top"]))
            except ValueError as exc:
                if not str(exc).startswith("No finite MD found"):
                    raise
                horizon_audit.append({
                    "well_name": well_name, "horizon_name": name,
                    "well_top_surface": horizon["well_top"], "inline_float": il,
                    "xline_float": xl, "well_top_md_m": np.nan,
                    "well_top_tvdss_m": np.nan, "interpreted_tvdss_m": np.nan,
                    "delta_interpretation_minus_well_top_m": np.nan,
                    "sample_method": "", "support_status": "missing_well_top",
                    "status": "not_contributed", "reason": str(exc),
                })
                continue
            top_tvdss = top_md - kb_m
            # Interpretation support failures are structural and must not be
            # downgraded to a missing-well-top contribution.
            sample = target_zone.get_horizon_surface(name).sample_at_line(il, xl)
            tops[name] = (top_md, top_tvdss)
            horizon_audit.append({
                "well_name": well_name, "horizon_name": name,
                "well_top_surface": horizon["well_top"], "inline_float": il,
                "xline_float": xl, "well_top_md_m": top_md,
                "well_top_tvdss_m": top_tvdss,
                "interpreted_tvdss_m": float(sample.value),
                "delta_interpretation_minus_well_top_m": float(sample.value - top_tvdss),
                "sample_method": sample.method, "support_status": sample.support_status,
                "status": "ok", "reason": "",
            })
        accepted_zones = 0
        for top_horizon, bottom_horizon in zip(script_cfg["horizons"][:-1], script_cfg["horizons"][1:]):
            top_name, bottom_name = str(top_horizon["name"]), str(bottom_horizon["name"])
            zone_id = f"{top_name}__to__{bottom_name}"
            if top_name not in tops or bottom_name not in tops:
                zone_status.append({"well_name": well_name, "zone_id": zone_id, "status": "not_contributed", "n_valid_cells": 0, "reason": "incomplete_well_top_bounds"})
                continue
            top = tops[top_name][1]
            bottom = tops[bottom_name][1]
            if not top < bottom:
                raise ValueError(f"misordered_well_tops:{well_name}:{zone_id}")
            first_edge_index = int(np.ceil((top - origin) / truth_dz - 1e-12))
            last_edge_index = int(np.floor((bottom - origin) / truth_dz + 1e-12))
            edges = origin + np.arange(first_edge_index, last_edge_index + 1) * truth_dz
            centers = 0.5 * (edges[:-1] + edges[1:])
            full_log_ai = _continuous_cell_average(full_tvdss, full_values, centers_m=centers, cell_size_m=truth_dz)
            filtered_log_ai = _continuous_cell_average(
                filtered_tvdss,
                filtered_values,
                centers_m=centers,
                cell_size_m=truth_dz,
            )
            valid = np.isfinite(full_log_ai) & np.isfinite(filtered_log_ai)
            n_valid = int(np.count_nonzero(valid))
            minimum = int(script_cfg["calibration"]["minimum_valid_cells_per_well_zone"])
            if n_valid < minimum:
                zone_status.append({"well_name": well_name, "zone_id": zone_id, "status": "rejected", "n_valid_cells": n_valid, "reason": "insufficient_valid_cells"})
                continue
            zone_centers = centers[valid]
            full_observed = full_log_ai[valid]
            filtered_observed = filtered_log_ai[valid]
            zeta = (zone_centers - top) / (bottom - top)
            fitted, weights, metrics = _huber_background(
                zeta, filtered_observed, delta_sigma=float(script_cfg["calibration"]["huber_delta_sigma"])
            )
            inputs.append(depth_well_zone_curves_for_object_core(
                well_name=well_name,
                spatial_cluster_id=cluster_by_well[well_name],
                zone_id=zone_id,
                top_horizon=top_name,
                bottom_horizon=bottom_name,
                tvdss_m=zone_centers,
                filtered_log_ai=filtered_observed,
                observed_log_ai=full_observed,
                zone_top_tvdss_m=top,
                zone_bottom_tvdss_m=bottom,
            ))
            accepted_zones += 1
            zone_status.append({"well_name": well_name, "zone_id": zone_id, "status": "ok", "n_valid_cells": n_valid, "reason": ""})
            background_qc.append({"well_name": well_name, "zone_id": zone_id, "zone_thickness_m": bottom - top, "n_valid_cells": n_valid, **metrics})
            for depth, coordinate, filtered_value, full_value, background, weight in zip(zone_centers, zeta, filtered_observed, full_observed, fitted, weights):
                background_samples.append({
                    "well_name": well_name,
                    "zone_id": zone_id,
                    "tvdss_m": depth,
                    "zeta": coordinate,
                    "filtered_log_ai": filtered_value,
                    "full_log_ai": full_value,
                    "background_log_ai": background,
                    "filtered_residual": filtered_value - background,
                    "full_residual": full_value - background,
                    "irls_weight": weight,
                })
        well_status.append({"well_name": well_name, "status": "ok" if accepted_zones else "rejected", "n_contributed_zones": accepted_zones, "reason": "" if accepted_zones else "no_complete_valid_zone"})

    if not inputs:
        raise ValueError("Depth calibration produced no valid well-zone inputs.")
    input_contracts = {
        key.removesuffix("_dir"): {
            "path": repo_relative_path(sources[key] / "run_summary.json", root=repo_root),
            "contract_fingerprint_sha256": str(value["contract_fingerprint_sha256"]),
        }
        for key, value in source_provenance.items()
    }
    legacy, objects, qc, samples, backgrounds, profile_samples = calibrate_depth_object_core(
        inputs,
        truth_dz_m=truth_dz,
        ordered_horizons=[item["name"] for item in script_cfg["horizons"]],
        source_runs={key: repo_relative_path(path, root=repo_root) for key, path in sources.items()},
        input_contracts=input_contracts,
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
    with (
        sources["rock_physics_analysis_dir"] / "run_summary.json"
    ).open("r", encoding="utf-8") as handle:
        rock_summary = json.load(handle)
    locked_step3_run = str(
        dict(rock_summary.get("source_run") or {}).get("well_preprocess_dir") or ""
    )
    if not locked_step3_run:
        raise ValueError("Rock-physics summary does not identify its Step-3 source run.")
    payload = depth_payload_from_object_core_calibration(legacy, extra={
        "background_estimator": "per_well_zone_huber",
        "background_huber_delta_sigma": float(script_cfg["calibration"]["huber_delta_sigma"]),
        "horizon_contract": list(script_cfg["horizons"]),
        "source_provenance": dict(source_provenance),
        "config_provenance": dict(config_provenance),
        "locked_step3_run": locked_step3_run,
        "locked_step5_run": repo_relative_path(sources["wavelet_batch_synthetic_depth_dir"], root=repo_root),
        "well_curve_source_contract": {
            "full_log_ai": "Step 5 shifted_preprocessed_las/AI",
            "filtered_log_ai": "Step 5 shifted_filtered_las/AI, used for background fitting",
        },
        "shifted_las_sources": shifted_las_sources,
    })
    relation = forward_inputs["ai_velocity_relation"]
    maximum_ai = float(np.exp(payload["generation_log_ai_bounds"]["global"]["maximum"]))
    maximum_vp = (maximum_ai - float(relation["b"])) / float(relation["a"])
    if not np.isfinite(maximum_vp) or maximum_vp <= 0.0:
        raise ValueError("Calibration generation bound produces non-positive maximum Vp.")
    payload["maximum_allowed_vp_mps"] = maximum_vp
    payload.update(SCIENCE_CONTRACT)

    frames = {
        "well_status.csv": pd.DataFrame.from_records(well_status),
        "well_zone_status.csv": pd.DataFrame.from_records(zone_status),
        "well_horizon_consistency.csv": pd.DataFrame.from_records(horizon_audit),
        "well_calibration_samples.csv": depth_frame_from_object_core(samples),
        "well_background_fits.csv": pd.DataFrame.from_records(background_qc),
        "well_background_samples.csv": pd.DataFrame.from_records(background_samples),
        "well_object_catalog.csv": depth_frame_from_object_core(objects),
        "well_object_profile_samples.csv": depth_frame_from_object_core(profile_samples),
        "calibration_qc.csv": qc,
    }
    for name, frame in frames.items():
        frame.to_csv(output_dir / name, index=False)
    calibration_path = output_dir / "impedance_calibration.json"
    write_json(calibration_path, payload)
    contract_fingerprint = contract_fingerprint_sha256(
        contract_schema_version=CALIBRATION_SCHEMA,
        semantics={
            **SCIENCE_CONTRACT,
            "sample_domain": "depth",
            "sample_unit": "m",
            "depth_basis": "tvdss",
            "truth_dz_m": truth_dz,
            "ordered_horizons": list(payload["ordered_horizons"]),
            "generator_family": GENERATOR_FAMILY,
            "calibration_model": {
                "parent": payload["parent"],
                "zones": payload["zones"],
                "zone_models": payload["zone_models"],
            },
        },
        business_config=script_cfg,
        input_contracts=input_contracts,
        primary_artifacts={
            "impedance_calibration": calibration_path,
            "well_object_catalog": output_dir / "well_object_catalog.csv",
            "well_calibration_samples": output_dir / "well_calibration_samples.csv",
            "well_background_fits": output_dir / "well_background_fits.csv",
            "well_object_profile_samples": output_dir / "well_object_profile_samples.csv",
        },
    )

    figure_summary = write_calibration_figures(output_dir, script_cfg.get("figures", {}))

    summary = {
        "schema_version": CALIBRATION_SCHEMA,
        **SCIENCE_CONTRACT,
        "status": "success",
        "sample_domain": "depth",
        "depth_basis": "tvdss",
        "generator_family": GENERATOR_FAMILY,
        "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
        "contract_fingerprint_sha256": contract_fingerprint,
        "input_contracts": input_contracts,
        "source_runs": payload["source_runs"],
        "config_provenance": dict(config_provenance),
        "n_well_zone_inputs": len(inputs),
        "n_objects": int(len(objects)),
        "outputs": {
            "impedance_calibration": repo_relative_path(calibration_path, root=repo_root),
            **{name.removesuffix(".csv"): repo_relative_path(output_dir / name, root=repo_root) for name in frames},
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


__all__ = ["load_depth_calibration_as_legacy", "run_depth_calibration"]
