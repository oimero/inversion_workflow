"""Build the Step 7 real-field LFM and model-grid well-target package.

This module is intentionally separate from the historical time-domain LFM
pipeline.  It does not consume well-constraint control points and does not use
the old proportional-slice modeling entrypoints.  The first implementation is
the narrow contract described in ``docs/spec/real-field-lfm-input.md``:

1. fit one first-order log(AI) trend per successful fourth-step well over the
   full target interval;
2. model the two trend coefficients as spatial parameter fields;
3. reconstruct a finite LFM only inside the target interval and export the
   target mask explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, QhullError, cKDTree

from cup.petrel.load import import_interpretation_petrel
from cup.seismic.horizon import (
    HorizonSurface,
    build_horizon_surface,
    normalize_interpretation_unit_for_geometry,
)
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.seismic.volume_export import export_volume_like_source
from cup.synthetic.forward import antialias_taps, downsample_continuous
from cup.utils.io import (
    array_sha256,
    load_yaml_config,
    repo_relative_path,
    resolve_artifact_path,
    resolve_relative_path,
    sha256_file,
    write_json,
)
from cup.utils.statistics import radius_connected_components
from cup.well.assets import normalize_well_name
from cup.well.las import read_las_curve
from cup.well.td import load_workflow_time_depth_table_csv


SCHEMA_VERSION = "real_field_model_inputs_v2"


@dataclass(frozen=True)
class RealFieldModelInputsConfig:
    source_runs: dict[str, str]
    well_inventory_file: str
    synthetic_benchmark_dir: str
    spatial_cluster_radius_m: float
    seismic: dict[str, Any]
    horizons: tuple[dict[str, str], ...]
    trend_fit: dict[str, Any]
    parameter_modeling: dict[str, Any]
    output_geometry: dict[str, Any]
    lfm_qc: dict[str, Any]


@dataclass(frozen=True)
class OutputGeometry:
    mode: str
    ilines: np.ndarray
    xlines: np.ndarray
    samples: np.ndarray
    is_section: bool
    section_id: str = ""
    x_m: np.ndarray | None = None
    y_m: np.ndarray | None = None
    x_grid_m: np.ndarray | None = None
    y_grid_m: np.ndarray | None = None


def parse_real_field_model_inputs_config(raw: Mapping[str, Any]) -> RealFieldModelInputsConfig:
    if "real_field_lfm" in raw:
        raise ValueError("real_field_lfm is retired; use real_field_model_inputs and rerun Step 7.")
    root = _mapping(raw.get("real_field_model_inputs"), "real_field_model_inputs")
    source_runs = _mapping(root.get("source_runs"), "real_field_model_inputs.source_runs")
    if "seismic" in root:
        raise ValueError("real_field_model_inputs.seismic is retired; use top-level seismic.")
    if "target_interval" in root:
        raise ValueError("real_field_model_inputs.target_interval is retired; use top-level target_interval.")
    horizons_parent = _mapping(raw.get("target_interval"), "target_interval")
    horizons_raw = horizons_parent.get("horizons")
    if not isinstance(horizons_raw, (list, tuple)) or len(horizons_raw) < 2:
        raise ValueError("target_interval.horizons must contain at least two horizons.")
    horizons: list[dict[str, str]] = []
    seen: set[str] = set()
    for idx, item in enumerate(horizons_raw):
        entry = _mapping(item, f"target_interval.horizons[{idx}]")
        name = _required_text(entry, "name", path=f"target_interval.horizons[{idx}]")
        file = _required_text(entry, "file", path=f"target_interval.horizons[{idx}]")
        key = name.casefold()
        if key in seen:
            raise ValueError(f"Duplicate horizon name in target_interval.horizons: {name}")
        seen.add(key)
        horizons.append({"name": name, "file": file})
    output_geometry = dict(root.get("output_geometry") or {"mode": "volume"})
    if isinstance(output_geometry.get("section"), Mapping):
        raise ValueError("real_field_model_inputs.output_geometry.section is retired; use sections_file.")
    if str(output_geometry.get("mode") or "volume").casefold() == "section":
        sections_file = _required_text(root, "sections_file", path="real_field_model_inputs")
        sections_payload = load_yaml_config(resolve_relative_path(sections_file, root=Path.cwd()))
        sections = sections_payload.get("sections")
        if not isinstance(sections, list) or len(sections) != 1:
            raise ValueError("real_field_model_inputs section mode requires sections_file with exactly one section.")
        output_geometry["section"] = dict(sections[0])
    spatial = _mapping(raw.get("spatial_debias"), "spatial_debias")
    return RealFieldModelInputsConfig(
        source_runs={
            "well_auto_tie_dir": _required_text(
                source_runs,
                "well_auto_tie_dir",
                path="real_field_model_inputs.source_runs",
            ),
            "well_preprocess_dir": _required_text(
                source_runs,
                "well_preprocess_dir",
                path="real_field_model_inputs.source_runs",
            ),
        },
        well_inventory_file=_required_text(root, "well_inventory_file", path="real_field_model_inputs"),
        synthetic_benchmark_dir=_required_text(root, "synthetic_benchmark_dir", path="real_field_model_inputs"),
        spatial_cluster_radius_m=float(spatial.get("cluster_radius_m", 600.0)),
        seismic=_mapping(raw.get("seismic"), "seismic"),
        horizons=tuple(horizons),
        trend_fit=dict(root.get("trend_fit") or {}),
        parameter_modeling=dict(root.get("parameter_modeling") or {}),
        output_geometry=output_geometry,
        lfm_qc=dict(root.get("lfm_qc") or {}),
    )


def run_real_field_model_inputs(
    *,
    config: RealFieldModelInputsConfig,
    repo_root: Path,
    data_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    well_auto_tie_dir = resolve_relative_path(config.source_runs["well_auto_tie_dir"], root=repo_root)
    well_preprocess_dir = resolve_relative_path(config.source_runs["well_preprocess_dir"], root=repo_root)
    well_inventory_file = resolve_relative_path(config.well_inventory_file, root=repo_root)
    benchmark_dir = resolve_relative_path(config.synthetic_benchmark_dir, root=repo_root)
    benchmark_manifest_path = benchmark_dir / "benchmark_manifest.json"
    metrics_path = well_auto_tie_dir / "well_tie_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"well_tie_metrics.csv not found: {metrics_path}")
    if not well_inventory_file.is_file():
        raise FileNotFoundError(f"well_inventory_file not found: {well_inventory_file}")
    preprocess_status_path = well_preprocess_dir / "well_preprocess_status.csv"
    if not preprocess_status_path.is_file():
        raise FileNotFoundError(f"well_preprocess_status.csv not found: {preprocess_status_path}")
    projection = _load_projection_contract(benchmark_manifest_path)
    projection["benchmark_manifest"] = repo_relative_path(benchmark_manifest_path, root=repo_root)
    _validate_auto_tie_preprocess_source(
        well_auto_tie_dir=well_auto_tie_dir,
        well_preprocess_dir=well_preprocess_dir,
        repo_root=repo_root,
    )

    seismic_path = resolve_relative_path(
        _required_text(config.seismic, "file", path="real_field_model_inputs.seismic"),
        root=data_root,
    )
    seismic_type = str(config.seismic.get("type", "zgy")).casefold()
    segy_options = segy_options_from_config(dict(config.seismic)) if seismic_type == "segy" else None
    survey = open_survey(seismic_path, seismic_type, segy_options=segy_options)
    survey_geometry = survey.describe_geometry(domain="time")
    sample_axis = np.asarray(survey.sample_axis("time").values, dtype=np.float64)

    surfaces, horizon_qc = _load_horizon_surfaces(
        config.horizons,
        survey_geometry=survey_geometry,
        data_root=data_root,
    )
    output_geometry = _resolve_output_geometry(
        config.output_geometry,
        survey=survey,
        sample_axis=sample_axis,
        surfaces=surfaces,
        horizon_names=[item["name"] for item in config.horizons],
    )
    controls = _fit_well_trends(
        metrics_path=metrics_path,
        inventory_path=well_inventory_file,
        well_auto_tie_dir=well_auto_tie_dir,
        repo_root=repo_root,
        surfaces=surfaces,
        horizon_names=[item["name"] for item in config.horizons],
        sample_axis=sample_axis,
        trend_cfg=config.trend_fit,
    )
    controls = _attach_control_xy(controls, survey.line_geometry)
    controls_path = output_dir / "well_trend_controls.csv"
    controls.to_csv(controls_path, index=False)

    targets, target_qc = _build_well_targets(
        metrics_path=metrics_path,
        preprocess_status_path=preprocess_status_path,
        inventory_path=well_inventory_file,
        well_auto_tie_dir=well_auto_tie_dir,
        well_preprocess_dir=well_preprocess_dir,
        repo_root=repo_root,
        sample_axis=output_geometry.samples,
        line_geometry=survey.line_geometry,
        controls=controls,
        projection=projection,
        cluster_radius_m=config.spatial_cluster_radius_m,
    )
    targets_path = output_dir / "well_model_targets.csv"
    target_qc_path = output_dir / "well_model_target_qc.csv"
    targets.to_csv(targets_path, index=False)
    target_qc.to_csv(target_qc_path, index=False)

    accepted = controls[controls["status"].eq("ok")].copy()
    min_wells = int(config.parameter_modeling.get("min_wells", 3))
    allow_constant = _as_bool(config.parameter_modeling.get("allow_constant_fallback", False))
    if accepted.shape[0] < min_wells and not allow_constant:
        status = "insufficient_control_wells"
        parameter_qc = pd.DataFrame(
            [
                {
                    "parameter": "a",
                    "status": status,
                    "n_controls": int(accepted.shape[0]),
                    "min_wells": min_wells,
                    "reason": "not_enough_valid_well_trends",
                },
                {
                    "parameter": "b",
                    "status": status,
                    "n_controls": int(accepted.shape[0]),
                    "min_wells": min_wells,
                    "reason": "not_enough_valid_well_trends",
                },
            ]
        )
        parameter_qc_path = output_dir / "parameter_field_qc.csv"
        parameter_qc.to_csv(parameter_qc_path, index=False)
        summary = _summary_payload(
            status=status,
            config=config,
            repo_root=repo_root,
            data_root=data_root,
            output_dir=output_dir,
            well_auto_tie_dir=well_auto_tie_dir,
            well_inventory_file=well_inventory_file,
            seismic_path=seismic_path,
            controls=controls,
            parameter_qc=parameter_qc,
            horizon_qc=horizon_qc,
            lfm_stats={},
            outputs={
                "well_trend_controls": controls_path,
                "well_model_targets": targets_path,
                "well_model_target_qc": target_qc_path,
                "parameter_field_qc": parameter_qc_path,
            },
        )
        summary["target_projection"] = projection
        write_json(output_dir / "real_field_model_inputs_summary.json", summary)
        return summary

    fields, parameter_qc, distance_to_control = _build_parameter_fields(
        accepted,
        output_geometry,
        config.parameter_modeling,
    )
    log_ai, valid_mask, continuity_qc = _reconstruct_lfm(
        a_field=fields["a"],
        b_field=fields["b"],
        output_geometry=output_geometry,
        surfaces=surfaces,
        horizon_names=[item["name"] for item in config.horizons],
    )
    lfm_stats = _lfm_stats(log_ai, valid_mask)
    lfm_status = _lfm_status(lfm_stats, config.lfm_qc)
    if lfm_status == "ok" and not target_qc["status"].astype(str).eq("ok").all():
        lfm_status = "warning"

    npz_path = output_dir / "real_field_lfm.npz"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "value_key": "log_ai",
        "value_domain": "log(AI)",
        "valid_mask_key": "valid_mask_model",
        "output_geometry_mode": output_geometry.mode,
        "horizon_names": [item["name"] for item in config.horizons],
        "well_auto_tie_dir": repo_relative_path(well_auto_tie_dir, root=repo_root),
        "well_preprocess_dir": repo_relative_path(well_preprocess_dir, root=repo_root),
        "benchmark_manifest": repo_relative_path(benchmark_manifest_path, root=repo_root),
        "benchmark_manifest_sha256": sha256_file(benchmark_manifest_path),
        "well_inventory_file": repo_relative_path(well_inventory_file, root=repo_root),
        "seismic_file": repo_relative_path(seismic_path, root=repo_root),
        "seismic_sha256": sha256_file(seismic_path),
    }
    np.savez_compressed(
        npz_path,
        log_ai=log_ai.astype(np.float32),
        valid_mask_model=valid_mask.astype(bool),
        lfm_support_mask=np.isfinite(distance_to_control),
        distance_to_control=distance_to_control.astype(np.float32),
        ilines=output_geometry.ilines.astype(np.float64),
        xlines=output_geometry.xlines.astype(np.float64),
        samples=output_geometry.samples.astype(np.float64),
        a_field=fields["a"].astype(np.float32),
        b_field=fields["b"].astype(np.float32),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    export_payload = _export_lfm_volume(
        output_dir=output_dir,
        log_ai=log_ai,
        output_geometry=output_geometry,
        seismic_path=seismic_path,
        seismic_type=seismic_type,
        seismic_cfg=config.seismic,
    )

    parameter_qc_path = output_dir / "parameter_field_qc.csv"
    continuity_path = output_dir / "internal_horizon_continuity_qc.csv"
    horizon_qc_path = output_dir / "horizon_qc.csv"
    parameter_qc.to_csv(parameter_qc_path, index=False)
    continuity_qc.to_csv(continuity_path, index=False)
    horizon_qc.to_csv(horizon_qc_path, index=False)
    figure_outputs = _write_figures(
        figures_dir=figures_dir,
        output_geometry=output_geometry,
        log_ai=log_ai,
        valid_mask=valid_mask,
        a_field=fields["a"],
        b_field=fields["b"],
        distance_to_control=distance_to_control,
        controls=controls,
        repo_root=repo_root,
    )

    summary = _summary_payload(
        status=lfm_status,
        config=config,
        repo_root=repo_root,
        data_root=data_root,
        output_dir=output_dir,
        well_auto_tie_dir=well_auto_tie_dir,
        well_inventory_file=well_inventory_file,
        seismic_path=seismic_path,
        controls=controls,
        parameter_qc=parameter_qc,
        horizon_qc=horizon_qc,
        lfm_stats=lfm_stats,
        outputs={
            "real_field_lfm": npz_path,
            "well_model_targets": targets_path,
            "well_model_target_qc": target_qc_path,
            **({"real_field_lfm_export": Path(export_payload["path"])} if export_payload.get("path") else {}),
            "well_trend_controls": controls_path,
            "parameter_field_qc": parameter_qc_path,
            "internal_horizon_continuity_qc": continuity_path,
            "horizon_qc": horizon_qc_path,
            **{f"figure:{key}": Path(value) for key, value in figure_outputs.items()},
        },
    )
    summary["volume_export"] = export_payload
    summary["target_projection"] = projection
    summary["target_wells"] = {
        "n_wells": int(targets["well_name"].nunique()),
        "n_valid_samples": int(targets["target_valid"].sum()),
    }
    write_json(output_dir / "real_field_model_inputs_summary.json", summary)
    return summary


def _load_horizon_surfaces(
    horizons: Sequence[Mapping[str, str]],
    *,
    survey_geometry: Mapping[str, Any],
    data_root: Path,
) -> tuple[dict[str, HorizonSurface], pd.DataFrame]:
    il_axis = _axis_from_geometry(survey_geometry, "inline")
    xl_axis = _axis_from_geometry(survey_geometry, "xline")
    surfaces: dict[str, HorizonSurface] = {}
    qc_rows: list[dict[str, Any]] = []
    for item in horizons:
        name = str(item["name"])
        path = resolve_relative_path(str(item["file"]), root=data_root)
        frame = import_interpretation_petrel(path)
        frame = normalize_interpretation_unit_for_geometry(frame, dict(survey_geometry))
        frame = frame.copy()
        frame["interpretation"] = np.abs(pd.to_numeric(frame["interpretation"], errors="coerce").to_numpy(dtype=np.float64))
        surface, interpolation = build_horizon_surface(
            frame,
            il_axis,
            xl_axis,
            name=name,
            nearest_distance_limit=None,
            outlier_threshold=None,
            outlier_min_neighbor_count=2,
            value_domain="time",
            value_unit="s",
        )
        surfaces[name] = surface
        finite = np.isfinite(surface.values)
        thickness_placeholder = np.full(surface.values.shape, np.nan)
        qc_rows.append(
            {
                "horizon_name": name,
                "horizon_file": str(path),
                "finite_fraction": float(np.mean(finite)),
                "raw_pick_fraction": float(np.mean(interpolation.raw_mask)),
                "linear_support_fraction": float(np.mean(interpolation.linear_support_mask)),
                "nearest_distance_p50": _nanquantile(interpolation.nearest_distance_grid, 0.50),
                "nearest_distance_p95": _nanquantile(interpolation.nearest_distance_grid, 0.95),
                "twt_p01_s": _nanquantile(surface.values, 0.01),
                "twt_p50_s": _nanquantile(surface.values, 0.50),
                "twt_p99_s": _nanquantile(surface.values, 0.99),
                "thickness_p50_s": _nanquantile(thickness_placeholder, 0.50),
                "status": "ok",
            }
        )
    names = [str(item["name"]) for item in horizons]
    values = [surfaces[name].values for name in names]
    for idx, (top, bottom) in enumerate(zip(values[:-1], values[1:])):
        finite = np.isfinite(top) & np.isfinite(bottom)
        thickness = np.where(finite, bottom - top, np.nan)
        crossing = finite & (thickness <= 0.0)
        qc_rows.append(
            {
                "horizon_name": f"{names[idx]}__to__{names[idx + 1]}",
                "horizon_file": "",
                "finite_fraction": float(np.mean(finite)),
                "raw_pick_fraction": float("nan"),
                "linear_support_fraction": float("nan"),
                "nearest_distance_p50": float("nan"),
                "nearest_distance_p95": float("nan"),
                "twt_p01_s": float("nan"),
                "twt_p50_s": float("nan"),
                "twt_p99_s": float("nan"),
                "thickness_p01_s": _nanquantile(thickness, 0.01),
                "thickness_p50_s": _nanquantile(thickness, 0.50),
                "thickness_p99_s": _nanquantile(thickness, 0.99),
                "crossing_trace_fraction": float(np.mean(crossing)),
                "status": "ok" if not np.any(crossing) else "has_crossing_traces",
            }
        )
    return surfaces, pd.DataFrame.from_records(qc_rows)


def _load_projection_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark_manifest.json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    antialias = _mapping(payload.get("antialias_filter"), "benchmark_manifest.antialias_filter")
    output_dt_s = float(payload["output_dt_s"])
    truth_dt_s = float(payload["truth_dt_s"])
    factor = int(antialias["factor"])
    if not np.isclose(output_dt_s / truth_dt_s, factor, rtol=0.0, atol=1e-9):
        raise ValueError("benchmark projection dt/factor mismatch")
    implementation = str(antialias["implementation"])
    if implementation != "scipy.signal.firwin/resample_poly":
        raise ValueError(f"unsupported benchmark antialias implementation: {implementation}")
    numtaps = int(antialias["numtaps"])
    if numtaps != 32 * factor + 1:
        raise ValueError(f"benchmark antialias numtaps mismatch: {numtaps} != {32 * factor + 1}")
    taps = antialias_taps(
        factor,
        cutoff_output_nyquist_fraction=float(antialias["cutoff_output_nyquist_fraction"]),
        kaiser_beta=float(antialias["kaiser_beta"]),
    )
    taps_hash = array_sha256(taps)
    expected_hash = str(antialias["taps_sha256"])
    if taps_hash != expected_hash:
        raise ValueError(f"benchmark antialias taps hash mismatch: {taps_hash} != {expected_hash}")
    return {
        "benchmark_manifest": str(path),
        "benchmark_manifest_sha256": sha256_file(path),
        "output_dt_s": output_dt_s,
        "truth_dt_s": truth_dt_s,
        "implementation": implementation,
        "factor": factor,
        "numtaps": numtaps,
        "cutoff_output_nyquist_fraction": float(antialias["cutoff_output_nyquist_fraction"]),
        "kaiser_beta": float(antialias["kaiser_beta"]),
        "taps_sha256": taps_hash,
    }


def _validate_auto_tie_preprocess_source(
    *, well_auto_tie_dir: Path, well_preprocess_dir: Path, repo_root: Path
) -> None:
    summary_path = well_auto_tie_dir / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"well auto-tie run_summary.json not found: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    recorded = _required_text(_mapping(payload.get("inputs"), "auto_tie.inputs"), "preprocess_status_file", path="auto_tie.inputs")
    recorded_path = resolve_relative_path(recorded, root=repo_root)
    expected = (well_preprocess_dir / "well_preprocess_status.csv").resolve()
    if recorded_path.resolve() != expected:
        raise ValueError(f"auto-tie preprocess source mismatch: {recorded_path} != {expected}")


def _build_well_targets(
    *,
    metrics_path: Path,
    preprocess_status_path: Path,
    inventory_path: Path,
    well_auto_tie_dir: Path,
    well_preprocess_dir: Path,
    repo_root: Path,
    sample_axis: np.ndarray,
    line_geometry: Any,
    controls: pd.DataFrame,
    projection: Mapping[str, Any],
    cluster_radius_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(metrics_path)
    preprocess = pd.read_csv(preprocess_status_path)
    inventory = pd.read_csv(inventory_path)
    for label, frame in (("metrics", metrics), ("preprocess", preprocess), ("inventory", inventory)):
        if "well_name" not in frame:
            raise ValueError(f"{label} table lacks well_name")
        frame["_well_key"] = frame["well_name"].map(normalize_well_name)
        if frame["_well_key"].duplicated().any():
            raise ValueError(f"duplicate normalized well in {label} table")
    pre_by_key = {str(row["_well_key"]): row for _, row in preprocess.iterrows()}
    inv_by_key = {str(row["_well_key"]): row for _, row in inventory.iterrows()}
    controls_by_key = {
        normalize_well_name(str(row["well_name"])): row for _, row in controls.iterrows()
    }
    successful = metrics[metrics["tie_status"].astype(str).str.casefold().eq("success")].copy()
    if successful.empty:
        raise ValueError("Step 7 has no successful auto-tie wells for target construction")

    model_dt = _regular_dt(sample_axis)
    expected_dt = float(projection["output_dt_s"])
    if not np.isclose(model_dt, expected_dt, rtol=0.0, atol=1e-9):
        raise ValueError(f"model sample dt does not match benchmark output dt: {model_dt} != {expected_dt}")
    target_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    positions: list[tuple[str, float, float]] = []

    for _, metric in successful.iterrows():
        well_name = str(metric["well_name"])
        key = normalize_well_name(well_name)
        pre = pre_by_key.get(key)
        inv = inv_by_key.get(key)
        control = controls_by_key.get(key)
        is_lfm_control = bool(control is not None and str(control.get("status", "")) == "ok")
        lfm_status = "ok" if is_lfm_control else str(control.get("reason", "not_evaluated") if control is not None else "not_evaluated")
        base_qc = {
            "well_name": well_name,
            "status": "rejected",
            "reason": "",
            "valid_sample_count": 0,
            "gap_rejected_count": 0,
            "fir_support_rejected_count": 0,
            "full_ai_source": "",
            "full_ai_sha256": "",
            "optimized_tdt_sha256": "",
        }
        try:
            if pre is None or str(pre.get("preprocess_status", "")).casefold() != "passed":
                raise ValueError("preprocess_status_not_passed")
            if inv is None:
                raise ValueError("missing_well_inventory")
            full_las = _required_artifact(
                pre.get("preprocessed_las"), run_dir=well_preprocess_dir, repo_root=repo_root,
                label=f"{well_name} preprocessed_las",
            )
            tdt_path = _required_artifact(
                metric.get("optimized_tdt_file"), run_dir=well_auto_tie_dir, repo_root=repo_root,
                label=f"{well_name} optimized_tdt_file",
            )
            table = load_workflow_time_depth_table_csv(tdt_path)
            if not table.is_md_domain:
                raise ValueError("optimized_tdt_not_md_domain")
            full = read_las_curve(full_las, "AI", match_policy="exact", allow_all_nan=True)
            target, projection_qc = _project_full_ai_to_model_grid(
                full.basis, full.values, table=table, model_axis=sample_axis, projection=projection
            )
            trace_plan = _load_trace_plan(
                metric.get("optimized_trace_sample_plan_file"),
                run_dir=well_auto_tie_dir,
                repo_root=repo_root,
            )
            inline, xline, x_m, y_m, sample_method = _well_coordinates_on_axis(
                sample_axis=sample_axis, inv=inv, trace_plan=trace_plan, line_geometry=line_geometry
            )
            trajectory_valid = np.isfinite(inline) & np.isfinite(xline) & np.isfinite(x_m) & np.isfinite(y_m)
            target_valid = np.isfinite(target) & trajectory_valid
            wellbore_class = str(inv.get("wellbore_class", "unknown"))
            tie_start = _coerce_float(metric.get("tie_window_start_s"))
            tie_end = _coerce_float(metric.get("tie_window_end_s"))
            rep_valid = trajectory_valid
            if not np.any(rep_valid):
                raise ValueError("no_valid_trajectory_positions")
            positions.append((well_name, float(np.nanmedian(x_m[rep_valid])), float(np.nanmedian(y_m[rep_valid]))))
            for index, twt in enumerate(sample_axis):
                if target_valid[index]:
                    reason = "ok"
                elif not trajectory_valid[index]:
                    reason = "trajectory_invalid"
                else:
                    reason = "target_projection_invalid"
                target_rows.append({
                    "well_name": well_name,
                    "sample_index": int(index),
                    "twt_s": float(twt),
                    "inline": float(inline[index]) if np.isfinite(inline[index]) else np.nan,
                    "xline": float(xline[index]) if np.isfinite(xline[index]) else np.nan,
                    "x_m": float(x_m[index]) if np.isfinite(x_m[index]) else np.nan,
                    "y_m": float(y_m[index]) if np.isfinite(y_m[index]) else np.nan,
                    "target_log_ai": float(target[index]) if np.isfinite(target[index]) else np.nan,
                    "target_valid": bool(target_valid[index]),
                    "target_reason": reason,
                    "sample_method": sample_method,
                    "wellbore_class": wellbore_class,
                    "is_lfm_control": is_lfm_control,
                    "lfm_control_status": lfm_status,
                    "tie_window_start_s": tie_start,
                    "tie_window_end_s": tie_end,
                })
            qc_rows.append({
                **base_qc,
                "status": "ok" if np.any(target_valid) else "rejected",
                "reason": "" if np.any(target_valid) else "no_valid_projected_samples",
                "valid_sample_count": int(np.count_nonzero(target_valid)),
                "valid_twt_min_s": _nanquantile(sample_axis[target_valid], 0.0),
                "valid_twt_max_s": _nanquantile(sample_axis[target_valid], 1.0),
                "gap_rejected_count": int(projection_qc["gap_rejected_count"]),
                "fir_support_rejected_count": int(projection_qc["fir_support_rejected_count"]),
                "full_ai_source": repo_relative_path(full_las, root=repo_root),
                "full_ai_sha256": sha256_file(full_las),
                "optimized_tdt_sha256": sha256_file(tdt_path),
            })
        except Exception as exc:
            qc_rows.append({**base_qc, "reason": f"{type(exc).__name__}:{exc}"})

    if not target_rows:
        raise ValueError("Step 7 produced no well target rows")
    frame = pd.DataFrame.from_records(target_rows)
    if not frame["target_valid"].astype(bool).any():
        raise ValueError("Step 7 produced no valid target_log_ai samples")
    pos_frame = pd.DataFrame(positions, columns=["well_name", "cluster_x_m", "cluster_y_m"])
    if pos_frame.empty or pos_frame[["cluster_x_m", "cluster_y_m"]].isna().any().any():
        raise ValueError("cannot build spatial clusters from target-well positions")
    pos_frame["spatial_cluster_id"] = radius_connected_components(
        pos_frame[["cluster_x_m", "cluster_y_m"]].to_numpy(dtype=np.float64),
        float(cluster_radius_m),
    )
    pos_frame["spatial_cluster_size"] = pos_frame.groupby("spatial_cluster_id")["well_name"].transform("count").astype(int)
    frame = frame.merge(
        pos_frame[["well_name", "spatial_cluster_id", "spatial_cluster_size"]],
        on="well_name", how="left", validate="many_to_one",
    )
    ordered = [
        "well_name", "sample_index", "twt_s", "inline", "xline", "x_m", "y_m",
        "spatial_cluster_id", "spatial_cluster_size", "target_log_ai", "target_valid",
        "target_reason", "sample_method", "wellbore_class", "is_lfm_control",
        "lfm_control_status", "tie_window_start_s", "tie_window_end_s",
    ]
    return frame[ordered].sort_values(["well_name", "sample_index"]), pd.DataFrame.from_records(qc_rows)


def _project_full_ai_to_model_grid(
    md_m: np.ndarray,
    ai: np.ndarray,
    *,
    table: Any,
    model_axis: np.ndarray,
    projection: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    factor = int(projection["factor"])
    truth_dt = float(projection["truth_dt_s"])
    n_high = (int(model_axis.size) - 1) * factor + 1
    high_axis = float(model_axis[0]) + np.arange(n_high, dtype=np.float64) * truth_dt
    high_values = _piecewise_cell_average_log_ai(md_m, ai, table=table, centers_s=high_axis, dt_s=truth_dt)
    finite = np.isfinite(high_values)
    taps = antialias_taps(
        factor,
        cutoff_output_nyquist_fraction=float(projection["cutoff_output_nyquist_fraction"]),
        kaiser_beta=float(projection["kaiser_beta"]),
    )
    projected = downsample_continuous(np.where(finite, high_values, 0.0), factor, taps)
    projected = projected[: model_axis.size]
    support = np.zeros(model_axis.size, dtype=bool)
    half = taps.size // 2
    for index in range(model_axis.size):
        center = index * factor
        left, right = center - half, center + half + 1
        support[index] = left >= 0 and right <= n_high and bool(np.all(finite[left:right]))
    projected[~support] = np.nan
    model_cell = _piecewise_cell_average_log_ai(
        md_m, ai, table=table, centers_s=model_axis, dt_s=float(projection["output_dt_s"])
    )
    gap_rejected = np.isfinite(model_cell) & ~support
    return projected, {
        "gap_rejected_count": int(np.count_nonzero(gap_rejected)),
        "fir_support_rejected_count": int(np.count_nonzero(~support)),
    }


def _well_coordinates_on_axis(
    *, sample_axis: np.ndarray, inv: pd.Series, trace_plan: pd.DataFrame | None, line_geometry: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if trace_plan is not None and not trace_plan.empty:
        required = {"twt_s", "inline_float", "xline_float", "x_m", "y_m"}
        missing = required - set(trace_plan.columns)
        if missing:
            raise ValueError(f"trace plan missing coordinate columns: {sorted(missing)}")
        twt = pd.to_numeric(trace_plan["twt_s"], errors="coerce").to_numpy(dtype=np.float64)
        arrays = [pd.to_numeric(trace_plan[key], errors="coerce").to_numpy(dtype=np.float64) for key in ("inline_float", "xline_float", "x_m", "y_m")]
        finite = np.isfinite(twt)
        for values in arrays:
            finite &= np.isfinite(values)
        if np.count_nonzero(finite) < 2:
            raise ValueError("invalid_optimized_trace_sample_plan")
        order = np.argsort(twt[finite])
        source_twt = twt[finite][order]
        output = [np.interp(sample_axis, source_twt, values[finite][order], left=np.nan, right=np.nan) for values in arrays]
        return output[0], output[1], output[2], output[3], "volume_trace_plan_bilinear"
    inline_value = _coerce_float(inv.get("inline_float"))
    xline_value = _coerce_float(inv.get("xline_float"))
    if not np.isfinite(inline_value) or not np.isfinite(xline_value):
        raise ValueError("missing_inventory_inline_xline")
    x_value, y_value = line_geometry.line_to_coord(inline_value, xline_value)
    shape = sample_axis.shape
    return (
        np.full(shape, inline_value), np.full(shape, xline_value),
        np.full(shape, float(x_value)), np.full(shape, float(y_value)), "volume_vertical_fixed",
    )


def _fit_well_trends(
    *,
    metrics_path: Path,
    inventory_path: Path,
    well_auto_tie_dir: Path,
    repo_root: Path,
    surfaces: Mapping[str, HorizonSurface],
    horizon_names: Sequence[str],
    sample_axis: np.ndarray,
    trend_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    inventory = pd.read_csv(inventory_path)
    metrics["_well_key"] = metrics["well_name"].map(normalize_well_name)
    inventory["_well_key"] = inventory["well_name"].map(normalize_well_name)
    inv_by_key = {str(row["_well_key"]): row for _, row in inventory.iterrows()}

    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        well_name = str(row.get("well_name", ""))
        base = {
            "well_name": well_name,
            "route": str(row.get("route", "")),
            "tie_status": str(row.get("tie_status", "")),
            "status": "rejected",
            "reason": "",
        }
        if str(row.get("tie_status", "")).casefold() != "success":
            rows.append({**base, "reason": "tie_status_not_success"})
            continue
        try:
            inv = inv_by_key.get(normalize_well_name(well_name))
            if inv is None:
                raise ValueError("missing_well_inventory")
            filtered_las = _required_artifact(
                row.get("filtered_las_file"),
                run_dir=well_auto_tie_dir,
                repo_root=repo_root,
                label=f"{well_name} filtered_las_file",
            )
            optimized_tdt = _required_artifact(
                row.get("optimized_tdt_file"),
                run_dir=well_auto_tie_dir,
                repo_root=repo_root,
                label=f"{well_name} optimized_tdt_file",
            )
            table = load_workflow_time_depth_table_csv(optimized_tdt)
            if not table.is_md_domain:
                raise ValueError("optimized_tdt_not_md_domain")
            ai_log = read_las_curve(filtered_las, "AI", match_policy="exact", allow_all_nan=True)
            cell_log_ai = _piecewise_cell_average_log_ai(
                ai_log.basis,
                ai_log.values,
                table=table,
                centers_s=sample_axis,
                dt_s=_regular_dt(sample_axis),
            )
            trace_plan = _load_trace_plan(
                row.get("optimized_trace_sample_plan_file"),
                run_dir=well_auto_tie_dir,
                repo_root=repo_root,
            )
            fit = _fit_one_well(
                well_name=well_name,
                route=str(row.get("route", "")),
                inv=inv,
                trace_plan=trace_plan,
                sample_axis=sample_axis,
                log_ai=cell_log_ai,
                surfaces=surfaces,
                horizon_names=horizon_names,
                trend_cfg=trend_cfg,
            )
            rows.append({**base, **fit})
        except Exception as exc:
            rows.append({**base, "reason": f"{type(exc).__name__}:{exc}"})
    out = pd.DataFrame.from_records(rows)
    return _apply_trend_range_qc(out, trend_cfg)


def _fit_one_well(
    *,
    well_name: str,
    route: str,
    inv: pd.Series,
    trace_plan: pd.DataFrame | None,
    sample_axis: np.ndarray,
    log_ai: np.ndarray,
    surfaces: Mapping[str, HorizonSurface],
    horizon_names: Sequence[str],
    trend_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    deviated = "deviated" in str(route).casefold() and trace_plan is not None and not trace_plan.empty
    if deviated:
        plan_twt = pd.to_numeric(trace_plan["twt_s"], errors="coerce").to_numpy(dtype=np.float64)
        plan_inline = pd.to_numeric(trace_plan["inline_float"], errors="coerce").to_numpy(dtype=np.float64)
        plan_xline = pd.to_numeric(trace_plan["xline_float"], errors="coerce").to_numpy(dtype=np.float64)
        finite_plan = np.isfinite(plan_twt) & np.isfinite(plan_inline) & np.isfinite(plan_xline)
        if int(np.count_nonzero(finite_plan)) < 2:
            raise ValueError("invalid_optimized_trace_sample_plan")
        order = np.argsort(plan_twt[finite_plan])
        plan_twt = plan_twt[finite_plan][order]
        plan_inline = plan_inline[finite_plan][order]
        plan_xline = plan_xline[finite_plan][order]
        inline_by_sample = np.interp(sample_axis, plan_twt, plan_inline, left=np.nan, right=np.nan)
        xline_by_sample = np.interp(sample_axis, plan_twt, plan_xline, left=np.nan, right=np.nan)
    else:
        inline = _coerce_float(inv.get("inline_float"))
        xline = _coerce_float(inv.get("xline_float"))
        if not np.isfinite(inline) or not np.isfinite(xline):
            raise ValueError("missing_inventory_inline_xline")
        inline_by_sample = np.full(sample_axis.shape, inline, dtype=np.float64)
        xline_by_sample = np.full(sample_axis.shape, xline, dtype=np.float64)

    top_values = _sample_horizon_along(samples_inline=inline_by_sample, samples_xline=xline_by_sample, surface=surfaces[horizon_names[0]])
    bottom_values = _sample_horizon_along(samples_inline=inline_by_sample, samples_xline=xline_by_sample, surface=surfaces[horizon_names[-1]])
    thickness = bottom_values - top_values
    u_total = (sample_axis - top_values) / thickness
    valid = (
        np.isfinite(log_ai)
        & np.isfinite(u_total)
        & np.isfinite(inline_by_sample)
        & np.isfinite(xline_by_sample)
        & (thickness > 0.0)
        & (u_total >= 0.0)
        & (u_total <= 1.0)
    )
    n_valid = int(np.count_nonzero(valid))
    min_valid = int(trend_cfg.get("min_valid_samples_per_well", 32))
    if n_valid < min_valid:
        raise ValueError(f"insufficient_valid_samples:{n_valid}<{min_valid}")
    x = 2.0 * u_total[valid] - 1.0
    y = log_ai[valid]
    fit = _huber_fit_line(x, y, f_scale=float(trend_cfg.get("huber_f_scale_log_ai", 0.05)))
    pred = fit["a"] + fit["b"] * x
    residual = y - pred
    weights = np.full(y.shape, _regular_dt(sample_axis), dtype=np.float64)
    rep_inline = float(np.average(inline_by_sample[valid], weights=weights))
    rep_xline = float(np.average(xline_by_sample[valid], weights=weights))
    rep_twt = float(np.average(sample_axis[valid], weights=weights))
    rep_u = float(np.average(u_total[valid], weights=weights))
    return {
        "status": "ok",
        "reason": "",
        "n_valid": n_valid,
        "a": float(fit["a"]),
        "b": float(fit["b"]),
        "trend_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "trend_residual_median": float(np.median(residual)),
        "representative_inline": rep_inline,
        "representative_xline": rep_xline,
        "representative_twt": rep_twt,
        "representative_u": rep_u,
        "trajectory_sample_count": int(0 if trace_plan is None else trace_plan.shape[0]),
        "weighted_twt_min": float(np.nanmin(sample_axis[valid])),
        "weighted_twt_max": float(np.nanmax(sample_axis[valid])),
        "weighted_u_min": float(np.nanmin(u_total[valid])),
        "weighted_u_max": float(np.nanmax(u_total[valid])),
        "position_source": "optimized_trace_sample_plan" if deviated else "well_inventory",
    }


def _apply_trend_range_qc(frame: pd.DataFrame, trend_cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    if "qc_warning" not in out.columns:
        out["qc_warning"] = ""
    ok = out["status"].eq("ok") if "status" in out else pd.Series(False, index=out.index)
    if not np.any(ok):
        return out
    center = pd.to_numeric(out.loc[ok, "a"], errors="coerce").to_numpy(dtype=np.float64)
    top = pd.to_numeric(out.loc[ok, "a"], errors="coerce").to_numpy(dtype=np.float64) - pd.to_numeric(out.loc[ok, "b"], errors="coerce").to_numpy(dtype=np.float64)
    bottom = pd.to_numeric(out.loc[ok, "a"], errors="coerce").to_numpy(dtype=np.float64) + pd.to_numeric(out.loc[ok, "b"], errors="coerce").to_numpy(dtype=np.float64)
    all_values = np.r_[center, top, bottom]
    lo = trend_cfg.get("log_ai_min")
    hi = trend_cfg.get("log_ai_max")
    lo_value = float(lo) if lo is not None else _nanquantile(all_values, 0.01)
    hi_value = float(hi) if hi is not None else _nanquantile(all_values, 0.99)
    max_abs_b = float(trend_cfg.get("max_abs_b_log_ai", 0.35))
    for idx in out.index[ok]:
        reasons: list[str] = []
        a = _coerce_float(out.at[idx, "a"])
        b = _coerce_float(out.at[idx, "b"])
        if not (lo_value <= a <= hi_value):
            reasons.append("center_out_of_range")
        if not (lo_value <= a - b <= hi_value and lo_value <= a + b <= hi_value):
            reasons.append("top_or_bottom_out_of_range")
        if not (abs(b) <= max_abs_b):
            reasons.append("slope_out_of_range")
        out.at[idx, "log_ai_qc_min"] = lo_value
        out.at[idx, "log_ai_qc_max"] = hi_value
        out.at[idx, "max_abs_b_log_ai"] = max_abs_b
        if reasons:
            out.at[idx, "qc_warning"] = ";".join(reasons)
    return out


def _attach_control_xy(frame: pd.DataFrame, line_geometry: Any) -> pd.DataFrame:
    out = frame.copy()
    out["representative_x_m"] = np.nan
    out["representative_y_m"] = np.nan
    ok = out["status"].eq("ok") if "status" in out else pd.Series(False, index=out.index)
    for idx in out.index[ok]:
        inline = _coerce_float(out.at[idx, "representative_inline"])
        xline = _coerce_float(out.at[idx, "representative_xline"])
        if not np.isfinite(inline) or not np.isfinite(xline):
            out.at[idx, "status"] = "rejected"
            out.at[idx, "reason"] = "missing_representative_line_position"
            continue
        try:
            x_m, y_m = line_geometry.line_to_coord(inline, xline)
        except Exception as exc:
            out.at[idx, "status"] = "rejected"
            out.at[idx, "reason"] = f"representative_xy_failed:{type(exc).__name__}:{exc}"
            continue
        out.at[idx, "representative_x_m"] = float(x_m)
        out.at[idx, "representative_y_m"] = float(y_m)
    return out


def _build_parameter_fields(
    controls: pd.DataFrame,
    output_geometry: OutputGeometry,
    cfg: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray]:
    control_x_m = controls["representative_x_m"].to_numpy(dtype=np.float64)
    control_y_m = controls["representative_y_m"].to_numpy(dtype=np.float64)
    range_hint = _nearest_neighbor_range(control_x_m, control_y_m)
    rows: list[dict[str, Any]] = []
    fields: dict[str, np.ndarray] = {}
    variances: dict[str, np.ndarray] = {}
    for parameter in ("a", "b"):
        values = controls[parameter].to_numpy(dtype=np.float64)
        field, variance = _krige_parameter(
            control_x_m,
            control_y_m,
            values,
            output_geometry,
            range_hint=range_hint,
            variogram=str(cfg.get("variogram", "spherical")),
            exact=_as_bool(cfg.get("exact", True)),
            nugget=float(cfg.get("nugget", 0.0)),
        )
        fields[parameter] = field
        variances[parameter] = variance
        rows.append(
            {
                "parameter": parameter,
                "status": "ok",
                "n_controls": int(values.size),
                "control_min": float(np.nanmin(values)),
                "control_max": float(np.nanmax(values)),
                "control_std": float(np.nanstd(values)),
                "range_hint_m": range_hint,
                "variance_p50": _nanquantile(variance, 0.50),
                "variance_p95": _nanquantile(variance, 0.95),
                "variogram": str(cfg.get("variogram", "spherical")),
                "nugget": float(cfg.get("nugget", 0.0)),
            }
        )
    distance = _distance_to_controls(control_x_m, control_y_m, output_geometry)
    outside_fraction = _outside_control_hull_fraction(control_x_m, control_y_m, output_geometry)
    for row in rows:
        row["distance_to_control_p50_m"] = _nanquantile(distance, 0.50)
        row["distance_to_control_p95_m"] = _nanquantile(distance, 0.95)
        row["outside_control_hull_fraction"] = outside_fraction
    return fields, pd.DataFrame.from_records(rows), distance


def _krige_parameter(
    control_x_m: np.ndarray,
    control_y_m: np.ndarray,
    control_values: np.ndarray,
    output_geometry: OutputGeometry,
    *,
    range_hint: float,
    variogram: str,
    exact: bool,
    nugget: float,
) -> tuple[np.ndarray, np.ndarray]:
    import gstools as gs

    finite = np.isfinite(control_x_m) & np.isfinite(control_y_m) & np.isfinite(control_values)
    x = control_x_m[finite]
    y = control_y_m[finite]
    values = control_values[finite]
    if values.size == 0:
        raise ValueError("No finite parameter controls.")
    if values.size == 1 or np.allclose(values, values[0]):
        shape = (output_geometry.ilines.size, output_geometry.xlines.size) if not output_geometry.is_section else (output_geometry.ilines.size,)
        return np.full(shape, float(values[0]), dtype=np.float64), np.zeros(shape, dtype=np.float64)
    model_map = {
        "spherical": gs.Spherical,
        "exponential": gs.Exponential,
        "gaussian": gs.Gaussian,
    }
    key = str(variogram).casefold()
    if key not in model_map:
        raise ValueError(f"Unsupported parameter variogram: {variogram}")
    model = model_map[key](
        dim=2,
        var=float(max(np.nanvar(values), 1e-8)),
        len_scale=float(max(range_hint, 1.0)),
        nugget=float(max(nugget, 0.0)),
    )
    krige = gs.krige.Ordinary(model, cond_pos=[x, y], cond_val=values, exact=bool(exact))
    if output_geometry.is_section:
        if output_geometry.x_m is None or output_geometry.y_m is None:
            raise ValueError("section output geometry lacks physical XY coordinates.")
        field, variance = krige(
            [output_geometry.x_m, output_geometry.y_m],
            mesh_type="unstructured",
            return_var=True,
        )
    else:
        if output_geometry.x_grid_m is None or output_geometry.y_grid_m is None:
            raise ValueError("volume output geometry lacks physical XY grids.")
        field, variance = krige(
            [output_geometry.x_grid_m.ravel(), output_geometry.y_grid_m.ravel()],
            mesh_type="unstructured",
            return_var=True,
        )
        field = np.asarray(field, dtype=np.float64).reshape(output_geometry.x_grid_m.shape)
        variance = np.asarray(variance, dtype=np.float64).reshape(output_geometry.x_grid_m.shape)
        return field, variance
    return np.asarray(field, dtype=np.float64), np.asarray(variance, dtype=np.float64)


def _reconstruct_lfm(
    *,
    a_field: np.ndarray,
    b_field: np.ndarray,
    output_geometry: OutputGeometry,
    surfaces: Mapping[str, HorizonSurface],
    horizon_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    top_name = horizon_names[0]
    bottom_name = horizon_names[-1]
    top = _sample_surface_for_output(surfaces[top_name], output_geometry)
    bottom = _sample_surface_for_output(surfaces[bottom_name], output_geometry)
    thickness = bottom - top
    if output_geometry.is_section:
        samples = output_geometry.samples[None, :]
        u = (samples - top[:, None]) / thickness[:, None]
        valid = np.isfinite(u) & np.isfinite(a_field[:, None]) & np.isfinite(b_field[:, None]) & (thickness[:, None] > 0) & (u >= 0.0) & (u <= 1.0)
        log_ai = a_field[:, None] + b_field[:, None] * (2.0 * u - 1.0)
    else:
        samples = output_geometry.samples[None, None, :]
        u = (samples - top[:, :, None]) / thickness[:, :, None]
        valid = np.isfinite(u) & np.isfinite(a_field[:, :, None]) & np.isfinite(b_field[:, :, None]) & (thickness[:, :, None] > 0) & (u >= 0.0) & (u <= 1.0)
        log_ai = a_field[:, :, None] + b_field[:, :, None] * (2.0 * u - 1.0)
    log_ai = np.where(valid, log_ai, np.nan)
    rows = []
    for name in horizon_names[1:-1]:
        surface = _sample_surface_for_output(surfaces[name], output_geometry)
        if output_geometry.is_section:
            below_idx = np.searchsorted(output_geometry.samples, surface, side="left")
            below_idx = np.clip(below_idx, 1, output_geometry.samples.size - 1)
            above_idx = below_idx - 1
            values_above = log_ai[np.arange(log_ai.shape[0]), above_idx]
            values_below = log_ai[np.arange(log_ai.shape[0]), below_idx]
        else:
            below_idx = np.searchsorted(output_geometry.samples, surface.ravel(), side="left")
            below_idx = np.clip(below_idx, 1, output_geometry.samples.size - 1)
            above_idx = below_idx - 1
            ii, jj = np.indices(surface.shape)
            values_above = log_ai[ii.ravel(), jj.ravel(), above_idx]
            values_below = log_ai[ii.ravel(), jj.ravel(), below_idx]
        diff = values_below - values_above
        rows.append(
            {
                "horizon_name": name,
                "n_samples": int(np.count_nonzero(np.isfinite(diff))),
                "lfm_delta_p50": _nanquantile(diff, 0.50),
                "lfm_delta_p95_abs": _nanquantile(np.abs(diff), 0.95),
                "status": "ok" if _nanquantile(np.abs(diff), 0.95) < 0.02 else "lfm_internal_horizon_discontinuity",
            }
        )
    return log_ai, valid, pd.DataFrame.from_records(rows)


def _resolve_output_geometry(
    cfg: Mapping[str, Any],
    *,
    survey: Any,
    sample_axis: np.ndarray,
    surfaces: Mapping[str, HorizonSurface],
    horizon_names: Sequence[str],
) -> OutputGeometry:
    mode = str(cfg.get("mode", "volume")).casefold()
    if mode == "volume":
        ilines = survey.line_geometry.inline_axis.values()
        xlines = survey.line_geometry.xline_axis.values()
        x_grid, y_grid = survey.line_geometry.trace_xy_grids(ilines, xlines)
        preliminary = OutputGeometry(
            mode="volume",
            ilines=ilines,
            xlines=xlines,
            samples=sample_axis,
            is_section=False,
            x_grid_m=x_grid,
            y_grid_m=y_grid,
        )
        samples = _crop_samples(
            sample_axis,
            cfg,
            output_geometry=preliminary,
            surfaces=surfaces,
            horizon_names=horizon_names,
        )
        return OutputGeometry(**{**preliminary.__dict__, "samples": samples})
    if mode != "section":
        raise ValueError(f"real_field_model_inputs.output_geometry.mode must be volume or section, got {mode!r}.")
    section_cfg = _mapping(cfg.get("section"), "real_field_model_inputs.output_geometry.section")
    points = section_cfg.get("path")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError("real_field_model_inputs.output_geometry.section.path must contain at least two points.")
    n_traces = int(section_cfg.get("n_traces") or _infer_trace_count(points[0], points[-1]))
    first = _mapping(points[0], "real_field_model_inputs.output_geometry.section.path[0]")
    last = _mapping(points[-1], "real_field_model_inputs.output_geometry.section.path[-1]")
    ilines = np.linspace(float(first["inline"]), float(last["inline"]), n_traces)
    xlines = np.linspace(float(first["xline"]), float(last["xline"]), n_traces)
    xy = np.asarray([survey.line_geometry.line_to_coord(il, xl) for il, xl in zip(ilines, xlines)], dtype=np.float64)
    preliminary = OutputGeometry(
        mode="section",
        ilines=ilines,
        xlines=xlines,
        samples=sample_axis,
        is_section=True,
        section_id=str(section_cfg.get("section_id") or "section"),
        x_m=xy[:, 0],
        y_m=xy[:, 1],
    )
    samples = _crop_samples(
        sample_axis,
        cfg,
        output_geometry=preliminary,
        surfaces=surfaces,
        horizon_names=horizon_names,
    )
    return OutputGeometry(**{**preliminary.__dict__, "samples": samples})


def _crop_samples(
    sample_axis: np.ndarray,
    cfg: Mapping[str, Any],
    *,
    output_geometry: OutputGeometry,
    surfaces: Mapping[str, HorizonSurface],
    horizon_names: Sequence[str],
) -> np.ndarray:
    start = cfg.get("sample_start_s")
    end = cfg.get("sample_end_s")
    if start is None and end is None:
        top = _sample_surface_for_output(surfaces[horizon_names[0]], output_geometry)
        bottom = _sample_surface_for_output(surfaces[horizon_names[-1]], output_geometry)
        finite = np.isfinite(top) & np.isfinite(bottom) & (bottom > top)
        if not np.any(finite):
            raise ValueError("Cannot derive output sample window: no valid top/bottom horizon support.")
        context = float(cfg.get("target_context_s", 0.05))
        start = max(float(sample_axis[0]), float(np.nanmin(top[finite])) - context)
        end = min(float(sample_axis[-1]), float(np.nanmax(bottom[finite])) + context)
    i0 = 0 if start is None else int(np.searchsorted(sample_axis, float(start), side="left"))
    i1 = sample_axis.size if end is None else int(np.searchsorted(sample_axis, float(end), side="right"))
    i0 = max(0, i0)
    i1 = min(sample_axis.size, i1)
    if i0 >= i1:
        raise ValueError("real_field_model_inputs.output_geometry selected an empty sample window.")
    return np.asarray(sample_axis[i0:i1], dtype=np.float64)


def _piecewise_cell_average_log_ai(
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
    half = 0.5 * float(dt_s)
    for start, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        if end - start < 2:
            continue
        run_md = md[start:end]
        run_twt = np.interp(run_md, table.md, table.twt)
        run_values = np.log(values[start:end])
        if np.any(np.diff(run_twt) <= 0.0):
            continue
        candidates = np.flatnonzero((centers - half >= run_twt[0]) & (centers + half <= run_twt[-1]))
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


def _huber_fit_line(x: np.ndarray, y: np.ndarray, *, f_scale: float) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        raise ValueError("not_enough_samples_for_trend_fit")
    initial = np.polyfit(x, y, deg=1)
    result = least_squares(
        lambda beta: beta[0] + beta[1] * x - y,
        x0=np.asarray([initial[1], initial[0]], dtype=np.float64),
        loss="huber",
        f_scale=float(f_scale),
    )
    if not result.success:
        raise ValueError(f"huber_fit_failed:{result.message}")
    return {"a": float(result.x[0]), "b": float(result.x[1])}


def _sample_horizon_along(
    *,
    samples_inline: np.ndarray,
    samples_xline: np.ndarray,
    surface: HorizonSurface,
) -> np.ndarray:
    out = np.full(samples_inline.shape, np.nan, dtype=np.float64)
    for idx, (il, xl) in enumerate(zip(samples_inline, samples_xline)):
        if not np.isfinite(il) or not np.isfinite(xl):
            continue
        try:
            out[idx] = surface.value_at_line(float(il), float(xl))
        except ValueError:
            continue
    return out


def _sample_surface_for_output(surface: HorizonSurface, output_geometry: OutputGeometry) -> np.ndarray:
    if output_geometry.is_section:
        return np.asarray(
            [surface.value_at_line(il, xl) for il, xl in zip(output_geometry.ilines, output_geometry.xlines)],
            dtype=np.float64,
        )
    return surface.values.astype(np.float64, copy=False)


def _distance_to_controls(control_x_m: np.ndarray, control_y_m: np.ndarray, output_geometry: OutputGeometry) -> np.ndarray:
    coords = np.column_stack([control_x_m, control_y_m])
    tree = cKDTree(coords)
    if output_geometry.is_section:
        if output_geometry.x_m is None or output_geometry.y_m is None:
            raise ValueError("section output geometry lacks physical XY coordinates.")
        query = np.column_stack([output_geometry.x_m, output_geometry.y_m])
        dist, _ = tree.query(query, k=1)
        return np.asarray(dist, dtype=np.float64)
    if output_geometry.x_grid_m is None or output_geometry.y_grid_m is None:
        raise ValueError("volume output geometry lacks physical XY grids.")
    dist, _ = tree.query(
        np.column_stack([output_geometry.x_grid_m.ravel(), output_geometry.y_grid_m.ravel()]),
        k=1,
    )
    return np.asarray(dist, dtype=np.float64).reshape(output_geometry.x_grid_m.shape)


def _outside_control_hull_fraction(control_x_m: np.ndarray, control_y_m: np.ndarray, output_geometry: OutputGeometry) -> float:
    coords = np.column_stack([control_x_m, control_y_m])
    if coords.shape[0] < 3:
        return 1.0
    try:
        hull = ConvexHull(coords)
    except QhullError:
        return 1.0
    equations = hull.equations
    if output_geometry.is_section:
        if output_geometry.x_m is None or output_geometry.y_m is None:
            raise ValueError("section output geometry lacks physical XY coordinates.")
        query = np.column_stack([output_geometry.x_m, output_geometry.y_m])
    else:
        if output_geometry.x_grid_m is None or output_geometry.y_grid_m is None:
            raise ValueError("volume output geometry lacks physical XY grids.")
        query = np.column_stack([output_geometry.x_grid_m.ravel(), output_geometry.y_grid_m.ravel()])
    inside = np.all(query @ equations[:, :-1].T + equations[:, -1] <= 1e-8, axis=1)
    return float(1.0 - np.mean(inside))


def _lfm_stats(log_ai: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(log_ai, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if not np.any(valid):
        return {}
    if values.ndim == 3:
        per_trace_std = _masked_std(values, valid, axis=2)
        time_diff = np.diff(values, axis=2)
        diff_valid = valid[:, :, 1:] & valid[:, :, :-1] & np.isfinite(time_diff)
        lateral_std = _masked_std(values, valid, axis=(0, 1))
    else:
        per_trace_std = _masked_std(values, valid, axis=1)
        time_diff = np.diff(values, axis=1)
        diff_valid = valid[:, 1:] & valid[:, :-1] & np.isfinite(time_diff)
        lateral_std = _masked_std(values, valid, axis=0)
    finite_values = values[valid]
    finite_diff = time_diff[diff_valid]
    return {
        "valid_fraction": float(np.mean(valid)),
        "global_mean": float(np.mean(finite_values)),
        "global_std": float(np.std(finite_values)),
        "global_rms": float(np.sqrt(np.mean(finite_values * finite_values))),
        "trace_time_std_median": _nanquantile(per_trace_std, 0.50),
        "trace_time_std_max": _nanquantile(per_trace_std, 1.00),
        "time_diff_rms": float(np.sqrt(np.mean(finite_diff * finite_diff))) if finite_diff.size else float("nan"),
        "lateral_std_median": _nanquantile(lateral_std, 0.50),
        "p01": _nanquantile(finite_values, 0.01),
        "p50": _nanquantile(finite_values, 0.50),
        "p99": _nanquantile(finite_values, 0.99),
    }


def _masked_std(values: np.ndarray, valid: np.ndarray, *, axis: int | tuple[int, ...]) -> np.ndarray:
    data = np.where(valid, values, np.nan)
    count = np.sum(np.isfinite(data), axis=axis)
    mean = np.divide(
        np.nansum(data, axis=axis),
        count,
        out=np.full(count.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )
    if isinstance(axis, tuple):
        expanded = mean
        for ax in sorted(axis):
            expanded = np.expand_dims(expanded, axis=ax)
    else:
        expanded = np.expand_dims(mean, axis=axis)
    sq = np.where(np.isfinite(data), (data - expanded) ** 2, np.nan)
    var = np.divide(
        np.nansum(sq, axis=axis),
        count,
        out=np.full(count.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )
    return np.sqrt(var)


def _lfm_status(stats: Mapping[str, float], qc: Mapping[str, Any]) -> str:
    if not stats:
        return "no_valid_lfm_samples"
    min_diff = float(qc.get("min_time_diff_rms", 1e-4))
    min_std = float(qc.get("min_trace_time_std_median", 1e-4))
    if float(stats.get("time_diff_rms", 0.0)) < min_diff:
        return "lfm_time_flat_or_invalid"
    if float(stats.get("trace_time_std_median", 0.0)) < min_std:
        return "lfm_time_structure_weak"
    return "ok"


def _write_figures(
    *,
    figures_dir: Path,
    output_geometry: OutputGeometry,
    log_ai: np.ndarray,
    valid_mask: np.ndarray,
    a_field: np.ndarray,
    b_field: np.ndarray,
    distance_to_control: np.ndarray,
    controls: pd.DataFrame,
    repo_root: Path,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if output_geometry.is_section:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        if np.allclose(output_geometry.ilines, output_geometry.ilines[0], rtol=0.0, atol=1e-8):
            lateral_axis = output_geometry.xlines
            lateral_label = "xline"
        else:
            lateral_axis = output_geometry.ilines
            lateral_label = "inline"
        extent = [
            float(lateral_axis[0]),
            float(lateral_axis[-1]),
            float(output_geometry.samples[-1]),
            float(output_geometry.samples[0]),
        ]
        axes[0].imshow(np.where(valid_mask, log_ai, np.nan).T, aspect="auto", extent=extent, cmap="viridis")
        axes[0].set_title("real_field_model_inputs LFM log(AI)")
        axes[0].set_ylabel("TWT s")
        axes[1].plot(lateral_axis, a_field, label="a")
        axes[1].plot(lateral_axis, b_field, label="b")
        axes[1].legend()
        axes[1].set_ylabel("coefficient")
        axes[2].plot(lateral_axis, distance_to_control)
        axes[2].set_ylabel("distance to control (m)")
        axes[2].set_xlabel(f"{lateral_label} along section")
        fig.tight_layout()
        path = figures_dir / "section_overview.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs["section_overview"] = repo_relative_path(path, root=repo_root)
    else:
        center_il = log_ai.shape[0] // 2
        center_xl = log_ai.shape[1] // 2
        center_t = log_ai.shape[2] // 2
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes[0, 0].imshow(np.where(valid_mask[center_il], log_ai[center_il], np.nan).T, aspect="auto", cmap="viridis")
        axes[0, 0].set_title("central inline log(AI)")
        axes[0, 1].imshow(np.where(valid_mask[:, center_xl, :], log_ai[:, center_xl, :], np.nan).T, aspect="auto", cmap="viridis")
        axes[0, 1].set_title("central xline log(AI)")
        axes[1, 0].imshow(a_field, aspect="auto", cmap="viridis")
        axes[1, 0].set_title("a_field")
        axes[1, 1].imshow(np.where(valid_mask[:, :, center_t], log_ai[:, :, center_t], np.nan), aspect="auto", cmap="viridis")
        axes[1, 1].set_title("mid-time log(AI)")
        fig.tight_layout()
        path = figures_dir / "volume_overview.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs["volume_overview"] = repo_relative_path(path, root=repo_root)

    ok = controls[controls["status"].eq("ok")]
    if not ok.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(ok["representative_inline"], ok["representative_xline"], c=ok["a"], cmap="viridis", s=48)
        ax.set_xlabel("inline")
        ax.set_ylabel("xline")
        ax.set_title("accepted well trend controls (color=a)")
        fig.colorbar(sc, ax=ax, label="a")
        fig.tight_layout()
        path = figures_dir / "well_trend_controls.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs["well_trend_controls"] = repo_relative_path(path, root=repo_root)
    return outputs


def _export_lfm_volume(
    *,
    output_dir: Path,
    log_ai: np.ndarray,
    output_geometry: OutputGeometry,
    seismic_path: Path,
    seismic_type: str,
    seismic_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    export_cfg = dict(seismic_cfg.get("volume_export") or {})
    if not _as_bool(export_cfg.get("enabled", True)):
        return {"status": "disabled", "format": "", "path": ""}
    if output_geometry.is_section:
        return {"status": "skipped_section_output", "format": "", "path": ""}
    if output_geometry.samples.size == 0:
        return {"status": "skipped_empty_sample_axis", "format": "", "path": ""}
    payload = export_volume_like_source(
        output_base=output_dir / "real_field_lfm_log_ai",
        volume=log_ai,
        ilines=output_geometry.ilines,
        xlines=output_geometry.xlines,
        samples=output_geometry.samples,
        source_seismic_file=seismic_path,
        source_seismic_type=seismic_type,
        title="Real-field LFM log(AI)",
        details=[
            f"schema={SCHEMA_VERSION}",
            "field=log_ai",
            "domain=log(AI)",
        ],
        seismic_options=seismic_cfg,
        inline_chunk_size=int(export_cfg.get("inline_chunk_size", seismic_cfg.get("zgy_inline_chunk_size", 16))),
        nan_fill=export_cfg.get("nan_fill"),
    )
    return payload


def _summary_payload(
    *,
    status: str,
    config: RealFieldModelInputsConfig,
    repo_root: Path,
    data_root: Path,
    output_dir: Path,
    well_auto_tie_dir: Path,
    well_inventory_file: Path,
    seismic_path: Path,
    controls: pd.DataFrame,
    parameter_qc: pd.DataFrame,
    horizon_qc: pd.DataFrame,
    lfm_stats: Mapping[str, Any],
    outputs: Mapping[str, Path],
) -> dict[str, Any]:
    status_counts = controls["status"].value_counts(dropna=False).to_dict() if "status" in controls else {}
    reason_counts = controls["reason"].value_counts(dropna=False).to_dict() if "reason" in controls else {}
    warning_counts = controls["qc_warning"].value_counts(dropna=False).to_dict() if "qc_warning" in controls else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "source_runs": {
            "well_auto_tie_dir": repo_relative_path(well_auto_tie_dir, root=repo_root),
            "well_preprocess_dir": repo_relative_path(
                resolve_relative_path(config.source_runs["well_preprocess_dir"], root=repo_root),
                root=repo_root,
            ),
        },
        "inputs": {
            "well_inventory_file": repo_relative_path(well_inventory_file, root=repo_root),
            "seismic_file": repo_relative_path(seismic_path, root=repo_root),
            "seismic_sha256": sha256_file(seismic_path),
            "benchmark_manifest": repo_relative_path(
                resolve_relative_path(config.synthetic_benchmark_dir, root=repo_root) / "benchmark_manifest.json",
                root=repo_root,
            ),
            "benchmark_manifest_sha256": sha256_file(
                resolve_relative_path(config.synthetic_benchmark_dir, root=repo_root) / "benchmark_manifest.json"
            ),
            "horizons": [
                {
                    "name": item["name"],
                    "file": repo_relative_path(resolve_relative_path(item["file"], root=data_root), root=repo_root),
                }
                for item in config.horizons
            ],
        },
        "output_dir": repo_relative_path(output_dir, root=repo_root),
        "output_field": "log_ai",
        "output_value_domain": "log(AI)",
        "valid_mask_field": "valid_mask_model",
        "control_wells": {
            "n_total_metrics": int(controls.shape[0]),
            "n_accepted": int(np.count_nonzero(controls["status"].eq("ok"))) if "status" in controls else 0,
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
            "warning_counts": {str(k): int(v) for k, v in warning_counts.items()},
        },
        "parameter_modeling": parameter_qc.to_dict(orient="records"),
        "horizon_qc": horizon_qc.to_dict(orient="records"),
        "lfm_stats": dict(lfm_stats),
        "outputs": {
            key: repo_relative_path(path, root=repo_root)
            for key, path in outputs.items()
        },
        "output_sha256": {
            key: sha256_file(path)
            for key, path in outputs.items()
            if path.is_file()
        },
        "legacy_lfm_statement": "No numeric values were read from historical lfm_precomputed_* artifacts.",
    }


def _axis_from_geometry(geometry: Mapping[str, Any], prefix: str) -> np.ndarray:
    minimum = float(geometry[f"{prefix}_min"])
    maximum = float(geometry[f"{prefix}_max"])
    step = float(geometry[f"{prefix}_step"])
    count_key = "n_il" if prefix == "inline" else "n_xl"
    count = int(geometry[count_key])
    return minimum + np.arange(count, dtype=np.float64) * step


def _required_artifact(value: Any, *, run_dir: Path, repo_root: Path, label: str) -> Path:
    path = resolve_artifact_path(value, root=repo_root, run_dir=run_dir)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _load_trace_plan(value: Any, *, run_dir: Path, repo_root: Path) -> pd.DataFrame | None:
    text = "" if value is None else str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    path = resolve_artifact_path(text, root=repo_root, run_dir=run_dir)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"optimized_trace_sample_plan_file does not exist: {path}")
    frame = pd.read_csv(path)
    required = {"twt_s", "inline_float", "xline_float"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"optimized_trace_sample_plan_file is missing columns {sorted(missing)}: {path}")
    return frame


def _regular_dt(axis: np.ndarray) -> float:
    values = np.asarray(axis, dtype=np.float64)
    if values.size < 2:
        raise ValueError("sample axis needs at least two samples.")
    diffs = np.diff(values)
    if np.any(diffs <= 0.0):
        raise ValueError("sample axis must be strictly increasing.")
    dt = float(np.median(diffs))
    if not np.allclose(diffs, dt, rtol=1e-4, atol=1e-9):
        raise ValueError("sample axis must be regular.")
    return dt


def _nearest_neighbor_range(inlines: np.ndarray, xlines: np.ndarray) -> float:
    coords = np.column_stack([inlines, xlines]).astype(np.float64)
    if coords.shape[0] <= 1:
        return 1.0
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    finite = nearest[np.isfinite(nearest)]
    return float(max(np.median(finite), 1.0)) if finite.size else 1.0


def _infer_trace_count(first: Mapping[str, Any], last: Mapping[str, Any]) -> int:
    return int(max(abs(float(last["inline"]) - float(first["inline"])), abs(float(last["xline"]) - float(first["xline"]))) + 1)


def _coerce_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _nanquantile(values: Any, q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, float(q)))


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _required_text(config: Mapping[str, Any], key: str, *, path: str) -> str:
    text = "" if config.get(key) is None else str(config.get(key)).strip()
    if not text:
        raise ValueError(f"{path}.{key} must be a non-empty string.")
    return text


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}
