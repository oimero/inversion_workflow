"""Calibration pipeline adapters for project data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cup.petrel.load import import_interpretation_petrel, import_well_tops_petrel
from cup.seismic.survey import open_survey
from cup.seismic.target_zone import TargetZone
from cup.seismic.wavelet import infer_wavelet_dt, load_wavelet_csv
from cup.synthetic.core.calibration import (
    GENERATOR_FAMILY,
    SCHEMA_VERSION as CALIBRATION_SCHEMA,
    WellZoneCurves,
    calibrate_impedance,
)
from cup.synthetic.time.config import IMPLEMENTATION_SCOPE
from cup.synthetic.reporting.figures import write_calibration_figures
from cup.config.workflow import WorkflowConfig
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    repo_relative_path,
    require_contract_fingerprint,
    resolve_artifact_path,
    resolve_relative_path,
    write_json,
)
from cup.well.assets import normalize_well_name
from cup.well.las import read_las_curve
from cup.well.td import find_well_top_md, load_workflow_time_depth_table_csv


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


def _artifact(value: Any, *, run_dir: Path, repo_root: Path, label: str) -> Path:
    path = resolve_artifact_path(value, root=repo_root, run_dir=run_dir)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _horizon_times_from_well_tops(
    well_tops_df: pd.DataFrame,
    *,
    well_name: str,
    horizons: list[Mapping[str, Any]],
    table: Any,
) -> list[float]:
    """Map explicitly configured well tops to TWT under stable horizon IDs."""
    horizon_times: list[float] = []
    for horizon in horizons:
        horizon_name = str(horizon["name"])
        well_top = str(horizon["well_top"])
        md = find_well_top_md(well_tops_df, well_name=well_name, surface=well_top)
        if not float(table.md[0]) <= md <= float(table.md[-1]):
            raise ValueError(f"outside_tdt_support:{horizon_name}:well_top={well_top}")
        horizon_times.append(float(np.interp(md, table.md, table.twt)))
    return horizon_times


def build_calibration_inputs(
    *,
    workflow: WorkflowConfig,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    repo_root: Path,
) -> tuple[list[WellZoneCurves], dict[str, Any]]:
    clusters = pd.read_csv(
        sources["wavelet_generation_dir"] / "evaluation_well_spatial_clusters.csv"
    )
    metrics = pd.read_csv(sources["well_auto_tie_dir"] / "well_tie_metrics.csv")
    preprocess = pd.read_csv(
        sources["well_preprocess_dir"] / "well_preprocess_status.csv"
    )
    with (sources["well_auto_tie_dir"] / "run_summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        auto_tie_summary = json.load(handle)
    inventory_text = str(
        dict(auto_tie_summary.get("inputs") or {}).get("inventory_file") or ""
    ).strip()
    if not inventory_text:
        raise ValueError("well_auto_tie run_summary.json lacks inputs.inventory_file")
    inventory_path = resolve_relative_path(inventory_text, root=repo_root)
    inventory = pd.read_csv(inventory_path)
    inventory_required = {"well_name", "inline_float", "xline_float"}
    missing_inventory = sorted(inventory_required - set(inventory.columns))
    if missing_inventory:
        raise ValueError(
            f"well_inventory.csv is missing columns: {missing_inventory}"
        )
    for frame, column in (
        (clusters, "well_name"),
        (metrics, "well_name"),
        (preprocess, "well_name"),
        (inventory, "well_name"),
    ):
        frame["_well_key"] = frame[column].map(normalize_well_name)
        if frame["_well_key"].duplicated().any():
            raise ValueError(f"Duplicate normalized well name in {column} table.")
    wells = clusters.merge(
        metrics, on=["well_name", "_well_key"], validate="one_to_one"
    ).merge(
        preprocess[["_well_key", "preprocess_status", "preprocessed_las"]],
        on="_well_key",
        validate="one_to_one",
    ).merge(
        inventory[["_well_key", "inline_float", "xline_float"]],
        on="_well_key",
        validate="one_to_one",
    )
    well_tops = import_well_tops_petrel(
        resolve_relative_path(
            workflow.assets.well_tops_file,
            root=resolve_relative_path(workflow.data_root, root=repo_root),
        )
    )
    data_root = resolve_relative_path(workflow.data_root, root=repo_root)
    ordered_horizons = [item["name"] for item in script_cfg["horizons"]]
    seismic_path = resolve_relative_path(workflow.seismic.file, root=data_root)
    survey = open_survey(
        seismic_path,
        workflow.seismic.type,
        segy_options={
            key: value
            for key, value in workflow.seismic.as_dict().items()
            if key in {"iline", "xline", "istep", "xstep"} and value is not None
        }
        or None,
    )
    raw_horizons = {}
    for horizon in script_cfg["horizons"]:
        frame = import_interpretation_petrel(
            resolve_relative_path(horizon["file"], root=data_root)
        ).copy()
        frame["interpretation"] = np.abs(
            frame["interpretation"].to_numpy(dtype=np.float64)
        )
        raw_horizons[str(horizon["name"])] = frame
    target_zone = TargetZone(
        raw_horizons,
        survey.describe_geometry(domain="time"),
        ordered_horizons,
        nearest_distance_limit=script_cfg["target_zone"].get("nearest_distance_limit"),
        outlier_threshold=script_cfg["target_zone"].get("outlier_threshold"),
        outlier_min_neighbor_count=int(
            script_cfg["target_zone"].get("outlier_min_neighbor_count", 2)
        ),
        min_thickness=script_cfg["target_zone"].get("min_thickness_s"),
    )
    wavelet_time, _ = load_wavelet_csv(
        sources["wavelet_generation_dir"] / "selected_wavelet.csv"
    )
    output_dt = infer_wavelet_dt(wavelet_time)
    expected = float(script_cfg["sampling"]["expected_output_dt_s"])
    if not np.isclose(output_dt, expected, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"sampling_mismatch:wavelet_dt={output_dt}:expected={expected}"
        )
    truth_dt = output_dt / int(script_cfg["sampling"]["vertical_oversampling_factor"])
    records: list[WellZoneCurves] = []
    well_status: list[dict[str, Any]] = []
    horizon_audit: list[dict[str, Any]] = []
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
            filtered = read_las_curve(
                filtered_path, "AI", match_policy="exact", allow_all_nan=True
            )
            full = read_las_curve(
                full_path, "AI", match_policy="exact", allow_all_nan=True
            )
            inline = float(row["inline_float"])
            xline = float(row["xline_float"])
            horizon_times: list[float] = []
            for horizon in script_cfg["horizons"]:
                horizon_name = str(horizon["name"])
                well_top = str(horizon["well_top"])
                md = find_well_top_md(well_tops, well_name=well_name, surface=well_top)
                if not float(table.md[0]) <= md <= float(table.md[-1]):
                    raise ValueError(
                        f"outside_tdt_support:{horizon_name}:well_top={well_top}"
                    )
                well_top_twt = float(np.interp(md, table.md, table.twt))
                sample = target_zone.get_horizon_surface(horizon_name).sample_at_line(
                    inline, xline
                )
                interpreted = float(sample.value)
                horizon_audit.append(
                    {
                        "well_name": well_name,
                        "horizon_name": horizon_name,
                        "well_top_surface": well_top,
                        "inline_float": inline,
                        "xline_float": xline,
                        "well_top_md_m": float(md),
                        "well_top_twt_s": well_top_twt,
                        "interpreted_twt_s": interpreted,
                        "delta_interpretation_minus_well_top_s": interpreted
                        - well_top_twt,
                        "sample_method": str(sample.method),
                        "support_status": str(sample.support_status),
                        "status": "ok",
                        "reason": "",
                    }
                )
                horizon_times.append(well_top_twt)
            if np.any(np.diff(horizon_times) <= 0.0):
                raise ValueError("misordered_horizons")
            zone_count = 0
            for zone_index, (top, bottom) in enumerate(
                zip(horizon_times[:-1], horizon_times[1:])
            ):
                centers = np.arange(
                    top + 0.5 * truth_dt, bottom, truth_dt, dtype=np.float64
                )
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
            well_status.append(
                {
                    "well_name": well_name,
                    "status": "ok",
                    "n_zones": zone_count,
                    "reasons": "",
                }
            )
        except Exception as exc:
            well_status.append(
                {
                    "well_name": well_name,
                    "status": "rejected",
                    "n_zones": 0,
                    "reasons": f"{type(exc).__name__}:{exc}",
                }
            )
    return records, {
        "well_status": well_status,
        "well_horizon_consistency": horizon_audit,
        "output_dt_s": output_dt,
        "truth_dt_s": truth_dt,
    }


def run_calibration(
    *,
    workflow: WorkflowConfig,
    script_cfg: Mapping[str, Any],
    sources: Mapping[str, Path],
    config_provenance: Mapping[str, str],
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
    input_contracts: dict[str, dict[str, str]] = {}
    contract_source_keys = [
        "well_preprocess_dir",
        "well_auto_tie_dir",
        "wavelet_generation_dir",
    ]
    for directory_key in contract_source_keys:
        summary_path = sources[directory_key] / "run_summary.json"
        with summary_path.open("r", encoding="utf-8") as handle:
            source_summary = json.load(handle)
        input_contracts[directory_key.removesuffix("_dir")] = {
            "path": repo_relative_path(summary_path, root=repo_root),
            "contract_fingerprint_sha256": require_contract_fingerprint(
                source_summary, label=f"{directory_key} {summary_path.parent}"
            ),
        }
    calibration, objects, qc, samples, backgrounds, profile_samples = (
        calibrate_impedance(
            inputs,
            truth_dt_s=float(input_qc["truth_dt_s"]),
            ordered_horizons=[item["name"] for item in script_cfg["horizons"]],
            source_runs={
                key: repo_relative_path(path, root=repo_root)
                for key, path in sources.items()
            },
            input_contracts=input_contracts,
            state_threshold_sigma=float(
                script_cfg["impedance"]["state_threshold_sigma"]
            ),
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
    )
    objects_path = output_dir / "well_object_catalog.csv"
    qc_path = output_dir / "calibration_qc.csv"
    status_path = output_dir / "well_status.csv"
    horizon_consistency_path = output_dir / "well_horizon_consistency.csv"
    samples_path = output_dir / "well_calibration_samples.csv"
    backgrounds_path = output_dir / "well_background_fits.csv"
    profile_samples_path = output_dir / "well_object_profile_samples.csv"
    objects.to_csv(objects_path, index=False)
    qc.to_csv(qc_path, index=False)
    samples.to_csv(samples_path, index=False)
    backgrounds.to_csv(backgrounds_path, index=False)
    profile_samples.to_csv(profile_samples_path, index=False)
    pd.DataFrame.from_records(input_qc["well_status"]).to_csv(status_path, index=False)
    pd.DataFrame.from_records(input_qc["well_horizon_consistency"]).to_csv(
        horizon_consistency_path, index=False
    )
    payload = calibration.to_dict()
    payload["sample_domain"] = "time"
    payload["config_provenance"] = dict(config_provenance)
    calibration_path = output_dir / "impedance_calibration.json"
    write_json(calibration_path, payload)
    contract_fingerprint = contract_fingerprint_sha256(
        contract_schema_version=CALIBRATION_SCHEMA,
        semantics={
            "sample_domain": "time",
            "truth_dt_s": calibration.truth_dt_s,
            "ordered_horizons": list(calibration.ordered_horizons),
            "generator_family": calibration.generator_family,
            "calibration_model": calibration.to_dict(),
        },
        business_config=script_cfg,
        input_contracts=input_contracts,
        primary_artifacts={
            "impedance_calibration": calibration_path,
            "well_object_catalog": objects_path,
            "well_calibration_samples": samples_path,
            "well_background_fits": backgrounds_path,
            "well_object_profile_samples": profile_samples_path,
        },
    )
    figure_summary = write_calibration_figures(
        output_dir,
        script_cfg.get("figures", {}),
    )
    summary = {
        "schema_version": CALIBRATION_SCHEMA,
        "status": "success",
        "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
        "contract_fingerprint_sha256": contract_fingerprint,
        "input_contracts": input_contracts,
        "sample_domain": "time",
        "generator_family": GENERATOR_FAMILY,
        "implementation_scope": IMPLEMENTATION_SCOPE,
        "source_runs": calibration.source_runs,
        "config_provenance": dict(config_provenance),
        "n_well_zone_inputs": len(inputs),
        "n_objects": int(len(objects)),
        "well_status_counts": pd.Series(
            [row["status"] for row in input_qc["well_status"]]
        )
        .value_counts()
        .to_dict(),
        "outputs": {
            "impedance_calibration": repo_relative_path(
                calibration_path, root=repo_root
            ),
            "well_object_catalog": repo_relative_path(objects_path, root=repo_root),
            "calibration_qc": repo_relative_path(qc_path, root=repo_root),
            "well_calibration_samples": repo_relative_path(
                samples_path, root=repo_root
            ),
            "well_background_fits": repo_relative_path(
                backgrounds_path, root=repo_root
            ),
            "well_object_profile_samples": repo_relative_path(
                profile_samples_path, root=repo_root
            ),
            "well_horizon_consistency": repo_relative_path(
                horizon_consistency_path, root=repo_root
            ),
            "well_status": repo_relative_path(status_path, root=repo_root),
        },
        "figures": {
            "generated_count": int(figure_summary.get("generated_count", 0)),
            "skipped_count": int(figure_summary.get("skipped_count", 0)),
            "figure_manifest": repo_relative_path(
                Path(
                    str(
                        figure_summary.get(
                            "figure_manifest",
                            output_dir / "figures" / "figure_manifest.json",
                        )
                    )
                ),
                root=repo_root,
            ),
        },
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary
