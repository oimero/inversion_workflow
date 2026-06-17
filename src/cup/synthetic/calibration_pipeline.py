"""Calibration pipeline adapters for project data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cup.petrel.load import import_well_tops_petrel
from cup.seismic.wavelet import infer_wavelet_dt, load_wavelet_csv
from cup.synthetic.calibration import (
    GENERATOR_FAMILY,
    SCHEMA_VERSION as CALIBRATION_SCHEMA,
    WellZoneCurves,
    calibrate_impedance,
)
from cup.synthetic.constants import IMPLEMENTATION_SCOPE
from cup.synthetic.figures import write_calibration_figures
from cup.synthetic.hashing import sha256_file
from cup.synthetic.sources import _artifact
from cup.time_config import TimeWorkflowConfig
from cup.utils.io import repo_relative_path, resolve_relative_path, write_json
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
