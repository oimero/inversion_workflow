"""Project-data adapters for the synthoseis-lite impedance-truth slice."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

from cup.petrel.load import (
    import_interpretation_petrel,
    import_well_tops_petrel,
)
from cup.seismic.horizon import HorizonSurface
from cup.seismic.survey import open_survey
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
from cup.synthetic.generation import (
    GenerationRejected,
    GenerationScenario,
    generate_field_section,
)
from cup.synthetic.io import sha256_file, write_generated_section
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
IMPLEMENTATION_SCOPE = "impedance_truth_and_nominal_forward"


@dataclass(frozen=True)
class SectionGeometry:
    section_id: str
    lateral_m: np.ndarray
    inline_float: np.ndarray
    xline_float: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    horizon_twt_s: np.ndarray


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _required_text(value: Mapping[str, Any], key: str, *, path: str) -> str:
    text = "" if value.get(key) is None else str(value[key]).strip()
    if not text:
        raise ValueError(f"{path}.{key} must be a non-empty string.")
    return text


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
    return {
        "global_seed": int(root.get("global_seed", 20260615)),
        "source_runs": source_runs,
        "sampling": {
            "expected_output_dt_s": float(sampling.get("expected_output_dt_s", 0.002)),
            "vertical_oversampling_factor": int(sampling.get("vertical_oversampling_factor", 8)),
        },
        "horizons": horizon_items,
        "sections": section_items,
        "lateral_sample_interval_m": float(geometry.get("lateral_sample_interval_m", 25.0)),
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
    }


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
    calibration, objects, qc = calibrate_impedance(
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
    objects.to_csv(objects_path, index=False)
    qc.to_csv(qc_path, index=False)
    pd.DataFrame.from_records(input_qc["well_status"]).to_csv(status_path, index=False)
    payload = calibration.to_dict()
    payload["artifact_hashes"] = {
        "well_object_catalog.csv": sha256_file(objects_path),
        "calibration_qc.csv": sha256_file(qc_path),
    }
    write_json(output_dir / "impedance_calibration.json", payload)
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
    surfaces: list[HorizonSurface] = []
    for horizon in script_cfg["horizons"]:
        path = resolve_relative_path(
            horizon["file"],
            root=resolve_relative_path(workflow.data_root, root=repo_root),
        )
        frame = import_interpretation_petrel(path)
        values = np.abs(frame["interpretation"].to_numpy(dtype=np.float64))
        if np.nanmedian(values) > 20.0:
            values = values / 1000.0
        frame = frame.copy()
        frame["interpretation"] = values
        surfaces.append(HorizonSurface.from_petrel_dataframe(frame, name=horizon["name"]))
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
        sections.append(
            SectionGeometry(
                section_id=section["section_id"],
                lateral_m=lateral,
                inline_float=inline,
                xline_float=xline,
                x_m=x,
                y_m=y,
                horizon_twt_s=horizon_values,
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
    sections = build_section_geometries(
        workflow=workflow,
        script_cfg=script_cfg,
        repo_root=repo_root,
    )
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
    qc_records: list[dict[str, Any]] = []
    rejection_records: list[dict[str, Any]] = []
    h5_path = output_dir / "synthetic_benchmark.h5"
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["schema_version"] = DATA_SCHEMA
        h5.attrs["generator_family"] = calibration.generator_family
        h5.attrs["implementation_scope"] = IMPLEMENTATION_SCOPE
        for section in sections:
            for scenario in scenarios:
                for attempt_id in range(attempts):
                    realization_id = f"{section.section_id}__{scenario.scenario_id}__a{attempt_id:03d}"
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
                        hdf5_group = "" if qc_only else write_generated_section(h5, generated)
                        object_records.extend(generated.object_catalog)
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
                    }
                    index_records.append(record)
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
    pd.DataFrame.from_records(qc_records).to_csv(output_dir / "generation_qc.csv", index=False)
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
        index.groupby(["section_id", "scenario_id"], dropna=False)["status"]
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
    manifest = {
        "schema_version": DATA_SCHEMA,
        "generator_family": calibration.generator_family,
        "implementation_scope": IMPLEMENTATION_SCOPE,
        "development_limited": development_limited,
        "qc_only": bool(qc_only),
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
        "not_yet_implemented": [
            "canonical_suite",
            "frequency_probe_matrix",
            "lfm_ideal",
            "lfm_controlled_degraded",
            "noise_gain_and_wavelet_mismatch_scenarios",
            "highres_forward_mismatch_qc",
        ],
        "files": {
            "synthetic_benchmark.h5": sha256_file(h5_path),
            "sample_index.csv": sha256_file(output_dir / "sample_index.csv"),
            "object_catalog.csv": sha256_file(output_dir / "object_catalog.csv"),
            "generation_qc.csv": sha256_file(output_dir / "generation_qc.csv"),
            "generation_rejection_details.csv": sha256_file(
                output_dir / "generation_rejection_details.csv"
            ),
            "scenario_catalog.csv": sha256_file(output_dir / "scenario_catalog.csv"),
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
        "accepted_realizations": int(index["status"].eq("ok").sum()),
        "rejected_realizations": int(index["status"].eq("rejected").sum()),
        "failed_scenario_count": int(failed_scenarios.sum()),
    }
    write_json(output_dir / "run_summary.json", summary)
    return summary
