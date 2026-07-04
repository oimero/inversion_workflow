"""Run the time-domain forward-observability side-route analysis.

Third-, fourth-, and fifth-step run directories are discovered from
``output_root`` by default.  Set ``forward_observability.source_runs`` in the
common config to pin a reproducible run.  This analysis does not occupy a
numbered workflow step and does not gate the production chain.

Usage::

    python scripts/forward_observability.py
    python scripts/forward_observability.py --config path/to/common.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.petrel.load import import_well_tops_petrel
from cup.seismic.contracts import FORWARD_OBSERVABILITY_SCHEMA_VERSION
from cup.seismic.observability import (
    ObservabilityWindow,
    WaveletScenario,
    aggregate_frequency_evidence,
    aggregate_well_scenarios,
    analyze_frequency_scenario,
    contiguous_experiment_ranges,
    frequency_grid,
    make_artificial_wavelet_scenarios,
    operator_transfer_rows,
    regular_dt,
)
from cup.seismic.wavelet import (
    load_wavelet_csv,
    validate_wavelet_normalization,
    wavelet_half_amplitude_frequencies,
)
from cup.config.workflow import WorkflowConfig
from cup.config.sources import assert_recorded_source_matches, require_source_files, resolve_source_run
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    load_yaml_config,
    published_contract_reference,
    repo_relative_path,
    resolve_artifact_path,
    resolve_relative_path,
    sanitize_filename,
    write_json,
)
from cup.utils.masks import true_runs
from cup.well.assets import normalize_well_name
from cup.well.las import read_las_curve
from cup.well.td import find_well_top_md, load_workflow_time_depth_table_csv
from cup.well.tie import load_saved_seismic_trace_csv


DEFAULT_COMMON_CONFIG = Path("experiments/common/common.yaml")
SCHEMA_VERSION = FORWARD_OBSERVABILITY_SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_COMMON_CONFIG,
        help="YAML config containing the main workflow configuration.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--well", type=str, default=None, help="Optional single-well debug filter.")
    return parser.parse_args()


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    return dict(value)


def _required_text(config: Mapping[str, Any], key: str, *, path: str) -> str:
    value = config.get(key)
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{path}.{key} must be a non-empty string.")
    return text


def _positive_float(config: Mapping[str, Any], key: str, *, path: str, default: float | None = None) -> float:
    value = config.get(key, default)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}.{key} must be a positive number.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path}.{key} must be a positive number.")
    return result


def _positive_int(config: Mapping[str, Any], key: str, *, path: str, default: int) -> int:
    value = config.get(key, default)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}.{key} must be a positive integer.") from exc
    if result <= 0:
        raise ValueError(f"{path}.{key} must be a positive integer.")
    return result


def _script_config(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(config.get("forward_observability"), path="forward_observability")
    source_runs = dict(root.get("source_runs") or {})
    sources = {
        key: str(source_runs.get(key) or "").strip()
        for key in ("wavelet_generation_dir", "well_auto_tie_dir", "well_preprocess_dir")
    }

    if "horizons" in root:
        raise ValueError("forward_observability.horizons is retired; use top-level target_interval.horizons.")
    target_interval = _mapping(config.get("target_interval"), path="target_interval")
    raw_horizons = target_interval.get("horizons")
    if not isinstance(raw_horizons, (list, tuple)):
        raise ValueError("target_interval.horizons must be a list.")
    horizons = []
    for idx, item in enumerate(raw_horizons):
        path = f"target_interval.horizons[{idx}]"
        entry = _mapping(item, path=path)
        horizons.append(
            {
                "name": _required_text(entry, "name", path=path),
                "well_top": _required_text(entry, "well_top", path=path),
                "file": _required_text(entry, "file", path=path),
            }
        )
    ordered_names = [item["name"] for item in horizons]
    if len(ordered_names) < 2 or any(not name for name in ordered_names):
        raise ValueError("target_interval.horizons needs at least two non-empty names.")
    normalized_names = [name.casefold() for name in ordered_names]
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("target_interval.horizons must not contain duplicate names.")

    frequency_cfg = _mapping(root.get("frequency"), path="forward_observability.frequency")
    if frequency_cfg.get("max_hz") is None:
        raise ValueError("forward_observability.frequency.max_hz must be explicitly configured.")

    perturbation = dict(root.get("perturbation") or {})
    thresholds = dict(root.get("thresholds") or {})
    tukey_alpha = float(perturbation.get("tukey_alpha", 0.5))
    if not np.isfinite(tukey_alpha) or not 0.0 <= tukey_alpha <= 1.0:
        raise ValueError("forward_observability.perturbation.tukey_alpha must be within [0, 1].")
    required_artificial = _positive_int(
        thresholds,
        "required_artificial_scenarios",
        path="forward_observability.thresholds",
        default=3,
    )
    if required_artificial > 4:
        raise ValueError(
            "forward_observability.thresholds.required_artificial_scenarios cannot exceed 4."
        )
    return {
        "source_runs": sources,
        "horizons": horizons,
        "ordered_horizons": ordered_names,
        "frequency": {
            "start_hz": _positive_float(
                frequency_cfg, "start_hz", path="forward_observability.frequency", default=5.0
            ),
            "step_hz": _positive_float(
                frequency_cfg, "step_hz", path="forward_observability.frequency", default=5.0
            ),
            "max_hz": _positive_float(
                frequency_cfg, "max_hz", path="forward_observability.frequency"
            ),
        },
        "perturbation": {
            "epsilon_log_ai": _positive_float(
                perturbation,
                "epsilon_log_ai",
                path="forward_observability.perturbation",
                default=1e-3,
            ),
            "tukey_alpha": tukey_alpha,
            "phase_degrees": _positive_float(
                perturbation,
                "phase_degrees",
                path="forward_observability.perturbation",
                default=10.0,
            ),
            "fractional_shift_samples": _positive_float(
                perturbation,
                "fractional_shift_samples",
                path="forward_observability.perturbation",
                default=0.5,
            ),
            "max_basis_condition_number": _positive_float(
                perturbation,
                "max_basis_condition_number",
                path="forward_observability.perturbation",
                default=1e6,
            ),
        },
        "thresholds": {
            "min_valid_samples": _positive_int(
                thresholds,
                "min_valid_samples",
                path="forward_observability.thresholds",
                default=50,
            ),
            "min_cycles": _positive_float(
                thresholds,
                "min_cycles",
                path="forward_observability.thresholds",
                default=2.0,
            ),
            "min_wells": _positive_int(
                thresholds,
                "min_wells",
                path="forward_observability.thresholds",
                default=5,
            ),
            "min_clusters": _positive_int(
                thresholds,
                "min_clusters",
                path="forward_observability.thresholds",
                default=3,
            ),
            "min_synthetic_rms": _positive_float(
                thresholds,
                "min_synthetic_rms",
                path="forward_observability.thresholds",
                default=1e-6,
            ),
            "required_artificial_scenarios": required_artificial,
            "max_short_log_gap_s": _positive_float(
                thresholds,
                "max_short_log_gap_s",
                path="forward_observability.thresholds",
                default=0.010,
            ),
        },
    }


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_relative_path(value, root=REPO_ROOT)


def _resolve_output_dir(args: argparse.Namespace, workflow: WorkflowConfig) -> Path:
    if args.output_dir is not None:
        return _resolve_repo_path(args.output_dir)
    output_root = _resolve_repo_path(workflow.output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"forward_observability_{timestamp}"


def _resolve_sources(script_cfg: Mapping[str, Any]) -> dict[str, Path]:
    source_cfg = script_cfg["source_runs"]
    output_root = _resolve_repo_path(script_cfg["output_root"])
    sources = {
        "wavelet_generation_dir": resolve_source_run(
            source_cfg.get("wavelet_generation_dir"),
            output_root=output_root,
            prefix="wavelet_generation",
            required_files=[
                "selected_wavelet.csv",
                "selected_wavelet_summary.json",
                "wavelet_candidate_aggregate.csv",
                "evaluation_well_spatial_clusters.csv",
                "batch_synthetic_metrics.csv",
            ],
            root=REPO_ROOT,
            label="wavelet_generation",
        ),
        "well_auto_tie_dir": resolve_source_run(
            source_cfg.get("well_auto_tie_dir"),
            output_root=output_root,
            prefix="well_auto_tie",
            required_files=["well_tie_metrics.csv", "well_tie_plan.csv", "wavelet_inventory.csv"],
            root=REPO_ROOT,
            label="well_auto_tie",
        ),
        "well_preprocess_dir": resolve_source_run(
            source_cfg.get("well_preprocess_dir"),
            output_root=output_root,
            prefix="well_preprocess",
            required_files=["well_preprocess_status.csv"],
            root=REPO_ROOT,
            label="well_preprocess",
        ),
    }
    require_source_files(
        sources["wavelet_generation_dir"],
        [
            "selected_wavelet.csv",
            "selected_wavelet_summary.json",
            "wavelet_candidate_aggregate.csv",
            "evaluation_well_spatial_clusters.csv",
            "batch_synthetic_metrics.csv",
        ],
        label="wavelet_generation",
    )
    require_source_files(
        sources["well_auto_tie_dir"],
        ["well_tie_metrics.csv", "well_tie_plan.csv", "wavelet_inventory.csv"],
        label="well_auto_tie",
    )
    require_source_files(
        sources["well_preprocess_dir"],
        ["well_preprocess_status.csv"],
        label="well_preprocess",
    )
    with (sources["wavelet_generation_dir"] / "selected_wavelet_summary.json").open(
        "r", encoding="utf-8"
    ) as handle:
        summary = json.load(handle)
    source_auto_tie = summary.get("source_auto_tie_dir")
    if not source_auto_tie:
        raise ValueError("selected_wavelet_summary.json is missing source_auto_tie_dir.")
    assert_recorded_source_matches(
        {"source_auto_tie_dir": source_auto_tie},
        "source_auto_tie_dir",
        sources["well_auto_tie_dir"],
        root=REPO_ROOT,
        message=(
            "source_run_mismatch: selected wavelet source_auto_tie_dir does not match "
            "forward_observability.source_runs.well_auto_tie_dir."
        ),
    )
    return sources


def _normalize_wavelet(
    *,
    name: str,
    kind: str,
    path: Path,
    source_well: str = "",
    expected_time_s: np.ndarray | None = None,
) -> WaveletScenario:
    time_s, amplitude = load_wavelet_csv(path)
    amplitude, qc = validate_wavelet_normalization(
        time_s,
        amplitude,
        expected_l2_energy=1.0,
        l2_energy_tolerance=1e-5,
        max_center_abs_time_s=1e-9,
        allow_small_renormalization=True,
    )
    if qc.status != "ok":
        raise ValueError(f"invalid_wavelet:{name}:{qc.reasons}")
    if time_s.size % 2 == 0:
        raise ValueError(f"invalid_wavelet:{name}:even_sample_count")
    if expected_time_s is not None and not np.allclose(time_s, expected_time_s, rtol=0.0, atol=1e-9):
        raise ValueError(f"sampling_mismatch:{name}:time_axis")
    return WaveletScenario(
        name=name,
        kind=kind,
        time_s=time_s,
        amplitude=amplitude,
        source_well=source_well,
    )


def _load_wavelet_scenarios(
    sources: Mapping[str, Path],
    script_cfg: Mapping[str, Any],
) -> tuple[list[WaveletScenario], int, list[dict[str, Any]]]:
    wavelet_dir = sources["wavelet_generation_dir"]
    auto_tie_dir = sources["well_auto_tie_dir"]
    nominal = _normalize_wavelet(
        name="nominal_selected",
        kind="nominal",
        path=wavelet_dir / "selected_wavelet.csv",
    )

    aggregate = pd.read_csv(wavelet_dir / "wavelet_candidate_aggregate.csv")
    inventory = pd.read_csv(auto_tie_dir / "wavelet_inventory.csv")
    required_aggregate = {"candidate_wavelet", "source_well"}
    required_inventory = {"source_well", "wavelet_file", "usable_as_candidate"}
    if missing := required_aggregate - set(aggregate.columns):
        raise ValueError(f"wavelet_candidate_aggregate.csv is missing columns: {sorted(missing)}")
    if missing := required_inventory - set(inventory.columns):
        raise ValueError(f"wavelet_inventory.csv is missing columns: {sorted(missing)}")

    aggregate = aggregate.copy()
    inventory = inventory.copy()
    aggregate["_well_key"] = aggregate["source_well"].map(normalize_well_name)
    inventory["_well_key"] = inventory["source_well"].map(normalize_well_name)
    if aggregate["_well_key"].duplicated().any() or inventory["_well_key"].duplicated().any():
        raise ValueError("candidate_join_failed: source_well is not unique in candidate tables.")
    usable = inventory["usable_as_candidate"].astype(str).str.casefold().isin({"true", "1", "yes"})
    joined = aggregate.merge(
        inventory.loc[usable, ["_well_key", "wavelet_file"]],
        on="_well_key",
        how="left",
        validate="one_to_one",
    )
    join_qc: list[dict[str, Any]] = []
    scenarios: list[WaveletScenario] = [nominal]
    for row in joined.to_dict(orient="records"):
        name = str(row["candidate_wavelet"])
        source_well = str(row["source_well"])
        wavelet_file = row.get("wavelet_file")
        if wavelet_file is None or not str(wavelet_file).strip() or str(wavelet_file) == "nan":
            join_qc.append(
                {
                    "candidate_wavelet": name,
                    "source_well": source_well,
                    "status": "candidate_join_failed",
                    "reasons": "source_well_missing_from_usable_wavelet_inventory",
                }
            )
            continue
        try:
            path = resolve_artifact_path(
                wavelet_file,
                root=REPO_ROOT,
                run_dir=auto_tie_dir,
            )
            if path is None or not path.is_file():
                raise FileNotFoundError(str(path))
            scenario = _normalize_wavelet(
                name=name,
                kind="candidate",
                path=path,
                source_well=source_well,
                expected_time_s=nominal.time_s,
            )
            scenarios.append(scenario)
            join_qc.append(
                {
                    "candidate_wavelet": name,
                    "source_well": source_well,
                    "wavelet_file": repo_relative_path(path, root=REPO_ROOT),
                    "status": "ok",
                    "reasons": "",
                }
            )
        except Exception as exc:
            join_qc.append(
                {
                    "candidate_wavelet": name,
                    "source_well": source_well,
                    "wavelet_file": str(wavelet_file),
                    "status": "invalid_wavelet",
                    "reasons": f"{type(exc).__name__}:{exc}",
                }
            )
    admitted_candidate_count = sum(scenario.kind == "candidate" for scenario in scenarios)
    if admitted_candidate_count == 0:
        raise ValueError("candidate_join_failed: no fifth-step admitted candidate wavelet could be loaded.")

    perturbation = script_cfg["perturbation"]
    scenarios.extend(
        make_artificial_wavelet_scenarios(
            nominal,
            phase_degrees=float(perturbation["phase_degrees"]),
            fractional_shift_samples=float(perturbation["fractional_shift_samples"]),
        )
    )
    return scenarios, admitted_candidate_count, join_qc


def _resolve_artifact(value: Any, *, run_dir: Path, label: str) -> Path:
    path = resolve_artifact_path(value, root=REPO_ROOT, run_dir=run_dir)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _project_md_curve_to_twt(
    md_m: np.ndarray,
    values: np.ndarray,
    *,
    table: Any,
    target_twt_s: np.ndarray,
) -> np.ndarray:
    """Project finite MD runs independently so long gaps are never bridged."""
    md = np.asarray(md_m, dtype=np.float64).reshape(-1)
    curve = np.asarray(values, dtype=np.float64).reshape(-1)
    target = np.asarray(target_twt_s, dtype=np.float64).reshape(-1)
    if md.shape != curve.shape:
        raise ValueError("MD and curve arrays must have matching shapes.")
    if not table.is_md_domain:
        raise ValueError("Optimized TDT must be in MD domain.")
    output = np.full(target.shape, np.nan, dtype=np.float64)
    in_tdt = (
        np.isfinite(md)
        & (md >= float(table.md[0]))
        & (md <= float(table.md[-1]))
        & np.isfinite(curve)
        & (curve > 0.0)
    )
    for start, end in true_runs(in_tdt):
        if end - start < 2:
            continue
        run_md = md[start:end]
        run_values = curve[start:end]
        run_twt = np.interp(run_md, table.md, table.twt)
        if np.any(np.diff(run_twt) <= 0.0):
            continue
        mask = (target >= float(run_twt[0])) & (target <= float(run_twt[-1]))
        output[mask] = np.interp(target[mask], run_twt, run_values)
    return output


def _load_ai_on_seismic_axis(
    las_path: Path,
    *,
    table: Any,
    seismic_twt_s: np.ndarray,
    max_short_gap_s: float,
) -> np.ndarray:
    ai = read_las_curve(
        las_path,
        "AI",
        match_policy="exact",
        allow_all_nan=True,
    )
    projected = _project_md_curve_to_twt(
        ai.basis,
        ai.values,
        table=table,
        target_twt_s=seismic_twt_s,
    )
    dt_s = regular_dt(seismic_twt_s, label=f"{las_path.name} target TWT")
    finite = np.isfinite(projected) & (projected > 0.0)
    max_short_samples = int(np.floor(float(max_short_gap_s) / dt_s + 1e-9))
    for start, end in true_runs(~finite):
        gap_size = end - start
        if (
            gap_size <= max_short_samples
            and start > 0
            and end < projected.size
            and finite[start - 1]
            and finite[end]
        ):
            projected[start:end] = np.interp(
                seismic_twt_s[start:end],
                [seismic_twt_s[start - 1], seismic_twt_s[end]],
                [projected[start - 1], projected[end]],
            )
            finite[start:end] = True
    valid = np.isfinite(projected) & (projected > 0.0)
    result = np.full(projected.shape, np.nan, dtype=np.float64)
    result[valid] = np.log(projected[valid])
    return result


def _build_windows(
    *,
    well_name: str,
    horizons: Sequence[Mapping[str, str]],
    well_tops: pd.DataFrame,
    table: Any,
) -> list[ObservabilityWindow]:
    horizon_times: list[tuple[str, float]] = []
    for horizon in horizons:
        horizon_name = str(horizon["name"])
        well_top = str(horizon["well_top"])
        md_m = find_well_top_md(well_tops, well_name=well_name, surface=well_top)
        if md_m < float(table.md[0]) or md_m > float(table.md[-1]):
            raise ValueError(
                f"outside_tdt_support:{horizon_name}:well_top={well_top}:md={md_m:g}"
            )
        horizon_times.append((horizon_name, float(np.interp(md_m, table.md, table.twt))))
    times = np.asarray([value for _, value in horizon_times], dtype=np.float64)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("misordered_horizons")
    first_name, first_time = horizon_times[0]
    last_name, last_time = horizon_times[-1]
    windows = [
        ObservabilityWindow(
            window_id=f"whole__{sanitize_filename(first_name)}__to__{sanitize_filename(last_name)}",
            window_type="whole_target",
            top_horizon=first_name,
            bottom_horizon=last_name,
            start_s=first_time,
            end_s=last_time,
        )
    ]
    for (top_name, top_time), (bottom_name, bottom_time) in zip(
        horizon_times[:-1], horizon_times[1:]
    ):
        windows.append(
            ObservabilityWindow(
                window_id=f"zone__{sanitize_filename(top_name)}__to__{sanitize_filename(bottom_name)}",
                window_type="adjacent_zone",
                top_horizon=top_name,
                bottom_horizon=bottom_name,
                start_s=top_time,
                end_s=bottom_time,
            )
        )
    return windows


def _continuous_analysis_segment(
    *,
    time_s: np.ndarray,
    filtered_log_ai: np.ndarray,
    preprocessed_log_ai: np.ndarray,
    window: ObservabilityWindow,
) -> tuple[slice, np.ndarray]:
    valid = np.isfinite(filtered_log_ai) & np.isfinite(preprocessed_log_ai)
    candidate_indices = np.flatnonzero(
        (time_s >= float(window.start_s))
        & (time_s <= float(window.end_s))
        & (np.arange(time_s.size) >= 1)
    )
    if candidate_indices.size == 0:
        raise ValueError("outside_seismic_support")
    required_start = int(candidate_indices[0] - 1)
    required_end = int(candidate_indices[-1] + 1)
    if required_start < 0 or not np.all(valid[required_start:required_end]):
        raise ValueError("long_gap_inside_window")
    containing_runs = [
        (start, end)
        for start, end in true_runs(valid)
        if start <= required_start and end >= required_end
    ]
    if len(containing_runs) != 1:
        raise ValueError("long_gap_inside_window")
    context_start, context_end = containing_runs[0]
    if context_start < 0 or context_end - context_start < 2:
        raise ValueError("insufficient_valid_samples")
    local_indices = candidate_indices - context_start
    return slice(context_start, context_end), local_indices.astype(np.int64)


def _failure_row(
    *,
    base: Mapping[str, Any],
    scenario: WaveletScenario,
    frequency_hz: float,
    status: str,
    reasons: str,
    n_valid_samples: int = 0,
    n_cycles: float = float("nan"),
) -> dict[str, Any]:
    return {
        **base,
        "wavelet_scenario": scenario.name,
        "wavelet_scenario_kind": scenario.kind,
        "wavelet_source_well": scenario.source_well,
        "frequency_hz": float(frequency_hz),
        "n_valid_samples": int(n_valid_samples),
        "n_cycles": float(n_cycles),
        "status": status,
        "reasons": reasons,
    }


def _analyze_well(
    *,
    well_name: str,
    route: str,
    spatial_cluster_id: int,
    filtered_las: Path,
    preprocessed_las: Path,
    optimized_tdt: Path,
    seismic_trace: Path,
    horizons: Sequence[Mapping[str, str]],
    well_tops: pd.DataFrame,
    scenarios: Sequence[WaveletScenario],
    frequencies_hz: np.ndarray,
    script_cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    table = load_workflow_time_depth_table_csv(optimized_tdt)
    if not table.is_md_domain:
        raise ValueError("optimized TDT is not MD-domain")
    seismic = load_saved_seismic_trace_csv(seismic_trace)
    time_s = np.asarray(seismic.basis, dtype=np.float64)
    dt_s = regular_dt(time_s, label=f"{well_name} seismic time")
    for scenario in scenarios:
        scenario_dt = regular_dt(scenario.time_s, label=f"{scenario.name} wavelet time")
        if not np.isclose(dt_s, scenario_dt, rtol=1e-5, atol=1e-9):
            raise ValueError(f"sampling_mismatch:{scenario.name}:wavelet_dt={scenario_dt}:seismic_dt={dt_s}")
    filtered_log_ai = _load_ai_on_seismic_axis(
        filtered_las,
        table=table,
        seismic_twt_s=time_s,
        max_short_gap_s=float(script_cfg["thresholds"]["max_short_log_gap_s"]),
    )
    preprocessed_log_ai = _load_ai_on_seismic_axis(
        preprocessed_las,
        table=table,
        seismic_twt_s=time_s,
        max_short_gap_s=float(script_cfg["thresholds"]["max_short_log_gap_s"]),
    )
    windows = _build_windows(
        well_name=well_name,
        horizons=horizons,
        well_tops=well_tops,
        table=table,
    )
    rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    plot_payload: dict[str, Any] | None = None
    thresholds = script_cfg["thresholds"]
    perturbation = script_cfg["perturbation"]
    observed_full = np.asarray(seismic.values, dtype=np.float64)
    if observed_full.size != time_s.size:
        raise ValueError("seismic values and time basis do not match")

    for window in windows:
        base = {
            "well_name": well_name,
            "route": route,
            "spatial_cluster_id": int(spatial_cluster_id),
            "window_id": window.window_id,
            "window_type": window.window_type,
            "top_horizon": window.top_horizon,
            "bottom_horizon": window.bottom_horizon,
            "window_start_s": float(window.start_s),
            "window_end_s": float(window.end_s),
        }
        try:
            context_slice, local_indices = _continuous_analysis_segment(
                time_s=time_s,
                filtered_log_ai=filtered_log_ai,
                preprocessed_log_ai=preprocessed_log_ai,
                window=window,
            )
            local_time = time_s[context_slice]
            local_filtered = filtered_log_ai[context_slice]
            local_preprocessed = preprocessed_log_ai[context_slice]
            local_observed = observed_full[context_slice][1:]
            n_samples = int(local_indices.size)
            window_rows.append(
                {
                    **base,
                    "analysis_start_s": float(local_time[local_indices[0]]),
                    "analysis_end_s": float(local_time[local_indices[-1]]),
                    "n_valid_samples": n_samples,
                    "status": "ok",
                    "reasons": "",
                }
            )
            if plot_payload is None and window.window_type == "whole_target":
                plot_payload = {
                    "well_name": well_name,
                    "time_s": local_time,
                    "filtered_log_ai": local_filtered,
                    "preprocessed_log_ai": local_preprocessed,
                    "observed": local_observed,
                    "output_indices": local_indices,
                    "window": window,
                }
            for frequency_hz in frequencies_hz:
                duration_s = float(n_samples * dt_s)
                n_cycles = duration_s * float(frequency_hz)
                required_samples = max(
                    int(thresholds["min_valid_samples"]),
                    int(np.ceil(float(thresholds["min_cycles"]) / (float(frequency_hz) * dt_s))),
                )
                if n_samples < int(thresholds["min_valid_samples"]):
                    for scenario in scenarios:
                        rows.append(
                            _failure_row(
                                base=base,
                                scenario=scenario,
                                frequency_hz=float(frequency_hz),
                                status="insufficient_valid_samples",
                                reasons=f"n={n_samples}<minimum={thresholds['min_valid_samples']}",
                                n_valid_samples=n_samples,
                                n_cycles=n_cycles,
                            )
                        )
                    continue
                if n_samples < required_samples:
                    for scenario in scenarios:
                        rows.append(
                            _failure_row(
                                base=base,
                                scenario=scenario,
                                frequency_hz=float(frequency_hz),
                                status="insufficient_cycles",
                                reasons=(
                                    f"n={n_samples}<required={required_samples};"
                                    f"cycles={n_cycles:.6g}<minimum={thresholds['min_cycles']}"
                                ),
                                n_valid_samples=n_samples,
                                n_cycles=n_cycles,
                            )
                        )
                    continue
                for scenario in scenarios:
                    try:
                        metrics = analyze_frequency_scenario(
                            time_s=local_time,
                            filtered_log_ai=local_filtered,
                            preprocessed_log_ai=local_preprocessed,
                            observed=local_observed,
                            output_indices=local_indices,
                            frequency_hz=float(frequency_hz),
                            scenario=scenario,
                            epsilon=float(perturbation["epsilon_log_ai"]),
                            tukey_alpha=float(perturbation["tukey_alpha"]),
                            max_basis_condition_number=float(
                                perturbation["max_basis_condition_number"]
                            ),
                            min_synthetic_rms=float(thresholds["min_synthetic_rms"]),
                        )
                        rows.append(
                            {
                                **base,
                                "wavelet_scenario": scenario.name,
                                "wavelet_scenario_kind": scenario.kind,
                                "wavelet_source_well": scenario.source_well,
                                "frequency_hz": float(frequency_hz),
                                "tukey_alpha": float(perturbation["tukey_alpha"]),
                                **metrics,
                                "status": "ok",
                                "reasons": "",
                            }
                        )
                    except Exception as exc:
                        message = str(exc)
                        status = (
                            message
                            if message
                            in {
                                "ill_conditioned_phase_basis",
                                "invalid_observed_energy",
                                "invalid_low_synthetic_energy",
                                "invalid_nonpositive_scale",
                                "invalid_sensitivity",
                            }
                            else "invalid_sensitivity"
                        )
                        rows.append(
                            _failure_row(
                                base=base,
                                scenario=scenario,
                                frequency_hz=float(frequency_hz),
                                status=status,
                                reasons=f"{type(exc).__name__}:{exc}",
                                n_valid_samples=n_samples,
                                n_cycles=n_cycles,
                            )
                        )
        except Exception as exc:
            message = str(exc)
            status = (
                message
                if message
                in {
                    "outside_seismic_support",
                    "insufficient_valid_samples",
                    "long_gap_inside_window",
                    "misordered_horizons",
                }
                else "insufficient_valid_samples"
            )
            window_rows.append({**base, "status": status, "reasons": f"{type(exc).__name__}:{exc}"})
            for frequency_hz in frequencies_hz:
                for scenario in scenarios:
                    rows.append(
                        _failure_row(
                            base=base,
                            scenario=scenario,
                            frequency_hz=float(frequency_hz),
                            status=status,
                            reasons=f"{type(exc).__name__}:{exc}",
                        )
                    )
    return rows, window_rows, plot_payload


def _operator_lookup(operator_rows: pd.DataFrame) -> dict[tuple[str, float], tuple[float, str]]:
    lookup: dict[tuple[str, float], tuple[float, str]] = {}
    for row in operator_rows.to_dict(orient="records"):
        lookup[(str(row["wavelet_scenario"]), float(row["frequency_hz"]))] = (
            float(row["combined_magnitude_normalized"]),
            str(row["operator_support_class"]),
        )
    return lookup


def _add_operator_fields(rows: pd.DataFrame, operator_rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    lookup = _operator_lookup(operator_rows)
    magnitudes = []
    classes = []
    for scenario, frequency in zip(output["wavelet_scenario"], output["frequency_hz"]):
        magnitude, support = lookup.get((str(scenario), float(frequency)), (float("nan"), "unsupported"))
        magnitudes.append(magnitude)
        classes.append(support)
    output["operator_magnitude_normalized"] = magnitudes
    output["operator_support_class"] = classes
    return output


def _zone_warnings(evidence: pd.DataFrame) -> pd.DataFrame:
    output = evidence.copy()
    output["zone_warnings"] = ""
    whole = output[output["window_type"].eq("whole_target")]
    zones = output[output["window_type"].eq("adjacent_zone")]
    for index, row in whole.iterrows():
        frequency = float(row["frequency_hz"])
        matching = zones[np.isclose(zones["frequency_hz"].to_numpy(dtype=float), frequency)]
        warnings = [
            f"{zone.window_id}:{zone.evidence_status}"
            for zone in matching.itertuples()
            if zone.evidence_status != "robust_detectable"
        ]
        output.at[index, "zone_warnings"] = ";".join(warnings)
    return output


def _plot_operator(operator_rows: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for kind, group in operator_rows.groupby("wavelet_scenario_kind"):
        for _, scenario in group.groupby("wavelet_scenario"):
            ax.plot(
                scenario["frequency_hz"],
                scenario["combined_magnitude_normalized"],
                color={
                    "nominal": "black",
                    "candidate": "tab:blue",
                    "artificial_phase": "tab:orange",
                    "artificial_shift": "tab:green",
                }.get(str(kind), "0.5"),
                alpha=1.0 if kind == "nominal" else 0.25,
                linewidth=2.0 if kind == "nominal" else 1.0,
                label="nominal" if kind == "nominal" else None,
            )
    ax.axhline(0.5, color="0.4", linestyle="--", linewidth=1.0, label="core threshold")
    ax.axhline(0.1, color="0.6", linestyle=":", linewidth=1.0, label="weak threshold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized |W(f)D(f)|")
    ax.set_title("Analytical forward-operator support")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_evidence(evidence: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for window_id, group in evidence.groupby("window_id"):
        ordered = group.sort_values("frequency_hz")
        is_whole = bool(ordered["window_type"].eq("whole_target").iloc[0])
        ax.plot(
            ordered["frequency_hz"],
            ordered["cluster_p25_detectability_ratio"],
            marker="o" if is_whole else None,
            linewidth=2.0 if is_whole else 1.0,
            alpha=1.0 if is_whole else 0.45,
            label=str(window_id),
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Cluster P25 detectability ratio")
    ax.set_title("Empirical frequency evidence")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_well_qc(
    payload: Mapping[str, Any],
    well_aggregate: pd.DataFrame,
    path: Path,
) -> None:
    well_name = str(payload["well_name"])
    local_time = np.asarray(payload["time_s"], dtype=np.float64)
    indices = np.asarray(payload["output_indices"], dtype=np.int64)
    filtered = np.asarray(payload["filtered_log_ai"], dtype=np.float64)
    preprocessed = np.asarray(payload["preprocessed_log_ai"], dtype=np.float64)
    observed = np.asarray(payload["observed"], dtype=np.float64)
    well_rows = well_aggregate[
        well_aggregate["well_name"].eq(well_name)
        & well_aggregate["window_type"].eq("whole_target")
    ].sort_values("frequency_hz")

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=False, constrained_layout=True)
    axes[0].plot(preprocessed[indices], local_time[indices], label="step 3", color="0.45")
    axes[0].plot(filtered[indices], local_time[indices], label="step 4", color="tab:blue")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("log(AI)")
    axes[0].set_ylabel("TWT (s)")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(observed[indices - 1], local_time[indices], color="black")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Observed seismic")
    axes[1].grid(alpha=0.2)

    axes[2].plot(well_rows["frequency_hz"], well_rows["detectability_ratio"], marker="o")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Scenario P25 detectability")
    axes[2].grid(alpha=0.2)
    fig.suptitle(f"Forward observability | {well_name}")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_empty_or_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(list(rows))
    frame.to_csv(path, index=False)
    return frame


def main() -> None:
    args = parse_args()
    config_path = _resolve_repo_path(args.config)
    config = load_yaml_config(config_path)
    workflow = WorkflowConfig.from_mapping(config)
    script_cfg = _script_config(config)
    script_cfg["output_root"] = workflow.output_root
    sources = _resolve_sources(script_cfg)
    output_dir = _resolve_output_dir(args, workflow)
    figures_dir = output_dir / "figures"
    well_figures_dir = figures_dir / "wells"
    output_dir.mkdir(parents=True, exist_ok=False)
    figures_dir.mkdir(parents=True, exist_ok=True)
    well_figures_dir.mkdir(parents=True, exist_ok=True)

    scenarios, admitted_candidate_count, candidate_join_qc = _load_wavelet_scenarios(
        sources,
        script_cfg,
    )
    nominal = next(scenario for scenario in scenarios if scenario.kind == "nominal")
    dt_s = regular_dt(nominal.time_s, label="nominal wavelet time")
    frequencies_hz = frequency_grid(
        dt_s=dt_s,
        start_hz=float(script_cfg["frequency"]["start_hz"]),
        step_hz=float(script_cfg["frequency"]["step_hz"]),
        configured_max_hz=float(script_cfg["frequency"]["max_hz"]),
    )
    operator_rows = operator_transfer_rows(scenarios, frequencies_hz)
    operator_rows.insert(0, "schema_version", SCHEMA_VERSION)
    operator_rows.to_csv(output_dir / "operator_transfer.csv", index=False)
    pd.DataFrame.from_records(candidate_join_qc).to_csv(
        output_dir / "wavelet_scenario_qc.csv",
        index=False,
    )

    wavelet_dir = sources["wavelet_generation_dir"]
    auto_tie_dir = sources["well_auto_tie_dir"]
    preprocess_dir = sources["well_preprocess_dir"]
    clusters = pd.read_csv(wavelet_dir / "evaluation_well_spatial_clusters.csv")
    batch_metrics = pd.read_csv(wavelet_dir / "batch_synthetic_metrics.csv")
    metrics = pd.read_csv(auto_tie_dir / "well_tie_metrics.csv")
    plans = pd.read_csv(auto_tie_dir / "well_tie_plan.csv")
    preprocess = pd.read_csv(preprocess_dir / "well_preprocess_status.csv")
    for frame, columns, label in [
        (clusters, {"well_name", "spatial_cluster_id"}, "evaluation_well_spatial_clusters.csv"),
        (
            batch_metrics,
            {"eval_well", "status", "corr", "nmae", "scale"},
            "batch_synthetic_metrics.csv",
        ),
        (
            metrics,
            {
                "well_name",
                "route",
                "tie_status",
                "filtered_las_file",
                "optimized_tdt_file",
                "seismic_trace_file",
            },
            "well_tie_metrics.csv",
        ),
        (plans, {"well_name", "input_las"}, "well_tie_plan.csv"),
        (
            preprocess,
            {"well_name", "preprocess_status", "preprocessed_las"},
            "well_preprocess_status.csv",
        ),
    ]:
        if missing := columns - set(frame.columns):
            raise ValueError(f"{label} is missing columns: {sorted(missing)}")
        name_column = "eval_well" if label == "batch_synthetic_metrics.csv" else "well_name"
        frame["_well_key"] = frame[name_column].map(normalize_well_name)
        if frame["_well_key"].duplicated().any():
            raise ValueError(f"{label} contains duplicate normalized well names.")

    cluster_keys = set(clusters["_well_key"])
    batch_keys = set(batch_metrics["_well_key"])
    if cluster_keys != batch_keys:
        raise ValueError(
            "source_run_mismatch: fifth-step evaluation clusters and batch metrics contain "
            "different well sets."
        )
    if not batch_metrics["status"].astype(str).str.casefold().eq("ok").all():
        raise ValueError("source_run_mismatch: fifth-step batch metrics contain failed evaluation wells.")

    wells = clusters.merge(
        metrics,
        on=["well_name", "_well_key"],
        how="left",
        validate="one_to_one",
    ).merge(
        plans[["_well_key", "input_las"]],
        on="_well_key",
        how="left",
        validate="one_to_one",
    ).merge(
        preprocess[["_well_key", "preprocess_status", "preprocessed_las"]],
        on="_well_key",
        how="left",
        validate="one_to_one",
    ).merge(
        batch_metrics[["_well_key", "corr", "nmae", "scale"]].rename(
            columns={
                "corr": "fifth_batch_corr",
                "nmae": "fifth_batch_nmae",
                "scale": "fifth_batch_scale",
            }
        ),
        on="_well_key",
        how="left",
        validate="one_to_one",
    )
    if args.well is not None:
        requested_key = normalize_well_name(args.well)
        wells = wells[wells["_well_key"].eq(requested_key)]
        if wells.empty:
            raise ValueError(f"Requested well is not in fifth-step evaluation set: {args.well}")

    data_root = _resolve_repo_path(workflow.data_root)
    well_tops_path = resolve_relative_path(workflow.assets.well_tops_file, root=data_root)
    well_tops = import_well_tops_petrel(well_tops_path)
    sensitivity_records: list[dict[str, Any]] = []
    window_records: list[dict[str, Any]] = []
    well_status_records: list[dict[str, Any]] = []
    plot_payloads: list[dict[str, Any]] = []

    for row in wells.to_dict(orient="records"):
        well_name = str(row["well_name"])
        try:
            if str(row.get("tie_status", "")).casefold() != "success":
                raise ValueError("tie_status_not_success")
            if str(row.get("preprocess_status", "")).casefold() != "passed":
                raise ValueError("preprocess_status_not_passed")
            filtered_las = _resolve_artifact(
                row.get("filtered_las_file"),
                run_dir=auto_tie_dir,
                label=f"{well_name} filtered LAS",
            )
            optimized_tdt = _resolve_artifact(
                row.get("optimized_tdt_file"),
                run_dir=auto_tie_dir,
                label=f"{well_name} optimized TDT",
            )
            seismic_trace = _resolve_artifact(
                row.get("seismic_trace_file"),
                run_dir=auto_tie_dir,
                label=f"{well_name} seismic trace",
            )
            preprocessed_las = _resolve_artifact(
                row.get("preprocessed_las"),
                run_dir=preprocess_dir,
                label=f"{well_name} preprocessed LAS",
            )
            planned_las = _resolve_artifact(
                row.get("input_las"),
                run_dir=auto_tie_dir,
                label=f"{well_name} planned input LAS",
            )
            if not _same_path(preprocessed_las, planned_las):
                raise ValueError("source_run_mismatch: fourth-step input_las differs from explicit preprocess run")

            well_rows, window_rows, plot_payload = _analyze_well(
                well_name=well_name,
                route=str(row["route"]),
                spatial_cluster_id=int(row["spatial_cluster_id"]),
                filtered_las=filtered_las,
                preprocessed_las=preprocessed_las,
                optimized_tdt=optimized_tdt,
                seismic_trace=seismic_trace,
                horizons=script_cfg["horizons"],
                well_tops=well_tops,
                scenarios=scenarios,
                frequencies_hz=frequencies_hz,
                script_cfg=script_cfg,
            )
            sensitivity_records.extend(well_rows)
            window_records.extend(window_rows)
            if plot_payload is not None:
                plot_payloads.append(plot_payload)
            well_status_records.append(
                {
                    "well_name": well_name,
                    "route": str(row["route"]),
                    "spatial_cluster_id": int(row["spatial_cluster_id"]),
                    "fifth_batch_corr": row.get("fifth_batch_corr"),
                    "fifth_batch_nmae": row.get("fifth_batch_nmae"),
                    "fifth_batch_scale": row.get("fifth_batch_scale"),
                    "status": "ok",
                    "reasons": "",
                }
            )
        except Exception as exc:
            message = str(exc)
            status = "missing_input"
            if message.startswith("source_run_mismatch"):
                status = "source_run_mismatch"
            elif message.startswith("No finite MD found"):
                status = "missing_horizon"
            elif message.startswith("outside_tdt_support"):
                status = "outside_tdt_support"
            elif message == "misordered_horizons":
                status = "misordered_horizons"
            well_status_records.append(
                {
                    "well_name": well_name,
                    "route": str(row.get("route", "")),
                    "spatial_cluster_id": row.get("spatial_cluster_id"),
                    "fifth_batch_corr": row.get("fifth_batch_corr"),
                    "fifth_batch_nmae": row.get("fifth_batch_nmae"),
                    "fifth_batch_scale": row.get("fifth_batch_scale"),
                    "status": status,
                    "reasons": f"{type(exc).__name__}:{exc}",
                }
            )

    well_status = _write_empty_or_rows(output_dir / "well_status.csv", well_status_records)
    _write_empty_or_rows(output_dir / "well_window_status.csv", window_records)
    sensitivity = pd.DataFrame.from_records(sensitivity_records)
    if sensitivity.empty:
        rejection_counts = (
            well_status["status"]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .sort_index()
        )
        rejection_summary = ", ".join(
            f"{status}={int(count)}" for status, count in rejection_counts.items()
        )
        raise ValueError(
            "No well/window observability records were produced. "
            f"Rejections: {rejection_summary or 'none recorded'}. "
            f"Inspect {output_dir / 'well_status.csv'}."
        )
    sensitivity = _add_operator_fields(sensitivity, operator_rows)
    sensitivity.insert(0, "schema_version", SCHEMA_VERSION)
    sensitivity.to_csv(output_dir / "well_frequency_sensitivity.csv", index=False)

    well_aggregate = aggregate_well_scenarios(
        sensitivity,
        admitted_candidate_count=admitted_candidate_count,
        required_artificial_count=int(
            script_cfg["thresholds"]["required_artificial_scenarios"]
        ),
    )
    well_aggregate.insert(0, "schema_version", SCHEMA_VERSION)
    well_aggregate.to_csv(output_dir / "well_frequency_aggregate.csv", index=False)

    cluster_aggregate, evidence = aggregate_frequency_evidence(
        well_aggregate,
        operator_rows,
        min_wells=int(script_cfg["thresholds"]["min_wells"]),
        min_clusters=int(script_cfg["thresholds"]["min_clusters"]),
    )
    if not cluster_aggregate.empty:
        cluster_aggregate.insert(0, "schema_version", SCHEMA_VERSION)
    cluster_aggregate.to_csv(output_dir / "cluster_frequency_aggregate.csv", index=False)
    evidence = _zone_warnings(evidence)
    if not evidence.empty:
        evidence.insert(0, "schema_version", SCHEMA_VERSION)
    evidence.to_csv(output_dir / "frequency_evidence_bands.csv", index=False)

    ranges = contiguous_experiment_ranges(
        evidence,
        frequency_step_hz=float(script_cfg["frequency"]["step_hz"]),
    )
    write_json(
        output_dir / "recommended_experiment_ranges.json",
        {
            "schema_version": SCHEMA_VERSION,
            "semantics": (
                "Experimental ranges for synthoseis-lite only; these are not model cutoffs "
                "or production frequency-band boundaries."
            ),
            "ranges": ranges,
        },
    )

    peak_hz, half_left_hz, half_right_hz = wavelet_half_amplitude_frequencies(
        nominal.time_s,
        nominal.amplitude,
    )
    warnings: list[str] = []
    if float(script_cfg["frequency"]["max_hz"]) > 1.5 * half_right_hz:
        warnings.append("configured_max_beyond_wavelet_support")
    rejection_counts = (
        sensitivity[sensitivity["status"].ne("ok")]
        .groupby(["frequency_hz", "status"])
        .size()
        .rename("count")
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "success",
        "config_file": repo_relative_path(config_path, root=REPO_ROOT),
        "source_runs": {
            key: repo_relative_path(path, root=REPO_ROOT) for key, path in sources.items()
        },
        "well_tops_file": repo_relative_path(well_tops_path, root=REPO_ROOT),
        "ordered_horizons": script_cfg["ordered_horizons"],
        "frequency": script_cfg["frequency"],
        "frequencies_hz": frequencies_hz,
        "perturbation": script_cfg["perturbation"],
        "thresholds": script_cfg["thresholds"],
        "wavelet_scenarios": [
            {
                "name": scenario.name,
                "kind": scenario.kind,
                "source_well": scenario.source_well,
            }
            for scenario in scenarios
        ],
        "admitted_candidate_count": admitted_candidate_count,
        "nominal_wavelet": {
            "dt_s": dt_s,
            "n_samples": int(nominal.amplitude.size),
            "peak_frequency_hz": peak_hz,
            "half_amplitude_left_hz": half_left_hz,
            "half_amplitude_right_hz": half_right_hz,
        },
        "well_status_counts": well_status["status"].value_counts(dropna=False).to_dict(),
        "rejections_by_frequency_and_status": rejection_counts,
        "warnings": warnings,
        "recommended_ranges": ranges,
    }
    input_contracts = {
        key.removesuffix("_dir"): published_contract_reference(
            path / "run_summary.json",
            root=REPO_ROOT,
            label=f"{key} {path}",
        )
        for key, path in sources.items()
    }
    summary["contract_fingerprint_schema"] = CONTRACT_FINGERPRINT_SCHEMA
    summary["contract_fingerprint_sha256"] = contract_fingerprint_sha256(
        contract_schema_version=SCHEMA_VERSION,
        semantics={
            "sample_domain": "time",
            "frequencies_hz": frequencies_hz,
            "ordered_horizons": script_cfg["ordered_horizons"],
        },
        business_config={
            "frequency": script_cfg["frequency"],
            "perturbation": script_cfg["perturbation"],
            "thresholds": script_cfg["thresholds"],
        },
        input_contracts=input_contracts,
        primary_artifacts={
            "frequency_evidence_bands": output_dir / "frequency_evidence_bands.csv",
            "well_frequency_sensitivity": output_dir / "well_frequency_sensitivity.csv",
            "recommended_experiment_ranges": output_dir / "recommended_experiment_ranges.json",
        },
    )
    summary["input_contracts"] = input_contracts
    write_json(output_dir / "run_summary.json", summary)

    _plot_operator(operator_rows, figures_dir / "operator_transfer.png")
    if not evidence.empty:
        _plot_evidence(evidence, figures_dir / "frequency_evidence.png")
    for payload in plot_payloads:
        path = well_figures_dir / f"well_observability_{sanitize_filename(payload['well_name'])}.png"
        _plot_well_qc(payload, well_aggregate, path)

    print("=== Forward Observability Gate ===")
    print(f"Output: {output_dir}")
    print(f"Wells: {(well_status['status'] == 'ok').sum()} ok / {len(well_status)} total")
    print(f"Wavelet scenarios: {len(scenarios)} ({admitted_candidate_count} candidates)")
    print(f"Frequencies: {frequencies_hz[0]:g}-{frequencies_hz[-1]:g} Hz")


if __name__ == "__main__":
    main()
