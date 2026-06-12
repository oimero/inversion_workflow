"""Build shared time-domain well constraints for GINN, LFM, and enhance.

Usage::

    python scripts/well_constraints.py
    python scripts/well_constraints.py --config experiments/common.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfreqz

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.seismic.wavelet import load_wavelet_csv, wavelet_half_amplitude_frequencies
from cup.time_config import TimeWorkflowConfig
from cup.utils.config import deep_merge_dict
from cup.utils.io import (
    latest_run,
    load_yaml_config,
    repo_relative_path,
    resolve_artifact_path,
    resolve_relative_path,
    sanitize_filename,
    write_json,
)
from cup.well.assets import normalize_well_name
from cup.well.constraints import (
    aggregate_trace_arrays,
    build_deviated_point_facts,
    build_point_conflict_report,
    build_vertical_point_facts,
    confidence_from_corr,
    high_frequency_stats,
    layer_shrinkage_stats,
)
from cup.well.frequency_bands import (
    ConditionedWellLog,
    ReferenceConditioningConfig,
    ThreeBandSplitConfig,
    WellFrequencyBands,
    build_conditioned_reference_log,
    build_frequency_bands,
    ginn_cutoff_candidates,
    segmented_lowpass,
)
from cup.well.las import load_standard_vp_rho_logs
from cup.well.td import load_workflow_time_depth_table_csv
from cup.well.tie import load_saved_seismic_trace_csv

# Step 06 depends on downstream bundle schemas only. Keep GINN/enhance training
# logic out of this script; move schemas to a neutral package when they stabilize.
from enhance.supervision import (
    WellHighSupervisionBundle,
    save_well_high_supervision_npz,
    validate_well_high_supervision,
)
from ginn.anchor import build_log_ai_anchor_bundle, save_log_ai_anchor_npz, validate_log_ai_anchor
from wtie.modeling.modeling import ConvModeler
from wtie.processing import grid


DEFAULT_CONFIG: dict[str, Any] = {
    "source_runs": {
        "well_auto_tie_dir": None,
        "wavelet_generation_dir": None,
    },
    "target_interval": {"horizons": ["interpre/H3-1", "interpre/H7-1"], "twt_unit": "auto"},
    "control_wells": {
        "min_batch_corr": 0.35,
        "max_batch_nmae": None,
        "include_wells": None,
        "exclude_wells": [],
    },
    "reference_conditioning": {
        "max_short_gap_s": 0.010,
        "hampel_window_samples": 7,
        "hampel_sigma": 4.0,
    },
    "frequency_bands": {
        "filter_order": 6,
        "buffer_seconds": None,
        "buffer_mode": "reflect",
        "qc_envelope_window_samples": 31,
        "lfm": {
            "cutoff_mode": "wavelet_left_half_amplitude",
            "cutoff_scale": 1.0,
            "manual_cutoff_hz": None,
        },
        "ginn": {
            "mode": "diagnose",
            "manual_cutoff_hz": None,
            "candidate_cutoff_hz": None,
            "candidate_min_right_half_ratio": 0.4,
            "candidate_max_right_half_ratio": 1.3,
            "candidate_step_hz": 5.0,
            "selection_corr_tolerance": 0.02,
            "selection_nmae_tolerance": 0.03,
            "fail_on_candidate_boundary": True,
        },
        "reference": {
            "cutoff_mode": "ginn_octave",
            "ginn_multiplier": 2.0,
            "max_nyquist_fraction": 0.4,
            "manual_cutoff_hz": None,
        },
    },
    "anchor": {"include_deviated": False},
    "high_supervision": {"include_deviated": False},
    "weights": {"mode": "corr", "corr_floor": 0.3, "corr_span": 0.4, "corr_min_weight": 0.6},
    "lfm_controls": {"min_control_samples_per_well": 16},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _resolve_source_dirs(script_cfg: dict[str, Any], output_root: Path) -> dict[str, Path]:
    source_cfg = dict(script_cfg.get("source_runs") or {})
    auto_value = source_cfg.get("well_auto_tie_dir")
    wavelet_value = source_cfg.get("wavelet_generation_dir")
    auto_dir = (
        latest_run(output_root, "well_auto_tie", "well_tie_metrics.csv")
        if auto_value in {None, ""}
        else resolve_relative_path(auto_value, root=REPO_ROOT)
    )
    wavelet_dir = (
        latest_run(output_root, "wavelet_generation", "batch_synthetic_metrics.csv")
        if wavelet_value in {None, ""}
        else resolve_relative_path(wavelet_value, root=REPO_ROOT)
    )
    return {"well_auto_tie_dir": auto_dir, "wavelet_generation_dir": wavelet_dir}


def _validate_wavelet_generation_source(*, auto_dir: Path, wavelet_dir: Path) -> None:
    summary_candidates = [
        wavelet_dir / "run_summary.json",
        wavelet_dir / "selected_wavelet_summary.json",
    ]
    summary_path = next((path for path in summary_candidates if path.exists()), None)
    if summary_path is None:
        raise FileNotFoundError(
            f"Wavelet-generation run has no run_summary.json or selected_wavelet_summary.json: {wavelet_dir}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_value = summary.get("source_auto_tie_dir")
    if not source_value:
        raise ValueError(f"Wavelet-generation summary does not identify source_auto_tie_dir: {summary_path}")
    source_dir = resolve_relative_path(source_value, root=REPO_ROOT)
    if source_dir.resolve() != auto_dir.resolve():
        raise ValueError(
            "wavelet_generation source_auto_tie_dir does not match well_constraints auto-tie input: "
            f"wavelet_generation={source_dir}, well_constraints={auto_dir}. "
            "Set well_constraints.source_runs.well_auto_tie_dir and wavelet_generation_dir to matching runs."
        )


def _resolve_seismic_trace_file(metric: pd.Series, *, run_dir: Path, well_name: str) -> Path | None:
    direct = resolve_artifact_path(metric.get("seismic_trace_file"), root=REPO_ROOT, run_dir=run_dir)
    if direct is not None and direct.exists():
        return direct
    fallback = run_dir / "seismic_trace" / f"seismic_trace_{sanitize_filename(well_name)}.csv"
    return fallback if fallback.exists() else direct


def _as_optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _batch_metric_lookup(batch_df: pd.DataFrame) -> dict[str, pd.Series]:
    well_col = "eval_well"
    if well_col not in batch_df.columns:
        raise ValueError("batch_synthetic_metrics.csv must contain eval_well.")
    return {
        normalize_well_name(str(row[well_col])): row
        for _, row in batch_df.iterrows()
        if str(row[well_col]).strip() and str(row[well_col]).casefold() != "nan"
    }


def _plan_lookup(plan_df: pd.DataFrame) -> dict[str, pd.Series]:
    if plan_df.empty or "well_name" not in plan_df.columns:
        return {}
    return {normalize_well_name(str(row["well_name"])): row for _, row in plan_df.iterrows()}


def _metric_value(row: pd.Series | None, keys: tuple[str, ...]) -> float | None:
    if row is None:
        return None
    for key in keys:
        if key in row:
            value = _as_optional_float(row.get(key))
            if value is not None:
                return value
    return None


def _resolve_segy_options(seismic_cfg: dict[str, Any]) -> dict[str, int] | None:
    from cup.seismic.survey import segy_options_from_config

    return segy_options_from_config(seismic_cfg) or None


def _open_survey(seismic_cfg: dict[str, Any], data_root: Path) -> tuple[Any, Path, str]:
    from cup.seismic.survey import open_survey

    seismic_file = resolve_relative_path(seismic_cfg.get("file"), root=data_root)
    seismic_type = str(seismic_cfg.get("type", "segy"))
    survey = open_survey(seismic_file, seismic_type=seismic_type, segy_options=_resolve_segy_options(seismic_cfg))
    return survey, seismic_file, seismic_type


def _normalize_horizon_twt_df(df: pd.DataFrame, *, unit: str) -> pd.DataFrame:
    unit_norm = str(unit or "auto").strip().casefold()
    out = df.copy()
    if "interpretation" not in out.columns:
        return out
    values = out["interpretation"].to_numpy(dtype=float, copy=True)
    finite = np.isfinite(values)
    if unit_norm in {"ms", "msec", "millisecond", "milliseconds"}:
        values[finite] = np.abs(values[finite]) / 1000.0
    elif unit_norm in {"s", "sec", "second", "seconds"}:
        values[finite] = np.abs(values[finite])
    elif unit_norm == "auto":
        if np.any(finite):
            abs_values = np.abs(values[finite])
            values[finite] = abs_values / 1000.0 if float(np.nanmax(abs_values)) > 20.0 else abs_values
    else:
        raise ValueError(f"Unsupported target_interval.twt_unit: {unit}")
    out["interpretation"] = values
    return out


def _build_target_layer(script_cfg: dict[str, Any], geometry: dict[str, Any], qc_dir: Path, data_root: Path) -> tuple[Any, list[dict[str, Any]]]:
    from cup.petrel.load import import_interpretation_petrel
    from cup.seismic.target_zone import TargetZone

    target_cfg = dict(script_cfg["target_interval"])
    horizon_values = target_cfg.get("horizons")
    if not isinstance(horizon_values, list) or len(horizon_values) < 2:
        raise ValueError("well_constraints.target_interval.horizons must contain at least two horizon files.")
    twt_unit = str(target_cfg.get("twt_unit", "auto"))
    raw_entries: list[tuple[float, str, Path, pd.DataFrame]] = []
    for index, value in enumerate(horizon_values):
        horizon_file = resolve_relative_path(value, root=data_root)
        horizon_df = _normalize_horizon_twt_df(import_interpretation_petrel(horizon_file), unit=twt_unit)
        finite = horizon_df["interpretation"].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError(f"Horizon contains no finite TWT values: {horizon_file}")
        raw_entries.append((float(np.mean(finite)), f"horizon_{index}", horizon_file, horizon_df))
    raw_entries.sort(key=lambda item: item[0])
    raw_horizons = {name: horizon_df for _, name, _, horizon_df in raw_entries}
    target_layer = TargetZone(
        raw_horizon_dfs=raw_horizons,
        geometry=geometry,
        horizon_names=list(raw_horizons.keys()),
        qc_output_dir=qc_dir,
        min_thickness=target_cfg.get("min_thickness"),
        nearest_distance_limit=target_cfg.get("nearest_distance_limit"),
        outlier_threshold=target_cfg.get("outlier_threshold"),
        outlier_min_neighbor_count=target_cfg.get("outlier_min_neighbor_count", 2),
    )
    horizons = []
    for mean_twt, name, path, _ in raw_entries:
        grid = target_layer.get_horizon_grid(name)
        horizons.append(
            {
                "file": repo_relative_path(path, root=REPO_ROOT),
                "mean_twt_s": float(np.nanmean(grid)),
                "input_mean_twt_s": float(mean_twt),
            }
        )
    return target_layer, horizons


def _resolve_frequency_bands(
    conditioned_by_key: dict[str, ConditionedWellLog],
    *,
    metrics_df: pd.DataFrame,
    auto_dir: Path,
    wavelet_dir: Path,
    batch_lookup: dict[str, pd.Series],
    sample_step_s: float,
    script_cfg: dict[str, Any],
) -> tuple[ThreeBandSplitConfig, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    bands_cfg = dict(script_cfg.get("frequency_bands") or {})
    order = int(bands_cfg.get("filter_order", 6))
    buffer_seconds = bands_cfg.get("buffer_seconds")
    buffer_seconds = None if buffer_seconds is None else float(buffer_seconds)
    buffer_mode = str(bands_cfg.get("buffer_mode", "reflect"))
    wavelet_time, wavelet_amplitude, wavelet_path = _load_selected_wavelet(wavelet_dir)
    peak_hz, left_half_hz, right_half_hz = wavelet_half_amplitude_frequencies(
        wavelet_time,
        wavelet_amplitude,
    )

    lfm_cfg = dict(bands_cfg.get("lfm") or {})
    lfm_mode = str(lfm_cfg.get("cutoff_mode", "wavelet_left_half_amplitude")).strip().lower()
    if lfm_mode == "manual":
        if lfm_cfg.get("manual_cutoff_hz") is None:
            raise ValueError("well_constraints.frequency_bands.lfm.manual_cutoff_hz is required.")
        lfm_cutoff = float(lfm_cfg["manual_cutoff_hz"])
    elif lfm_mode == "wavelet_left_half_amplitude":
        lfm_cutoff = left_half_hz * float(lfm_cfg.get("cutoff_scale", 1.0))
    else:
        raise ValueError(f"Unsupported LFM cutoff_mode: {lfm_mode!r}.")

    ginn_cfg = dict(bands_cfg.get("ginn") or {})
    mode = str(ginn_cfg.get("mode", "diagnose")).strip().lower()
    if mode == "manual":
        cutoff = ginn_cfg.get("manual_cutoff_hz")
        if cutoff is None:
            raise ValueError("well_constraints.frequency_bands.ginn.manual_cutoff_hz is required.")
        well_rows = [
            {
                "well_name": str(key),
                "route": "",
                "cutoff_hz": float(cutoff),
                "status": "manual",
                "corr": np.nan,
                "nmae": np.nan,
                "scale": np.nan,
                "n_eval_samples": 0,
                "wavelet_file": "",
                "reason": "",
            }
            for key in conditioned_by_key
        ]
        diag = pd.DataFrame.from_records(well_rows)
        cluster_aggregate = pd.DataFrame()
        aggregate = pd.DataFrame(
            {
                "cutoff_hz": [float(cutoff)],
                "mode": ["manual"],
                "n_wells": [len(well_rows)],
                "n_clusters": [0],
                "median_corr": [np.nan],
                "mean_corr": [np.nan],
                "p25_corr": [np.nan],
                "p75_corr": [np.nan],
                "median_nmae": [np.nan],
                "mean_nmae": [np.nan],
                "p25_nmae": [np.nan],
                "p75_nmae": [np.nan],
                "median_scale": [np.nan],
                "median_n_eval_samples": [np.nan],
            }
        )
        selection = {"selected_cutoff_hz": float(cutoff), "reason": "manual", "candidate_boundary_hit": False}
    elif mode == "diagnose":
        configured_candidates = ginn_cfg.get("candidate_cutoff_hz")
        candidates = (
            [float(value) for value in configured_candidates]
            if configured_candidates
            else ginn_cutoff_candidates(
                right_half_hz,
                minimum_ratio=float(ginn_cfg.get("candidate_min_right_half_ratio", 0.4)),
                maximum_ratio=float(ginn_cfg.get("candidate_max_right_half_ratio", 1.3)),
                step_hz=float(ginn_cfg.get("candidate_step_hz", 5.0)),
            )
        )
        reference_preview = dict(bands_cfg.get("reference") or {})
        reference_preview_mode = str(
            reference_preview.get("cutoff_mode", "ginn_octave")
        ).strip().lower()
        if reference_preview_mode == "manual":
            manual_reference = reference_preview.get("manual_cutoff_hz")
            if manual_reference is None:
                raise ValueError("well_constraints.frequency_bands.reference.manual_cutoff_hz is required.")
            candidate_upper_hz = float(manual_reference)
        elif reference_preview_mode == "ginn_octave":
            multiplier = float(reference_preview.get("ginn_multiplier", 2.0))
            if multiplier <= 1.0:
                raise ValueError("reference.ginn_multiplier must be greater than 1.")
            candidate_upper_hz = (
                float(reference_preview.get("max_nyquist_fraction", 0.4))
                / float(sample_step_s)
            )
        else:
            raise ValueError(f"Unsupported reference cutoff_mode: {reference_preview_mode!r}.")
        candidates = sorted(
            {
                value
                for value in candidates
                if np.isfinite(value) and lfm_cutoff < value < candidate_upper_hz
            }
        )
        if len(candidates) < 3:
            raise ValueError(
                "GINN cutoff diagnosis requires at least three candidates satisfying "
                "lfm_cutoff < candidate < reference_limit."
            )
        diag, cluster_aggregate, aggregate, selection = _diagnose_ginn_cutoff_by_forward_modeling(
            conditioned_by_key,
            metrics_df=metrics_df,
            auto_dir=auto_dir,
            wavelet_time_s=wavelet_time,
            wavelet_amplitude=wavelet_amplitude,
            wavelet_path=wavelet_path,
            batch_lookup=batch_lookup,
            candidate_cutoff_hz=candidates,
            filter_order=order,
            buffer_seconds=buffer_seconds,
            buffer_mode=buffer_mode,
            corr_tolerance=float(ginn_cfg.get("selection_corr_tolerance", 0.02)),
            nmae_tolerance=float(ginn_cfg.get("selection_nmae_tolerance", 0.03)),
        )
    else:
        raise ValueError(f"Unsupported GINN cutoff mode: {mode!r}.")

    ginn_cutoff = float(selection["selected_cutoff_hz"])
    reference_cfg = dict(bands_cfg.get("reference") or {})
    reference_mode = str(reference_cfg.get("cutoff_mode", "ginn_octave")).strip().lower()
    fs = 1.0 / float(sample_step_s)
    if reference_mode == "manual":
        if reference_cfg.get("manual_cutoff_hz") is None:
            raise ValueError("well_constraints.frequency_bands.reference.manual_cutoff_hz is required.")
        reference_cutoff = float(reference_cfg["manual_cutoff_hz"])
    elif reference_mode == "ginn_octave":
        reference_cutoff = min(
            float(reference_cfg.get("ginn_multiplier", 2.0)) * ginn_cutoff,
            float(reference_cfg.get("max_nyquist_fraction", 0.4)) * fs,
        )
    else:
        raise ValueError(f"Unsupported reference cutoff_mode: {reference_mode!r}.")
    resolved = ThreeBandSplitConfig(
        lfm_cutoff_hz=lfm_cutoff,
        ginn_cutoff_hz=ginn_cutoff,
        reference_cutoff_hz=reference_cutoff,
        filter_order=order,
        buffer_seconds=buffer_seconds,
        buffer_mode=buffer_mode,
    )
    resolved.validate(sample_step_s)
    selection.update(
        {
            "mode": mode,
            "wavelet_file": repo_relative_path(wavelet_path, root=REPO_ROOT),
            "wavelet_peak_hz": peak_hz,
            "wavelet_left_half_amplitude_hz": left_half_hz,
            "wavelet_right_half_amplitude_hz": right_half_hz,
            "lfm_cutoff_hz": lfm_cutoff,
            "reference_cutoff_hz": reference_cutoff,
            "fail_on_candidate_boundary": bool(ginn_cfg.get("fail_on_candidate_boundary", True)),
        }
    )
    return resolved, diag, cluster_aggregate, aggregate, selection


def _load_selected_wavelet(wavelet_dir: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    summary_path = wavelet_dir / "selected_wavelet_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        value = summary.get("selected_wavelet_file")
        if value:
            path = resolve_relative_path(value, root=REPO_ROOT)
            if path.exists():
                time_s, amplitude = load_wavelet_csv(path)
                return time_s, amplitude, path
    fallback = wavelet_dir / "selected_wavelet.csv"
    time_s, amplitude = load_wavelet_csv(fallback)
    return time_s, amplitude, fallback


def _reflectivity_from_ai(ai: np.ndarray) -> np.ndarray:
    values = np.asarray(ai, dtype=np.float64)
    out = np.zeros(values.shape, dtype=np.float32)
    valid = np.isfinite(values) & (values > 0.0)
    pair_valid = valid[:-1] & valid[1:]
    upper = values[:-1][pair_valid]
    lower = values[1:][pair_valid]
    out[np.flatnonzero(pair_valid) + 1] = ((lower - upper) / np.maximum(lower + upper, 1e-12)).astype(np.float32)
    return out


def _masked_scaled_synthetic_metrics(
    *,
    modeler: Any,
    wavelet: Any,
    reflectivity: Any,
    seismic: Any,
    eval_mask: np.ndarray,
) -> tuple[float, float, float, int]:
    synthetic_raw = np.asarray(modeler(wavelet.values, reflectivity.values), dtype=np.float64)
    seismic_values = np.asarray(seismic.values, dtype=np.float64)
    mask = np.asarray(eval_mask, dtype=bool) & np.isfinite(synthetic_raw) & np.isfinite(seismic_values)
    n_eval = int(np.count_nonzero(mask))
    if n_eval < 8:
        raise ValueError(f"too_few_eval_samples:{n_eval}")
    seismic_eval = seismic_values[mask]
    seismic_norm = seismic_eval - float(np.mean(seismic_eval))
    std = float(np.std(seismic_norm))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("Seismic eval window has zero standard deviation.")
    seismic_norm = seismic_norm / std
    synthetic_eval = synthetic_raw[mask]
    denom = max(float(np.dot(synthetic_eval, synthetic_eval)), 1e-12)
    scale = float(np.dot(seismic_norm, synthetic_eval) / denom)
    synthetic_scaled = scale * synthetic_eval
    corr = float(np.corrcoef(seismic_norm, synthetic_scaled)[0, 1]) if np.std(synthetic_scaled) > 0.0 else np.nan
    nmae = float(np.sum(np.abs(seismic_norm - synthetic_scaled)) / max(np.sum(np.abs(seismic_norm)), 1e-12))
    if not np.isfinite(corr) or not np.isfinite(nmae):
        raise ValueError("Forward diagnostic produced non-finite corr or NMAE.")
    return corr, nmae, scale, n_eval


def _diagnose_ginn_cutoff_by_forward_modeling(
    conditioned_by_key: dict[str, ConditionedWellLog],
    *,
    metrics_df: pd.DataFrame,
    auto_dir: Path,
    wavelet_time_s: np.ndarray,
    wavelet_amplitude: np.ndarray,
    wavelet_path: Path,
    batch_lookup: dict[str, pd.Series],
    candidate_cutoff_hz: list[float],
    filter_order: int,
    buffer_seconds: float | None,
    buffer_mode: str,
    corr_tolerance: float,
    nmae_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not conditioned_by_key:
        raise ValueError("Cannot diagnose GINN cutoff without conditioned well logs.")
    wavelet_dt = float(np.median(np.diff(wavelet_time_s)))
    wavelet = grid.Wavelet(wavelet_amplitude, wavelet_time_s, name=wavelet_path.stem)
    modeler = ConvModeler()
    rows: list[dict[str, Any]] = []

    for _, metric in metrics_df.iterrows():
        well_name = str(metric.get("well_name", "")).strip()
        key = normalize_well_name(well_name)
        conditioned = conditioned_by_key.get(key)
        if conditioned is None:
            continue
        route = str(metric.get("route", ""))
        seismic_file = _resolve_seismic_trace_file(metric, run_dir=auto_dir, well_name=well_name)
        if seismic_file is None:
            for cutoff in candidate_cutoff_hz:
                rows.append(
                    {
                        "well_name": well_name,
                        "route": route,
                        "cutoff_hz": cutoff,
                        "status": "failed",
                        "reason": "missing_seismic_trace",
                    }
                )
            continue
        try:
            seismic = load_saved_seismic_trace_csv(seismic_file)
            seismic_basis = np.asarray(seismic.basis, dtype=np.float64)
            seismic_dt = float(np.median(np.diff(seismic_basis)))
            if not np.isclose(wavelet_dt, seismic_dt, rtol=1e-5, atol=1e-9):
                raise ValueError(f"wavelet_dt_mismatch:{wavelet_dt:g}!={seismic_dt:g}")
            source_basis = np.asarray(conditioned.log_ai.basis, dtype=np.float64)
            observed_float = conditioned.observed_mask.astype(np.float64)
            eval_mask = np.interp(seismic_basis, source_basis, observed_float, left=0.0, right=0.0) > 0.999
            filter_cfg = ThreeBandSplitConfig(
                lfm_cutoff_hz=1.0,
                ginn_cutoff_hz=2.0,
                reference_cutoff_hz=3.0,
                filter_order=int(filter_order),
                buffer_seconds=buffer_seconds,
                buffer_mode=buffer_mode,
            )
            batch_row = batch_lookup.get(key)
            cluster_value = None if batch_row is None else batch_row.get("spatial_cluster_id")
            if cluster_value is None or pd.isna(cluster_value) or not str(cluster_value).strip():
                raise ValueError("missing_spatial_cluster_id")
            cluster_id = str(cluster_value)
            for cutoff in candidate_cutoff_hz:
                try:
                    low_log = segmented_lowpass(conditioned.log_ai, float(cutoff), filter_cfg)
                    low_log_on_seis = np.interp(
                        seismic_basis,
                        source_basis,
                        low_log.values,
                        left=np.nan,
                        right=np.nan,
                    )
                    low_ai_on_seis = np.exp(low_log_on_seis)
                    context_valid = np.convolve(
                        np.isfinite(low_ai_on_seis).astype(np.int16),
                        np.ones(wavelet_amplitude.size, dtype=np.int16),
                        mode="same",
                    ) == int(wavelet_amplitude.size)
                    reflectivity = grid.Reflectivity(_reflectivity_from_ai(low_ai_on_seis), seismic_basis)
                    corr, nmae, scale, n_eval = _masked_scaled_synthetic_metrics(
                        modeler=modeler,
                        wavelet=wavelet,
                        reflectivity=reflectivity,
                        seismic=seismic,
                        eval_mask=eval_mask & context_valid,
                    )
                    rows.append(
                        {
                            "well_name": well_name,
                            "route": route,
                            "spatial_cluster_id": cluster_id,
                            "cutoff_hz": float(cutoff),
                            "status": "ok",
                            "corr": corr,
                            "nmae": nmae,
                            "scale": scale,
                            "n_eval_samples": n_eval,
                            "wavelet_file": repo_relative_path(wavelet_path, root=REPO_ROOT),
                            "reason": "",
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "well_name": well_name,
                            "route": route,
                            "spatial_cluster_id": cluster_id,
                            "cutoff_hz": float(cutoff),
                            "status": "failed",
                            "reason": f"{type(exc).__name__}:{exc}",
                        }
                    )
        except Exception as exc:
            for cutoff in candidate_cutoff_hz:
                rows.append(
                    {
                        "well_name": well_name,
                        "route": route,
                        "cutoff_hz": float(cutoff),
                        "status": "failed",
                        "reason": f"{type(exc).__name__}:{exc}",
                    }
                )

    diag_df = pd.DataFrame(rows)
    ok = diag_df.loc[diag_df["status"].eq("ok")].copy()
    if ok.empty:
        raise ValueError("No successful forward-modeled GINN cutoff diagnostics.")
    cluster_aggregate = (
        ok.groupby(["cutoff_hz", "spatial_cluster_id"], dropna=False)
        .agg(
            n_wells=("well_name", "nunique"),
            median_corr=("corr", "median"),
            median_nmae=("nmae", "median"),
            median_scale=("scale", "median"),
            median_n_eval_samples=("n_eval_samples", "median"),
        )
        .reset_index()
    )
    aggregate = (
        cluster_aggregate.groupby("cutoff_hz")
        .agg(
            n_clusters=("spatial_cluster_id", "nunique"),
            n_wells=("n_wells", "sum"),
            median_corr=("median_corr", "median"),
            mean_corr=("median_corr", "mean"),
            p25_corr=("median_corr", lambda x: float(np.nanpercentile(x, 25))),
            p75_corr=("median_corr", lambda x: float(np.nanpercentile(x, 75))),
            median_nmae=("median_nmae", "median"),
            mean_nmae=("median_nmae", "mean"),
            p25_nmae=("median_nmae", lambda x: float(np.nanpercentile(x, 25))),
            p75_nmae=("median_nmae", lambda x: float(np.nanpercentile(x, 75))),
            median_scale=("median_scale", "median"),
            median_n_eval_samples=("median_n_eval_samples", "median"),
        )
        .reset_index()
        .sort_values("cutoff_hz")
    )
    finite = aggregate[np.isfinite(aggregate["median_corr"]) & np.isfinite(aggregate["median_nmae"])].copy()
    if finite.empty:
        raise ValueError("No finite aggregate forward-modeled frequency split metrics.")
    best = finite.loc[finite["median_corr"].idxmax()]
    best_corr = float(best["median_corr"])
    best_nmae = float(best["median_nmae"])
    plateau = finite[
        (finite["median_corr"] >= best_corr - float(corr_tolerance))
        & (finite["median_nmae"] <= best_nmae + float(nmae_tolerance))
    ]
    if plateau.empty:
        chosen = best
        reason = "No cutoff passed plateau tolerances; selected max median_corr."
    else:
        chosen = plateau.loc[plateau["cutoff_hz"].idxmin()]
        reason = "Selected the lowest GINN cutoff on the cluster-debiased near-best waveform-fit plateau."
    selected = float(chosen["cutoff_hz"])
    candidate_min = float(min(candidate_cutoff_hz))
    candidate_max = float(max(candidate_cutoff_hz))
    boundary_hit = bool(np.isclose(selected, candidate_min) or np.isclose(selected, candidate_max))
    selection = {
        "selected_cutoff_hz": selected,
        "selected_median_corr": float(chosen["median_corr"]),
        "selected_median_nmae": float(chosen["median_nmae"]),
        "best_median_corr": best_corr,
        "best_corr_cutoff_hz": float(best["cutoff_hz"]),
        "best_corr_median_nmae": best_nmae,
        "corr_tolerance": float(corr_tolerance),
        "nmae_tolerance": float(nmae_tolerance),
        "reason": reason,
        "wavelet_file": repo_relative_path(wavelet_path, root=REPO_ROOT),
        "candidate_min_hz": candidate_min,
        "candidate_max_hz": candidate_max,
        "candidate_boundary_hit": boundary_hit,
        "aggregation": "median_within_spatial_cluster_then_median_across_clusters",
    }
    return diag_df, cluster_aggregate, aggregate, selection


def _save_frequency_band_qc(
    bands_by_well: dict[str, WellFrequencyBands],
    qc_dir: Path,
    *,
    envelope_window: int,
) -> list[dict[str, Any]]:
    qc_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = qc_dir / "figures"
    trace_dir = qc_dir / "traces"
    fig_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for well_name, bands in bands_by_well.items():
        group = pd.DataFrame(
            {
                "well_name": well_name,
                "twt_s": bands.reference_log_ai.basis,
                "reference_ai": bands.reference_ai.values,
                "reference_log_ai": bands.reference_log_ai.values,
                "lfm_ai": bands.lfm_ai.values,
                "lfm_log_ai": bands.lfm_log_ai.values,
                "ginn_target_ai": bands.ginn_target_ai.values,
                "ginn_target_log_ai": bands.ginn_target_log_ai.values,
                "ginn_band_log_ai": bands.ginn_band_log_ai.values,
                "enhance_residual_log_ai": bands.enhance_residual_log_ai.values,
                "observed_well_sample": bands.observed_mask,
                "short_gap_interpolated": bands.interpolation_mask,
                "hampel_conditioned": bands.conditioned_mask,
                "frequency_band_valid": bands.valid_band_mask,
            }
        )
        safe = sanitize_filename(str(well_name))
        trace_path = trace_dir / f"frequency_bands_{safe}.csv"
        group[
            [
                "well_name",
                "twt_s",
                "reference_ai",
                "reference_log_ai",
                "lfm_ai",
                "lfm_log_ai",
                "ginn_target_ai",
                "ginn_target_log_ai",
                "ginn_band_log_ai",
                "enhance_residual_log_ai",
                "observed_well_sample",
                "short_gap_interpolated",
                "hampel_conditioned",
                "frequency_band_valid",
            ]
        ].to_csv(trace_path, index=False, encoding="utf-8-sig")
        fig_path = fig_dir / f"frequency_bands_{safe}.png"
        fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
        axes[0].plot(group["twt_s"], group["reference_log_ai"], label="reference", linewidth=1.0)
        axes[0].plot(group["twt_s"], group["ginn_target_log_ai"], label="GINN target", linewidth=1.0)
        axes[0].plot(group["twt_s"], group["lfm_log_ai"], label="LFM", linewidth=1.0)
        axes[0].legend(loc="best")
        axes[1].plot(group["twt_s"], group["ginn_band_log_ai"], label="GINN band", linewidth=1.0)
        axes[1].legend(loc="best")
        high = group["enhance_residual_log_ai"].to_numpy(dtype=float)
        axes[2].plot(group["twt_s"], high, label="enhance residual", linewidth=1.0)
        if high.size and int(envelope_window) > 1:
            window = min(int(envelope_window), max(1, high.size))
            kernel = np.ones(window, dtype=float) / float(window)
            envelope = np.convolve(np.abs(high), kernel, mode="same")
            axes[2].plot(group["twt_s"], envelope, color="tab:red", alpha=0.8, label="abs envelope")
            axes[2].plot(group["twt_s"], -envelope, color="tab:red", alpha=0.8)
        axes[2].legend(loc="best")
        for offset, (column, label) in enumerate(
            (
                ("observed_well_sample", "observed"),
                ("short_gap_interpolated", "short-gap support"),
                ("hampel_conditioned", "Hampel support"),
                ("frequency_band_valid", "band valid"),
            )
        ):
            axes[3].step(
                group["twt_s"],
                group[column].astype(float) + 1.25 * offset,
                where="mid",
                linewidth=0.9,
                label=label,
            )
        axes[3].set_yticks([])
        axes[3].legend(loc="best", ncol=2)
        axes[3].set_xlabel("TWT (s)")
        fig.suptitle(str(well_name))
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        rows.append(
            {
                "well_name": str(well_name),
                "frequency_band_qc_trace_path": repo_relative_path(trace_path, root=REPO_ROOT),
                "frequency_band_qc_figure_path": repo_relative_path(fig_path, root=REPO_ROOT),
            }
        )
    return rows


def _plot_frequency_split_aggregate(
    aggregate_df: pd.DataFrame,
    *,
    figure_path: Path,
    selected_cutoff_hz: float | None,
    best_cutoff_hz: float | None,
) -> None:
    if aggregate_df.empty or "cutoff_hz" not in aggregate_df.columns:
        return
    if not {"median_corr", "median_nmae"}.issubset(aggregate_df.columns):
        return
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = aggregate_df.sort_values("cutoff_hz").copy()
    x = plot_df["cutoff_hz"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(x, plot_df["median_corr"], marker="o", color="tab:blue", label="median corr")
    if {"p25_corr", "p75_corr"}.issubset(plot_df.columns):
        axes[0].fill_between(x, plot_df["p25_corr"], plot_df["p75_corr"], color="tab:blue", alpha=0.15, label="p25-p75")
    axes[0].set_ylabel("Correlation")
    axes[0].set_title("Forward synthetic correlation")
    axes[0].legend(loc="best")

    axes[1].plot(x, plot_df["median_nmae"], marker="o", color="tab:orange", label="median nmae")
    if {"p25_nmae", "p75_nmae"}.issubset(plot_df.columns):
        axes[1].fill_between(x, plot_df["p25_nmae"], plot_df["p75_nmae"], color="tab:orange", alpha=0.15, label="p25-p75")
    axes[1].set_ylabel("NMAE")
    axes[1].set_title("Forward synthetic error")
    axes[1].legend(loc="best")

    for ax in axes:
        if best_cutoff_hz is not None and np.isfinite(best_cutoff_hz):
            ax.axvline(float(best_cutoff_hz), color="black", linestyle=":", linewidth=1.0, alpha=0.75, label="best corr")
        if selected_cutoff_hz is not None and np.isfinite(selected_cutoff_hz):
            ax.axvline(float(selected_cutoff_hz), color="tab:red", linestyle="--", linewidth=1.2, alpha=0.85, label="selected")
        ax.set_xlabel("Cutoff (Hz)")
        ax.grid(True, alpha=0.25)

    fig.suptitle("GINN cutoff forward-model sweep")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def _plot_frequency_split_well_sweeps(diag_df: pd.DataFrame, figure_dir: Path) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    ok = diag_df.loc[diag_df.get("status", "").eq("ok")].copy() if not diag_df.empty else pd.DataFrame()
    for well_name, group in ok.groupby("well_name", sort=False):
        plot_df = group.sort_values("cutoff_hz")
        path = figure_dir / f"ginn_cutoff_sweep_{sanitize_filename(str(well_name))}.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].plot(plot_df["cutoff_hz"], plot_df["corr"], marker="o", color="tab:blue")
        axes[0].set_title(str(well_name))
        axes[0].set_ylabel("Correlation")
        axes[1].plot(plot_df["cutoff_hz"], plot_df["nmae"], marker="o", color="tab:orange")
        axes[1].set_title("Forward synthetic error")
        axes[1].set_ylabel("NMAE")
        for ax in axes:
            ax.set_xlabel("Cutoff (Hz)")
            ax.grid(True, alpha=0.25)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        out[str(well_name)] = repo_relative_path(path, root=REPO_ROOT)
    return out


def _plot_frequency_band_response(
    *,
    wavelet_time_s: np.ndarray,
    wavelet_amplitude: np.ndarray,
    bands: ThreeBandSplitConfig,
    figure_path: Path,
) -> None:
    """Plot wavelet amplitude spectrum and the effective zero-phase filters."""
    dt_s = float(np.median(np.diff(wavelet_time_s)))
    fs = 1.0 / dt_s
    n_fft = max(4096, 1 << int(np.ceil(np.log2(wavelet_amplitude.size * 16))))
    frequency = np.fft.rfftfreq(n_fft, d=dt_s)
    spectrum = np.abs(np.fft.rfft(wavelet_amplitude, n=n_fft))
    spectrum /= max(float(np.max(spectrum)), 1e-12)
    peak_hz, left_hz, right_hz = wavelet_half_amplitude_frequencies(
        wavelet_time_s,
        wavelet_amplitude,
    )

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(frequency, spectrum, color="black", linewidth=1.4, label="consensus wavelet amplitude")
    for cutoff, label, color in (
        (bands.lfm_cutoff_hz, "LFM low-pass", "tab:blue"),
        (bands.ginn_cutoff_hz, "GINN target low-pass", "tab:orange"),
        (bands.reference_cutoff_hz, "reference low-pass", "tab:green"),
    ):
        sos = butter(int(bands.filter_order) // 2, float(cutoff), btype="low", fs=fs, output="sos")
        response_frequency, response = sosfreqz(sos, worN=4096, fs=fs)
        ax.plot(response_frequency, np.abs(response) ** 2, color=color, linewidth=1.1, label=label)
        ax.axvline(float(cutoff), color=color, linestyle="--", linewidth=0.9, alpha=0.75)
    ax.axvline(peak_hz, color="black", linestyle=":", linewidth=0.9, label="wavelet peak")
    ax.axvline(left_hz, color="tab:purple", linestyle=":", linewidth=0.9, label="left/right amplitude 0.5")
    ax.axvline(right_hz, color="tab:purple", linestyle=":", linewidth=0.9)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlim(0.0, min(0.5 * fs, max(bands.reference_cutoff_hz * 1.5, right_hz * 1.5)))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def _make_high_supervision_bundle(
    *,
    point_df: pd.DataFrame,
    samples: np.ndarray,
    metadata: dict[str, Any],
) -> WellHighSupervisionBundle:
    arrays, _summary_df = aggregate_trace_arrays(
        point_df,
        samples,
        target_col="ginn_target_ai",
        value_cols=["reference_log_ai", "ginn_target_log_ai", "enhance_residual_log_ai"],
        include_anchor_only=False,
    )
    keep = arrays["mask"].sum(axis=1) >= 2
    if not np.all(keep):
        for key, value in list(arrays.items()):
            if key == "samples":
                continue
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] == keep.size:
                arrays[key] = arr[keep]
    n_sample = int(samples.size)
    mask = arrays["mask"].astype(bool)
    samples_1d = np.asarray(samples, dtype=np.float32)
    bundle = WellHighSupervisionBundle(
        sample_domain="time",
        sample_unit="s",
        samples=samples_1d,
        flat_indices=arrays["flat_indices"],
        well_names=arrays["well_names"],
        inline=arrays["inline"],
        xline=arrays["xline"],
        reference_log_ai=arrays["reference_log_ai"].astype(np.float32),
        ginn_target_log_ai=arrays["ginn_target_log_ai"].astype(np.float32),
        enhance_residual_log_ai=arrays["enhance_residual_log_ai"].astype(np.float32),
        well_mask=mask,
        well_weight=arrays["weight"].astype(np.float32),
        native_samples=samples_1d.copy(),
        native_reference_log_ai=arrays["reference_log_ai"].astype(np.float32),
        native_ginn_target_log_ai=arrays["ginn_target_log_ai"].astype(np.float32),
        native_enhance_residual_log_ai=arrays["enhance_residual_log_ai"].astype(np.float32),
        native_well_mask=mask.copy(),
        summary=high_frequency_stats(point_df, sample_step_s=float(np.median(np.diff(samples))) if samples.size > 1 else None),
        metadata=metadata,
    )
    validate_well_high_supervision(bundle, sample_domain="time", n_sample=n_sample)
    return bundle


def _make_lfm_control_table(point_df: pd.DataFrame) -> pd.DataFrame:
    """Export point-level LFM controls without slice aggregation."""
    base_columns = [
        "well_name",
        "route",
        "source",
        "twt_s",
        "md_m",
        "x_m",
        "y_m",
        "inline_float",
        "xline_float",
        "nearest_inline",
        "nearest_xline",
        "inline_index",
        "xline_index",
        "flat_idx",
        "seismic_sample_index",
        "zone_name",
        "u_in_zone",
        "weight",
        "batch_corr",
        "batch_nmae",
    ]
    ai_df = point_df.loc[point_df["observed_well_sample"].astype(bool), base_columns + ["lfm_ai"]].copy()
    return ai_df.rename(columns={"lfm_ai": "ai"})


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    workflow = TimeWorkflowConfig.from_mapping(cfg)
    script_cfg = deep_merge_dict(DEFAULT_CONFIG, dict(cfg.get("well_constraints") or {}))
    if "frequency_split" in dict(cfg.get("well_constraints") or {}):
        raise ValueError(
            "well_constraints.frequency_split was replaced by well_constraints.frequency_bands. "
            "Migrate the configuration; the old two-band contract is not supported."
        )
    if "n_slices" in dict(script_cfg.get("lfm_controls") or {}):
        raise ValueError(
            "well_constraints.lfm_controls.n_slices has moved to lfm_precomputed.modeling.n_slices; "
            "step 06 exports point-level LFM controls only."
        )
    data_root = REPO_ROOT / workflow.data_root
    output_root = REPO_ROOT / workflow.output_root
    source_dirs = _resolve_source_dirs(script_cfg, output_root)

    if args.output_dir is None:
        output_dir = output_root / f"well_constraints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    target_qc_dir = output_dir / "target_layer_qc"

    survey, seismic_file, seismic_type = _open_survey(workflow.seismic.as_dict(), data_root)
    geometry = survey.describe_geometry(domain="time")
    if str(geometry.get("sample_domain")).lower() != "time" or str(geometry.get("sample_unit")).lower() != "s":
        raise ValueError(f"Expected time-domain seismic geometry in seconds, got {geometry}")
    samples = survey.sample_axis("time").values.astype(np.float64)
    sample_step_s = float(np.median(np.diff(samples))) if samples.size > 1 else None
    target_layer, horizon_metadata = _build_target_layer(script_cfg, geometry, target_qc_dir, data_root)

    auto_dir = source_dirs["well_auto_tie_dir"]
    wavelet_dir = source_dirs["wavelet_generation_dir"]
    _validate_wavelet_generation_source(auto_dir=auto_dir, wavelet_dir=wavelet_dir)
    metrics_df = pd.read_csv(auto_dir / "well_tie_metrics.csv")
    plan_df = pd.read_csv(auto_dir / "well_tie_plan.csv") if (auto_dir / "well_tie_plan.csv").exists() else pd.DataFrame()
    batch_df = pd.read_csv(wavelet_dir / "batch_synthetic_metrics.csv")
    batch_lookup = _batch_metric_lookup(batch_df)
    plan_by_key = _plan_lookup(plan_df)

    include_wells = script_cfg["control_wells"].get("include_wells")
    include_keys = {normalize_well_name(w) for w in include_wells} if include_wells else None
    exclude_keys = {normalize_well_name(w) for w in script_cfg["control_wells"].get("exclude_wells", [])}
    min_corr = script_cfg["control_wells"].get("min_batch_corr")
    max_nmae = script_cfg["control_wells"].get("max_batch_nmae")
    min_points = int(script_cfg["lfm_controls"].get("min_control_samples_per_well", 16))
    weights_cfg = dict(script_cfg.get("weights") or {})
    include_deviated_anchor = bool(script_cfg.get("anchor", {}).get("include_deviated", False))
    include_deviated_high = bool(script_cfg.get("high_supervision", {}).get("include_deviated", False))

    conditioning_cfg = ReferenceConditioningConfig(**dict(script_cfg.get("reference_conditioning") or {}))
    conditioned_by_key: dict[str, ConditionedWellLog] = {}
    candidate_context: dict[str, dict[str, Any]] = {}
    qc_rows: list[dict[str, Any]] = []
    for _, metric in metrics_df.iterrows():
        well_name = str(metric.get("well_name", "")).strip()
        if not well_name:
            continue
        key = normalize_well_name(well_name)
        route = str(metric.get("route", ""))
        batch_row = batch_lookup.get(key)
        batch_corr = _metric_value(batch_row, ("corr", "batch_corr", "selected_corr"))
        batch_nmae = _metric_value(batch_row, ("nmae", "batch_nmae", "selected_nmae"))
        reasons: list[str] = []
        status = "selected"
        if str(metric.get("tie_status", "")) != "success":
            reasons.append(f"tie_status_{metric.get('tie_status', 'unknown')}")
        if include_keys is not None and key not in include_keys:
            reasons.append("not_in_include_wells")
        if key in exclude_keys:
            reasons.append("excluded_well")
        if batch_corr is None:
            reasons.append("missing_batch_corr")
        elif min_corr is not None and batch_corr < float(min_corr):
            reasons.append("batch_corr_below_threshold")
        if max_nmae is not None and (batch_nmae is None or batch_nmae > float(max_nmae)):
            reasons.append("batch_nmae_above_threshold")

        plan = plan_by_key.get(key)
        las_file = (
            None
            if plan is None
            else resolve_artifact_path(plan.get("input_las"), root=REPO_ROOT, run_dir=auto_dir)
        )
        tdt_file = resolve_artifact_path(metric.get("optimized_tdt_file"), root=REPO_ROOT, run_dir=auto_dir)
        if las_file is None or not las_file.exists():
            reasons.append("missing_preprocessed_las")
        if tdt_file is None or not tdt_file.exists():
            reasons.append("missing_optimized_tdt_file")

        conditioning_qc: dict[str, Any] = {}
        if reasons:
            status = "rejected"
        else:
            try:
                weight = confidence_from_corr(
                    batch_corr,
                    mode=str(weights_cfg.get("mode", "corr")),
                    floor=float(weights_cfg.get("corr_floor", 0.3)),
                    span=float(weights_cfg.get("corr_span", 0.4)),
                    min_weight=float(weights_cfg.get("corr_min_weight", 0.6)),
                )
                standard = load_standard_vp_rho_logs(las_file)
                table = load_workflow_time_depth_table_csv(tdt_file)
                conditioned = build_conditioned_reference_log(
                    standard,
                    table,
                    samples,
                    conditioning_cfg,
                )
                observed_count = int(np.count_nonzero(conditioned.observed_mask))
                conditioning_qc = {
                    "observed_samples": observed_count,
                    "short_gap_interpolated_samples": int(np.count_nonzero(conditioned.interpolation_mask)),
                    "hampel_conditioned_samples": int(np.count_nonzero(conditioned.conditioned_mask)),
                    "valid_conditioned_samples": int(np.count_nonzero(np.isfinite(conditioned.log_ai.values))),
                }
                if observed_count < min_points:
                    status = "rejected"
                    reasons.append("too_few_control_samples")
                else:
                    conditioned_by_key[key] = conditioned
                    candidate_context[key] = {
                        "well_name": well_name,
                        "route": route,
                        "metric": metric,
                        "plan": plan,
                        "tdt_file": tdt_file,
                        "weight": weight,
                        "batch_corr": batch_corr,
                        "batch_nmae": batch_nmae,
                    }
            except Exception as exc:
                status = "failed"
                reasons.append(str(exc) or type(exc).__name__)

        qc_rows.append(
            {
                "well_name": well_name,
                "status": status,
                "route": route,
                "batch_corr": batch_corr,
                "batch_nmae": batch_nmae,
                "control_point_count": 0,
                "invalid_point_count": None,
                "invalid_point_fraction": None,
                "unique_trace_count": 0,
                **conditioning_qc,
                "reasons": ";".join(dict.fromkeys(reasons)),
            }
        )

    if not conditioned_by_key:
        qc_path = output_dir / "well_constraint_qc.csv"
        pd.DataFrame(qc_rows).to_csv(qc_path, index=False, encoding="utf-8-sig")
        raise ValueError(f"No selected conditioned well logs. Inspect {qc_path}.")

    bands_cfg, split_diag_df, split_cluster_df, split_aggregate_df, split_selection = _resolve_frequency_bands(
        conditioned_by_key,
        metrics_df=metrics_df,
        auto_dir=auto_dir,
        wavelet_dir=wavelet_dir,
        batch_lookup=batch_lookup,
        sample_step_s=float(sample_step_s),
        script_cfg=script_cfg,
    )
    split_diag_path = output_dir / "ginn_cutoff_diagnostics.csv"
    split_cluster_path = output_dir / "ginn_cutoff_cluster_aggregate.csv"
    split_aggregate_path = output_dir / "ginn_cutoff_aggregate.csv"
    split_diag_df.to_csv(split_diag_path, index=False, encoding="utf-8-sig")
    split_cluster_df.to_csv(split_cluster_path, index=False, encoding="utf-8-sig")
    split_aggregate_df.to_csv(split_aggregate_path, index=False, encoding="utf-8-sig")
    split_diag_figure_path = output_dir / "figures" / "ginn_cutoff_sweep.png"
    _plot_frequency_split_aggregate(
        split_aggregate_df,
        figure_path=split_diag_figure_path,
        selected_cutoff_hz=split_selection.get("selected_cutoff_hz"),
        best_cutoff_hz=split_selection.get("best_corr_cutoff_hz"),
    )
    split_well_figure_paths = _plot_frequency_split_well_sweeps(
        split_diag_df,
        output_dir / "figures" / "ginn_cutoff_wells",
    )
    wavelet_time_s, wavelet_amplitude, _wavelet_path = _load_selected_wavelet(wavelet_dir)
    frequency_response_path = output_dir / "figures" / "frequency_band_response.png"
    _plot_frequency_band_response(
        wavelet_time_s=wavelet_time_s,
        wavelet_amplitude=wavelet_amplitude,
        bands=bands_cfg,
        figure_path=frequency_response_path,
    )
    if (
        bool(split_selection.get("candidate_boundary_hit"))
        and bool(split_selection.get("fail_on_candidate_boundary"))
    ):
        write_json(
            output_dir / "run_summary.json",
            {
                "frequency_bands": split_selection,
                "status": "failed_candidate_boundary",
                "outputs": {
                    "ginn_cutoff_diagnostics": repo_relative_path(split_diag_path, root=REPO_ROOT),
                    "ginn_cutoff_cluster_aggregate": repo_relative_path(split_cluster_path, root=REPO_ROOT),
                    "ginn_cutoff_aggregate": repo_relative_path(split_aggregate_path, root=REPO_ROOT),
                    "ginn_cutoff_sweep_figure": repo_relative_path(split_diag_figure_path, root=REPO_ROOT),
                    "ginn_cutoff_well_sweep_figures": split_well_figure_paths,
                    "frequency_band_response_figure": repo_relative_path(
                        frequency_response_path,
                        root=REPO_ROOT,
                    ),
                },
            },
        )
        raise ValueError(
            "Selected GINN cutoff lies on the candidate boundary. Diagnostics were written; "
            "expand well_constraints.frequency_bands.ginn candidates and rerun."
        )

    point_frames: list[pd.DataFrame] = []
    bands_by_well: dict[str, WellFrequencyBands] = {}
    qc_by_key = {normalize_well_name(str(row["well_name"])): row for row in qc_rows}
    for key, context in candidate_context.items():
        qc_row = qc_by_key[key]
        try:
            bands = build_frequency_bands(conditioned_by_key[key], bands_cfg)
            metric = context["metric"]
            plan = context["plan"]
            route = context["route"]
            well_name = context["well_name"]
            if route.startswith("deviated"):
                trace_file = resolve_artifact_path(
                    metric.get("optimized_trace_sample_plan_file"),
                    root=REPO_ROOT,
                    run_dir=auto_dir,
                )
                if trace_file is None:
                    trace_file = (
                        auto_dir
                        / "trace_sample_plan"
                        / f"optimized_trace_sample_plan_{sanitize_filename(well_name)}.csv"
                    )
                if not trace_file.exists():
                    raise FileNotFoundError(f"missing_optimized_trace_sample_plan:{well_name}")
                points, point_qc = build_deviated_point_facts(
                    well_name=well_name,
                    route=route,
                    trace_plan_file=trace_file,
                    bands=bands,
                    target_layer=target_layer,
                    survey=survey,
                    samples=samples,
                    weight=context["weight"],
                    batch_corr=context["batch_corr"],
                    batch_nmae=context["batch_nmae"],
                    sample_step_s=sample_step_s,
                    anchor_eligible=include_deviated_anchor,
                )
            else:
                surface_x = _as_optional_float(plan.get("surface_x"))
                surface_y = _as_optional_float(plan.get("surface_y"))
                if surface_x is None or surface_y is None:
                    raise ValueError(f"missing_surface_xy:{well_name}")
                points, point_qc = build_vertical_point_facts(
                    well_name=well_name,
                    route=route,
                    tdt_file=context["tdt_file"],
                    bands=bands,
                    surface_x=surface_x,
                    surface_y=surface_y,
                    target_layer=target_layer,
                    survey=survey,
                    samples=samples,
                    weight=context["weight"],
                    batch_corr=context["batch_corr"],
                    batch_nmae=context["batch_nmae"],
                    anchor_eligible=True,
                )
            attempted = int(point_qc.get("attempted_samples", 0))
            invalid = int(point_qc.get("invalid_point_count", 0))
            qc_row["invalid_point_count"] = invalid
            qc_row["invalid_point_fraction"] = float(invalid / attempted) if attempted else None
            qc_row["zone_sample_errors"] = int(point_qc.get("zone_sample_errors", 0))
            if int(point_qc.get("valid_points", 0)) >= min_points:
                point_frames.append(points)
                bands_by_well[well_name] = bands
            else:
                qc_row["status"] = "rejected"
                qc_row["reasons"] = ";".join(
                    value
                    for value in [str(qc_row.get("reasons") or ""), "too_few_valid_frequency_band_points"]
                    if value
                )
        except Exception as exc:
            qc_row["status"] = "failed"
            qc_row["reasons"] = ";".join(
                value
                for value in [
                    str(qc_row.get("reasons") or ""),
                    f"{type(exc).__name__}:{exc}",
                ]
                if value
            )
    if not point_frames:
        raise ValueError("No wells retained enough valid three-band control samples.")
    point_df = pd.concat(point_frames, ignore_index=True)
    high_points = (
        point_df.copy()
        if include_deviated_high
        else point_df.loc[~point_df["source"].astype(str).eq("deviated_trajectory")].copy()
    )
    high_points = high_points.loc[high_points["observed_well_sample"].astype(bool)].copy()

    freq_cfg = dict(script_cfg.get("frequency_bands") or {})
    qc_split_rows = _save_frequency_band_qc(
        bands_by_well,
        output_dir / "frequency_band_qc",
        envelope_window=int(freq_cfg.get("qc_envelope_window_samples", 31)),
    )
    qc_split_by_well = {row["well_name"]: row for row in qc_split_rows}
    qc_df = pd.DataFrame(qc_rows)
    point_counts = point_df.groupby("well_name").size().to_dict()
    unique_trace_counts = point_df.groupby("well_name")["flat_idx"].nunique().to_dict()
    qc_df["control_point_count"] = [int(point_counts.get(str(name), 0)) for name in qc_df["well_name"]]
    qc_df["unique_trace_count"] = [int(unique_trace_counts.get(str(name), 0)) for name in qc_df["well_name"]]
    qc_df.loc[qc_df["control_point_count"].eq(0) & qc_df["status"].eq("selected"), "status"] = "rejected"
    high_counts_by_well = high_points.groupby("well_name").size().to_dict() if not high_points.empty else {}
    qc_df["high_supervision_point_count"] = [
        int(high_counts_by_well.get(str(well_name), 0)) for well_name in qc_df["well_name"]
    ]
    qc_df["high_supervision_eligible"] = qc_df["high_supervision_point_count"].gt(0)
    for key in ["frequency_band_qc_trace_path", "frequency_band_qc_figure_path"]:
        qc_df[key] = [qc_split_by_well.get(str(w), {}).get(key) for w in qc_df["well_name"]]

    point_path = output_dir / "well_constraint_points.csv"
    point_df.to_csv(point_path, index=False, encoding="utf-8-sig")
    qc_path = output_dir / "well_constraint_qc.csv"
    qc_df.to_csv(qc_path, index=False, encoding="utf-8-sig")

    anchor_points = point_df.loc[
        point_df["anchor_eligible"].astype(bool) & point_df["observed_well_sample"].astype(bool)
    ].copy()
    conflicts = build_point_conflict_report(anchor_points, value_col="ginn_target_log_ai")
    conflicts_path = output_dir / "well_anchor_conflicts.csv" if not conflicts.empty else None
    if conflicts_path is not None:
        conflicts.to_csv(conflicts_path, index=False, encoding="utf-8-sig")

    high_conflicts = build_point_conflict_report(high_points, value_col="enhance_residual_log_ai")
    high_conflicts_path = output_dir / "well_high_supervision_conflicts.csv" if not high_conflicts.empty else None
    if high_conflicts_path is not None:
        high_conflicts.to_csv(high_conflicts_path, index=False, encoding="utf-8-sig")

    anchor_arrays, _anchor_trace_summary = aggregate_trace_arrays(
        anchor_points,
        samples,
        target_col="ginn_target_ai",
        value_cols=["ginn_target_log_ai"],
        include_anchor_only=False,
    )
    anchor_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": Path(__file__).name,
        "artifact_family": "well_constraints_time",
        "artifact_role": "log_ai_anchor",
        "anchor_target_band": "lowpass_reference_to_ginn_cutoff",
        "include_deviated": include_deviated_anchor,
        "conflict_strategy": "weighted_average",
        "frequency_bands": {
            "lfm_cutoff_hz": bands_cfg.lfm_cutoff_hz,
            "ginn_cutoff_hz": bands_cfg.ginn_cutoff_hz,
            "reference_cutoff_hz": bands_cfg.reference_cutoff_hz,
            "filter_order": bands_cfg.filter_order,
            "buffer_seconds": bands_cfg.buffer_seconds,
            "buffer_mode": bands_cfg.buffer_mode,
        },
        "point_table": repo_relative_path(point_path, root=REPO_ROOT),
        "conflicts": repo_relative_path(conflicts_path, root=REPO_ROOT) if conflicts_path is not None else None,
    }
    anchor_bundle = build_log_ai_anchor_bundle(
        sample_domain="time",
        sample_unit="s",
        samples=np.asarray(samples, dtype=np.float32),
        flat_indices=anchor_arrays["flat_indices"],
        target_ai=anchor_arrays["target_ai"],
        anchor_mask=anchor_arrays["mask"],
        anchor_weight=anchor_arrays["weight"],
        anchor_names=anchor_arrays["well_names"],
        anchor_types=np.array(["well"] * int(anchor_arrays["flat_indices"].size)),
        inline=anchor_arrays["inline"],
        xline=anchor_arrays["xline"],
        metadata=anchor_metadata,
    )
    validate_log_ai_anchor(anchor_bundle, sample_domain="time", n_sample=int(samples.size))
    anchor_path = output_dir / "log_ai_anchor_time.npz"
    save_log_ai_anchor_npz(anchor_path, anchor_bundle)

    supervision_metadata = {
        "created_at_utc": anchor_metadata["created_at_utc"],
        "source_script": Path(__file__).name,
        "artifact_family": "well_constraints_time",
        "artifact_role": "well_high_supervision",
        "schema_compatibility": "enhance_residual_supervision_v2",
        "sample_domain": "time",
        "include_deviated": include_deviated_high,
        "excluded_sources": [] if include_deviated_high else ["deviated_trajectory"],
        "native_sample_semantics": "TWT seconds; native_samples currently matches samples in time-domain step 06.",
        "base_ai_fields_available": False,
        "base_ai_fields_note": "This package contains well high-frequency supervision only. Read base AI from LFM or GINN outputs.",
        "frequency_bands": anchor_metadata["frequency_bands"],
        "point_table": repo_relative_path(point_path, root=REPO_ROOT),
        "high_conflicts": (
            repo_relative_path(high_conflicts_path, root=REPO_ROOT) if high_conflicts_path is not None else None
        ),
    }
    supervision_bundle = _make_high_supervision_bundle(
        point_df=high_points,
        samples=samples,
        metadata=supervision_metadata,
    )
    supervision_path = output_dir / "well_high_supervision_time.npz"
    save_well_high_supervision_npz(supervision_path, supervision_bundle)

    global_stats, layer_stats_df, shrinkage = layer_shrinkage_stats(high_points, sample_step_s=sample_step_s)
    global_stats_path = output_dir / "well_high_stats_global.json"
    layer_stats_path = output_dir / "well_high_stats_by_layer.csv"
    shrinkage_path = output_dir / "well_high_stats_shrinkage.json"
    write_json(global_stats_path, global_stats)
    layer_stats_df.to_csv(layer_stats_path, index=False, encoding="utf-8-sig")
    write_json(shrinkage_path, shrinkage)

    lfm_control_df = _make_lfm_control_table(point_df)
    lfm_control_path = output_dir / "lfm_control_points.csv"
    lfm_control_df.to_csv(lfm_control_path, index=False, encoding="utf-8-sig")

    outputs = {
        "well_constraint_points": repo_relative_path(point_path, root=REPO_ROOT),
        "well_constraint_qc": repo_relative_path(qc_path, root=REPO_ROOT),
        "lfm_control_points": repo_relative_path(lfm_control_path, root=REPO_ROOT),
        "log_ai_anchor_time": repo_relative_path(anchor_path, root=REPO_ROOT),
        "well_high_supervision_time": repo_relative_path(supervision_path, root=REPO_ROOT),
        "well_high_stats_global": repo_relative_path(global_stats_path, root=REPO_ROOT),
        "well_high_stats_by_layer": repo_relative_path(layer_stats_path, root=REPO_ROOT),
        "well_high_stats_shrinkage": repo_relative_path(shrinkage_path, root=REPO_ROOT),
        "ginn_cutoff_diagnostics": repo_relative_path(split_diag_path, root=REPO_ROOT),
        "ginn_cutoff_cluster_aggregate": repo_relative_path(split_cluster_path, root=REPO_ROOT),
        "ginn_cutoff_aggregate": repo_relative_path(split_aggregate_path, root=REPO_ROOT),
        "ginn_cutoff_sweep_figure": repo_relative_path(split_diag_figure_path, root=REPO_ROOT),
        "ginn_cutoff_well_sweep_figures": split_well_figure_paths,
        "frequency_band_response_figure": repo_relative_path(frequency_response_path, root=REPO_ROOT),
    }
    if conflicts_path is not None:
        outputs["well_anchor_conflicts"] = repo_relative_path(conflicts_path, root=REPO_ROOT)
    if high_conflicts_path is not None:
        outputs["well_high_supervision_conflicts"] = repo_relative_path(high_conflicts_path, root=REPO_ROOT)

    run_summary = {
        "created_at_utc": anchor_metadata["created_at_utc"],
        "source_script": Path(__file__).name,
        "artifact_family": "well_constraints_time",
        "source_dirs": {key: repo_relative_path(value, root=REPO_ROOT) for key, value in source_dirs.items()},
        "seismic_file": repo_relative_path(seismic_file, root=REPO_ROOT),
        "seismic_type": seismic_type,
        "horizons": horizon_metadata,
        "config": script_cfg,
        "frequency_bands": {
            "lfm_cutoff_hz": bands_cfg.lfm_cutoff_hz,
            "ginn_cutoff_hz": bands_cfg.ginn_cutoff_hz,
            "reference_cutoff_hz": bands_cfg.reference_cutoff_hz,
            "filter_order": bands_cfg.filter_order,
            "buffer_seconds": bands_cfg.buffer_seconds,
            "buffer_mode": bands_cfg.buffer_mode,
            "diagnostics": repo_relative_path(split_diag_path, root=REPO_ROOT),
            "cluster_aggregate": repo_relative_path(split_cluster_path, root=REPO_ROOT),
            "aggregate": repo_relative_path(split_aggregate_path, root=REPO_ROOT),
            "diagnostic_figure": repo_relative_path(split_diag_figure_path, root=REPO_ROOT),
            "selection": split_selection,
        },
        "conflicts": {
            "strategy": "weighted_average",
            "point_conflict_count": int(len(conflicts)),
            "report": repo_relative_path(conflicts_path, root=REPO_ROOT) if conflicts_path is not None else None,
            "high_supervision_conflict_count": int(len(high_conflicts)),
            "high_supervision_report": (
                repo_relative_path(high_conflicts_path, root=REPO_ROOT) if high_conflicts_path is not None else None
            ),
            "high_supervision_include_deviated": include_deviated_high,
        },
        "counts": {
            "selected_wells": int(qc_df["control_point_count"].gt(0).sum()),
            "point_count": int(len(point_df)),
            "deviated_point_count": int(point_df["source"].astype(str).eq("deviated_trajectory").sum()),
            "high_supervision_point_count": int(len(high_points)),
            "high_supervision_deviated_point_count": int(
                high_points["source"].astype(str).eq("deviated_trajectory").sum()
            )
            if not high_points.empty
            else 0,
            "excluded_deviated_high_supervision_point_count": int(
                point_df["source"].astype(str).eq("deviated_trajectory").sum()
            )
            if not include_deviated_high
            else 0,
            "anchor_trace_count": int(anchor_bundle.n_anchors),
            "lfm_control_point_count": int(len(lfm_control_df)),
        },
        "outputs": outputs,
    }
    write_json(output_dir / "run_summary.json", run_summary)

    print("=== Well Constraints ===")
    print(f"Output: {output_dir}")
    print(f"Selected wells: {run_summary['counts']['selected_wells']}")
    print(f"Point facts: {len(point_df)}")
    print(f"Anchor traces: {anchor_bundle.n_anchors}")
    print(f"LFM control points: {len(lfm_control_df)}")


if __name__ == "__main__":
    main()
