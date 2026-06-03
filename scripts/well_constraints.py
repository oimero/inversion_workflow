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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sanitize_filename, write_json
from cup.well.assets import normalize_well_name
from cup.well.constraints import (
    FrequencySplitConfig,
    aggregate_lfm_control_points,
    aggregate_trace_arrays,
    apply_frequency_split,
    build_deviated_point_facts,
    build_point_conflict_report,
    build_vertical_point_facts,
    confidence_from_corr,
    high_frequency_stats,
    layer_shrinkage_stats,
    lowpass_values_on_twt,
)
from cup.well.las import load_vp_rho_logset_from_standard_las
from cup.well.td import load_workflow_time_depth_table_csv
from cup.well.tie import load_saved_seismic_trace_csv
from cup.well.wavelet import load_wavelet_csv
# Step 06 depends on downstream bundle schemas only. Keep GINN/enhance training
# logic out of this script; move schemas to a neutral package when they stabilize.
from enhance.prior import WellResolutionPriorBundle, save_well_resolution_prior_npz, validate_well_resolution_prior
from ginn.anchor import build_log_ai_anchor_bundle, save_log_ai_anchor_npz, validate_log_ai_anchor
from wtie.modeling.modeling import ConvModeler
from wtie.processing import grid


DEFAULT_CONFIG: dict[str, Any] = {
    "source_runs": {
        "mode": "latest",
        "well_auto_tie_dir": None,
        "wavelet_generation_dir": None,
    },
    "seismic": {"file": "raw/obn-clipped-240-912-872-1544.zgy", "type": "zgy"},
    "target_interval": {"horizons": ["interpre/H3-1", "interpre/H7-1"], "twt_unit": "auto"},
    "control_wells": {
        "min_batch_corr": 0.35,
        "max_batch_nmae": None,
        "include_wells": None,
        "exclude_wells": [],
    },
    "frequency_split": {
        "mode": "diagnose",
        "manual_cutoff_hz": None,
        "filter_order": 6,
        "candidate_cutoff_hz": [6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0],
        "selection_corr_tolerance": 0.02,
        "selection_nmae_tolerance": 0.03,
        "buffer_seconds": None,
        "buffer_mode": "reflect",
        "qc_enabled": True,
        "qc_envelope_window_samples": 31,
    },
    "anchor": {"include_deviated": False, "min_points_per_trace": 2},
    "conflicts": {"strategy": "weighted_average"},
    "weights": {"mode": "corr", "corr_floor": 0.3, "corr_span": 0.4, "corr_min_weight": 0.6},
    "lfm_controls": {"n_slices": 20, "min_control_samples_per_well": 16},
    "motif": {"write_manifest": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = {key: (value.copy() if isinstance(value, dict) else value) for key, value in base.items()}
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _latest_run(output_root: Path, prefix: str, required_file: str) -> Path:
    candidates = [p for p in output_root.glob(f"{prefix}_*") if p.is_dir() and (p / required_file).exists()]
    if not candidates:
        raise FileNotFoundError(f"No run found under {output_root} for {prefix}_* containing {required_file}")
    return sorted(candidates, key=lambda p: (p.stat().st_mtime, p.name))[-1]


def _resolve_source_dirs(script_cfg: dict[str, Any], output_root: Path) -> dict[str, Path]:
    source_cfg = dict(script_cfg.get("source_runs") or {})
    mode = str(source_cfg.get("mode", "latest")).strip().lower()
    if mode != "latest":
        raise ValueError(f"well_constraints.source_runs.mode only supports 'latest', got {mode!r}.")
    auto_value = source_cfg.get("well_auto_tie_dir")
    wavelet_value = source_cfg.get("wavelet_generation_dir")
    auto_dir = (
        _latest_run(output_root, "well_auto_tie", "well_tie_metrics.csv")
        if auto_value in {None, ""}
        else resolve_relative_path(auto_value, root=REPO_ROOT)
    )
    wavelet_dir = (
        _latest_run(output_root, "wavelet_generation", "batch_synthetic_metrics.csv")
        if wavelet_value in {None, ""}
        else resolve_relative_path(wavelet_value, root=REPO_ROOT)
    )
    return {"well_auto_tie_dir": auto_dir, "wavelet_generation_dir": wavelet_dir}


def _resolve_artifact_path(value: Any, *, run_dir: Path) -> Path | None:
    text = "" if value is None else str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, run_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_seismic_trace_file(metric: pd.Series, *, run_dir: Path, well_name: str) -> Path | None:
    direct = _resolve_artifact_path(metric.get("seismic_trace_file"), run_dir=run_dir)
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
    well_col = "eval_well" if "eval_well" in batch_df.columns else "well_name"
    if well_col not in batch_df.columns:
        raise ValueError("batch_synthetic_metrics.csv must contain eval_well or well_name.")
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


def _resolve_segy_options(cfg: dict[str, Any]) -> dict[str, int] | None:
    from cup.seismic.survey import segy_options_from_config

    if "segy" in cfg:
        return segy_options_from_config(dict(cfg["segy"])) or None
    return None


def _open_survey(script_cfg: dict[str, Any], cfg: dict[str, Any], data_root: Path) -> tuple[Any, Path, str]:
    from cup.seismic.survey import open_survey

    seismic_cfg = dict(script_cfg.get("seismic") or {})
    seismic_file = resolve_relative_path(seismic_cfg.get("file"), root=data_root)
    seismic_type = str(seismic_cfg.get("type", "segy"))
    survey = open_survey(seismic_file, seismic_type=seismic_type, segy_options=_resolve_segy_options(cfg))
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


def _resolve_frequency_split(
    point_df: pd.DataFrame,
    *,
    metrics_df: pd.DataFrame,
    auto_dir: Path,
    wavelet_dir: Path,
    script_cfg: dict[str, Any],
) -> tuple[FrequencySplitConfig, pd.DataFrame, pd.DataFrame, dict[str, Any], str]:
    split_cfg = dict(script_cfg.get("frequency_split") or {})
    mode = str(split_cfg.get("mode", "diagnose")).strip().lower()
    order = int(split_cfg.get("filter_order", 6))
    buffer_seconds = split_cfg.get("buffer_seconds")
    buffer_seconds = None if buffer_seconds is None else float(buffer_seconds)
    buffer_mode = str(split_cfg.get("buffer_mode", "reflect"))
    if mode == "manual":
        cutoff = split_cfg.get("manual_cutoff_hz")
        if cutoff is None:
            raise ValueError("well_constraints.frequency_split.manual_cutoff_hz is required for mode=manual.")
        cfg = FrequencySplitConfig(float(cutoff), filter_order=order, buffer_seconds=buffer_seconds, buffer_mode=buffer_mode)
        diag = pd.DataFrame([{"cutoff_hz": float(cutoff), "mode": "manual", "status": "manual"}])
        aggregate = pd.DataFrame([{"cutoff_hz": float(cutoff), "mode": "manual"}])
        selection = {"selected_cutoff_hz": float(cutoff), "reason": "manual"}
        return cfg, diag, aggregate, selection, "manual"
    if mode != "diagnose":
        raise ValueError(f"Unsupported frequency split mode: {mode!r}")
    candidates = [float(v) for v in split_cfg.get("candidate_cutoff_hz", [6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0])]
    return (
        *_diagnose_frequency_split_by_forward_modeling(
            point_df,
            metrics_df=metrics_df,
            auto_dir=auto_dir,
            wavelet_dir=wavelet_dir,
            candidate_cutoff_hz=candidates,
            filter_order=order,
            buffer_seconds=buffer_seconds,
            buffer_mode=buffer_mode,
            corr_tolerance=float(split_cfg.get("selection_corr_tolerance", 0.02)),
            nmae_tolerance=float(split_cfg.get("selection_nmae_tolerance", 0.03)),
        ),
        "diagnose",
    )


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
    fallback = wavelet_dir / "global_wavelet_201ms.csv"
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
    return corr, nmae, scale, n_eval


def _diagnose_frequency_split_by_forward_modeling(
    point_df: pd.DataFrame,
    *,
    metrics_df: pd.DataFrame,
    auto_dir: Path,
    wavelet_dir: Path,
    candidate_cutoff_hz: list[float],
    filter_order: int,
    buffer_seconds: float | None,
    buffer_mode: str,
    corr_tolerance: float,
    nmae_tolerance: float,
) -> tuple[FrequencySplitConfig, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if point_df.empty:
        raise ValueError("Cannot diagnose frequency split from an empty point table.")
    wavelet_time_s, wavelet_amplitude, wavelet_path = _load_selected_wavelet(wavelet_dir)
    wavelet_dt = float(np.median(np.diff(wavelet_time_s)))
    wavelet = grid.Wavelet(wavelet_amplitude, wavelet_time_s, name=wavelet_path.stem)
    modeler = ConvModeler()
    selected_keys = {normalize_well_name(name) for name in point_df["well_name"].astype(str).unique()}
    point_by_key = {normalize_well_name(str(name)): group.copy() for name, group in point_df.groupby("well_name", sort=False)}
    rows: list[dict[str, Any]] = []

    for _, metric in metrics_df.iterrows():
        well_name = str(metric.get("well_name", "")).strip()
        key = normalize_well_name(well_name)
        if key not in selected_keys:
            continue
        route = str(metric.get("route", ""))
        las_file = _resolve_artifact_path(metric.get("filtered_las_file"), run_dir=auto_dir)
        tdt_file = _resolve_artifact_path(metric.get("optimized_tdt_file"), run_dir=auto_dir)
        seismic_file = _resolve_seismic_trace_file(metric, run_dir=auto_dir, well_name=well_name)
        if las_file is None or tdt_file is None or seismic_file is None:
            for cutoff in candidate_cutoff_hz:
                rows.append({"well_name": well_name, "route": route, "cutoff_hz": cutoff, "status": "failed", "reason": "missing_las_tdt_or_seismic"})
            continue
        try:
            logset = load_vp_rho_logset_from_standard_las(las_file)
            table = load_workflow_time_depth_table_csv(tdt_file)
            seismic = load_saved_seismic_trace_csv(seismic_file)
            seismic_basis = np.asarray(seismic.basis, dtype=np.float64)
            seismic_dt = float(np.median(np.diff(seismic_basis)))
            if not np.isclose(wavelet_dt, seismic_dt, rtol=1e-5, atol=1e-9):
                raise ValueError(f"wavelet_dt_mismatch:{wavelet_dt:g}!={seismic_dt:g}")
            ai_twt = grid.convert_log_from_md_to_twt(logset.AI, table, None, seismic_dt)
            ai_twt_basis = np.asarray(ai_twt.basis, dtype=np.float64)
            ai_values = np.asarray(ai_twt.values, dtype=np.float64)
            valid_ai = np.isfinite(ai_values) & (ai_values > 0.0)
            if int(np.count_nonzero(valid_ai)) < 8:
                raise ValueError("too_few_finite_ai_samples")
            log_ai_values = np.full(ai_values.shape, np.nan, dtype=np.float64)
            log_ai_values[valid_ai] = np.log(ai_values[valid_ai])
            eval_group = point_by_key[key]
            eval_twt = eval_group["twt_s"].to_numpy(dtype=np.float64)
            eval_mask = np.isin(np.round(seismic_basis, 9), np.round(eval_twt, 9))
            if int(np.count_nonzero(eval_mask)) < 8:
                eval_mask = (seismic_basis >= float(np.nanmin(eval_twt)) - 0.5 * seismic_dt) & (
                    seismic_basis <= float(np.nanmax(eval_twt)) + 0.5 * seismic_dt
                )
            for cutoff in candidate_cutoff_hz:
                low_log_ai = lowpass_values_on_twt(
                    ai_twt_basis,
                    log_ai_values,
                    dt_s=seismic_dt,
                    cutoff_hz=float(cutoff),
                    order=int(filter_order),
                    buffer_seconds=buffer_seconds,
                    buffer_mode=buffer_mode,
                )
                low_ai_on_seis = np.exp(np.interp(seismic_basis, ai_twt_basis, low_log_ai, left=np.nan, right=np.nan))
                reflectivity = grid.Reflectivity(_reflectivity_from_ai(low_ai_on_seis), seismic_basis)
                corr, nmae, scale, n_eval = _masked_scaled_synthetic_metrics(
                    modeler=modeler,
                    wavelet=wavelet,
                    reflectivity=reflectivity,
                    seismic=seismic,
                    eval_mask=eval_mask,
                )
                rows.append(
                    {
                        "well_name": well_name,
                        "route": route,
                        "cutoff_hz": float(cutoff),
                        "status": "ok",
                        "corr": corr,
                        "nmae": nmae,
                        "scale": scale,
                        "n_eval_samples": n_eval,
                        "wavelet_file": repo_relative_path(wavelet_path, root=REPO_ROOT),
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
        raise ValueError("No successful forward-modeled frequency split diagnostics.")
    aggregate = (
        ok.groupby("cutoff_hz")
        .agg(
            n_wells=("well_name", "nunique"),
            median_corr=("corr", "median"),
            mean_corr=("corr", "mean"),
            p25_corr=("corr", lambda x: float(np.nanpercentile(x, 25))),
            p75_corr=("corr", lambda x: float(np.nanpercentile(x, 75))),
            median_nmae=("nmae", "median"),
            mean_nmae=("nmae", "mean"),
            p25_nmae=("nmae", lambda x: float(np.nanpercentile(x, 25))),
            p75_nmae=("nmae", lambda x: float(np.nanpercentile(x, 75))),
            median_scale=("scale", "median"),
            median_n_eval_samples=("n_eval_samples", "median"),
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
        reason = "Selected the lowest Hz on the near-best waveform-fit plateau to keep the low-frequency branch conservative."
    selected = float(chosen["cutoff_hz"])
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
    }
    return (
        FrequencySplitConfig(
            cutoff_hz=selected,
            filter_order=int(filter_order),
            buffer_seconds=buffer_seconds,
            buffer_mode=buffer_mode,
        ),
        diag_df,
        aggregate,
        selection,
    )


def _save_frequency_qc(point_df: pd.DataFrame, qc_dir: Path, *, envelope_window: int) -> list[dict[str, Any]]:
    qc_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = qc_dir / "figures"
    trace_dir = qc_dir / "traces"
    fig_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for well_name, group in point_df.groupby("well_name", sort=False):
        group = group.sort_values("twt_s")
        safe = sanitize_filename(str(well_name))
        trace_path = trace_dir / f"frequency_split_{safe}.csv"
        group[
            [
                "well_name",
                "twt_s",
                "ai_full",
                "log_ai_full",
                "well_low_ai",
                "well_low_log_ai",
                "well_high_log_ai",
                "zone_name",
                "weight",
            ]
        ].to_csv(trace_path, index=False, encoding="utf-8-sig")
        fig_path = fig_dir / f"frequency_split_{safe}.png"
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
        axes[0].plot(group["twt_s"], group["log_ai_full"], label="full log-AI", linewidth=1.0)
        axes[0].plot(group["twt_s"], group["well_low_log_ai"], label="low log-AI", linewidth=1.0)
        axes[0].legend(loc="best")
        high = group["well_high_log_ai"].to_numpy(dtype=float)
        axes[1].plot(group["twt_s"], high, label="high residual", linewidth=1.0)
        if high.size and int(envelope_window) > 1:
            window = min(int(envelope_window), max(1, high.size))
            kernel = np.ones(window, dtype=float) / float(window)
            envelope = np.convolve(np.abs(high), kernel, mode="same")
            axes[1].plot(group["twt_s"], envelope, color="tab:red", alpha=0.8, label="abs envelope")
            axes[1].plot(group["twt_s"], -envelope, color="tab:red", alpha=0.8)
        axes[1].legend(loc="best")
        axes[1].set_xlabel("TWT (s)")
        fig.suptitle(str(well_name))
        fig.savefig(fig_path, dpi=180)
        plt.close(fig)
        rows.append(
            {
                "well_name": str(well_name),
                "frequency_split_qc_trace_path": repo_relative_path(trace_path, root=REPO_ROOT),
                "frequency_split_qc_figure_path": repo_relative_path(fig_path, root=REPO_ROOT),
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

    fig.suptitle("Frequency split forward-model cutoff sweep")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def _plot_frequency_split_well_sweeps(diag_df: pd.DataFrame, figure_dir: Path) -> dict[str, str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    ok = diag_df.loc[diag_df.get("status", "").eq("ok")].copy() if not diag_df.empty else pd.DataFrame()
    for well_name, group in ok.groupby("well_name", sort=False):
        plot_df = group.sort_values("cutoff_hz")
        path = figure_dir / f"frequency_split_sweep_{sanitize_filename(str(well_name))}.png"
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


def _make_high_supervision_bundle(
    *,
    point_df: pd.DataFrame,
    samples: np.ndarray,
    metadata: dict[str, Any],
) -> WellResolutionPriorBundle:
    arrays, _summary_df = aggregate_trace_arrays(
        point_df,
        samples,
        target_col="well_low_ai",
        value_cols=["log_ai_full", "well_low_ai", "well_low_log_ai", "well_high_log_ai"],
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
    n_trace = int(arrays["flat_indices"].size)
    n_sample = int(samples.size)
    lfm_zero = np.zeros((n_trace, n_sample), dtype=np.float32)
    mask = arrays["mask"].astype(bool)
    well_ai = np.zeros((n_trace, n_sample), dtype=np.float32)
    well_ai[mask] = np.exp(arrays["log_ai_full"][mask]).astype(np.float32)
    residual = arrays["well_high_log_ai"].astype(np.float32)
    bundle = WellResolutionPriorBundle(
        sample_domain="time",
        sample_unit="s",
        samples=np.asarray(samples, dtype=np.float32),
        flat_indices=arrays["flat_indices"],
        well_log_ai=arrays["log_ai_full"].astype(np.float32),
        lfm_log_ai=lfm_zero,
        residual_log_ai=residual,
        well_ai=well_ai,
        well_low_ai=arrays["well_low_ai"].astype(np.float32),
        well_low_log_ai=arrays["well_low_log_ai"].astype(np.float32),
        well_high_log_ai=residual,
        lfm_ai=np.zeros_like(well_ai, dtype=np.float32),
        well_mask=mask,
        well_weight=arrays["weight"].astype(np.float32),
        highres_depth=np.tile(np.asarray(samples, dtype=np.float32), (n_trace, 1)),
        highres_well_ai=well_ai.copy(),
        highres_well_log_ai=arrays["log_ai_full"].astype(np.float32),
        highres_well_low_ai=arrays["well_low_ai"].astype(np.float32),
        highres_well_low_log_ai=arrays["well_low_log_ai"].astype(np.float32),
        highres_well_high_log_ai=residual.copy(),
        highres_lfm_log_ai=lfm_zero.copy(),
        highres_residual_log_ai=residual.copy(),
        highres_well_mask=mask.copy(),
        well_names=arrays["well_names"],
        inline=arrays["inline"],
        xline=arrays["xline"],
        summary=high_frequency_stats(point_df, sample_step_s=float(np.median(np.diff(samples))) if samples.size > 1 else None),
        metadata=metadata,
    )
    validate_well_resolution_prior(bundle, sample_domain="time", n_sample=n_sample)
    return bundle


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = _deep_update(DEFAULT_CONFIG, dict(cfg.get("well_constraints") or {}))
    data_root = REPO_ROOT / str(cfg.get("data_root", "data"))
    output_root = REPO_ROOT / str(cfg.get("output_root", "scripts/output"))
    source_dirs = _resolve_source_dirs(script_cfg, output_root)

    if str(script_cfg.get("conflicts", {}).get("strategy", "weighted_average")) != "weighted_average":
        raise ValueError("well_constraints.conflicts.strategy currently supports only 'weighted_average'.")

    if args.output_dir is None:
        output_dir = output_root / f"well_constraints_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    target_qc_dir = output_dir / "target_layer_qc"

    survey, seismic_file, seismic_type = _open_survey(script_cfg, cfg, data_root)
    geometry = survey.describe_geometry(domain="time")
    if str(geometry.get("sample_domain")).lower() != "time" or str(geometry.get("sample_unit")).lower() != "s":
        raise ValueError(f"Expected time-domain seismic geometry in seconds, got {geometry}")
    samples = survey.sample_axis("time").values.astype(np.float64)
    sample_step_s = float(np.median(np.diff(samples))) if samples.size > 1 else None
    target_layer, horizon_metadata = _build_target_layer(script_cfg, geometry, target_qc_dir, data_root)

    auto_dir = source_dirs["well_auto_tie_dir"]
    wavelet_dir = source_dirs["wavelet_generation_dir"]
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

    point_frames: list[pd.DataFrame] = []
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

        las_file = _resolve_artifact_path(metric.get("filtered_las_file"), run_dir=auto_dir)
        tdt_file = _resolve_artifact_path(metric.get("optimized_tdt_file"), run_dir=auto_dir)
        if las_file is None or not las_file.exists():
            reasons.append("missing_filtered_las_file")
        if not route.startswith("deviated") and (tdt_file is None or not tdt_file.exists()):
            reasons.append("missing_optimized_tdt_file")

        point_df = pd.DataFrame()
        point_qc: dict[str, Any] = {}
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
                if route.startswith("deviated"):
                    trace_file = _resolve_artifact_path(metric.get("optimized_trace_sample_plan_file"), run_dir=auto_dir)
                    if trace_file is None:
                        trace_file = auto_dir / "trace_sample_plan" / f"optimized_trace_sample_plan_{sanitize_filename(well_name)}.csv"
                    if not trace_file.exists():
                        raise FileNotFoundError("missing_optimized_trace_sample_plan")
                    point_df, point_qc = build_deviated_point_facts(
                        well_name=well_name,
                        route=route,
                        las_file=las_file,
                        trace_plan_file=trace_file,
                        target_layer=target_layer,
                        survey=survey,
                        weight=weight,
                        batch_corr=batch_corr,
                        batch_nmae=batch_nmae,
                        sample_step_s=sample_step_s,
                        anchor_eligible=include_deviated_anchor,
                    )
                else:
                    plan = plan_by_key.get(key)
                    if plan is None:
                        raise ValueError("missing_well_tie_plan_row")
                    surface_x = _as_optional_float(plan.get("surface_x"))
                    surface_y = _as_optional_float(plan.get("surface_y"))
                    if surface_x is None or surface_y is None:
                        raise ValueError("missing_surface_xy")
                    point_df, point_qc = build_vertical_point_facts(
                        well_name=well_name,
                        route=route,
                        las_file=las_file,
                        tdt_file=tdt_file,
                        surface_x=surface_x,
                        surface_y=surface_y,
                        target_layer=target_layer,
                        survey=survey,
                        samples=samples,
                        weight=weight,
                        batch_corr=batch_corr,
                        batch_nmae=batch_nmae,
                        anchor_eligible=True,
                    )
                if int(point_qc.get("valid_points", 0)) < min_points:
                    status = "rejected"
                    reasons.append("too_few_control_samples")
                else:
                    point_frames.append(point_df)
            except Exception as exc:
                status = "rejected" if str(exc) == "missing_optimized_trace_sample_plan" else "failed"
                reasons.append(str(exc) or type(exc).__name__)

        qc_rows.append(
            {
                "well_name": well_name,
                "status": status,
                "route": route,
                "batch_corr": batch_corr,
                "batch_nmae": batch_nmae,
                "control_point_count": int(len(point_df)) if status == "selected" else 0,
                "invalid_point_count": point_qc.get("invalid_point_count"),
                "invalid_point_fraction": (
                    float(point_qc.get("invalid_point_count", 0) / point_qc.get("attempted_samples", 1))
                    if point_qc.get("attempted_samples")
                    else None
                ),
                "unique_trace_count": point_qc.get("unique_trace_count", 0),
                "reasons": ";".join(dict.fromkeys(reasons)),
            }
        )

    if not point_frames:
        qc_path = output_dir / "well_high_supervision_qc.csv"
        pd.DataFrame(qc_rows).to_csv(qc_path, index=False, encoding="utf-8-sig")
        raise ValueError(f"No selected well constraint points. Inspect {qc_path}.")

    raw_points = pd.concat(point_frames, ignore_index=True)
    split_cfg, split_diag_df, split_aggregate_df, split_selection, split_source = _resolve_frequency_split(
        raw_points,
        metrics_df=metrics_df,
        auto_dir=auto_dir,
        wavelet_dir=wavelet_dir,
        script_cfg=script_cfg,
    )
    point_df = apply_frequency_split(raw_points, split_cfg)

    qc_split_rows: list[dict[str, Any]] = []
    freq_cfg = dict(script_cfg.get("frequency_split") or {})
    if bool(freq_cfg.get("qc_enabled", True)):
        qc_split_rows = _save_frequency_qc(
            point_df,
            output_dir / "frequency_split_qc",
            envelope_window=int(freq_cfg.get("qc_envelope_window_samples", 31)),
        )
    qc_split_by_well = {row["well_name"]: row for row in qc_split_rows}
    qc_df = pd.DataFrame(qc_rows)
    for key in ["frequency_split_qc_trace_path", "frequency_split_qc_figure_path"]:
        qc_df[key] = [qc_split_by_well.get(str(w), {}).get(key) for w in qc_df["well_name"]]

    point_path = output_dir / "well_constraint_points.csv"
    point_df.to_csv(point_path, index=False, encoding="utf-8-sig")
    qc_path = output_dir / "well_high_supervision_qc.csv"
    qc_df.to_csv(qc_path, index=False, encoding="utf-8-sig")

    anchor_points = point_df.loc[point_df["anchor_eligible"].astype(bool)].copy()
    anchor_points_path = output_dir / "well_anchor_points.csv"
    anchor_points.to_csv(anchor_points_path, index=False, encoding="utf-8-sig")
    conflicts = build_point_conflict_report(anchor_points, value_col="well_low_log_ai")
    conflicts_path = output_dir / "well_anchor_conflicts.csv"
    conflicts.to_csv(conflicts_path, index=False, encoding="utf-8-sig")

    high_conflicts = build_point_conflict_report(point_df, value_col="well_high_log_ai")
    high_conflicts_path = output_dir / "well_high_supervision_conflicts.csv"
    high_conflicts.to_csv(high_conflicts_path, index=False, encoding="utf-8-sig")

    anchor_arrays, anchor_trace_summary = aggregate_trace_arrays(
        point_df,
        samples,
        target_col="well_low_ai",
        value_cols=["well_low_log_ai"],
        include_anchor_only=True,
    )
    anchor_trace_summary_path = output_dir / "well_anchor_trace_summary.csv"
    anchor_trace_summary.to_csv(anchor_trace_summary_path, index=False, encoding="utf-8-sig")
    anchor_metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": Path(__file__).name,
        "artifact_family": "well_constraints_time",
        "artifact_role": "log_ai_anchor",
        "anchor_target_band": "lowpass_log_ai",
        "include_deviated": include_deviated_anchor,
        "conflict_strategy": "weighted_average",
        "frequency_split": {
            "mode": split_source,
            "cutoff_hz": split_cfg.cutoff_hz,
            "filter_order": split_cfg.filter_order,
            "buffer_seconds": split_cfg.buffer_seconds,
            "buffer_mode": split_cfg.buffer_mode,
        },
        "point_table": repo_relative_path(point_path, root=REPO_ROOT),
        "conflicts": repo_relative_path(conflicts_path, root=REPO_ROOT),
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

    prior_metadata = {
        "created_at_utc": anchor_metadata["created_at_utc"],
        "source_script": Path(__file__).name,
        "artifact_family": "well_constraints_time",
        "artifact_role": "well_high_supervision",
        "schema_compatibility": "ginn_well_resolution_prior_v3",
        "sample_domain": "time",
        "highres_depth_field_semantics": "TWT seconds, kept under legacy field name for enhance.prior compatibility",
        "lfm_fields_available": False,
        "lfm_fields_note": "Step 06 runs before LFM. lfm_log_ai, lfm_ai, and highres_lfm_log_ai are zero placeholders and must not be used as base AI.",
        "frequency_split": anchor_metadata["frequency_split"],
        "point_table": repo_relative_path(point_path, root=REPO_ROOT),
        "high_conflicts": repo_relative_path(high_conflicts_path, root=REPO_ROOT),
    }
    prior_bundle = _make_high_supervision_bundle(point_df=point_df, samples=samples, metadata=prior_metadata)
    prior_path = output_dir / "well_high_supervision_time.npz"
    save_well_resolution_prior_npz(prior_path, prior_bundle)

    global_stats, layer_stats_df, shrinkage = layer_shrinkage_stats(point_df, sample_step_s=sample_step_s)
    global_stats_path = output_dir / "well_high_stats_global.json"
    layer_stats_path = output_dir / "well_high_stats_by_layer.csv"
    shrinkage_path = output_dir / "well_high_stats_shrinkage.json"
    write_json(global_stats_path, global_stats)
    layer_stats_df.to_csv(layer_stats_path, index=False, encoding="utf-8-sig")
    write_json(shrinkage_path, shrinkage)

    motif_path = output_dir / "well_high_motif_manifest.csv"
    pd.DataFrame(
        columns=["motif_id", "well_name", "zone_name", "start_twt_s", "end_twt_s", "quality_tag", "reason"]
    ).to_csv(motif_path, index=False, encoding="utf-8-sig")

    lfm_control_df, lfm_aggregated_count = aggregate_lfm_control_points(
        point_df,
        n_slices=int(script_cfg["lfm_controls"]["n_slices"]),
    )
    lfm_control_path = output_dir / "lfm_layer_control_points.csv"
    lfm_control_df.to_csv(lfm_control_path, index=False, encoding="utf-8-sig")
    lfm_qc = qc_df.rename(columns={"status": "well_constraint_status"}).copy()
    lfm_qc["status"] = np.where(lfm_qc["well_constraint_status"].eq("selected"), "selected", lfm_qc["well_constraint_status"])
    lfm_qc["raw_point_count"] = lfm_qc["control_point_count"]
    lfm_qc["aggregated_point_count"] = [
        int(lfm_control_df.loc[lfm_control_df["well_name"].astype(str).eq(str(w))].shape[0]) for w in lfm_qc["well_name"]
    ]
    lfm_qc_path = output_dir / "lfm_control_qc.csv"
    lfm_qc.to_csv(lfm_qc_path, index=False, encoding="utf-8-sig")

    split_diag_path = output_dir / "frequency_split_diagnostics.csv"
    split_diag_df.to_csv(split_diag_path, index=False, encoding="utf-8-sig")
    split_aggregate_path = output_dir / "frequency_split_aggregate.csv"
    split_aggregate_df.to_csv(split_aggregate_path, index=False, encoding="utf-8-sig")
    split_diag_figure_path = output_dir / "figures" / "frequency_split_cutoff_sweep.png"
    _plot_frequency_split_aggregate(
        split_aggregate_df,
        figure_path=split_diag_figure_path,
        selected_cutoff_hz=split_selection.get("selected_cutoff_hz"),
        best_cutoff_hz=split_selection.get("best_corr_cutoff_hz"),
    )
    split_well_figure_paths = _plot_frequency_split_well_sweeps(
        split_diag_df,
        output_dir / "figures" / "frequency_split_wells",
    )
    run_summary = {
        "created_at_utc": anchor_metadata["created_at_utc"],
        "source_script": Path(__file__).name,
        "artifact_family": "well_constraints_time",
        "source_dirs": {key: repo_relative_path(value, root=REPO_ROOT) for key, value in source_dirs.items()},
        "seismic_file": repo_relative_path(seismic_file, root=REPO_ROOT),
        "seismic_type": seismic_type,
        "horizons": horizon_metadata,
        "config": script_cfg,
        "frequency_split": {
            "mode": split_source,
            "cutoff_hz": split_cfg.cutoff_hz,
            "filter_order": split_cfg.filter_order,
            "buffer_seconds": split_cfg.buffer_seconds,
            "buffer_mode": split_cfg.buffer_mode,
            "diagnostics": repo_relative_path(split_diag_path, root=REPO_ROOT),
            "aggregate": repo_relative_path(split_aggregate_path, root=REPO_ROOT),
            "diagnostic_figure": repo_relative_path(split_diag_figure_path, root=REPO_ROOT),
            "selection": split_selection,
        },
        "conflicts": {
            "strategy": "weighted_average",
            "point_conflict_count": int(len(conflicts)),
            "report": repo_relative_path(conflicts_path, root=REPO_ROOT),
            "high_supervision_conflict_count": int(len(high_conflicts)),
            "high_supervision_report": repo_relative_path(high_conflicts_path, root=REPO_ROOT),
        },
        "counts": {
            "selected_wells": int(qc_df["control_point_count"].gt(0).sum()),
            "point_count": int(len(point_df)),
            "anchor_trace_count": int(anchor_bundle.n_anchors),
            "lfm_control_point_count": int(len(lfm_control_df)),
            "lfm_aggregated_point_count": int(lfm_aggregated_count),
        },
        "outputs": {
            "well_constraint_points": repo_relative_path(point_path, root=REPO_ROOT),
            "log_ai_anchor_time": repo_relative_path(anchor_path, root=REPO_ROOT),
            "well_anchor_points": repo_relative_path(anchor_points_path, root=REPO_ROOT),
            "well_anchor_conflicts": repo_relative_path(conflicts_path, root=REPO_ROOT),
            "well_anchor_trace_summary": repo_relative_path(anchor_trace_summary_path, root=REPO_ROOT),
            "well_high_supervision_time": repo_relative_path(prior_path, root=REPO_ROOT),
            "well_high_supervision_conflicts": repo_relative_path(high_conflicts_path, root=REPO_ROOT),
            "well_high_supervision_qc": repo_relative_path(qc_path, root=REPO_ROOT),
            "well_high_stats_global": repo_relative_path(global_stats_path, root=REPO_ROOT),
            "well_high_stats_by_layer": repo_relative_path(layer_stats_path, root=REPO_ROOT),
            "well_high_stats_shrinkage": repo_relative_path(shrinkage_path, root=REPO_ROOT),
            "well_high_motif_manifest": repo_relative_path(motif_path, root=REPO_ROOT),
            "lfm_layer_control_points": repo_relative_path(lfm_control_path, root=REPO_ROOT),
            "lfm_control_qc": repo_relative_path(lfm_qc_path, root=REPO_ROOT),
            "frequency_split_diagnostics": repo_relative_path(split_diag_path, root=REPO_ROOT),
            "frequency_split_aggregate": repo_relative_path(split_aggregate_path, root=REPO_ROOT),
            "frequency_split_cutoff_sweep_figure": repo_relative_path(split_diag_figure_path, root=REPO_ROOT),
            "frequency_split_well_sweep_figures": split_well_figure_paths,
        },
    }
    write_json(output_dir / "run_summary.json", run_summary)

    print("=== Well Constraints ===")
    print(f"Output: {output_dir}")
    print(f"Selected wells: {run_summary['counts']['selected_wells']}")
    print(f"Point facts: {len(point_df)}")
    print(f"Anchor traces: {anchor_bundle.n_anchors}")
    print(f"LFM control points: {len(lfm_control_df)}")
    print("Motif bank: not populated in this version")


if __name__ == "__main__":
    main()
