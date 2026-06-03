"""Time-domain well-constraint helpers.

The functions in this module keep well curves and spatial facts together until
the final export step.  Scripts may convert the returned data frames to bundle
arrays, but the canonical point rows are always keyed by floating line
coordinates and positive TWT seconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cup.utils.io import to_json_compatible


@dataclass(frozen=True)
class FrequencySplitConfig:
    """Resolved frequency split configuration for time-domain log-AI."""

    cutoff_hz: float
    filter_order: int = 6
    buffer_seconds: float | None = None
    buffer_mode: str = "reflect"


POINT_COLUMNS = [
    "well_name",
    "route",
    "source",
    "anchor_eligible",
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
    "sample_index",
    "zone_name",
    "u_in_zone",
    "ai_full",
    "log_ai_full",
    "well_low_ai",
    "well_low_log_ai",
    "well_high_log_ai",
    "weight",
    "batch_corr",
    "batch_nmae",
]

CONFLICT_COLUMNS = [
    "flat_idx",
    "sample_index",
    "n_points",
    "well_names",
    "sources",
    "min_value",
    "max_value",
    "range_value",
    "strategy",
    "point_rows_json",
]

__all__ = [
    "CONFLICT_COLUMNS",
    "FrequencySplitConfig",
    "POINT_COLUMNS",
    "aggregate_lfm_control_points",
    "aggregate_trace_arrays",
    "apply_frequency_split",
    "build_deviated_point_facts",
    "build_point_conflict_report",
    "build_vertical_point_facts",
    "confidence_from_corr",
    "diagnose_frequency_split",
    "high_frequency_stats",
    "layer_shrinkage_stats",
    "lowpass_values_on_twt",
    "nearest_trace_fields",
    "robust_stats",
    "sample_zone",
    "split_log_ai_frequency_bands",
    "true_runs",
]


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs where a 1D boolean mask is true."""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0 or not np.any(values):
        return []
    padded = np.concatenate(([False], values, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, edges.size, 2)]


def robust_stats(values: np.ndarray) -> dict[str, Any]:
    """Small JSON-friendly robust summary."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "rms": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "abs_p95": None,
            "abs_p99": None,
        }
    abs_arr = np.abs(arr)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "rms": float(np.sqrt(np.mean(arr * arr))),
        "p10": float(np.percentile(arr, 10.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "abs_p95": float(np.percentile(abs_arr, 95.0)),
        "abs_p99": float(np.percentile(abs_arr, 99.0)),
    }


def confidence_from_corr(
    corr: object,
    *,
    mode: str = "corr",
    floor: float = 0.3,
    span: float = 0.4,
    min_weight: float = 0.6,
) -> float:
    """Map batch well-tie correlation to a non-negative sample weight."""
    if str(mode).strip().lower() == "uniform":
        return 1.0
    if str(mode).strip().lower() != "corr":
        raise ValueError("weight mode must be 'corr' or 'uniform'.")
    if span <= 0.0:
        raise ValueError(f"confidence span must be positive, got {span}.")
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError(f"min_weight must be within [0, 1], got {min_weight}.")
    try:
        value = float(corr)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    scaled = float(np.clip((value - float(floor)) / float(span), 0.0, 1.0))
    return float(min_weight + (1.0 - min_weight) * scaled)


def lowpass_values_on_twt(
    twt: np.ndarray,
    values: np.ndarray,
    *,
    dt_s: float,
    cutoff_hz: float,
    order: int,
    buffer_seconds: float | None = None,
    buffer_mode: str = "reflect",
) -> np.ndarray:
    """Low-pass a trace on an irregular TWT basis and interpolate back."""
    from cup.seismic.lfm_time import lowpass_twt_log
    from wtie.processing import grid

    original_twt = np.asarray(twt, dtype=np.float64).reshape(-1)
    original_values = np.asarray(values, dtype=np.float64).reshape(-1)
    if original_twt.shape != original_values.shape:
        raise ValueError("twt and values must have matching 1D shapes.")
    if dt_s <= 0.0 or not np.isfinite(dt_s):
        raise ValueError(f"dt_s must be positive and finite, got {dt_s}.")

    out = np.full(original_values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(original_twt) & np.isfinite(original_values)
    finite_indices = np.flatnonzero(finite)
    min_filter_samples = max(4, 3 * int(order) + 2)
    if finite_indices.size < min_filter_samples:
        out[finite_indices] = original_values[finite_indices]
        return out

    twt_valid = original_twt[finite_indices]
    values_valid = original_values[finite_indices]
    order_idx = np.argsort(twt_valid)
    twt_valid = twt_valid[order_idx]
    values_valid = values_valid[order_idx]
    unique_twt, inverse = np.unique(twt_valid, return_inverse=True)
    if unique_twt.size != twt_valid.size:
        value_sum = np.zeros(unique_twt.size, dtype=np.float64)
        count = np.zeros(unique_twt.size, dtype=np.float64)
        np.add.at(value_sum, inverse, values_valid)
        np.add.at(count, inverse, 1.0)
        values_valid = value_sum / np.maximum(count, 1.0)
        twt_valid = unique_twt
    if twt_valid.size < min_filter_samples:
        out[finite_indices] = np.interp(original_twt[finite_indices], twt_valid, values_valid)
        return out

    regular_twt = np.arange(float(twt_valid[0]), float(twt_valid[-1]) + 0.5 * dt_s, float(dt_s))
    regular_values = np.interp(regular_twt, twt_valid, values_valid)
    log = grid.Log(regular_values, regular_twt, "twt", name="log_AI", unit="ln(m/s*g/cm3)", allow_nan=False)
    filtered = lowpass_twt_log(
        log,
        cutoff_hz=float(cutoff_hz),
        order=int(order),
        buffer_seconds=buffer_seconds,
        buffer_mode=buffer_mode,
    )
    out[finite_indices] = np.interp(original_twt[finite_indices], filtered.basis, filtered.values)
    return out


def split_log_ai_frequency_bands(
    twt_s: np.ndarray,
    ai: np.ndarray,
    mask: np.ndarray,
    cfg: FrequencySplitConfig,
) -> dict[str, np.ndarray]:
    """Split positive AI samples into full, low-pass, and high-pass log-AI."""
    twt = np.asarray(twt_s, dtype=np.float64).reshape(-1)
    ai_values = np.asarray(ai, dtype=np.float64).reshape(-1)
    valid = np.asarray(mask, dtype=bool).reshape(-1) & np.isfinite(twt) & np.isfinite(ai_values) & (ai_values > 0.0)
    if twt.shape != ai_values.shape or twt.shape != valid.shape:
        raise ValueError("twt_s, ai, and mask must have matching 1D shapes.")

    log_ai = np.zeros(ai_values.shape, dtype=np.float32)
    low_log_ai = np.zeros(ai_values.shape, dtype=np.float32)
    high_log_ai = np.zeros(ai_values.shape, dtype=np.float32)
    low_ai = np.zeros(ai_values.shape, dtype=np.float32)
    log_ai[valid] = np.log(np.clip(ai_values[valid], 1e-6, None)).astype(np.float32)

    if int(np.count_nonzero(valid)) >= 2:
        steps = np.diff(np.sort(twt[valid]))
        steps = steps[np.isfinite(steps) & (steps > 0.0)]
        dt_s = float(np.median(steps)) if steps.size else np.nan
    else:
        dt_s = np.nan
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        low_log_ai[valid] = log_ai[valid]
    else:
        low_values = lowpass_values_on_twt(
            twt,
            np.where(valid, log_ai, np.nan),
            dt_s=dt_s,
            cutoff_hz=float(cfg.cutoff_hz),
            order=int(cfg.filter_order),
            buffer_seconds=cfg.buffer_seconds,
            buffer_mode=cfg.buffer_mode,
        )
        low_log_ai[valid] = low_values[valid].astype(np.float32)
    high_log_ai[valid] = (log_ai[valid] - low_log_ai[valid]).astype(np.float32)
    low_ai[valid] = np.exp(low_log_ai[valid]).astype(np.float32)
    return {
        "log_ai_full": log_ai,
        "well_low_ai": low_ai,
        "well_low_log_ai": low_log_ai,
        "well_high_log_ai": high_log_ai,
    }


def diagnose_frequency_split(
    point_df: pd.DataFrame,
    candidate_cutoff_hz: list[float],
    *,
    filter_order: int,
    buffer_seconds: float | None,
    buffer_mode: str,
) -> tuple[FrequencySplitConfig, pd.DataFrame]:
    """Choose one shared cutoff from simple well-log split diagnostics."""
    if point_df.empty:
        raise ValueError("Cannot diagnose frequency split from an empty point table.")
    diagnostics: list[dict[str, Any]] = []
    for cutoff in candidate_cutoff_hz:
        cfg = FrequencySplitConfig(
            cutoff_hz=float(cutoff),
            filter_order=int(filter_order),
            buffer_seconds=buffer_seconds,
            buffer_mode=buffer_mode,
        )
        split_df = apply_frequency_split(point_df, cfg)
        valid = split_df["well_high_log_ai"].to_numpy(dtype=float)
        full = split_df["log_ai_full"].to_numpy(dtype=float)
        mask = np.isfinite(valid) & np.isfinite(full) & (split_df["weight"].to_numpy(dtype=float) > 0.0)
        high_rms = float(np.sqrt(np.mean(valid[mask] ** 2))) if np.any(mask) else np.nan
        full_std = float(np.std(full[mask])) if np.any(mask) else np.nan
        ratio = high_rms / max(full_std, 1e-6) if np.isfinite(high_rms) and np.isfinite(full_std) else np.nan
        roughness_values = []
        boundary_values = []
        for _, group in split_df.loc[mask].groupby("well_name", sort=False):
            low = group.sort_values("twt_s")["well_low_log_ai"].to_numpy(dtype=float)
            high = group.sort_values("twt_s")["well_high_log_ai"].to_numpy(dtype=float)
            if low.size > 1:
                roughness_values.append(float(np.mean(np.abs(np.diff(low)))))
            if high.size >= 6:
                edge = max(1, min(5, high.size // 5))
                boundary_values.append(float(np.mean(np.abs(np.r_[high[:edge], high[-edge:]]))))
        roughness = float(np.mean(roughness_values)) if roughness_values else np.nan
        boundary = float(np.mean(boundary_values)) if boundary_values else np.nan
        normalizer = max(full_std, 1e-6) if np.isfinite(full_std) else np.nan
        roughness_norm = roughness / normalizer if np.isfinite(roughness) and np.isfinite(normalizer) else np.nan
        boundary_norm = boundary / normalizer if np.isfinite(boundary) and np.isfinite(normalizer) else np.nan
        ratio_penalty = 0.0 if np.isfinite(ratio) and 0.05 <= ratio <= 0.8 else 1.0
        score = (
            (0.0 if np.isfinite(roughness_norm) else 1.0)
            + (roughness_norm if np.isfinite(roughness_norm) else 0.0)
            + 0.5 * (boundary_norm if np.isfinite(boundary_norm) else 0.0)
            + ratio_penalty
        )
        diagnostics.append(
            {
                "cutoff_hz": float(cutoff),
                "score": float(score),
                "high_rms": high_rms,
                "full_std": full_std,
                "high_to_full_std_ratio": ratio,
                "low_log_ai_mean_abs_diff": roughness,
                "low_log_ai_mean_abs_diff_norm": roughness_norm,
                "edge_high_abs_mean": boundary,
                "edge_high_abs_mean_norm": boundary_norm,
                "valid_samples": int(np.count_nonzero(mask)),
            }
        )
    diag_df = pd.DataFrame(diagnostics).sort_values(["score", "cutoff_hz"], ascending=[True, True])
    best = float(diag_df.iloc[0]["cutoff_hz"])
    return (
        FrequencySplitConfig(
            cutoff_hz=best,
            filter_order=int(filter_order),
            buffer_seconds=buffer_seconds,
            buffer_mode=buffer_mode,
        ),
        pd.DataFrame(diagnostics),
    )


def apply_frequency_split(point_df: pd.DataFrame, cfg: FrequencySplitConfig) -> pd.DataFrame:
    """Add low/high log-AI columns to a point table."""
    out = point_df.copy()
    for col in ("log_ai_full", "well_low_ai", "well_low_log_ai", "well_high_log_ai"):
        if col not in out.columns:
            out[col] = np.nan
    for _, index in out.groupby("well_name", sort=False).groups.items():
        group = out.loc[index].sort_values("twt_s")
        split = split_log_ai_frequency_bands(
            group["twt_s"].to_numpy(dtype=float),
            group["ai_full"].to_numpy(dtype=float),
            np.isfinite(group["ai_full"].to_numpy(dtype=float)),
            cfg,
        )
        for key, values in split.items():
            out.loc[group.index, key] = values
    return out


def nearest_trace_fields(survey: Any, inline_float: float, xline_float: float) -> dict[str, Any]:
    """Resolve nearest integer trace fields from floating line coordinates."""
    nearest_inline = survey.line_geometry.snap_inline(float(inline_float))
    nearest_xline = survey.line_geometry.snap_xline(float(xline_float))
    inline_index_f, xline_index_f = survey.line_geometry.line_to_index(nearest_inline, nearest_xline)
    inline_index = min(int(survey.line_geometry.inline_axis.count) - 1, max(0, int(round(inline_index_f))))
    xline_index = min(int(survey.line_geometry.xline_axis.count) - 1, max(0, int(round(xline_index_f))))
    return {
        "nearest_inline": float(nearest_inline),
        "nearest_xline": float(nearest_xline),
        "inline_index": int(inline_index),
        "xline_index": int(xline_index),
        "flat_idx": int(survey.trace_flat_index(inline_index, xline_index)),
    }


def sample_zone(target_layer: Any, inline_float: float, xline_float: float, twt_s: float) -> tuple[str | None, float | None]:
    """Sample target-layer zone name and proportional vertical position."""
    values = target_layer.get_interpretation_values_at_location(float(inline_float), float(xline_float))
    for top_name, bottom_name in target_layer.iter_zones():
        top = float(values[top_name])
        bottom = float(values[bottom_name])
        if not np.isfinite(top) or not np.isfinite(bottom) or bottom <= top:
            continue
        if top <= float(twt_s) <= bottom:
            return f"{top_name}->{bottom_name}", float((float(twt_s) - top) / (bottom - top))
    return None, None


def build_vertical_point_facts(
    *,
    well_name: str,
    route: str,
    las_file: Any,
    tdt_file: Any,
    surface_x: float,
    surface_y: float,
    target_layer: Any,
    survey: Any,
    samples: np.ndarray,
    weight: float,
    batch_corr: float | None,
    batch_nmae: float | None,
    anchor_eligible: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build target-window point facts for one vertical well."""
    from cup.well.las import load_vp_rho_logset_from_standard_las
    from cup.well.td import load_workflow_time_depth_table_csv
    from wtie.processing import grid

    logset = load_vp_rho_logset_from_standard_las(las_file)
    table = load_workflow_time_depth_table_csv(tdt_file)
    dt_s = float(np.median(np.diff(np.asarray(samples, dtype=float))))
    ai_twt = grid.convert_log_from_md_to_twt(logset.AI, table, None, dt_s)
    ai_at_samples = np.interp(samples, np.asarray(ai_twt.basis, dtype=float), ai_twt.values, left=np.nan, right=np.nan)
    md_at_samples = np.interp(samples, table.twt, table.depth, left=np.nan, right=np.nan)
    inline_float, xline_float = survey.line_geometry.coord_to_line(float(surface_x), float(surface_y))
    trace_fields = nearest_trace_fields(survey, inline_float, xline_float)

    rows: list[dict[str, Any]] = []
    attempted = 0
    zone_errors = 0
    for sample_index, (twt_s, ai, md_m) in enumerate(zip(samples, ai_at_samples, md_at_samples)):
        if not np.isfinite(ai) or ai <= 0.0 or not np.isfinite(md_m):
            continue
        attempted += 1
        try:
            zone_name, u_in_zone = sample_zone(target_layer, inline_float, xline_float, float(twt_s))
        except Exception:
            zone_errors += 1
            zone_name, u_in_zone = None, None
        if zone_name is None or u_in_zone is None:
            continue
        rows.append(
            {
                "well_name": well_name,
                "route": route,
                "source": "vertical_trace",
                "anchor_eligible": bool(anchor_eligible),
                "twt_s": float(twt_s),
                "md_m": float(md_m),
                "x_m": float(surface_x),
                "y_m": float(surface_y),
                "inline_float": float(inline_float),
                "xline_float": float(xline_float),
                "sample_index": int(sample_index),
                "zone_name": zone_name,
                "u_in_zone": float(u_in_zone),
                "ai_full": float(ai),
                "weight": float(weight),
                "batch_corr": batch_corr,
                "batch_nmae": batch_nmae,
                **trace_fields,
            }
        )
    return (
        pd.DataFrame(rows, columns=[c for c in POINT_COLUMNS if c not in {"log_ai_full", "well_low_ai", "well_low_log_ai", "well_high_log_ai"}]),
        {
            "attempted_samples": int(attempted),
            "valid_points": int(len(rows)),
            "invalid_point_count": int(max(0, attempted - len(rows))),
            "zone_sample_errors": int(zone_errors),
            "unique_trace_count": 1 if rows else 0,
        },
    )


def build_deviated_point_facts(
    *,
    well_name: str,
    route: str,
    las_file: Any,
    trace_plan_file: Any,
    target_layer: Any,
    survey: Any,
    weight: float,
    batch_corr: float | None,
    batch_nmae: float | None,
    sample_step_s: float | None = None,
    anchor_eligible: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build target-window point facts for one deviated well from a trace plan."""
    from cup.well.las import load_vp_rho_logset_from_standard_las

    logset = load_vp_rho_logset_from_standard_las(las_file)
    ai_log = logset.AI
    ai_basis = np.asarray(ai_log.basis, dtype=float)
    if ai_basis.size < 2 or np.any(np.diff(ai_basis) <= 0.0):
        raise ValueError(f"AI log MD basis must be strictly increasing: {las_file}")
    plan_df = pd.read_csv(trace_plan_file)
    required = {"twt_s", "md_m", "x_m", "y_m", "inline_float", "xline_float", "survey_position"}
    missing = required - set(plan_df.columns)
    if missing:
        raise ValueError(f"trace_sample_plan missing columns: {sorted(missing)}")
    plan_df = plan_df.loc[plan_df["survey_position"].astype(str).eq("inside")].copy()
    plan_df = plan_df.sort_values("twt_s").reset_index(drop=True)
    if sample_step_s is not None and sample_step_s > 0.0 and not plan_df.empty:
        keep: list[int] = []
        last_twt = -np.inf
        for idx, twt in enumerate(plan_df["twt_s"].to_numpy(dtype=float)):
            if twt >= last_twt + float(sample_step_s) - 1e-9:
                keep.append(idx)
                last_twt = float(twt)
        plan_df = plan_df.iloc[keep].reset_index(drop=True)

    attempted = int(len(plan_df))
    if plan_df.empty:
        return pd.DataFrame(columns=POINT_COLUMNS), {
            "attempted_samples": attempted,
            "valid_points": 0,
            "invalid_point_count": attempted,
            "zone_sample_errors": 0,
            "unique_trace_count": 0,
        }
    plan_md = plan_df["md_m"].to_numpy(dtype=float)
    ai_values = np.interp(plan_md, ai_basis, np.asarray(ai_log.values, dtype=float), left=np.nan, right=np.nan)
    rows: list[dict[str, Any]] = []
    zone_errors = 0
    for row_index, row in plan_df.iterrows():
        ai = float(ai_values[row_index])
        if not np.isfinite(ai) or ai <= 0.0:
            continue
        inline_float = float(row["inline_float"])
        xline_float = float(row["xline_float"])
        twt_s = float(row["twt_s"])
        try:
            zone_name, u_in_zone = sample_zone(target_layer, inline_float, xline_float, twt_s)
        except Exception:
            zone_errors += 1
            zone_name, u_in_zone = None, None
        if zone_name is None or u_in_zone is None:
            continue
        trace_fields: dict[str, Any] = {}
        for key in ["flat_idx", "inline_index", "xline_index", "nearest_inline", "nearest_xline"]:
            if key in row and pd.notna(row[key]):
                trace_fields[key] = int(row[key]) if key.endswith("index") or key == "flat_idx" else float(row[key])
        if "flat_idx" not in trace_fields:
            trace_fields = nearest_trace_fields(survey, inline_float, xline_float)
        rows.append(
            {
                "well_name": well_name,
                "route": route,
                "source": "deviated_trajectory",
                "anchor_eligible": bool(anchor_eligible),
                "twt_s": twt_s,
                "md_m": float(row["md_m"]),
                "x_m": float(row["x_m"]),
                "y_m": float(row["y_m"]),
                "inline_float": inline_float,
                "xline_float": xline_float,
                "sample_index": int(row["sample_index"]) if "sample_index" in row and pd.notna(row["sample_index"]) else int(row_index),
                "zone_name": zone_name,
                "u_in_zone": float(u_in_zone),
                "ai_full": ai,
                "weight": float(weight),
                "batch_corr": batch_corr,
                "batch_nmae": batch_nmae,
                **trace_fields,
            }
        )
    return (
        pd.DataFrame(rows, columns=[c for c in POINT_COLUMNS if c not in {"log_ai_full", "well_low_ai", "well_low_log_ai", "well_high_log_ai"}]),
        {
            "attempted_samples": attempted,
            "valid_points": int(len(rows)),
            "invalid_point_count": int(max(0, attempted - len(rows))),
            "zone_sample_errors": int(zone_errors),
            "unique_trace_count": int(pd.DataFrame(rows)["flat_idx"].nunique()) if rows else 0,
        },
    )


def aggregate_lfm_control_points(point_df: pd.DataFrame, *, n_slices: int) -> tuple[pd.DataFrame, int]:
    """Aggregate point facts to LFM layer control points."""
    if point_df.empty:
        return pd.DataFrame(), 0
    frame = point_df.copy()
    frame["_slice_index"] = [_slice_index_from_u(float(v), int(n_slices)) for v in frame["u_in_zone"]]
    rows: list[dict[str, Any]] = []
    removed = 0
    group_cols = ["well_name", "route", "source", "zone_name", "_slice_index"]
    for (_, _, source, zone_name, _), group in frame.groupby(group_cols, sort=False):
        removed += max(0, int(len(group) - 1))
        weight = np.maximum(group["weight"].to_numpy(dtype=float), 0.0)
        if not np.any(weight > 0.0):
            weight = np.ones_like(weight)
        rows.append(
            {
                "well_name": str(group["well_name"].iloc[0]),
                "route": str(group["route"].iloc[0]),
                "source": str(source),
                "twt_s": _weighted_mean(group["twt_s"], weight),
                "md_m": _weighted_mean(group["md_m"], weight),
                "x_m": _weighted_mean(group["x_m"], weight),
                "y_m": _weighted_mean(group["y_m"], weight),
                "inline_float": _weighted_mean(group["inline_float"], weight),
                "xline_float": _weighted_mean(group["xline_float"], weight),
                "zone_name": str(zone_name),
                "u_in_zone": _weighted_mean(group["u_in_zone"], weight),
                "ai": _weighted_mean(group["well_low_ai"], weight),
                "weight": float(np.mean(weight)),
                "flat_idx": int(round(_weighted_mean(group["flat_idx"], weight))),
                "sample_index": int(round(_weighted_mean(group["sample_index"], weight))),
            }
        )
    return pd.DataFrame(rows), int(removed)


def build_point_conflict_report(point_df: pd.DataFrame, *, value_col: str = "well_low_log_ai") -> pd.DataFrame:
    """Report duplicate nearest trace/sample constraints before aggregation."""
    rows: list[dict[str, Any]] = []
    if point_df.empty:
        return pd.DataFrame(columns=CONFLICT_COLUMNS)
    for (flat_idx, sample_index), group in point_df.groupby(["flat_idx", "sample_index"], sort=False):
        if len(group) <= 1:
            continue
        values = group[value_col].to_numpy(dtype=float)
        rows.append(
            {
                "flat_idx": int(flat_idx),
                "sample_index": int(sample_index),
                "n_points": int(len(group)),
                "well_names": ";".join(sorted(group["well_name"].astype(str).unique())),
                "sources": ";".join(sorted(group["source"].astype(str).unique())),
                "min_value": float(np.nanmin(values)),
                "max_value": float(np.nanmax(values)),
                "range_value": float(np.nanmax(values) - np.nanmin(values)),
                "strategy": "weighted_average",
                "point_rows_json": json.dumps(
                    to_json_compatible(
                        group[
                            [
                                "well_name",
                                "source",
                                "twt_s",
                                "inline_float",
                                "xline_float",
                                value_col,
                                "weight",
                            ]
                        ].to_dict(orient="records")
                    ),
                    ensure_ascii=False,
                ),
            }
        )
    return pd.DataFrame(rows, columns=CONFLICT_COLUMNS)


def aggregate_trace_arrays(
    point_df: pd.DataFrame,
    samples: np.ndarray,
    *,
    target_col: str,
    value_cols: list[str],
    include_anchor_only: bool = False,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Aggregate point rows to unique flat-index trace arrays."""
    if include_anchor_only:
        frame = point_df.loc[point_df["anchor_eligible"].astype(bool)].copy()
    else:
        frame = point_df.copy()
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    n_sample = int(samples.size)
    if frame.empty:
        empty = {
            "samples": samples,
            "flat_indices": np.empty(0, dtype=np.int64),
            "target_ai": np.zeros((0, n_sample), dtype=np.float32),
            "mask": np.zeros((0, n_sample), dtype=bool),
            "weight": np.zeros((0, n_sample), dtype=np.float32),
            "well_names": np.asarray([], dtype=str),
            "inline": np.empty(0, dtype=np.float32),
            "xline": np.empty(0, dtype=np.float32),
        }
        for col in value_cols:
            empty[col] = np.zeros((0, n_sample), dtype=np.float32)
        return empty, pd.DataFrame()

    flat_indices = np.array(sorted(frame["flat_idx"].dropna().astype(int).unique()), dtype=np.int64)
    flat_to_row = {int(flat): idx for idx, flat in enumerate(flat_indices)}
    arrays: dict[str, np.ndarray] = {
        "samples": samples,
        "flat_indices": flat_indices,
        "target_ai": np.zeros((flat_indices.size, n_sample), dtype=np.float32),
        "mask": np.zeros((flat_indices.size, n_sample), dtype=bool),
        "weight": np.zeros((flat_indices.size, n_sample), dtype=np.float32),
    }
    for col in value_cols:
        arrays[col] = np.zeros((flat_indices.size, n_sample), dtype=np.float32)

    summary_rows: list[dict[str, Any]] = []
    for flat_idx, flat_group in frame.groupby("flat_idx", sort=True):
        row_idx = flat_to_row[int(flat_idx)]
        for sample_index, sample_group in flat_group.groupby("sample_index", sort=False):
            sample_idx = int(sample_index)
            if not 0 <= sample_idx < n_sample:
                continue
            weight = np.maximum(sample_group["weight"].to_numpy(dtype=float), 0.0)
            if not np.any(weight > 0.0):
                weight = np.ones_like(weight)
            arrays["target_ai"][row_idx, sample_idx] = _weighted_mean(sample_group[target_col], weight)
            arrays["weight"][row_idx, sample_idx] = float(np.mean(weight))
            arrays["mask"][row_idx, sample_idx] = True
            for col in value_cols:
                arrays[col][row_idx, sample_idx] = _weighted_mean(sample_group[col], weight)
        summary_rows.append(
            {
                "flat_idx": int(flat_idx),
                "well_names": ";".join(sorted(flat_group["well_name"].astype(str).unique())),
                "sources": ";".join(sorted(flat_group["source"].astype(str).unique())),
                "sample_count": int(arrays["mask"][row_idx].sum()),
                "weight_min": float(np.min(arrays["weight"][row_idx][arrays["mask"][row_idx]])),
                "weight_mean": float(np.mean(arrays["weight"][row_idx][arrays["mask"][row_idx]])),
                "weight_max": float(np.max(arrays["weight"][row_idx][arrays["mask"][row_idx]])),
                "inline": float(np.average(flat_group["nearest_inline"].to_numpy(dtype=float))),
                "xline": float(np.average(flat_group["nearest_xline"].to_numpy(dtype=float))),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    arrays["well_names"] = summary_df["well_names"].to_numpy(dtype=str)
    arrays["inline"] = summary_df["inline"].to_numpy(dtype=np.float32)
    arrays["xline"] = summary_df["xline"].to_numpy(dtype=np.float32)
    return arrays, summary_df


def high_frequency_stats(point_df: pd.DataFrame, *, sample_step_s: float | None = None) -> dict[str, Any]:
    """Compute high-frequency residual statistics for one point subset."""
    if point_df.empty:
        return _empty_high_stats()
    values = point_df["well_high_log_ai"].to_numpy(dtype=float)
    valid = np.isfinite(values) & (point_df["weight"].to_numpy(dtype=float) > 0.0)
    values = values[valid]
    event_threshold = float(np.percentile(np.abs(values), 90.0)) if values.size else np.nan
    events = np.abs(values) >= event_threshold if np.isfinite(event_threshold) else np.zeros(values.shape, dtype=bool)
    run_lengths_by_state = _state_run_lengths_by_state(point_df.loc[valid], event_threshold)
    run_lengths = np.r_[run_lengths_by_state["positive"], run_lengths_by_state["negative"]]
    transitions = _transition_matrix(point_df.loc[valid], event_threshold)
    reflectivity = []
    spectra = []
    for _, group in point_df.loc[valid].groupby("well_name", sort=False):
        high = group.sort_values("twt_s")["well_high_log_ai"].to_numpy(dtype=float)
        if high.size >= 2:
            reflectivity.append(np.tanh(0.5 * np.diff(high)))
        if high.size >= 8:
            centered = high - float(np.mean(high))
            spectra.append(np.abs(np.fft.rfft(centered)))
    refl_values = np.concatenate(reflectivity) if reflectivity else np.asarray([], dtype=float)
    sample_step = sample_step_s if sample_step_s is not None and sample_step_s > 0.0 else None
    event_density = float(np.mean(events)) if events.size else None
    return {
        "sample_count": int(values.size),
        "well_count": int(point_df.loc[valid, "well_name"].nunique()),
        "event_count": int(np.count_nonzero(events)),
        "event_threshold_abs": event_threshold if np.isfinite(event_threshold) else None,
        "event_density_per_sample": event_density,
        "event_density_per_second": None if event_density is None or sample_step is None else float(event_density / sample_step),
        "amplitude": robust_stats(values),
        "abs_amplitude": robust_stats(np.abs(values)),
        "run_length_samples": robust_stats(run_lengths),
        "run_length_by_state": {
            "positive": robust_stats(run_lengths_by_state["positive"]),
            "negative": robust_stats(run_lengths_by_state["negative"]),
            "quiet": robust_stats(run_lengths_by_state["quiet"]),
        },
        "transition_matrix": transitions,
        "reflectivity": robust_stats(refl_values),
        "spectrum": _spectrum_stats(spectra),
    }


def layer_shrinkage_stats(point_df: pd.DataFrame, *, sample_step_s: float | None = None) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Compute global, empirical layer, and shrinkage high-frequency stats."""
    global_stats = high_frequency_stats(point_df, sample_step_s=sample_step_s)
    rows: list[dict[str, Any]] = []
    shrinkage: dict[str, Any] = {}
    for zone_name, group in point_df.groupby("zone_name", sort=False):
        empirical = high_frequency_stats(group, sample_step_s=sample_step_s)
        reliability = _layer_reliability(empirical)
        alpha = reliability
        rows.append(
            {
                "zone_name": str(zone_name),
                "well_count": empirical["well_count"],
                "sample_count": empirical["sample_count"],
                "event_count": empirical["event_count"],
                "reliability": reliability,
                "alpha_to_layer": alpha,
                "event_density_per_sample": empirical["event_density_per_sample"],
                "event_density_per_second": empirical["event_density_per_second"],
                "amplitude_rms": empirical["amplitude"]["rms"],
                "amplitude_p10": empirical["amplitude"]["p10"],
                "amplitude_p50": empirical["amplitude"]["p50"],
                "amplitude_p90": empirical["amplitude"]["p90"],
                "amplitude_abs_p95": empirical["amplitude"]["abs_p95"],
                "run_length_p50": empirical["run_length_samples"]["p50"],
                "run_length_p90": empirical["run_length_samples"]["p90"],
                "transition_matrix_json": json.dumps(to_json_compatible(empirical["transition_matrix"]), ensure_ascii=False),
            }
        )
        shrinkage[str(zone_name)] = {
            "alpha": alpha,
            "reliability": reliability,
            "empirical": empirical,
            "global": global_stats,
            "final": _blend_stats(empirical, global_stats, alpha),
        }
    return global_stats, pd.DataFrame(rows), shrinkage


def _slice_index_from_u(u_in_zone: float, n_slices: int) -> int:
    if n_slices < 2:
        raise ValueError(f"n_slices must be >= 2, got {n_slices}.")
    slice_u = np.linspace(0.0, 1.0, int(n_slices), dtype=np.float64)
    boundaries = 0.5 * (slice_u[:-1] + slice_u[1:])
    return int(np.searchsorted(boundaries, float(u_in_zone), side="right"))


def _weighted_mean(values: Any, weight: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr) & np.isfinite(weight) & (weight >= 0.0)
    if not np.any(valid):
        return float("nan")
    w = weight[valid]
    if not np.any(w > 0.0):
        w = np.ones_like(w)
    return float(np.average(arr[valid], weights=w))


def _state_run_lengths(frame: pd.DataFrame, threshold: float) -> np.ndarray:
    runs = _state_run_lengths_by_state(frame, threshold)
    return np.r_[runs["positive"], runs["negative"]]


def _state_run_lengths_by_state(frame: pd.DataFrame, threshold: float) -> dict[str, np.ndarray]:
    empty = {
        "positive": np.asarray([], dtype=float),
        "negative": np.asarray([], dtype=float),
        "quiet": np.asarray([], dtype=float),
    }
    if frame.empty or not np.isfinite(threshold):
        return empty
    lengths: dict[str, list[int]] = {"positive": [], "negative": [], "quiet": []}
    label_by_value = {-1: "negative", 0: "quiet", 1: "positive"}
    for _, group in frame.groupby("well_name", sort=False):
        values = group.sort_values("twt_s")["well_high_log_ai"].to_numpy(dtype=float)
        state = np.where(values > threshold, 1, np.where(values < -threshold, -1, 0))
        if state.size == 0:
            continue
        start = 0
        for idx in range(1, state.size):
            if state[idx] != state[start]:
                lengths[label_by_value[int(state[start])]].append(idx - start)
                start = idx
        lengths[label_by_value[int(state[start])]].append(state.size - start)
    return {key: np.asarray(value, dtype=float) for key, value in lengths.items()}


def _transition_matrix(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    labels = [-1, 0, 1]
    counts = np.zeros((3, 3), dtype=float)
    if not frame.empty and np.isfinite(threshold):
        for _, group in frame.groupby("well_name", sort=False):
            values = group.sort_values("twt_s")["well_high_log_ai"].to_numpy(dtype=float)
            state = np.where(values > threshold, 1, np.where(values < -threshold, -1, 0))
            for a, b in zip(state[:-1], state[1:]):
                counts[labels.index(int(a)), labels.index(int(b))] += 1.0
    row_sum = counts.sum(axis=1, keepdims=True)
    prob = np.divide(counts, np.maximum(row_sum, 1.0))
    return {"labels": labels, "counts": counts.astype(int).tolist(), "probability": prob.tolist()}


def _spectrum_stats(spectra: list[np.ndarray]) -> dict[str, Any]:
    if not spectra:
        return {"frequency_cycles_per_sample": [], "amplitude_mean": [], "amplitude_p50": [], "amplitude_p90": []}
    n = max(arr.size for arr in spectra)
    freq = np.linspace(0.0, 0.5, num=n, dtype=float)
    stacked = np.stack([np.interp(freq, np.linspace(0.0, 0.5, num=arr.size), arr) for arr in spectra])
    return {
        "frequency_cycles_per_sample": freq.tolist(),
        "amplitude_mean": np.mean(stacked, axis=0).tolist(),
        "amplitude_p50": np.percentile(stacked, 50.0, axis=0).tolist(),
        "amplitude_p90": np.percentile(stacked, 90.0, axis=0).tolist(),
    }


def _empty_high_stats() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "well_count": 0,
        "event_count": 0,
        "event_threshold_abs": None,
        "event_density_per_sample": None,
        "event_density_per_second": None,
        "amplitude": robust_stats(np.asarray([])),
        "abs_amplitude": robust_stats(np.asarray([])),
        "run_length_samples": robust_stats(np.asarray([])),
        "run_length_by_state": {
            "positive": robust_stats(np.asarray([])),
            "negative": robust_stats(np.asarray([])),
            "quiet": robust_stats(np.asarray([])),
        },
        "transition_matrix": _transition_matrix(pd.DataFrame(), float("nan")),
        "reflectivity": robust_stats(np.asarray([])),
        "spectrum": _spectrum_stats([]),
    }


def _layer_reliability(stats: dict[str, Any]) -> float:
    # These thresholds are conservative defaults for a first-pass layer prior:
    # several wells, hundreds of samples, and dozens of detected events should
    # make a layer mostly trust its own empirical statistics.
    well_factor = min(1.0, float(stats.get("well_count", 0)) / 4.0)
    sample_factor = min(1.0, float(stats.get("sample_count", 0)) / 256.0)
    event_factor = min(1.0, float(stats.get("event_count", 0)) / 32.0)
    return float(np.clip(0.4 * well_factor + 0.4 * sample_factor + 0.2 * event_factor, 0.0, 1.0))


def _blend_number(layer_value: Any, global_value: Any, alpha: float) -> Any:
    if layer_value is None:
        return global_value
    if global_value is None:
        return layer_value
    try:
        return float(alpha * float(layer_value) + (1.0 - alpha) * float(global_value))
    except (TypeError, ValueError):
        return layer_value


def _blend_stats(layer: dict[str, Any], global_stats: dict[str, Any], alpha: float) -> dict[str, Any]:
    return {
        "event_density_per_sample": _blend_number(
            layer.get("event_density_per_sample"),
            global_stats.get("event_density_per_sample"),
            alpha,
        ),
        "event_density_per_second": _blend_number(
            layer.get("event_density_per_second"),
            global_stats.get("event_density_per_second"),
            alpha,
        ),
        "amplitude": {
            key: _blend_number(layer.get("amplitude", {}).get(key), global_stats.get("amplitude", {}).get(key), alpha)
            for key in ("rms", "p10", "p50", "p90", "abs_p95", "abs_p99")
        },
        "run_length_samples": {
            key: _blend_number(
                layer.get("run_length_samples", {}).get(key),
                global_stats.get("run_length_samples", {}).get(key),
                alpha,
            )
            for key in ("p10", "p50", "p90")
        },
        "run_length_by_state": {
            state: {
                key: _blend_number(
                    layer.get("run_length_by_state", {}).get(state, {}).get(key),
                    global_stats.get("run_length_by_state", {}).get(state, {}).get(key),
                    alpha,
                )
                for key in ("p10", "p50", "p90")
            }
            for state in ("positive", "negative", "quiet")
        },
        "transition_matrix": _blend_transition_matrix(
            layer.get("transition_matrix"),
            global_stats.get("transition_matrix"),
            alpha,
        ),
        "spectrum": _blend_spectrum(layer.get("spectrum"), global_stats.get("spectrum"), alpha),
    }


def _blend_transition_matrix(layer: Any, global_value: Any, alpha: float) -> Any:
    if not isinstance(layer, dict):
        return global_value
    if not isinstance(global_value, dict):
        return layer
    labels = layer.get("labels") or global_value.get("labels")
    layer_prob = np.asarray(layer.get("probability", []), dtype=float)
    global_prob = np.asarray(global_value.get("probability", []), dtype=float)
    if layer_prob.shape != global_prob.shape or layer_prob.size == 0:
        return layer if alpha >= 0.5 else global_value
    prob = alpha * layer_prob + (1.0 - alpha) * global_prob
    row_sum = prob.sum(axis=1, keepdims=True)
    prob = np.divide(prob, np.maximum(row_sum, 1e-12))
    return {
        "labels": labels,
        "counts": layer.get("counts"),
        "global_counts": global_value.get("counts"),
        "probability": prob.tolist(),
        "blend_alpha": float(alpha),
    }


def _blend_spectrum(layer: Any, global_value: Any, alpha: float) -> Any:
    if not isinstance(layer, dict):
        return global_value
    if not isinstance(global_value, dict):
        return layer
    freq = layer.get("frequency_cycles_per_sample") or global_value.get("frequency_cycles_per_sample") or []
    out: dict[str, Any] = {"frequency_cycles_per_sample": freq, "blend_alpha": float(alpha)}
    for key in ("amplitude_mean", "amplitude_p50", "amplitude_p90"):
        layer_values = np.asarray(layer.get(key, []), dtype=float)
        global_values = np.asarray(global_value.get(key, []), dtype=float)
        if layer_values.shape == global_values.shape and layer_values.size:
            out[key] = (alpha * layer_values + (1.0 - alpha) * global_values).tolist()
        else:
            out[key] = layer.get(key) if alpha >= 0.5 else global_value.get(key)
    return out
