"""Time-domain well-constraint helpers.

The functions in this module keep well curves and spatial facts together until
the final export step.  Scripts may convert the returned data frames to bundle
arrays, but the canonical point rows are always keyed by floating line
coordinates and positive TWT seconds.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from cup.seismic.geometry import nearest_sample_indices, validate_sample_indices
from cup.utils.io import to_json_compatible
from cup.well.frequency_bands import WellFrequencyBands


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
    "seismic_sample_index",
    "zone_name",
    "u_in_zone",
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
    "weight",
    "batch_corr",
    "batch_nmae",
]

CONFLICT_COLUMNS = [
    "flat_idx",
    "seismic_sample_index",
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
    "POINT_COLUMNS",
    "aggregate_trace_arrays",
    "build_deviated_point_facts",
    "build_point_conflict_report",
    "build_vertical_point_facts",
    "confidence_from_corr",
    "horizon_markers_from_zone_points",
    "high_frequency_stats",
    "layer_shrinkage_stats",
    "nearest_trace_fields",
    "robust_stats",
    "sample_zone",
]


def horizon_markers_from_zone_points(
    points: pd.DataFrame,
    *,
    horizon_names: dict[str, str] | None = None,
) -> list[tuple[float, str]]:
    """Recover per-well horizon TWT markers from canonical zone coordinates."""
    required = {"zone_name", "u_in_zone", "twt_s"}
    if points.empty or not required.issubset(points.columns):
        return []

    labels = horizon_names or {}
    estimates: dict[str, list[float]] = {}
    for zone_name, zone_group in points.groupby("zone_name", sort=False):
        parts = str(zone_name).split("->", maxsplit=1)
        if len(parts) != 2 or len(zone_group) < 2:
            continue
        u = zone_group["u_in_zone"].to_numpy(dtype=np.float64)
        twt = zone_group["twt_s"].to_numpy(dtype=np.float64)
        valid = np.isfinite(u) & np.isfinite(twt)
        if int(np.count_nonzero(valid)) < 2 or np.ptp(u[valid]) <= 0.0:
            continue
        slope, intercept = np.polyfit(u[valid], twt[valid], deg=1)
        estimates.setdefault(parts[0], []).append(float(intercept))
        estimates.setdefault(parts[1], []).append(float(intercept + slope))

    return sorted(
        [
            (float(np.mean(values)), labels.get(name, name))
            for name, values in estimates.items()
            if values
        ],
        key=lambda item: item[0],
    )


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
    tdt_file: Any,
    bands: WellFrequencyBands,
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
    from cup.well.td import load_workflow_time_depth_table_csv

    table = load_workflow_time_depth_table_csv(tdt_file)
    band_twt = np.asarray(bands.reference_log_ai.basis, dtype=float)
    if not np.allclose(band_twt, np.asarray(samples, dtype=float), rtol=0.0, atol=1e-9):
        raise ValueError("Vertical well frequency bands must use the seismic sample axis.")
    md_at_samples = np.interp(samples, table.twt, table.depth, left=np.nan, right=np.nan)
    inline_float, xline_float = survey.line_geometry.coord_to_line(float(surface_x), float(surface_y))
    trace_fields = nearest_trace_fields(survey, inline_float, xline_float)

    rows: list[dict[str, Any]] = []
    attempted = 0
    zone_errors = 0
    reference_ai = bands.reference_ai.values
    lfm_ai = bands.lfm_ai.values
    ginn_target_ai = bands.ginn_target_ai.values
    for seismic_sample_index, (twt_s, md_m) in enumerate(zip(samples, md_at_samples)):
        if not bands.valid_band_mask[seismic_sample_index] or not np.isfinite(md_m):
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
                "seismic_sample_index": int(seismic_sample_index),
                "zone_name": zone_name,
                "u_in_zone": float(u_in_zone),
                "reference_ai": float(reference_ai[seismic_sample_index]),
                "reference_log_ai": float(bands.reference_log_ai.values[seismic_sample_index]),
                "lfm_ai": float(lfm_ai[seismic_sample_index]),
                "lfm_log_ai": float(bands.lfm_log_ai.values[seismic_sample_index]),
                "ginn_target_ai": float(ginn_target_ai[seismic_sample_index]),
                "ginn_target_log_ai": float(bands.ginn_target_log_ai.values[seismic_sample_index]),
                "ginn_band_log_ai": float(bands.ginn_band_log_ai.values[seismic_sample_index]),
                "enhance_residual_log_ai": float(bands.enhance_residual_log_ai.values[seismic_sample_index]),
                "observed_well_sample": bool(bands.observed_mask[seismic_sample_index]),
                "short_gap_interpolated": bool(bands.interpolation_mask[seismic_sample_index]),
                "hampel_conditioned": bool(bands.conditioned_mask[seismic_sample_index]),
                "frequency_band_valid": True,
                "weight": float(weight),
                "batch_corr": batch_corr,
                "batch_nmae": batch_nmae,
                **trace_fields,
            }
        )
    return (
        pd.DataFrame(rows, columns=POINT_COLUMNS),
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
    trace_plan_file: Any,
    bands: WellFrequencyBands,
    target_layer: Any,
    survey: Any,
    samples: np.ndarray,
    weight: float,
    batch_corr: float | None,
    batch_nmae: float | None,
    sample_step_s: float | None = None,
    anchor_eligible: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build target-window point facts for one deviated well from a trace plan."""
    sample_axis = np.asarray(samples, dtype=np.float64).reshape(-1)
    if sample_axis.size < 2 or np.any(np.diff(sample_axis) <= 0.0):
        raise ValueError("samples must be a strictly increasing seismic sample axis.")
    band_twt = np.asarray(bands.reference_log_ai.basis, dtype=float)
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
    rows: list[dict[str, Any]] = []
    zone_errors = 0
    for row_index, row in plan_df.iterrows():
        inline_float = float(row["inline_float"])
        xline_float = float(row["xline_float"])
        twt_s = float(row["twt_s"])
        seismic_sample_index = int(nearest_sample_indices(sample_axis, np.asarray([twt_s]))[0])
        band_index = int(nearest_sample_indices(band_twt, np.asarray([twt_s]))[0])
        if not bands.valid_band_mask[band_index]:
            continue
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
                "seismic_sample_index": seismic_sample_index,
                "zone_name": zone_name,
                "u_in_zone": float(u_in_zone),
                "reference_ai": float(bands.reference_ai.values[band_index]),
                "reference_log_ai": float(bands.reference_log_ai.values[band_index]),
                "lfm_ai": float(bands.lfm_ai.values[band_index]),
                "lfm_log_ai": float(bands.lfm_log_ai.values[band_index]),
                "ginn_target_ai": float(bands.ginn_target_ai.values[band_index]),
                "ginn_target_log_ai": float(bands.ginn_target_log_ai.values[band_index]),
                "ginn_band_log_ai": float(bands.ginn_band_log_ai.values[band_index]),
                "enhance_residual_log_ai": float(bands.enhance_residual_log_ai.values[band_index]),
                "observed_well_sample": bool(bands.observed_mask[band_index]),
                "short_gap_interpolated": bool(bands.interpolation_mask[band_index]),
                "hampel_conditioned": bool(bands.conditioned_mask[band_index]),
                "frequency_band_valid": True,
                "weight": float(weight),
                "batch_corr": batch_corr,
                "batch_nmae": batch_nmae,
                **trace_fields,
            }
        )
    return (
        pd.DataFrame(rows, columns=POINT_COLUMNS),
        {
            "attempted_samples": attempted,
            "valid_points": int(len(rows)),
            "invalid_point_count": int(max(0, attempted - len(rows))),
            "zone_sample_errors": int(zone_errors),
            "unique_trace_count": int(pd.DataFrame(rows)["flat_idx"].nunique()) if rows else 0,
        },
    )


def build_point_conflict_report(point_df: pd.DataFrame, *, value_col: str = "lfm_log_ai") -> pd.DataFrame:
    """Report duplicate nearest trace/sample constraints before aggregation."""
    rows: list[dict[str, Any]] = []
    if point_df.empty:
        return pd.DataFrame(columns=CONFLICT_COLUMNS)
    for (flat_idx, seismic_sample_index), group in point_df.groupby(
        ["flat_idx", "seismic_sample_index"],
        sort=False,
    ):
        if len(group) <= 1:
            continue
        values = group[value_col].to_numpy(dtype=float)
        rows.append(
            {
                "flat_idx": int(flat_idx),
                "seismic_sample_index": int(seismic_sample_index),
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

    frame["_seismic_sample_index_used"] = validate_sample_indices(
        samples,
        frame["twt_s"].to_numpy(dtype=np.float64),
        frame["seismic_sample_index"].to_numpy(dtype=np.float64),
        field_name="seismic_sample_index",
    )
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
        for seismic_sample_index, sample_group in flat_group.groupby("_seismic_sample_index_used", sort=False):
            sample_idx = int(seismic_sample_index)
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
    values = point_df["enhance_residual_log_ai"].to_numpy(dtype=float)
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
        high = group.sort_values("twt_s")["enhance_residual_log_ai"].to_numpy(dtype=float)
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
        values = group.sort_values("twt_s")["enhance_residual_log_ai"].to_numpy(dtype=float)
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
            values = group.sort_values("twt_s")["enhance_residual_log_ai"].to_numpy(dtype=float)
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
