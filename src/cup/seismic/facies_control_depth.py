"""Depth-domain facies-control utilities for AI low-frequency models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

SCHEMA_VERSION = "cup_depth_facies_control_v1"
REQUIRED_COLUMNS = {"x", "y", "depth_m", "radius_xy_m", "radius_z_m", "target_ai"}
OPTIONAL_COLUMNS = {"name", "strength"}


class _SurveyLike(Protocol):
    def coord_to_line(self, x: float, y: float) -> tuple[float, float]:
        ...

    def line_to_coord(self, il_no: float, xl_no: float) -> tuple[float, float]:
        ...


@dataclass(frozen=True)
class FaciesControlPoint:
    name: str
    x: float
    y: float
    depth_m: float
    radius_xy_m: float
    radius_z_m: float
    target_ai: float
    strength: float = 1.0


def load_depth_facies_control_points_csv(path: str | Path) -> list[FaciesControlPoint]:
    """Load and validate depth-domain facies control points from CSV."""
    path = Path(path)
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Facies control CSV is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"Facies control CSV is empty: {path}")

    points: list[FaciesControlPoint] = []
    for row_idx, row in df.iterrows():
        name = str(row.get("name", "")).strip()
        if not name or name.lower() == "nan":
            name = f"control_{row_idx + 1:03d}"
        strength = 1.0 if "strength" not in df.columns or pd.isna(row.get("strength")) else float(row["strength"])
        point = FaciesControlPoint(
            name=name,
            x=float(row["x"]),
            y=float(row["y"]),
            depth_m=float(row["depth_m"]),
            radius_xy_m=float(row["radius_xy_m"]),
            radius_z_m=float(row["radius_z_m"]),
            target_ai=float(row["target_ai"]),
            strength=strength,
        )
        validate_control_point(point)
        points.append(point)
    return points


def validate_control_point(point: FaciesControlPoint) -> None:
    """Validate numeric constraints for one control point."""
    values = {
        "x": point.x,
        "y": point.y,
        "depth_m": point.depth_m,
        "radius_xy_m": point.radius_xy_m,
        "radius_z_m": point.radius_z_m,
        "target_ai": point.target_ai,
        "strength": point.strength,
    }
    bad = [name for name, value in values.items() if not np.isfinite(float(value))]
    if bad:
        raise ValueError(f"Facies control point {point.name!r} has non-finite values: {bad}")
    if point.radius_xy_m <= 0.0:
        raise ValueError(f"Facies control point {point.name!r} radius_xy_m must be positive.")
    if point.radius_z_m <= 0.0:
        raise ValueError(f"Facies control point {point.name!r} radius_z_m must be positive.")
    if point.target_ai <= 0.0:
        raise ValueError(f"Facies control point {point.name!r} target_ai must be positive.")
    if not (0.0 <= point.strength <= 1.0):
        raise ValueError(f"Facies control point {point.name!r} strength must be within [0, 1].")


def raised_cosine_weight(normalized_distance: np.ndarray | float) -> np.ndarray:
    """Return a compact raised-cosine weight for normalized distance in [0, 1]."""
    d = np.asarray(normalized_distance, dtype=np.float64)
    weight = np.zeros_like(d, dtype=np.float64)
    inside = (d >= 0.0) & (d <= 1.0)
    weight[inside] = 0.5 * (1.0 + np.cos(np.pi * d[inside]))
    return weight


def build_trace_xy_grids(
    survey: _SurveyLike,
    ilines: np.ndarray,
    xlines: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build trace-center XY grids from survey line coordinates."""
    ilines = np.asarray(ilines, dtype=np.float64)
    xlines = np.asarray(xlines, dtype=np.float64)
    if ilines.ndim != 1 or xlines.ndim != 1 or ilines.size == 0 or xlines.size == 0:
        raise ValueError("ilines and xlines must be non-empty 1D arrays.")

    x0, y0 = survey.line_to_coord(float(ilines[0]), float(xlines[0]))
    if ilines.size > 1:
        x1, y1 = survey.line_to_coord(float(ilines[-1]), float(xlines[0]))
        dx_i = (x1 - x0) / float(ilines.size - 1)
        dy_i = (y1 - y0) / float(ilines.size - 1)
    else:
        dx_i = dy_i = 0.0
    if xlines.size > 1:
        x2, y2 = survey.line_to_coord(float(ilines[0]), float(xlines[-1]))
        dx_j = (x2 - x0) / float(xlines.size - 1)
        dy_j = (y2 - y0) / float(xlines.size - 1)
    else:
        dx_j = dy_j = 0.0

    i_idx = np.arange(ilines.size, dtype=np.float64)[:, None]
    j_idx = np.arange(xlines.size, dtype=np.float64)[None, :]
    x_grid = x0 + i_idx * dx_i + j_idx * dx_j
    y_grid = y0 + i_idx * dy_i + j_idx * dy_j
    return x_grid.astype(np.float64), y_grid.astype(np.float64)


def build_target_layer_from_lfm_metadata(
    metadata: dict[str, Any],
    geometry: dict[str, Any],
    *,
    qc_output_dir: str | Path | None = None,
) -> Any:
    """Rebuild a TargetLayer from horizon metadata stored in an AI LFM NPZ."""
    from cup.petrel.load import import_interpretation_petrel
    from cup.seismic.target_layer import TargetLayer

    horizons = metadata.get("horizons", [])
    if not isinstance(horizons, list) or len(horizons) < 2:
        raise ValueError("AI LFM metadata must contain at least two horizons.")

    raw_horizons: dict[str, pd.DataFrame] = {}
    horizon_names: list[str] = []
    for idx, item in enumerate(horizons):
        if not isinstance(item, dict) or not item.get("file"):
            raise ValueError(f"Invalid horizon metadata entry at index {idx}: {item!r}")
        name = str(item.get("name") or f"horizon_{idx}")
        horizon_names.append(name)
        raw_horizons[name] = import_interpretation_petrel(Path(str(item["file"])))

    tl_meta = metadata.get("target_layer", {})
    if not isinstance(tl_meta, dict):
        tl_meta = {}
    return TargetLayer(
        raw_horizon_dfs=raw_horizons,
        geometry=geometry,
        horizon_names=horizon_names,
        qc_output_dir=qc_output_dir,
        min_thickness=tl_meta.get("min_thickness"),
        nearest_distance_limit=tl_meta.get("nearest_distance_limit"),
        outlier_threshold=tl_meta.get("outlier_threshold"),
        outlier_min_neighbor_count=tl_meta.get("outlier_min_neighbor_count", 2),
    )


def locate_control_zone(
    target_layer: Any,
    *,
    inline: float,
    xline: float,
    depth_m: float,
) -> tuple[str, str, dict[str, float]]:
    """Locate the adjacent horizon pair containing a control point depth."""
    horizon_values = target_layer.get_interpretation_values_at_location(inline, xline)
    names = list(target_layer.horizon_names)
    for top_name, bottom_name in zip(names[:-1], names[1:]):
        top = float(horizon_values[top_name])
        bottom = float(horizon_values[bottom_name])
        if not np.isfinite(top) or not np.isfinite(bottom):
            continue
        lo, hi = min(top, bottom), max(top, bottom)
        if lo <= float(depth_m) <= hi:
            return top_name, bottom_name, {name: float(horizon_values[name]) for name in names}
    raise ValueError(
        f"Control depth {float(depth_m):.3f} m is outside all target-layer zones at "
        f"inline={inline:.3f}, xline={xline:.3f}."
    )


def apply_depth_facies_controls(
    volume: np.ndarray,
    *,
    ilines: np.ndarray,
    xlines: np.ndarray,
    samples: np.ndarray,
    target_layer: Any,
    survey: _SurveyLike,
    control_points: list[FaciesControlPoint],
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
    error_on_empty: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply facies controls to an AI volume using layer-clipped log-AI blending."""
    ai = np.asarray(volume, dtype=np.float32)
    if ai.ndim != 3:
        raise ValueError(f"Expected volume ndim=3, got {ai.ndim}.")
    ilines = np.asarray(ilines, dtype=np.float64)
    xlines = np.asarray(xlines, dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    expected_shape = (ilines.size, xlines.size, samples.size)
    if ai.shape != expected_shape:
        raise ValueError(f"Volume shape {ai.shape} does not match axes {expected_shape}.")
    if np.any(~np.isfinite(ai)) or np.any(ai <= 0.0):
        raise ValueError("AI volume must contain only finite positive values.")

    if x_grid is None or y_grid is None:
        x_grid, y_grid = build_trace_xy_grids(survey, ilines, xlines)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)
    if x_grid.shape != ai.shape[:2] or y_grid.shape != ai.shape[:2]:
        raise ValueError("x_grid/y_grid shape must match volume trace shape.")

    controlled = ai.copy()
    affected_samples: set[int] = set()
    qc_rows: list[dict[str, Any]] = []
    n_xl = xlines.size
    n_sample = samples.size

    for order, point in enumerate(control_points):
        inline, xline = survey.coord_to_line(point.x, point.y)
        top_name, bottom_name, horizon_values = locate_control_zone(
            target_layer,
            inline=float(inline),
            xline=float(xline),
            depth_m=point.depth_m,
        )
        top_grid = np.asarray(target_layer.get_horizon_grid(top_name), dtype=np.float64)
        bottom_grid = np.asarray(target_layer.get_horizon_grid(bottom_name), dtype=np.float64)
        zone_top = np.minimum(top_grid, bottom_grid)
        zone_bottom = np.maximum(top_grid, bottom_grid)

        d_xy = np.hypot(x_grid - point.x, y_grid - point.y)
        trace_mask = d_xy <= point.radius_xy_m
        trace_indices = np.argwhere(trace_mask)

        affected_trace_count = 0
        affected_sample_count = 0
        overlap_sample_count = 0
        weight_sum = 0.0
        max_weight = 0.0
        before_sum = 0.0
        after_sum = 0.0
        delta_abs_max = 0.0

        for i, j in trace_indices:
            i_int = int(i)
            j_int = int(j)
            z_in_radius = np.abs(samples - point.depth_m) <= point.radius_z_m
            z_in_layer = (samples >= zone_top[i_int, j_int]) & (samples <= zone_bottom[i_int, j_int])
            z_mask = z_in_radius & z_in_layer
            if not np.any(z_mask):
                continue

            sample_idx = np.flatnonzero(z_mask)
            w_xy = float(raised_cosine_weight(float(d_xy[i_int, j_int]) / point.radius_xy_m).item())
            w_z = raised_cosine_weight(np.abs(samples[sample_idx] - point.depth_m) / point.radius_z_m)
            weights = (point.strength * w_xy * w_z).astype(np.float64)
            positive = weights > 0.0
            if not np.any(positive):
                continue
            sample_idx = sample_idx[positive]
            weights = weights[positive]

            before = controlled[i_int, j_int, sample_idx].astype(np.float64)
            after_log = (1.0 - weights) * np.log(before) + weights * np.log(point.target_ai)
            after = np.exp(after_log).astype(np.float64)
            controlled[i_int, j_int, sample_idx] = after.astype(np.float32)

            linear_ids = ((i_int * n_xl + j_int) * n_sample + sample_idx).astype(np.int64)
            overlap_sample_count += sum(int(idx) in affected_samples for idx in linear_ids)
            affected_samples.update(int(idx) for idx in linear_ids)

            affected_trace_count += 1
            affected_sample_count += int(sample_idx.size)
            weight_sum += float(weights.sum())
            max_weight = max(max_weight, float(weights.max()))
            before_sum += float(before.sum())
            after_sum += float(after.sum())
            if sample_idx.size:
                delta_abs_max = max(delta_abs_max, float(np.max(np.abs(after - before))))

        if affected_sample_count == 0 and error_on_empty:
            raise ValueError(f"Facies control point {point.name!r} affected no samples.")

        qc_rows.append(
            {
                "order": int(order),
                "name": point.name,
                "x": float(point.x),
                "y": float(point.y),
                "depth_m": float(point.depth_m),
                "inline": float(inline),
                "xline": float(xline),
                "zone_top": top_name,
                "zone_bottom": bottom_name,
                "radius_xy_m": float(point.radius_xy_m),
                "radius_z_m": float(point.radius_z_m),
                "target_ai": float(point.target_ai),
                "strength": float(point.strength),
                "affected_traces": int(affected_trace_count),
                "affected_samples": int(affected_sample_count),
                "overlap_samples": int(overlap_sample_count),
                "max_weight": float(max_weight),
                "mean_weight": float(weight_sum / affected_sample_count) if affected_sample_count else 0.0,
                "mean_ai_before": float(before_sum / affected_sample_count) if affected_sample_count else np.nan,
                "mean_ai_after": float(after_sum / affected_sample_count) if affected_sample_count else np.nan,
                "mean_ai_delta": float((after_sum - before_sum) / affected_sample_count)
                if affected_sample_count
                else np.nan,
                "max_abs_ai_delta": float(delta_abs_max),
                "horizon_values": horizon_values,
            }
        )

    if np.any(~np.isfinite(controlled)) or np.any(controlled <= 0.0):
        raise ValueError("Controlled AI volume contains non-finite or non-positive values.")
    return controlled.astype(np.float32, copy=False), pd.DataFrame.from_records(qc_rows)
