"""Depth-domain facies-control log-AI anchor helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np
import pandas as pd

from cup.seismic.facies_control_depth import FaciesControlPoint, locate_control_zone
from ginn.anchor import LogAIAnchorBundle, build_log_ai_anchor_bundle, validate_log_ai_anchor


class _SurveyLike(Protocol):
    def coord_to_line(self, x: float, y: float) -> tuple[float, float]:
        ...

    def line_to_coord(self, il_no: float, xl_no: float) -> tuple[float, float]:
        ...


@dataclass(frozen=True)
class FaciesAnchorBuildResult:
    bundle: LogAIAnchorBundle
    qc: pd.DataFrame


@dataclass(frozen=True)
class AnchorMergeResult:
    bundle: LogAIAnchorBundle
    qc: pd.DataFrame


def nearest_trace_from_xy(
    *,
    ilines: np.ndarray,
    xlines: np.ndarray,
    survey: _SurveyLike,
    x: float,
    y: float,
) -> tuple[int, int, int, float, float, float, float]:
    """Resolve XY coordinates to the nearest LFM trace."""
    inline, xline = survey.coord_to_line(float(x), float(y))
    il_idx = int(np.argmin(np.abs(np.asarray(ilines, dtype=np.float64) - float(inline))))
    xl_idx = int(np.argmin(np.abs(np.asarray(xlines, dtype=np.float64) - float(xline))))
    flat_idx = il_idx * int(np.asarray(xlines).size) + xl_idx
    nearest_inline = float(np.asarray(ilines, dtype=np.float64)[il_idx])
    nearest_xline = float(np.asarray(xlines, dtype=np.float64)[xl_idx])
    return flat_idx, il_idx, xl_idx, float(inline), float(xline), nearest_inline, nearest_xline


def build_facies_control_anchor_bundle(
    *,
    control_points: list[FaciesControlPoint],
    ilines: np.ndarray,
    xlines: np.ndarray,
    samples: np.ndarray,
    target_layer: Any,
    survey: _SurveyLike,
    metadata: dict[str, Any] | None = None,
) -> FaciesAnchorBuildResult:
    """Build stage-1 anchor rows from facies control points."""
    samples = np.asarray(samples, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for order, point in enumerate(control_points):
        flat_idx, il_idx, xl_idx, resolved_inline, resolved_xline, nearest_inline, nearest_xline = nearest_trace_from_xy(
            ilines=ilines,
            xlines=xlines,
            survey=survey,
            x=point.x,
            y=point.y,
        )
        zone_top, zone_bottom, horizon_values = locate_control_zone(
            target_layer,
            inline=resolved_inline,
            xline=resolved_xline,
            depth_m=point.depth_m,
        )
        top_grid = np.asarray(target_layer.get_horizon_grid(zone_top), dtype=np.float64)
        bottom_grid = np.asarray(target_layer.get_horizon_grid(zone_bottom), dtype=np.float64)
        layer_top = float(min(top_grid[il_idx, xl_idx], bottom_grid[il_idx, xl_idx]))
        layer_bottom = float(max(top_grid[il_idx, xl_idx], bottom_grid[il_idx, xl_idx]))
        depth_mask = np.abs(samples.astype(np.float64) - float(point.depth_m)) <= float(point.radius_z_m)
        layer_mask = (samples.astype(np.float64) >= layer_top) & (samples.astype(np.float64) <= layer_bottom)
        mask = depth_mask & layer_mask
        if not np.any(mask):
            raise ValueError(f"Facies control point {point.name!r} produced an empty anchor mask.")

        target_ai = np.zeros(samples.shape, dtype=np.float32)
        target_ai[mask] = float(point.target_ai)
        anchor_weight = np.zeros(samples.shape, dtype=np.float32)
        anchor_weight[mask] = float(point.strength)

        rows.append(
            {
                "flat_idx": int(flat_idx),
                "inline": float(nearest_inline),
                "xline": float(nearest_xline),
                "name": point.name,
                "target_ai": target_ai,
                "mask": mask.astype(bool),
                "weight": anchor_weight,
            }
        )
        qc_rows.append(
            {
                "order": int(order),
                "name": point.name,
                "x": float(point.x),
                "y": float(point.y),
                "depth_m": float(point.depth_m),
                "radius_xy_m": float(point.radius_xy_m),
                "radius_z_m": float(point.radius_z_m),
                "target_ai": float(point.target_ai),
                "strength": float(point.strength),
                "resolved_inline": float(resolved_inline),
                "resolved_xline": float(resolved_xline),
                "nearest_inline": float(nearest_inline),
                "nearest_xline": float(nearest_xline),
                "flat_idx": int(flat_idx),
                "zone_top": zone_top,
                "zone_bottom": zone_bottom,
                "layer_top_m": layer_top,
                "layer_bottom_m": layer_bottom,
                "valid_samples": int(np.count_nonzero(mask)),
                "horizon_values": horizon_values,
            }
        )

    duplicates = [flat for flat, count in Counter(row["flat_idx"] for row in rows).items() if count > 1]
    if duplicates:
        names = [row["name"] for row in rows if row["flat_idx"] in duplicates]
        raise ValueError(f"Duplicate nearest traces are not supported for facies anchors v1: {duplicates}, controls={names}")

    created_at = datetime.now(timezone.utc).isoformat()
    bundle = build_log_ai_anchor_bundle(
        sample_domain="depth",
        sample_unit="m",
        samples=samples,
        flat_indices=np.array([row["flat_idx"] for row in rows], dtype=np.int64),
        target_ai=np.stack([row["target_ai"] for row in rows]).astype(np.float32),
        anchor_mask=np.stack([row["mask"] for row in rows]).astype(bool),
        anchor_weight=np.stack([row["weight"] for row in rows]).astype(np.float32),
        anchor_names=np.array([row["name"] for row in rows]),
        anchor_types=np.array(["facies_control"] * len(rows)),
        inline=np.array([row["inline"] for row in rows], dtype=np.float32),
        xline=np.array([row["xline"] for row in rows], dtype=np.float32),
        metadata={
            "created_at_utc": created_at,
            "artifact_role": "facies_control_log_ai_anchor",
            **({} if metadata is None else metadata),
        },
    )
    return FaciesAnchorBuildResult(bundle=bundle, qc=pd.DataFrame.from_records(qc_rows))


def merge_well_and_facies_anchor_bundles(
    *,
    well_bundle: LogAIAnchorBundle,
    facies_bundle: LogAIAnchorBundle,
    survey: _SurveyLike,
    min_well_facies_separation_m: float,
    metadata: dict[str, Any] | None = None,
) -> AnchorMergeResult:
    """Merge well and facies anchors, skipping facies anchors too close to wells."""
    validate_log_ai_anchor(well_bundle)
    validate_log_ai_anchor(facies_bundle)
    if well_bundle.sample_domain != facies_bundle.sample_domain:
        raise ValueError("Anchor bundles must share sample_domain.")
    if well_bundle.sample_unit != facies_bundle.sample_unit:
        raise ValueError("Anchor bundles must share sample_unit.")
    if not np.array_equal(well_bundle.samples, facies_bundle.samples):
        raise ValueError("Anchor bundles must share identical sample axes.")
    if min_well_facies_separation_m < 0.0:
        raise ValueError("min_well_facies_separation_m must be non-negative.")

    well_xy = np.array(
        [survey.line_to_coord(float(il), float(xl)) for il, xl in zip(well_bundle.inline, well_bundle.xline)],
        dtype=np.float64,
    )
    keep_facies_rows: list[int] = []
    qc_rows: list[dict[str, Any]] = []

    for row, (name, inline, xline) in enumerate(zip(facies_bundle.anchor_names, facies_bundle.inline, facies_bundle.xline)):
        facies_x, facies_y = survey.line_to_coord(float(inline), float(xline))
        if well_xy.size:
            dists = np.hypot(well_xy[:, 0] - facies_x, well_xy[:, 1] - facies_y)
            nearest_idx = int(np.argmin(dists))
            nearest_distance = float(dists[nearest_idx])
            nearest_well_name = str(well_bundle.anchor_names[nearest_idx])
        else:
            nearest_distance = float("inf")
            nearest_well_name = ""
        skipped = nearest_distance < float(min_well_facies_separation_m)
        if not skipped:
            keep_facies_rows.append(row)
        qc_rows.append(
            {
                "facies_anchor_name": str(name),
                "facies_inline": float(inline),
                "facies_xline": float(xline),
                "facies_x": float(facies_x),
                "facies_y": float(facies_y),
                "nearest_well_anchor_name": nearest_well_name,
                "nearest_well_distance_m": nearest_distance,
                "min_well_facies_separation_m": float(min_well_facies_separation_m),
                "status": "skipped_too_close_to_well" if skipped else "kept",
            }
        )

    rows = np.asarray(keep_facies_rows, dtype=np.int64)
    merged_flat_indices = np.concatenate([well_bundle.flat_indices, facies_bundle.flat_indices[rows]])
    if np.unique(merged_flat_indices).size != merged_flat_indices.size:
        raise ValueError("Combined anchor would contain duplicate flat_indices after spacing checks.")

    merged = build_log_ai_anchor_bundle(
        sample_domain=well_bundle.sample_domain,
        sample_unit=well_bundle.sample_unit,
        samples=well_bundle.samples,
        flat_indices=merged_flat_indices,
        target_ai=np.concatenate([well_bundle.target_ai, facies_bundle.target_ai[rows]], axis=0),
        anchor_mask=np.concatenate([well_bundle.anchor_mask, facies_bundle.anchor_mask[rows]], axis=0),
        anchor_weight=np.concatenate([well_bundle.anchor_weight, facies_bundle.anchor_weight[rows]], axis=0),
        anchor_names=np.concatenate([well_bundle.anchor_names, facies_bundle.anchor_names[rows]], axis=0),
        anchor_types=np.concatenate([well_bundle.anchor_types, facies_bundle.anchor_types[rows]], axis=0),
        inline=np.concatenate([well_bundle.inline, facies_bundle.inline[rows]], axis=0),
        xline=np.concatenate([well_bundle.xline, facies_bundle.xline[rows]], axis=0),
        metadata={
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_role": "combined_log_ai_anchor",
            "n_well_anchors": int(well_bundle.n_anchors),
            "n_facies_input": int(facies_bundle.n_anchors),
            "n_facies_kept": int(rows.size),
            "n_facies_skipped": int(facies_bundle.n_anchors - rows.size),
            "min_well_facies_separation_m": float(min_well_facies_separation_m),
            **({} if metadata is None else metadata),
        },
    )
    return AnchorMergeResult(bundle=merged, qc=pd.DataFrame.from_records(qc_rows))
