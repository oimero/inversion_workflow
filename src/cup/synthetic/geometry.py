"""Section geometry construction for field-conditioned synthoseis-lite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cup.petrel.load import import_interpretation_petrel
from cup.seismic.survey import open_survey
from cup.seismic.target_zone import TargetZone
from cup.time_config import TimeWorkflowConfig
from cup.utils.io import resolve_relative_path


@dataclass(frozen=True)
class SectionGeometry:
    section_id: str
    lateral_m: np.ndarray
    inline_float: np.ndarray
    xline_float: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    horizon_twt_s: np.ndarray
    qc_rows: tuple[dict[str, Any], ...] = ()

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

def _nearest_grid_index(value: float, *, minimum: float, step: float, size: int) -> int:
    index = int(round((float(value) - float(minimum)) / float(step)))
    return max(0, min(index, int(size) - 1))

def _section_target_zone_qc_rows(
    *,
    section_id: str,
    lateral_m: np.ndarray,
    inline_float: np.ndarray,
    xline_float: np.ndarray,
    horizon_values: np.ndarray,
    target_zone: TargetZone,
    horizon_names: Sequence[str],
    geometry_mode: str,
) -> list[dict[str, Any]]:
    geometry = target_zone.geometry
    inline_min = float(geometry["inline_min"])
    xline_min = float(geometry["xline_min"])
    inline_step = float(geometry["inline_step"])
    xline_step = float(geometry["xline_step"])
    inline_size, xline_size = target_zone.valid_control_mask.shape
    rows: list[dict[str, Any]] = []
    for sample_index, (distance, il, xl) in enumerate(zip(lateral_m, inline_float, xline_float)):
        i = _nearest_grid_index(il, minimum=inline_min, step=inline_step, size=inline_size)
        j = _nearest_grid_index(xl, minimum=xline_min, step=xline_step, size=xline_size)
        trace_valid_control = bool(target_zone.valid_control_mask[i, j])
        trace_filled_model = bool(target_zone.filled_model_mask[i, j])
        trace_filled_by_thickness = bool(trace_filled_model and not trace_valid_control)
        for horizon_index, horizon_name in enumerate(horizon_names):
            surface = target_zone.get_horizon_surface(str(horizon_name))
            sample = surface.sample_at_line(float(il), float(xl))
            rows.append(
                {
                    "section_id": section_id,
                    "geometry_mode": geometry_mode,
                    "sample_index": int(sample_index),
                    "lateral_m": float(distance),
                    "inline_float": float(il),
                    "xline_float": float(xl),
                    "nearest_grid_inline": float(inline_min + i * inline_step),
                    "nearest_grid_xline": float(xline_min + j * xline_step),
                    "nearest_grid_index_inline": int(i),
                    "nearest_grid_index_xline": int(j),
                    "horizon_name": str(horizon_name),
                    "horizon_twt_s": float(horizon_values[sample_index, horizon_index]),
                    "horizon_sample_method": str(sample.method),
                    "horizon_support_status": str(sample.support_status),
                    "raw_pick": bool(target_zone.raw_pick_masks[str(horizon_name)][i, j]),
                    "linear_support": bool(
                        target_zone.interpolation_support_masks[str(horizon_name)][i, j]
                    ),
                    "nearest_distance_grid": float(
                        target_zone.nearest_distance_grids[str(horizon_name)][i, j]
                    ),
                    "trace_valid_control": trace_valid_control,
                    "trace_filled_model": trace_filled_model,
                    "trace_filled_by_thickness_interpolation": trace_filled_by_thickness,
                    "trace_no_support": bool(target_zone.no_support_mask[i, j]),
                    "trace_crossing": bool(target_zone.crossing_mask[i, j]),
                    "trace_thin": bool(target_zone.thin_mask[i, j]),
                }
            )
    return rows

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
    target_zone_cfg = dict(script_cfg.get("target_zone") or {})
    mode = str(target_zone_cfg.get("mode", "filled_target_zone"))
    if mode != "filled_target_zone":
        raise ValueError(
            "synthoseis_lite field-conditioned geometry only supports "
            f"filled_target_zone, got {mode!r}."
        )
    survey_geometry = survey.describe_geometry(domain="time")
    raw_horizon_dfs: dict[str, pd.DataFrame] = {}
    ordered_horizons = [str(item["name"]) for item in script_cfg["horizons"]]
    for horizon in script_cfg["horizons"]:
        path = resolve_relative_path(
            horizon["file"],
            root=resolve_relative_path(workflow.data_root, root=repo_root),
        )
        frame = import_interpretation_petrel(path)
        frame = frame.copy()
        frame["interpretation"] = np.abs(frame["interpretation"].to_numpy(dtype=np.float64))
        raw_horizon_dfs[str(horizon["name"])] = frame
    target_zone = TargetZone(
        raw_horizon_dfs,
        survey_geometry,
        ordered_horizons,
        nearest_distance_limit=target_zone_cfg.get("nearest_distance_limit"),
        outlier_threshold=target_zone_cfg.get("outlier_threshold"),
        outlier_min_neighbor_count=int(target_zone_cfg.get("outlier_min_neighbor_count", 2)),
        min_thickness=target_zone_cfg.get("min_thickness_s"),
    )
    surfaces = [target_zone.get_horizon_surface(name) for name in ordered_horizons]
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
        qc_rows = _section_target_zone_qc_rows(
            section_id=str(section["section_id"]),
            lateral_m=lateral,
            inline_float=inline,
            xline_float=xline,
            horizon_values=horizon_values,
            target_zone=target_zone,
            horizon_names=ordered_horizons,
            geometry_mode=mode,
        )
        sections.append(
            SectionGeometry(
                section_id=section["section_id"],
                lateral_m=lateral,
                inline_float=inline,
                xline_float=xline,
                x_m=x,
                y_m=y,
                horizon_twt_s=horizon_values,
                qc_rows=tuple(qc_rows),
            )
        )
    return sections
