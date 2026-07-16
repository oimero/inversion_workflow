"""Domain-neutral section path geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SectionGeometry:
    section_id: str
    lateral_m: np.ndarray
    inline_float: np.ndarray
    xline_float: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    horizon_coordinates: np.ndarray
    sample_domain: str
    axis_unit: str
    depth_basis: str | None = None
    qc_rows: tuple[dict[str, Any], ...] = ()


def validate_line_geometry(geometry: Any) -> None:
    for axis in (geometry.inline_axis, geometry.xline_axis):
        if axis.count <= 0 or not np.isfinite(axis.step) or (axis.count > 1 and axis.step == 0.0):
            raise ValueError(f"invalid survey line axis: {axis.name}")
    tolerance = 1e-8 * max(abs(geometry.inline_axis.step), abs(geometry.xline_axis.step), 1.0)
    for inline in (geometry.inline_axis.minimum, geometry.inline_axis.maximum):
        for xline in (geometry.xline_axis.minimum, geometry.xline_axis.maximum):
            restored = geometry.coord_to_line(*geometry.line_to_coord(inline, xline))
            if not np.allclose(restored, (inline, xline), rtol=0.0, atol=tolerance):
                raise ValueError("survey line/XY round-trip failed")


def resample_section_path(
    points: Sequence[Mapping[str, float]], *, geometry: Any, sample_interval_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    validate_line_geometry(geometry)
    lines = np.asarray([[float(item["inline"]), float(item["xline"])] for item in points])
    xy = np.asarray([geometry.line_to_coord(il, xl) for il, xl in lines], dtype=np.float64)
    cumulative = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if cumulative[-1] <= 0.0:
        raise ValueError("invalid_section_path")
    lateral = np.arange(0.0, cumulative[-1], float(sample_interval_m), dtype=np.float64)
    if lateral.size == 0 or not np.isclose(lateral[-1], cumulative[-1]):
        lateral = np.r_[lateral, cumulative[-1]]
    x = np.interp(lateral, cumulative, xy[:, 0])
    y = np.interp(lateral, cumulative, xy[:, 1])
    samples = np.asarray([geometry.coord_to_line(xi, yi) for xi, yi in zip(x, y)])
    roundtrip = np.asarray([geometry.line_to_coord(il, xl) for il, xl in samples])
    tolerance = max(1e-7, 1e-8 * max(float(np.ptp(x)), float(np.ptp(y)), 1.0))
    if not np.allclose(roundtrip, np.column_stack((x, y)), rtol=0.0, atol=tolerance):
        raise ValueError("section_line_xy_roundtrip_failed")
    return lateral, samples[:, 0], samples[:, 1], x, y


__all__ = ["SectionGeometry", "resample_section_path", "validate_line_geometry"]
