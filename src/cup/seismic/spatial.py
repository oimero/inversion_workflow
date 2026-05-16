"""XY spatial helpers for regular seismic trace grids."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class SurveyLineToCoord(Protocol):
    """Minimum survey interface needed to resolve inline/xline to XY."""

    def line_to_coord(self, il_no: float, xl_no: float) -> tuple[float, float]: ...


def build_trace_xy_grids(
    survey: SurveyLineToCoord,
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


def nominal_bin_spacing_m(x_grid: np.ndarray, y_grid: np.ndarray) -> float:
    """Return a robust nominal trace-center spacing in XY meters."""
    x = np.asarray(x_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x_grid and y_grid must be matching 2D arrays.")

    spacings: list[np.ndarray] = []
    if x.shape[0] > 1:
        spacings.append(np.hypot(np.diff(x, axis=0), np.diff(y, axis=0)).reshape(-1))
    if x.shape[1] > 1:
        spacings.append(np.hypot(np.diff(x, axis=1), np.diff(y, axis=1)).reshape(-1))
    if not spacings:
        return 0.0

    values = np.concatenate(spacings)
    values = values[np.isfinite(values) & (values > 0.0)]
    return float(np.median(values)) if values.size else 0.0


def xy_distance_grid(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    center_x: float,
    center_y: float,
) -> np.ndarray:
    """Compute XY Euclidean distance from every trace center to one center point."""
    x = np.asarray(x_grid, dtype=np.float64)
    y = np.asarray(y_grid, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x_grid and y_grid must be matching 2D arrays.")
    return np.hypot(x - float(center_x), y - float(center_y))


def xy_circle_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    radius_xy_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mask, distance_m)`` for a true XY circular radius."""
    radius = float(radius_xy_m)
    if radius < 0.0:
        raise ValueError(f"radius_xy_m must be non-negative, got {radius}.")
    distance = xy_distance_grid(x_grid, y_grid, center_x=center_x, center_y=center_y)
    if radius == 0.0:
        min_distance = float(np.nanmin(distance))
        mask = np.isclose(distance, min_distance, rtol=0.0, atol=1e-6)
    else:
        mask = distance <= radius + 1e-8
    return mask, distance
