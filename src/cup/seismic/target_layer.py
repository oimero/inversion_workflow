"""Fault-tolerant target-layer preparation from raw interpretation picks.

This module defines a ``TargetLayer`` that starts from raw, uninterpolated
horizon interpretations.  It builds interpolation support masks, trace-level QC
masks, and full-coverage horizon grids reconstructed from interpolated positive
thicknesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter
from scipy.spatial import QhullError, cKDTree  # type: ignore

OUTLIER_REMOVAL_WARNING_RATIO = 0.05


@dataclass
class _SurfaceInterpolation:
    raw_grid: np.ndarray
    despiked_grid: np.ndarray
    linear_grid: np.ndarray
    nearest_grid: np.ndarray
    linear_support_mask: np.ndarray
    raw_mask: np.ndarray
    nearest_distance_grid: np.ndarray
    outlier_stats: Dict[str, Any]


def _build_axis(axis_min: float, axis_max: float, axis_step: float, axis_name: str) -> np.ndarray:
    if axis_step <= 0:
        raise ValueError(f"{axis_name}_step must be positive, got {axis_step}.")
    return np.arange(float(axis_min), float(axis_max) + float(axis_step), float(axis_step), dtype=float)


def _validate_geometry_keys(geometry: Dict[str, Any]) -> None:
    required = {
        "inline_min",
        "inline_max",
        "inline_step",
        "xline_min",
        "xline_max",
        "xline_step",
        "sample_min",
        "sample_max",
        "sample_step",
    }
    missing = required - set(geometry)
    if missing:
        raise ValueError(f"geometry is missing required keys: {sorted(missing)}")


def _normalize_interpretation_unit_for_geometry(
    interpretation_df: pd.DataFrame,
    geometry: Dict[str, Any],
) -> pd.DataFrame:
    sample_domain = str(geometry.get("sample_domain", "")).lower()
    sample_unit = str(geometry.get("sample_unit", "")).lower()
    if sample_domain != "time" or sample_unit != "s" or "interpretation" not in interpretation_df.columns:
        return interpretation_df.copy()

    z = interpretation_df["interpretation"].to_numpy(dtype=float, copy=False)
    finite = np.isfinite(z)
    if not np.any(finite) or float(np.nanmax(np.abs(z[finite]))) <= 10.0:
        return interpretation_df.copy()

    out_df = interpretation_df.copy()
    converted = out_df["interpretation"].to_numpy(dtype=float, copy=True)
    converted[finite] = converted[finite] / 1000.0
    out_df["interpretation"] = converted
    return out_df


def _require_interpretation_columns(df: pd.DataFrame, name: str) -> None:
    required = {"inline", "xline", "interpretation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"horizon '{name}' is missing required columns: {sorted(missing)}")


def _grid_raw_interpretation(
    interpretation_df: pd.DataFrame,
    il_axis: np.ndarray,
    xl_axis: np.ndarray,
) -> np.ndarray:
    grid = np.full((il_axis.size, xl_axis.size), np.nan, dtype=float)

    values = interpretation_df[["inline", "xline", "interpretation"]].to_numpy(dtype=float, copy=False)
    finite = np.isfinite(values).all(axis=1)
    if not np.any(finite):
        return grid

    values = values[finite]
    il_min = float(il_axis[0])
    xl_min = float(xl_axis[0])
    il_step = float(il_axis[1] - il_axis[0]) if il_axis.size > 1 else 1.0
    xl_step = float(xl_axis[1] - xl_axis[0]) if xl_axis.size > 1 else 1.0

    il_float = (values[:, 0] - il_min) / il_step
    xl_float = (values[:, 1] - xl_min) / xl_step
    il_idx = np.rint(il_float).astype(np.int64)
    xl_idx = np.rint(xl_float).astype(np.int64)
    tol = 1e-8
    on_grid = (
        np.isclose(il_float, il_idx, atol=tol)
        & np.isclose(xl_float, xl_idx, atol=tol)
        & (il_idx >= 0)
        & (il_idx < il_axis.size)
        & (xl_idx >= 0)
        & (xl_idx < xl_axis.size)
    )
    if not np.any(on_grid):
        return grid

    gridded = pd.DataFrame(
        {
            "il_idx": il_idx[on_grid],
            "xl_idx": xl_idx[on_grid],
            "interpretation": values[on_grid, 2],
        }
    )
    averaged = gridded.groupby(["il_idx", "xl_idx"], as_index=False)["interpretation"].mean()
    grid[
        averaged["il_idx"].to_numpy(dtype=np.int64),
        averaged["xl_idx"].to_numpy(dtype=np.int64),
    ] = averaged["interpretation"].to_numpy(dtype=float)
    return grid


def _nanmedian(values: np.ndarray) -> float:
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmedian(values))


def _remove_isolated_outliers_with_stats(
    surface: np.ndarray,
    threshold: Optional[float],
    min_neighbor_count: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if min_neighbor_count < 1:
        raise ValueError(f"outlier_min_neighbor_count must be >= 1, got {min_neighbor_count}.")

    valid_mask = np.isfinite(surface)
    total_count = int(np.count_nonzero(valid_mask))
    stats: Dict[str, Any] = {
        "total_points": total_count,
        "removed_points": 0,
        "removed_ratio": 0.0,
        "warning_ratio": OUTLIER_REMOVAL_WARNING_RATIO,
        "threshold": None if threshold is None else float(threshold),
        "min_neighbor_count": int(min_neighbor_count),
        "enabled": threshold is not None,
    }
    if threshold is None or total_count == 0:
        return surface.copy(), stats

    out = surface.copy()
    footprint = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )
    local_median = generic_filter(
        out,
        function=_nanmedian,
        footprint=footprint,
        mode="constant",
        cval=np.nan,
    )
    local_valid_count = generic_filter(
        valid_mask.astype(float),
        function=np.nansum,
        footprint=footprint,
        mode="constant",
        cval=0.0,
    )
    isolated_mask = (
        valid_mask
        & np.isfinite(local_median)
        & (local_valid_count >= float(min_neighbor_count))
        & (np.abs(out - local_median) > float(threshold))
    )
    out[isolated_mask] = np.nan

    removed_count = int(np.count_nonzero(isolated_mask))
    stats["removed_points"] = removed_count
    stats["removed_ratio"] = float(removed_count / total_count) if total_count else 0.0
    return out, stats


def _linear_then_nearest_from_grid(
    control_grid: np.ndarray,
    *,
    nearest_distance_limit: Optional[float],
    raw_grid: Optional[np.ndarray] = None,
    outlier_stats: Optional[Dict[str, Any]] = None,
) -> _SurfaceInterpolation:
    despiked_grid = control_grid.astype(float, copy=True)
    original_grid = despiked_grid.copy() if raw_grid is None else raw_grid.astype(float, copy=True)
    raw_mask = np.isfinite(original_grid)
    control_mask = np.isfinite(despiked_grid)
    if not np.any(control_mask):
        raise ValueError("Cannot interpolate a horizon with no finite control points.")

    ii, jj = np.indices(despiked_grid.shape)
    known_points = np.column_stack((ii[control_mask], jj[control_mask]))
    known_values = despiked_grid[control_mask]
    all_points = np.column_stack((ii.ravel(), jj.ravel()))

    linear_grid = despiked_grid.copy()
    linear_values = np.full(all_points.shape[0], np.nan, dtype=float)
    if known_points.shape[0] >= 3:
        try:
            linear_values = griddata(known_points, known_values, all_points, method="linear")
        except (QhullError, ValueError):
            linear_values = np.full(all_points.shape[0], np.nan, dtype=float)

    linear_values_grid = linear_values.reshape(despiked_grid.shape)
    linear_fill_mask = ~control_mask & np.isfinite(linear_values_grid)
    linear_grid[linear_fill_mask] = linear_values_grid[linear_fill_mask]
    linear_support_mask = np.isfinite(linear_grid)

    nearest_grid = linear_grid.copy()
    nearest_distance_grid = np.full(despiked_grid.shape, np.nan, dtype=float)
    tree = cKDTree(known_points.astype(float, copy=False))
    distances, indices = tree.query(all_points.astype(float, copy=False), k=1)
    nearest_distance_grid = distances.reshape(despiked_grid.shape)

    nearest_fill_mask = ~np.isfinite(nearest_grid)
    if nearest_distance_limit is not None:
        nearest_fill_mask &= nearest_distance_grid <= float(nearest_distance_limit)
    if np.any(nearest_fill_mask):
        nearest_grid[nearest_fill_mask] = known_values[indices.reshape(despiked_grid.shape)[nearest_fill_mask]]

    return _SurfaceInterpolation(
        raw_grid=original_grid,
        despiked_grid=despiked_grid,
        linear_grid=linear_grid,
        nearest_grid=nearest_grid,
        linear_support_mask=linear_support_mask,
        raw_mask=raw_mask,
        nearest_distance_grid=nearest_distance_grid,
        outlier_stats={} if outlier_stats is None else dict(outlier_stats),
    )


def _interpolate_surface_from_raw_df(
    interpretation_df: pd.DataFrame,
    il_axis: np.ndarray,
    xl_axis: np.ndarray,
    *,
    nearest_distance_limit: Optional[float],
    outlier_threshold: Optional[float],
    outlier_min_neighbor_count: int,
) -> _SurfaceInterpolation:
    raw_grid = _grid_raw_interpretation(interpretation_df, il_axis, xl_axis)
    despiked_grid, outlier_stats = _remove_isolated_outliers_with_stats(
        raw_grid,
        threshold=outlier_threshold,
        min_neighbor_count=outlier_min_neighbor_count,
    )
    return _linear_then_nearest_from_grid(
        despiked_grid,
        nearest_distance_limit=nearest_distance_limit,
        raw_grid=raw_grid,
        outlier_stats=outlier_stats,
    )


def _resolve_axis_interpolation_window(
    axis: np.ndarray,
    coord_float: float,
    axis_name: str,
) -> tuple[int, int, float]:
    coord = float(coord_float)
    axis_min = float(axis[0])
    axis_max = float(axis[-1])
    if axis.size == 1:
        if not np.isclose(coord, axis_min, atol=1e-8):
            raise ValueError(f"{axis_name}_float={coord} is out of bounds [{axis_min}, {axis_max}].")
        return 0, 0, 0.0

    axis_step = float(axis[1] - axis[0])
    tol = max(abs(axis_step), 1.0) * 1e-8
    if coord < axis_min - tol or coord > axis_max + tol:
        raise ValueError(f"{axis_name}_float={coord} is out of bounds [{axis_min}, {axis_max}].")

    k = (coord - axis_min) / axis_step
    rounded = round(k)
    if np.isclose(k, rounded, atol=1e-8):
        k = float(rounded)
    k0 = max(0, min(axis.size - 1, int(np.floor(k))))
    k1 = max(0, min(axis.size - 1, int(np.ceil(k))))
    weight = 0.0 if k0 == k1 else float(k - k0)
    return k0, k1, weight


def _bilinear_interpolate_surface_at_location(
    surface: np.ndarray,
    il_axis: np.ndarray,
    xl_axis: np.ndarray,
    il_float: float,
    xl_float: float,
) -> float:
    il0, il1, wi = _resolve_axis_interpolation_window(il_axis, il_float, "inline")
    xl0, xl1, wj = _resolve_axis_interpolation_window(xl_axis, xl_float, "xline")

    node_indices = {(il0, xl0), (il0, xl1), (il1, xl0), (il1, xl1)}
    for il_idx, xl_idx in node_indices:
        value = float(surface[il_idx, xl_idx])
        if not np.isfinite(value):
            raise ValueError(
                f"missing interpretation node: inline={int(il_axis[il_idx])}, xline={int(xl_axis[xl_idx])}"
            )

    t00 = float(surface[il0, xl0])
    t01 = float(surface[il0, xl1])
    t10 = float(surface[il1, xl0])
    t11 = float(surface[il1, xl1])
    return float((1.0 - wi) * (1.0 - wj) * t00 + (1.0 - wi) * wj * t01 + wi * (1.0 - wj) * t10 + wi * wj * t11)


class TargetLayer:
    """Prepare ordered target layers from raw horizon interpretations.

    Parameters
    ----------
    raw_horizon_dfs
        Mapping from horizon name to raw, uninterpolated interpretation picks.
    geometry
        Survey geometry.  Inline/xline and sample axis keys are required.
    horizon_names
        Horizon order from shallow/top to deep/bottom.
    output_dir
        QC CSV output directory.  Defaults to ``./output``.
    min_thickness
        Minimum allowed adjacent-layer thickness.  Defaults to sample step.
    nearest_distance_limit
        Optional maximum nearest-neighbor distance in trace-index units.  The
        default ``None`` leaves nearest filling unrestricted.
    outlier_threshold
        Optional isolated-pick removal threshold in interpretation units.  When
        omitted, isolated outlier removal is disabled.
    outlier_min_neighbor_count
        Minimum valid cross-neighbor count required before testing a pick as an
        isolated outlier.
    """

    def __init__(
        self,
        raw_horizon_dfs: Dict[str, pd.DataFrame],
        geometry: Dict[str, Any],
        horizon_names: list[str],
        *,
        output_dir: Optional[str | Path] = None,
        min_thickness: Optional[float] = None,
        nearest_distance_limit: Optional[float] = None,
        outlier_threshold: Optional[float] = None,
        outlier_min_neighbor_count: int = 2,
    ) -> None:
        if len(raw_horizon_dfs) < 2:
            raise ValueError("raw_horizon_dfs must contain at least two horizons.")
        if len(horizon_names) < 2:
            raise ValueError("horizon_names must contain at least two ordered horizons.")
        if len(set(horizon_names)) != len(horizon_names):
            raise ValueError("horizon_names must be unique.")
        missing = [name for name in horizon_names if name not in raw_horizon_dfs]
        if missing:
            raise ValueError(f"horizon_names not found in raw_horizon_dfs: {missing}")

        _validate_geometry_keys(geometry)
        self.geometry = dict(geometry)
        self.horizon_names = list(horizon_names)
        self.output_dir = Path("output") if output_dir is None else Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_thickness = float(self.geometry["sample_step"]) if min_thickness is None else float(min_thickness)
        if self.min_thickness <= 0.0:
            raise ValueError(f"min_thickness must be positive, got {self.min_thickness}.")
        self.nearest_distance_limit = None if nearest_distance_limit is None else float(nearest_distance_limit)
        if self.nearest_distance_limit is not None and self.nearest_distance_limit <= 0.0:
            raise ValueError(f"nearest_distance_limit must be positive, got {self.nearest_distance_limit}.")
        self.outlier_threshold = None if outlier_threshold is None else float(outlier_threshold)
        self.outlier_min_neighbor_count = int(outlier_min_neighbor_count)
        if self.outlier_min_neighbor_count < 1:
            raise ValueError(
                f"outlier_min_neighbor_count must be >= 1, got {self.outlier_min_neighbor_count}."
            )

        self._il_axis, self._xl_axis, self._sample_axis = self._build_axes()
        self.raw_horizon_dfs = {
            name: _normalize_interpretation_unit_for_geometry(raw_horizon_dfs[name], self.geometry)
            for name in self.horizon_names
        }
        for name, df in self.raw_horizon_dfs.items():
            _require_interpretation_columns(df, name)

        self._surface_interpolations = {
            name: _interpolate_surface_from_raw_df(
                self.raw_horizon_dfs[name],
                self._il_axis,
                self._xl_axis,
                nearest_distance_limit=self.nearest_distance_limit,
                outlier_threshold=self.outlier_threshold,
                outlier_min_neighbor_count=self.outlier_min_neighbor_count,
            )
            for name in self.horizon_names
        }
        self.initial_horizon_grids = {
            name: interp.linear_grid.copy() for name, interp in self._surface_interpolations.items()
        }
        self.independent_filled_horizon_grids = {
            name: interp.nearest_grid.copy() for name, interp in self._surface_interpolations.items()
        }
        self.interpolation_support_masks = {
            name: interp.linear_support_mask.copy() for name, interp in self._surface_interpolations.items()
        }
        self.raw_pick_masks = {name: interp.raw_mask.copy() for name, interp in self._surface_interpolations.items()}
        self.nearest_distance_grids = {
            name: interp.nearest_distance_grid.copy() for name, interp in self._surface_interpolations.items()
        }
        self.outlier_stats = {
            name: dict(interp.outlier_stats) for name, interp in self._surface_interpolations.items()
        }

        self._build_trace_qc_masks()
        self._horizon_grids = self._build_final_horizon_grids()
        self.interpolated_horizon_dfs = {
            name: self._grid_to_horizon_df(grid) for name, grid in self._horizon_grids.items()
        }
        self._write_qc_csvs()

    def _build_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        il_axis = _build_axis(
            self.geometry["inline_min"],
            self.geometry["inline_max"],
            self.geometry["inline_step"],
            "inline",
        )
        xl_axis = _build_axis(
            self.geometry["xline_min"],
            self.geometry["xline_max"],
            self.geometry["xline_step"],
            "xline",
        )
        sample_axis = _build_axis(
            self.geometry["sample_min"],
            self.geometry["sample_max"],
            self.geometry["sample_step"],
            "sample",
        )
        return il_axis, xl_axis, sample_axis

    def _build_trace_qc_masks(self) -> None:
        shape = (self._il_axis.size, self._xl_axis.size)
        no_support_any = np.zeros(shape, dtype=bool)
        for support_mask in self.interpolation_support_masks.values():
            no_support_any |= ~support_mask

        crossing_any = np.zeros(shape, dtype=bool)
        thin_any = np.zeros(shape, dtype=bool)
        pair_records = []
        self._pair_qc_masks: dict[tuple[str, str], dict[str, np.ndarray]] = {}

        for top_name, bottom_name in self.iter_zones():
            top_grid = self.initial_horizon_grids[top_name]
            bottom_grid = self.initial_horizon_grids[bottom_name]
            finite_pair = np.isfinite(top_grid) & np.isfinite(bottom_grid)
            thickness = bottom_grid - top_grid
            crossing = finite_pair & (top_grid >= bottom_grid)
            thin = finite_pair & ~crossing & (thickness < self.min_thickness)
            pair_no_support = ~(
                self.interpolation_support_masks[top_name] & self.interpolation_support_masks[bottom_name]
            )

            crossing_any |= crossing
            thin_any |= thin
            self._pair_qc_masks[(top_name, bottom_name)] = {
                "finite_pair": finite_pair,
                "no_support": pair_no_support,
                "crossing": crossing,
                "thin": thin,
                "thickness": thickness,
            }

            pair_records.append(
                {
                    "horizon_pair": f"{top_name}->{bottom_name}",
                    "top_name": top_name,
                    "bottom_name": bottom_name,
                    "total_traces": int(np.prod(shape)),
                    "pair_no_support_count": int(np.count_nonzero(pair_no_support)),
                    "crossing_count": int(np.count_nonzero(crossing)),
                    "thin_count": int(np.count_nonzero(thin)),
                    "pair_valid_count": int(np.count_nonzero(finite_pair & ~pair_no_support & ~crossing & ~thin)),
                    "min_thickness": self.min_thickness,
                }
            )

        self.no_support_mask = no_support_any
        self.crossing_mask = crossing_any
        self.thin_mask = thin_any
        self.valid_control_mask = ~(no_support_any | crossing_any | thin_any)
        self.masked_trace_mask = ~self.valid_control_mask
        self._summary_pair_records = pair_records

    def _build_final_horizon_grids(self) -> dict[str, np.ndarray]:
        final_grids = {}
        top_name = self.horizon_names[0]
        top_grid = self.independent_filled_horizon_grids[top_name].copy()
        final_grids[top_name] = top_grid
        previous_grid = top_grid

        for top_name, bottom_name in self.iter_zones():
            pair_qc = self._pair_qc_masks[(top_name, bottom_name)]
            thickness = pair_qc["thickness"]
            control_mask = self.valid_control_mask & np.isfinite(thickness) & (thickness >= self.min_thickness)
            if not np.any(control_mask):
                positive_mask = np.isfinite(thickness) & (thickness >= self.min_thickness)
                if not np.any(positive_mask):
                    raise ValueError(f"Zone '{top_name}' -> '{bottom_name}' has no positive thickness controls.")
                control_mask = positive_mask

            thickness_control_grid = np.full(thickness.shape, np.nan, dtype=float)
            thickness_control_grid[control_mask] = thickness[control_mask]
            thickness_interp = _linear_then_nearest_from_grid(
                thickness_control_grid,
                nearest_distance_limit=self.nearest_distance_limit,
            ).nearest_grid
            finite = np.isfinite(thickness_interp)
            thickness_interp[finite] = np.maximum(thickness_interp[finite], self.min_thickness)
            bottom_grid = previous_grid + thickness_interp
            final_grids[bottom_name] = bottom_grid
            previous_grid = bottom_grid

        filled_model_mask = np.ones_like(self.valid_control_mask, dtype=bool)
        for grid in final_grids.values():
            filled_model_mask &= np.isfinite(grid)
        self.filled_model_mask = filled_model_mask
        return final_grids

    def _grid_to_horizon_df(self, grid: np.ndarray) -> pd.DataFrame:
        il_grid, xl_grid = np.meshgrid(self._il_axis, self._xl_axis, indexing="ij")
        return pd.DataFrame(
            {
                "inline": il_grid.ravel(),
                "xline": xl_grid.ravel(),
                "interpretation": grid.ravel(),
            }
        )

    def _trace_qc_dataframe(self) -> pd.DataFrame:
        rows = []
        for i, inline in enumerate(self._il_axis):
            for j, xline in enumerate(self._xl_axis):
                no_support_horizons = [
                    name for name in self.horizon_names if not bool(self.interpolation_support_masks[name][i, j])
                ]
                crossing_pairs = []
                thin_pairs = []
                for top_name, bottom_name in self.iter_zones():
                    pair_qc = self._pair_qc_masks[(top_name, bottom_name)]
                    if bool(pair_qc["crossing"][i, j]):
                        crossing_pairs.append(f"{top_name}->{bottom_name}")
                    if bool(pair_qc["thin"][i, j]):
                        thin_pairs.append(f"{top_name}->{bottom_name}")

                rows.append(
                    {
                        "inline": float(inline),
                        "xline": float(xline),
                        "valid_control": bool(self.valid_control_mask[i, j]),
                        "filled_model": bool(self.filled_model_mask[i, j]),
                        "masked_trace": bool(self.masked_trace_mask[i, j]),
                        "filled_by_thickness_interpolation": bool(
                            self.filled_model_mask[i, j] and not self.valid_control_mask[i, j]
                        ),
                        "no_support": bool(self.no_support_mask[i, j]),
                        "crossing": bool(self.crossing_mask[i, j]),
                        "thin": bool(self.thin_mask[i, j]),
                        "no_support_horizons": ";".join(no_support_horizons),
                        "crossing_pairs": ";".join(crossing_pairs),
                        "thin_pairs": ";".join(thin_pairs),
                    }
                )
        return pd.DataFrame.from_records(rows)

    def _pair_qc_dataframe(self) -> pd.DataFrame:
        records = []
        for top_name, bottom_name in self.iter_zones():
            pair_qc = self._pair_qc_masks[(top_name, bottom_name)]
            top_grid = self.initial_horizon_grids[top_name]
            bottom_grid = self.initial_horizon_grids[bottom_name]
            invalid = pair_qc["no_support"] | pair_qc["crossing"] | pair_qc["thin"]
            invalid_indices = np.argwhere(invalid)
            for i, j in invalid_indices:
                records.append(
                    {
                        "top_name": top_name,
                        "bottom_name": bottom_name,
                        "inline": float(self._il_axis[i]),
                        "xline": float(self._xl_axis[j]),
                        "interpretation_top": float(top_grid[i, j]) if np.isfinite(top_grid[i, j]) else np.nan,
                        "interpretation_bottom": float(bottom_grid[i, j]) if np.isfinite(bottom_grid[i, j]) else np.nan,
                        "thickness": float(pair_qc["thickness"][i, j])
                        if np.isfinite(pair_qc["thickness"][i, j])
                        else np.nan,
                        "no_support": bool(pair_qc["no_support"][i, j]),
                        "crossing": bool(pair_qc["crossing"][i, j]),
                        "thin": bool(pair_qc["thin"][i, j]),
                    }
                )
        columns = [
            "top_name",
            "bottom_name",
            "inline",
            "xline",
            "interpretation_top",
            "interpretation_bottom",
            "thickness",
            "no_support",
            "crossing",
            "thin",
        ]
        return pd.DataFrame.from_records(records, columns=columns)

    def _summary_dataframe(self) -> pd.DataFrame:
        records = []
        for horizon_name in self.horizon_names:
            stats = self.outlier_stats.get(horizon_name, {})
            records.append(
                {
                    "record_type": "horizon",
                    "horizon_pair": horizon_name,
                    "top_name": horizon_name,
                    "bottom_name": "",
                    "total_traces": int(self.valid_control_mask.size),
                    "pair_no_support_count": int(np.count_nonzero(~self.interpolation_support_masks[horizon_name])),
                    "crossing_count": 0,
                    "thin_count": 0,
                    "pair_valid_count": int(np.count_nonzero(self.interpolation_support_masks[horizon_name])),
                    "min_thickness": self.min_thickness,
                    "outlier_enabled": bool(stats.get("enabled", False)),
                    "outlier_threshold": stats.get("threshold"),
                    "outlier_min_neighbor_count": stats.get("min_neighbor_count"),
                    "outlier_total_points": stats.get("total_points"),
                    "outlier_removed_points": stats.get("removed_points"),
                    "outlier_removed_ratio": stats.get("removed_ratio"),
                }
            )

        for record in self._summary_pair_records:
            out_record = dict(record)
            out_record["record_type"] = "pair"
            records.append(out_record)

        records.append(
            {
                "record_type": "global",
                "horizon_pair": "__trace_global__",
                "top_name": "",
                "bottom_name": "",
                "total_traces": int(self.valid_control_mask.size),
                "pair_no_support_count": int(np.count_nonzero(self.no_support_mask)),
                "crossing_count": int(np.count_nonzero(self.crossing_mask)),
                "thin_count": int(np.count_nonzero(self.thin_mask)),
                "pair_valid_count": int(np.count_nonzero(self.valid_control_mask)),
                "min_thickness": self.min_thickness,
                "filled_model_count": int(np.count_nonzero(self.filled_model_mask)),
                "filled_by_thickness_interpolation_count": int(
                    np.count_nonzero(self.filled_model_mask & ~self.valid_control_mask)
                ),
                "nearest_distance_limit": self.nearest_distance_limit,
                "outlier_threshold": self.outlier_threshold,
                "outlier_min_neighbor_count": self.outlier_min_neighbor_count,
            }
        )
        return pd.DataFrame.from_records(records)

    def _write_qc_csvs(self) -> None:
        self.trace_qc_df = self._trace_qc_dataframe()
        self.pair_qc_df = self._pair_qc_dataframe()
        self.qc_summary_df = self._summary_dataframe()

        self.trace_qc_path = self.output_dir / "target_layer_trace_qc.csv"
        self.pair_qc_path = self.output_dir / "target_layer_pair_qc.csv"
        self.qc_summary_path = self.output_dir / "target_layer_qc_summary.csv"
        self.trace_qc_df.to_csv(self.trace_qc_path, index=False, encoding="utf-8-sig")
        self.pair_qc_df.to_csv(self.pair_qc_path, index=False, encoding="utf-8-sig")
        self.qc_summary_df.to_csv(self.qc_summary_path, index=False, encoding="utf-8-sig")

    @property
    def ilines(self) -> np.ndarray:
        return self._il_axis.copy()

    @property
    def xlines(self) -> np.ndarray:
        return self._xl_axis.copy()

    @property
    def samples(self) -> np.ndarray:
        return self._sample_axis.copy()

    def iter_zones(self) -> list[tuple[str, str]]:
        return list(zip(self.horizon_names[:-1], self.horizon_names[1:]))

    def get_trace_valid_mask(self) -> np.ndarray:
        """Return the reliable-control trace mask before delivery filling."""
        return self.valid_control_mask.copy()

    def get_filled_model_mask(self) -> np.ndarray:
        """Return where final horizon grids contain finite values."""
        return self.filled_model_mask.copy()

    def get_zone_valid_mask(self, zone: tuple[str, str], *, use_valid_control_mask: bool = True) -> np.ndarray:
        top_name, bottom_name = self._resolve_zone(zone)
        top_grid, bottom_grid = self.get_zone_sample_index_grids((top_name, bottom_name))
        valid = np.isfinite(top_grid) & np.isfinite(bottom_grid) & (bottom_grid > top_grid)
        if use_valid_control_mask:
            valid &= self.valid_control_mask
        return valid

    def _resolve_zone(self, zone: tuple[str, str]) -> tuple[str, str]:
        top_name, bottom_name = zone
        if top_name not in self.horizon_names or bottom_name not in self.horizon_names:
            raise ValueError(f"zone contains unknown horizons: {zone}")
        top_idx = self.horizon_names.index(top_name)
        bottom_idx = self.horizon_names.index(bottom_name)
        if bottom_idx != top_idx + 1:
            raise ValueError(f"zone must contain adjacent horizons, got {zone}")
        return top_name, bottom_name

    def get_horizon_grid(self, horizon_name: str) -> np.ndarray:
        if horizon_name not in self._horizon_grids:
            raise ValueError(f"horizon_name '{horizon_name}' is not in raw_horizon_dfs.")
        return self._horizon_grids[horizon_name].copy()

    def get_horizon_interpretation_at_location(
        self,
        horizon_name: str,
        il_float: float,
        xl_float: float,
    ) -> float:
        if horizon_name not in self._horizon_grids:
            raise ValueError(f"horizon_name '{horizon_name}' is not in raw_horizon_dfs.")
        return _bilinear_interpolate_surface_at_location(
            self._horizon_grids[horizon_name],
            self._il_axis,
            self._xl_axis,
            il_float,
            xl_float,
        )

    def get_interpretation_values_at_location(self, il_float: float, xl_float: float) -> Dict[str, float]:
        return {
            horizon_name: self.get_horizon_interpretation_at_location(horizon_name, il_float, xl_float)
            for horizon_name in self.horizon_names
        }

    def convert_horizon_to_relative_sample_index(self, horizon_name: str) -> pd.DataFrame:
        if horizon_name not in self._horizon_grids:
            raise ValueError(f"horizon_name '{horizon_name}' is not in raw_horizon_dfs.")

        sample_min = float(self._sample_axis[0])
        sample_max = float(self._sample_axis[-1])
        sample_step = float(self.geometry["sample_step"])
        grid = self._horizon_grids[horizon_name]
        finite = np.isfinite(grid)
        if np.any(finite):
            tol = 1e-6
            out_of_range = finite & ((grid < sample_min - tol) | (grid > sample_max + tol))
            if np.any(out_of_range):
                bad_indices = np.argwhere(out_of_range)[:5]
                examples = [
                    {
                        "inline": float(self._il_axis[i]),
                        "xline": float(self._xl_axis[j]),
                        "interpretation": float(grid[i, j]),
                    }
                    for i, j in bad_indices
                ]
                raise ValueError(
                    "Horizon values are out of sample range. "
                    f"Expected within [{sample_min}, {sample_max}], "
                    f"found {int(np.count_nonzero(out_of_range))} out-of-range points. "
                    f"Examples: {examples}"
                )

        out_df = self._grid_to_horizon_df(grid)
        sample_index = np.full(grid.size, np.nan, dtype=float)
        values = out_df["interpretation"].to_numpy(dtype=float, copy=False)
        value_finite = np.isfinite(values)
        sample_index[value_finite] = (values[value_finite] - sample_min) / sample_step
        out_df["sample_index"] = sample_index
        return out_df

    def _get_horizon_sample_index_grid(self, horizon_name: str) -> np.ndarray:
        df = self.convert_horizon_to_relative_sample_index(horizon_name)
        return df["sample_index"].to_numpy(dtype=float).reshape((self._il_axis.size, self._xl_axis.size))

    def get_zone_sample_index_grids(self, zone: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
        top_name, bottom_name = self._resolve_zone(zone)
        return self._get_horizon_sample_index_grid(top_name), self._get_horizon_sample_index_grid(bottom_name)

    def with_boundary_extension(
        self,
        extension_samples: int,
        *,
        top_extension_name: str = "top_extension",
        bottom_extension_name: str = "bottom_extension",
    ) -> "TargetLayer":
        """Return a lightweight copy with synthetic top/bottom extension horizons."""
        if extension_samples < 0:
            raise ValueError(f"extension_samples must be >= 0, got {extension_samples}.")
        if extension_samples == 0:
            return self
        if top_extension_name == bottom_extension_name:
            raise ValueError("top_extension_name and bottom_extension_name must be different.")
        duplicate_names = {
            name for name in (top_extension_name, bottom_extension_name) if name in self.horizon_names
        }
        if duplicate_names:
            raise ValueError(f"extension horizon names already exist: {sorted(duplicate_names)}")

        sample_min = float(self._sample_axis[0])
        sample_max = float(self._sample_axis[-1])
        offset = float(extension_samples) * float(self.geometry["sample_step"])

        top_source = self._horizon_grids[self.horizon_names[0]]
        bottom_source = self._horizon_grids[self.horizon_names[-1]]
        top_extension = np.where(np.isfinite(top_source), np.clip(top_source - offset, sample_min, sample_max), np.nan)
        bottom_extension = np.where(
            np.isfinite(bottom_source),
            np.clip(bottom_source + offset, sample_min, sample_max),
            np.nan,
        )

        out = object.__new__(TargetLayer)
        out.geometry = dict(self.geometry)
        out.horizon_names = [top_extension_name, *self.horizon_names, bottom_extension_name]
        out.output_dir = self.output_dir
        out.min_thickness = self.min_thickness
        out.nearest_distance_limit = self.nearest_distance_limit
        out.outlier_threshold = self.outlier_threshold
        out.outlier_min_neighbor_count = self.outlier_min_neighbor_count
        out._il_axis = self._il_axis.copy()
        out._xl_axis = self._xl_axis.copy()
        out._sample_axis = self._sample_axis.copy()
        out.raw_horizon_dfs = {name: df.copy() for name, df in self.raw_horizon_dfs.items()}
        out._surface_interpolations = dict(self._surface_interpolations)
        out.initial_horizon_grids = {name: grid.copy() for name, grid in self.initial_horizon_grids.items()}
        out.independent_filled_horizon_grids = {
            name: grid.copy() for name, grid in self.independent_filled_horizon_grids.items()
        }
        out.interpolation_support_masks = {
            name: mask.copy() for name, mask in self.interpolation_support_masks.items()
        }
        out.raw_pick_masks = {name: mask.copy() for name, mask in self.raw_pick_masks.items()}
        out.nearest_distance_grids = {name: grid.copy() for name, grid in self.nearest_distance_grids.items()}
        out.outlier_stats = {name: dict(stats) for name, stats in self.outlier_stats.items()}
        out.no_support_mask = self.no_support_mask.copy()
        out.crossing_mask = self.crossing_mask.copy()
        out.thin_mask = self.thin_mask.copy()
        out.valid_control_mask = self.valid_control_mask.copy()
        out.masked_trace_mask = self.masked_trace_mask.copy()
        out.filled_model_mask = self.filled_model_mask.copy()
        out._pair_qc_masks = dict(self._pair_qc_masks)
        out._summary_pair_records = list(self._summary_pair_records)
        out._horizon_grids = {
            top_extension_name: top_extension,
            **{name: grid.copy() for name, grid in self._horizon_grids.items()},
            bottom_extension_name: bottom_extension,
        }
        out.interpolated_horizon_dfs = {
            name: out._grid_to_horizon_df(grid) for name, grid in out._horizon_grids.items()
        }
        out.trace_qc_df = self.trace_qc_df.copy()
        out.pair_qc_df = self.pair_qc_df.copy()
        out.qc_summary_df = self.qc_summary_df.copy()
        out.trace_qc_path = self.trace_qc_path
        out.pair_qc_path = self.pair_qc_path
        out.qc_summary_path = self.qc_summary_path
        return out

    def to_mask(
        self,
        zone: Optional[tuple[str, str]] = None,
        *,
        use_valid_control_mask: bool = True,
    ) -> np.ndarray:
        """Build a 3D sample mask.

        By default, only reliable-control traces participate.  Pass
        ``use_valid_control_mask=False`` to build a full-coverage mask from the
        filled horizon grids.
        """
        n_il = int(self.geometry.get("n_il", self._il_axis.size))
        n_xl = int(self.geometry.get("n_xl", self._xl_axis.size))
        n_sample = int(self.geometry.get("n_sample", self._sample_axis.size))
        if n_il != self._il_axis.size:
            raise ValueError(f"geometry n_il={n_il} does not match axis size {self._il_axis.size}.")
        if n_xl != self._xl_axis.size:
            raise ValueError(f"geometry n_xl={n_xl} does not match axis size {self._xl_axis.size}.")
        if n_sample != self._sample_axis.size:
            raise ValueError(f"geometry n_sample={n_sample} does not match axis size {self._sample_axis.size}.")

        mask = np.zeros((n_il, n_xl, n_sample), dtype=bool)
        zones = [self._resolve_zone(zone)] if zone is not None else self.iter_zones()
        for top_name, bottom_name in zones:
            top_grid, bottom_grid = self.get_zone_sample_index_grids((top_name, bottom_name))
            valid = self.get_zone_valid_mask((top_name, bottom_name), use_valid_control_mask=use_valid_control_mask)
            for i in range(n_il):
                for j in range(n_xl):
                    if not valid[i, j]:
                        continue
                    idx_top = max(0, int(np.round(top_grid[i, j])))
                    idx_bottom = min(n_sample, int(np.round(bottom_grid[i, j])) + 1)
                    if idx_top < idx_bottom:
                        mask[i, j, idx_top:idx_bottom] = True
        return mask
