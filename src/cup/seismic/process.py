"""地震解释数据的预处理与插值工具。"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter
from scipy.spatial import QhullError


def _build_line_axis(line_min: int, line_max: int, line_step: int) -> np.ndarray:
    """根据最小值、最大值和步长构建完整轴。"""
    if line_step <= 0:
        raise ValueError(f"line_step must be positive, got {line_step}.")
    return np.arange(line_min, line_max + line_step, line_step, dtype=int)


def _nanmedian(values: np.ndarray) -> float:
    """计算窗口值的 NaN-aware 中值。"""
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmedian(values))


def _to_surface_grid(
    interpretation_df: pd.DataFrame,
    il_axis: np.ndarray,
    xl_axis: np.ndarray,
) -> np.ndarray:
    """将离散解释点映射到 2D 网格，空位填 NaN。"""
    required_cols = {"inline", "xline", "interpretation"}
    missing_cols = required_cols - set(interpretation_df.columns)
    if missing_cols:
        raise ValueError(f"interpretation_df is missing required columns: {sorted(missing_cols)}")

    grid = np.full((il_axis.size, xl_axis.size), np.nan, dtype=float)

    il_values = interpretation_df["inline"].to_numpy(dtype=float, copy=False)
    xl_values = interpretation_df["xline"].to_numpy(dtype=float, copy=False)
    z_values = interpretation_df["interpretation"].to_numpy(dtype=float, copy=False)

    il_min = int(il_axis[0])
    xl_min = int(xl_axis[0])
    il_step = int(il_axis[1] - il_axis[0]) if il_axis.size > 1 else 1
    xl_step = int(xl_axis[1] - xl_axis[0]) if xl_axis.size > 1 else 1

    il_rel = il_values - il_min
    xl_rel = xl_values - xl_min

    finite_mask = np.isfinite(il_values) & np.isfinite(xl_values) & np.isfinite(z_values)
    on_grid_mask = finite_mask & (np.mod(il_rel, il_step) == 0) & (np.mod(xl_rel, xl_step) == 0)
    if not np.any(on_grid_mask):
        return grid

    il_idx = (il_rel[on_grid_mask] / il_step).astype(np.int64)
    xl_idx = (xl_rel[on_grid_mask] / xl_step).astype(np.int64)
    val_idx = np.flatnonzero(on_grid_mask)

    in_bounds = (il_idx >= 0) & (il_idx < il_axis.size) & (xl_idx >= 0) & (xl_idx < xl_axis.size)
    if not np.any(in_bounds):
        return grid

    grid[il_idx[in_bounds], xl_idx[in_bounds]] = z_values[val_idx[in_bounds]]

    return grid


def _remove_isolated_outliers(
    surface: np.ndarray,
    threshold: float,
    min_neighbor_count: int = 2,
) -> np.ndarray:
    """基于十字邻域中值剔除孤立跳变点。"""
    if min_neighbor_count < 1:
        raise ValueError(f"min_neighbor_count must be >= 1, got {min_neighbor_count}.")

    out = surface.copy()
    valid_mask = np.isfinite(out)
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
    return out


def _interpolate_whole_mesh_linear_then_nearest(surface: np.ndarray) -> np.ndarray:
    """全域插值：先 linear，再对凸包外 NaN 用 nearest 兜底。"""
    out = surface.copy()
    valid_mask = np.isfinite(out)
    if np.count_nonzero(valid_mask) < 1:
        return out

    ii, jj = np.indices(out.shape)
    known_points = np.column_stack((ii[valid_mask], jj[valid_mask]))
    known_values = out[valid_mask]

    nan_mask = ~valid_mask
    if not np.any(nan_mask):
        return out

    target_points = np.column_stack((ii[nan_mask], jj[nan_mask]))
    target_i = target_points[:, 0].astype(np.int64)
    target_j = target_points[:, 1].astype(np.int64)

    linear_values: np.ndarray
    if np.count_nonzero(valid_mask) >= 3:
        try:
            linear_values = griddata(known_points, known_values, target_points, method="linear")
        except (QhullError, ValueError):
            linear_values = np.full(target_points.shape[0], np.nan, dtype=float)
    else:
        linear_values = np.full(target_points.shape[0], np.nan, dtype=float)

    finite_linear = np.isfinite(linear_values)
    if np.any(finite_linear):
        out[target_i[finite_linear], target_j[finite_linear]] = linear_values[finite_linear]

    remain_nan_mask = ~np.isfinite(out[target_i, target_j])
    if not np.any(remain_nan_mask):
        return out

    remain_points = target_points[remain_nan_mask]
    try:
        nearest_values = griddata(known_points, known_values, remain_points, method="nearest")
    except (QhullError, ValueError):
        nearest_values = None
    if nearest_values is None:
        return out

    finite_nearest = np.isfinite(nearest_values)
    if np.any(finite_nearest):
        remain_i = remain_points[:, 0].astype(np.int64)
        remain_j = remain_points[:, 1].astype(np.int64)
        out[remain_i[finite_nearest], remain_j[finite_nearest]] = nearest_values[finite_nearest]

    return out


def interpolate_interpretation_surface(
    interpretation_df: pd.DataFrame,
    geometry: Dict[str, Any],
    outlier_threshold: float,
    min_neighbor_count: int = 2,
    keep_nan: bool = True,
) -> pd.DataFrame:
    """层位面插值：先去孤立异常点，再执行全域插值。

    Parameters
    ----------
    interpretation_df : pd.DataFrame
            输入离散解释点，至少包含 `inline`, `xline`, `interpretation` 三列。
    geometry : Dict[str, Any]
            地震几何信息，至少包含 `inline_min`, `inline_max`, `inline_step`,
            `xline_min`, `xline_max`, `xline_step`。
    outlier_threshold : float
            孤立点判定阈值。若样点与邻域中值差值绝对值超过该阈值，则剔除为 NaN。
            单位与 `interpretation` 一致（例如 ms 或 m）。
    min_neighbor_count : int, default=2
            执行孤立点判断所需的最小有效邻居数（十字邻域，不含中心点）。
    keep_nan : bool, default=True
            为 True 时返回完整网格；
            为 False 时仅返回有效解释点。

    Returns
    -------
    pd.DataFrame
            列为 `inline`, `xline`, `interpretation`。
    """
    geometry_keys = {
        "inline_min",
        "inline_max",
        "inline_step",
        "xline_min",
        "xline_max",
        "xline_step",
    }
    missing_geometry_keys = geometry_keys - set(geometry)
    if missing_geometry_keys:
        raise ValueError(f"geometry is missing required keys: {sorted(missing_geometry_keys)}")

    il_axis = _build_line_axis(
        int(geometry["inline_min"]),
        int(geometry["inline_max"]),
        int(geometry["inline_step"]),
    )
    xl_axis = _build_line_axis(
        int(geometry["xline_min"]),
        int(geometry["xline_max"]),
        int(geometry["xline_step"]),
    )

    surface = _to_surface_grid(interpretation_df, il_axis, xl_axis)
    surface_despiked = _remove_isolated_outliers(
        surface,
        threshold=outlier_threshold,
        min_neighbor_count=min_neighbor_count,
    )
    surface_filled = _interpolate_whole_mesh_linear_then_nearest(surface_despiked)

    il_grid, xl_grid = np.meshgrid(il_axis, xl_axis, indexing="ij")
    out_df = pd.DataFrame(
        {
            "inline": il_grid.ravel(),
            "xline": xl_grid.ravel(),
            "interpretation": surface_filled.ravel(),
        }
    )

    if keep_nan:
        return out_df
    return out_df[np.isfinite(out_df["interpretation"])].reset_index(drop=True)


# def import_and_interpolate_interpretation_petrel(
#     interpretation_file: Path,
#     seismic_file: Path,
#     outlier_threshold: float,
#     seismic_type: str = "segy",
#     domain: Optional[str] = "time",
#     min_neighbor_count: int = 2,
#     keep_nan: bool = True,
# ) -> pd.DataFrame:
#     """导入 Petrel 层位并执行去孤立点+全域插值。"""
#     interpretation_df = import_interpretation_petrel(interpretation_file)
#     geometry = query_seismic_geometry(seismic_file, seismic_type=seismic_type, domain=domain)
#     return interpolate_interpretation_surface(
#         interpretation_df=interpretation_df,
#         geometry=geometry,
#         outlier_threshold=outlier_threshold,
#         min_neighbor_count=min_neighbor_count,
#         keep_nan=keep_nan,
#     )
