"""cup.seismic.process: 地震解释层位预处理与插值。

本模块提供离散层位解释点到规则 inline/xline 网格的预处理、异常剔除与插值能力，
用于单层位面清洗、网格化与插值补全。

边界说明
--------
- 本模块不负责解释点文件读取、地震体几何查询或低频模型反演流程。
- 本模块假定输入层位已经位于同一 inline/xline 坐标系，且 geometry 由上游提供。
- 当 ``sample_domain='time'`` 且 ``sample_unit='s'`` 时，会按经验规则自动识别
  毫秒输入并换算为秒。

核心公开对象
------------
1. interpolate_interpretation_surface: 清洗并插值单个层位面。
"""

import warnings
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter
from scipy.spatial import QhullError

OUTLIER_REMOVAL_WARNING_RATIO = 0.05


def _build_axis(axis_min: float, axis_max: float, axis_step: float, axis_name: str) -> np.ndarray:
    """根据最小值、最大值和步长构建完整轴。"""
    if axis_step <= 0:
        raise ValueError(f"{axis_name}_step must be positive, got {axis_step}.")
    return np.arange(axis_min, axis_max + axis_step, axis_step, dtype=float)


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


def _remove_isolated_outliers_with_stats(
    surface: np.ndarray,
    threshold: float,
    min_neighbor_count: int = 2,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """基于十字邻域中值剔除孤立跳变点，并返回剔除统计。"""
    if min_neighbor_count < 1:
        raise ValueError(f"min_neighbor_count must be >= 1, got {min_neighbor_count}.")

    out = surface.copy()
    valid_mask = np.isfinite(out)
    total_count = int(np.count_nonzero(valid_mask))
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
    removed_ratio = float(removed_count / total_count) if total_count else 0.0
    stats = {
        "total_points": total_count,
        "removed_points": removed_count,
        "removed_ratio": removed_ratio,
        "warning_ratio": OUTLIER_REMOVAL_WARNING_RATIO,
        "threshold": float(threshold),
        "min_neighbor_count": int(min_neighbor_count),
    }

    if total_count and removed_ratio > OUTLIER_REMOVAL_WARNING_RATIO:
        warnings.warn(
            "Isolated outlier removal dropped "
            f"{removed_count}/{total_count} interpretation point(s) "
            f"({removed_ratio:.2%}), exceeding the "
            f"{OUTLIER_REMOVAL_WARNING_RATIO:.0%} warning threshold.",
            RuntimeWarning,
            stacklevel=2,
        )

    return out, stats


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


def _validate_geometry_keys(geometry: Dict[str, Any]) -> None:
    """校验几何信息是否包含 inline/xline 插值必需键。"""
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


def _normalize_interpretation_unit_for_geometry(
    interpretation_df: pd.DataFrame,
    geometry: Dict[str, Any],
) -> pd.DataFrame:
    """按 geometry 自动归一化 interpretation 单位（当前支持 time+s 自动识别 ms）。"""
    sample_domain = str(geometry.get("sample_domain", "")).lower()
    sample_unit = str(geometry.get("sample_unit", "")).lower()
    if sample_domain != "time" or sample_unit != "s":
        return interpretation_df

    if "interpretation" not in interpretation_df.columns:
        return interpretation_df

    z = interpretation_df["interpretation"].to_numpy(dtype=float, copy=False)
    finite = np.isfinite(z)
    if not np.any(finite):
        return interpretation_df

    # 经验规则：秒域层位通常数量级不超过 10；若明显超出，按 ms 输入自动换算为 s。
    if float(np.nanmax(np.abs(z[finite]))) <= 10.0:
        return interpretation_df

    out_df = interpretation_df.copy()
    converted = out_df["interpretation"].to_numpy(dtype=float, copy=True)
    converted[finite] = converted[finite] / 1000.0
    out_df["interpretation"] = converted
    return out_df


def interpolate_interpretation_surface(
    interpretation_df: pd.DataFrame,
    geometry: Dict[str, Any],
    outlier_threshold: float,
    min_neighbor_count: int = 2,
    keep_nan: bool = True,
) -> pd.DataFrame:
    """对单个层位面执行去异常与规则网格插值。

    Parameters
    ----------
    interpretation_df : pd.DataFrame
        输入离散解释点，至少包含 ``inline``、``xline``、``interpretation`` 三列。
    geometry : Dict[str, Any]
        地震几何信息，至少包含 ``inline_min``、``inline_max``、``inline_step``、
        ``xline_min``、``xline_max``、``xline_step``。
    outlier_threshold : float
        孤立点判定阈值。若样点与邻域中值差值绝对值超过该阈值，则剔除为 NaN。
        单位与 ``interpretation`` 一致，例如 s 或 m。
    min_neighbor_count : int, default=2
        执行孤立点判断所需的最小有效邻居数（十字邻域，不含中心点）。
    keep_nan : bool, default=True
        为 ``True`` 时返回完整规则网格；
        为 ``False`` 时仅返回有效解释点。

    Notes
    -----
    当 geometry 指定为 ``sample_domain='time'`` 且 ``sample_unit='s'`` 时，
    若 interpretation 数值量级明显超过秒域常见范围（>10），会自动按 ms 输入除以 1000 转为 s。

    插值流程为：

    1. 将离散解释点映射到规则 inline/xline 网格；
    2. 基于十字邻域中值剔除孤立异常点；
    3. 先使用 linear 插值填补凸包内空洞，再使用 nearest 兜底。

    Returns
    -------
    pd.DataFrame
        列为 ``inline``、``xline``、``interpretation`` 的 DataFrame。
        剔除统计写入 ``df.attrs["outlier_removal"]``。

    Raises
    ------
    ValueError
        当 geometry 缺少规则轴信息，或输入 DataFrame 缺少必需列时抛出。

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {"inline": [0, 0, 1, 1], "xline": [0, 1, 0, 1], "interpretation": [1.0, 1.2, 1.1, 1.3]}
    ... )
    >>> geometry = {
    ...     "inline_min": 0, "inline_max": 1, "inline_step": 1,
    ...     "xline_min": 0, "xline_max": 1, "xline_step": 1,
    ... }
    >>> out = interpolate_interpretation_surface(df, geometry, outlier_threshold=0.05)
    >>> sorted(out.columns)
    ['inline', 'interpretation', 'xline']
    """
    _validate_geometry_keys(geometry)

    il_axis = _build_axis(
        float(geometry["inline_min"]),
        float(geometry["inline_max"]),
        float(geometry["inline_step"]),
        "inline",
    )
    xl_axis = _build_axis(
        float(geometry["xline_min"]),
        float(geometry["xline_max"]),
        float(geometry["xline_step"]),
        "xline",
    )

    normalized_interpretation_df = _normalize_interpretation_unit_for_geometry(interpretation_df, geometry)

    input_values = normalized_interpretation_df[["inline", "xline", "interpretation"]].to_numpy(
        dtype=float,
        copy=False,
    )
    valid_input_count = int(np.count_nonzero(np.isfinite(input_values).all(axis=1)))
    surface = _to_surface_grid(normalized_interpretation_df, il_axis, xl_axis)
    surface_despiked, outlier_stats = _remove_isolated_outliers_with_stats(
        surface,
        threshold=outlier_threshold,
        min_neighbor_count=min_neighbor_count,
    )
    outlier_stats["input_valid_points"] = valid_input_count
    outlier_stats["gridded_valid_points"] = outlier_stats["total_points"]
    surface_filled = _interpolate_whole_mesh_linear_then_nearest(surface_despiked)

    il_grid, xl_grid = np.meshgrid(il_axis, xl_axis, indexing="ij")
    out_df = pd.DataFrame(
        {
            "inline": il_grid.ravel(),
            "xline": xl_grid.ravel(),
            "interpretation": surface_filled.ravel(),
        }
    )
    out_df.attrs["outlier_removal"] = outlier_stats

    if keep_nan:
        return out_df
    out_df_valid = out_df[np.isfinite(out_df["interpretation"])].reset_index(drop=True)
    out_df_valid.attrs["outlier_removal"] = outlier_stats
    return out_df_valid
