"""cup.seismic.process: 地震解释层位预处理、插值与目的层对象。

本模块提供离散层位解释点到规则 inline/xline 网格的预处理与插值能力，
并封装 `TargetLayer` 对象，用于按层序组织多个层位、查询浮点位置解释值、
转换到相对采样索引以及生成层段三维布尔掩码。

边界说明
--------
- 本模块不负责解释点文件读取、地震体几何查询或低频模型反演流程。
- 本模块假定输入层位已经位于同一 inline/xline 坐标系，且 geometry 由上游提供。
- 当 ``sample_domain='time'`` 且 ``sample_unit='s'`` 时，会按经验规则自动识别
  毫秒输入并换算为秒。

核心公开对象
------------
1. interpolate_interpretation_surface: 清洗并插值单个层位面。
2. TargetLayer: 管理有序层位并提供层段相关操作。
3. TargetLayer.get_interpretation_values_at_location: 查询浮点位置上的层位值。
4. TargetLayer.to_mask: 将相邻层位转换为三维布尔掩码。

Examples
--------
>>> import pandas as pd
>>> from cup.seismic.process import TargetLayer, interpolate_interpretation_surface
>>> geometry = {
...     "inline_min": 0, "inline_max": 1, "inline_step": 1,
...     "xline_min": 0, "xline_max": 1, "xline_step": 1,
...     "sample_min": 0.0, "sample_max": 3.0, "sample_step": 1.0,
...     "sample_domain": "time", "sample_unit": "s",
... }
>>> h1 = pd.DataFrame(
...     {"inline": [0, 0, 1, 1], "xline": [0, 1, 0, 1], "interpretation": [1.0, 1.1, 1.2, 1.3]}
... )
>>> h2 = pd.DataFrame(
...     {"inline": [0, 0, 1, 1], "xline": [0, 1, 0, 1], "interpretation": [2.0, 2.1, 2.2, 2.3]}
... )
>>> h1_interp = interpolate_interpretation_surface(h1, geometry, outlier_threshold=0.05)
>>> h2_interp = interpolate_interpretation_surface(h2, geometry, outlier_threshold=0.05)
>>> target = TargetLayer({"h1": h1_interp, "h2": h2_interp}, geometry, ["h1", "h2"])
>>> sorted(target.get_interpretation_values_at_location(0.5, 0.5))
['h1', 'h2']
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter
from scipy.spatial import QhullError

OUTLIER_REMOVAL_WARNING_RATIO = 0.05


def _sanitize_debug_token(value: str) -> str:
    """将任意名称规整为适合作为调试文件名的片段。"""
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    token = token.strip("_")
    return token or "unnamed"


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


def _resolve_axis_interpolation_window(
    axis: np.ndarray,
    coord_float: float,
    axis_name: str,
) -> tuple[int, int, float]:
    """将浮点坐标映射到规则轴上的双线性插值窗口。"""
    if axis.ndim != 1 or axis.size < 1:
        raise ValueError(f"{axis_name} axis must be a non-empty 1D array.")

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
    k_rounded = round(k)
    if np.isclose(k, k_rounded, atol=1e-8):
        k = float(k_rounded)

    k0 = int(np.floor(k))
    k1 = int(np.ceil(k))
    max_idx = axis.size - 1
    if k0 < 0:
        k0 = 0
    if k1 > max_idx:
        k1 = max_idx
    if k0 > max_idx or k1 < 0:
        raise ValueError(f"{axis_name}_float={coord} is out of bounds [{axis_min}, {axis_max}].")

    weight = 0.0 if k1 == k0 else float(k - k0)
    return k0, k1, weight


def _bilinear_interpolate_surface_at_location(
    surface: np.ndarray,
    il_axis: np.ndarray,
    xl_axis: np.ndarray,
    il_float: float,
    xl_float: float,
) -> float:
    """在规则 inline/xline 网格上对单个层位面执行双线性插值。"""
    if surface.shape != (il_axis.size, xl_axis.size):
        raise ValueError(
            "surface shape does not match axis sizes: "
            f"surface.shape={surface.shape}, il_axis.size={il_axis.size}, xl_axis.size={xl_axis.size}."
        )

    il0_idx, il1_idx, wi = _resolve_axis_interpolation_window(il_axis, il_float, "inline")
    xl0_idx, xl1_idx, wj = _resolve_axis_interpolation_window(xl_axis, xl_float, "xline")

    node_indices = {
        (il0_idx, xl0_idx),
        (il0_idx, xl1_idx),
        (il1_idx, xl0_idx),
        (il1_idx, xl1_idx),
    }
    for il_idx, xl_idx in node_indices:
        value = float(surface[il_idx, xl_idx])
        if not np.isfinite(value):
            raise ValueError(
                f"missing interpretation node: inline={int(il_axis[il_idx])}, xline={int(xl_axis[xl_idx])}"
            )

    t00 = float(surface[il0_idx, xl0_idx])
    t01 = float(surface[il0_idx, xl1_idx])
    t10 = float(surface[il1_idx, xl0_idx])
    t11 = float(surface[il1_idx, xl1_idx])

    return float((1.0 - wi) * (1.0 - wj) * t00 + (1.0 - wi) * wj * t01 + wi * (1.0 - wj) * t10 + wi * wj * t11)


class TargetLayer:
    """按层序组织多个层位并提供层段边界操作。

    Parameters
    ----------
    interpolated_horizon_dfs : Dict[str, pd.DataFrame]
        层位名到层位 DataFrame 的映射。每个 DataFrame 至少包含
        ``inline``、``xline`` 和 ``interpretation`` 三列，通常为规则网格上的
        插值结果，也允许包含 NaN。
    geometry : Dict[str, Any]
        地震几何信息。
    horizon_names : list[str]
        从顶到底排序后的层位名列表。列表中的名称必须唯一，且都存在于
        ``interpolated_horizon_dfs`` 中。
    debug_dir : str or pathlib.Path, optional
        当相邻层位顺序校验失败时，将违例点 CSV 写入该目录，便于定位问题位置。

    Notes
    -----
    构造时会：

    - 根据 geometry 构建规则 inline/xline/sample 轴；
    - 在秒域场景下按经验规则归一化层位单位；
    - 校验相邻层位在共址样点上的解释值顺序；
    - 若配置 ``debug_dir``，在校验失败时导出违例样点日志；
    - 为后续插值查询预构建 2D 层位网格。
    """

    def __init__(
        self,
        interpolated_horizon_dfs: Dict[str, pd.DataFrame],
        geometry: Dict[str, Any],
        horizon_names: list[str],
        debug_dir: Optional[str | Path] = None,
    ) -> None:
        """初始化目的层对象。"""
        if len(interpolated_horizon_dfs) < 2:
            raise ValueError("interpolated_horizon_dfs must contain at least two horizons.")
        if len(horizon_names) < 2:
            raise ValueError("horizon_names must contain at least two ordered horizons.")
        if len(set(horizon_names)) != len(horizon_names):
            raise ValueError("horizon_names must be unique.")
        missing_horizon_names = [name for name in horizon_names if name not in interpolated_horizon_dfs]
        if missing_horizon_names:
            raise ValueError(f"horizon_names not found in interpolated_horizon_dfs: {missing_horizon_names}")

        _validate_geometry_keys(geometry)

        self.geometry = dict(geometry)
        self.interpolated_horizon_dfs = {name: df.copy() for name, df in interpolated_horizon_dfs.items()}
        self.horizon_names = list(horizon_names)
        self.debug_dir = Path(debug_dir) if debug_dir is not None else None
        self._il_axis, self._xl_axis, self._sample_axis = self._build_axes()
        self._normalized_horizon_dfs = {
            name: _normalize_interpretation_unit_for_geometry(df, self.geometry)
            for name, df in self.interpolated_horizon_dfs.items()
        }
        self._horizon_grids = {
            name: _to_surface_grid(df, self._il_axis, self._xl_axis)
            for name, df in self._normalized_horizon_dfs.items()
        }

        self._assert_horizon_sequence_is_strictly_increasing()

    def iter_zones(self) -> list[tuple[str, str]]:
        """返回按层位顺序构成的相邻层段列表。

        Returns
        -------
        list[tuple[str, str]]
            相邻层位对列表，每个元素为 ``(top_name, bottom_name)``。

        Examples
        --------
        >>> target_layer.iter_zones()
        [('h1', 'h2'), ('h2', 'h3')]
        """
        return list(zip(self.horizon_names[:-1], self.horizon_names[1:]))

    def _assert_horizon_pair_is_strictly_increasing(self, top_name: str, bottom_name: str) -> None:
        """在共址样点上断言相邻层位解释值严格递增。"""
        aligned = self._build_horizon_pair_alignment(top_name, bottom_name)
        if aligned.empty:
            raise AssertionError(
                "No overlapping (inline, xline) samples were found between adjacent horizons; "
                f"cannot assert order for '{top_name}' < '{bottom_name}'."
            )

        if self._horizon_pair_allows_equal_boundary(top_name, bottom_name):
            violated = aligned[aligned["interpretation_top"] > aligned["interpretation_bottom"]].copy()
        else:
            violated = aligned[aligned["interpretation_top"] >= aligned["interpretation_bottom"]].copy()
        if violated.empty:
            return

        debug_path = self._write_horizon_pair_violation_log(top_name, bottom_name, violated)
        error_message = (
            f"Horizon '{top_name}' must be strictly smaller than '{bottom_name}' on overlapping samples. "
            f"Found {len(violated)} violation(s)."
        )
        if debug_path is not None:
            error_message += f" Debug log saved to '{debug_path.as_posix()}'."
        raise AssertionError(error_message)

    def _horizon_pair_allows_equal_boundary(self, top_name: str, bottom_name: str) -> bool:
        """合成扩展层位裁剪到采样边界时允许与相邻原始层位重合。"""
        top_attrs = self.interpolated_horizon_dfs[top_name].attrs
        bottom_attrs = self.interpolated_horizon_dfs[bottom_name].attrs
        return bool(top_attrs.get("boundary_extension") or bottom_attrs.get("boundary_extension"))

    def _build_horizon_pair_alignment(self, top_name: str, bottom_name: str) -> pd.DataFrame:
        """返回相邻层位在共址样点上的对齐结果。"""
        required_cols = ["inline", "xline", "interpretation"]
        top_df = self.interpolated_horizon_dfs[top_name][required_cols].copy()
        bot_df = self.interpolated_horizon_dfs[bottom_name][required_cols].copy()

        top_df = top_df[np.isfinite(top_df["interpretation"])].rename(columns={"interpretation": "interpretation_top"})
        bot_df = bot_df[np.isfinite(bot_df["interpretation"])].rename(
            columns={"interpretation": "interpretation_bottom"}
        )

        return pd.merge(top_df, bot_df, on=["inline", "xline"], how="inner").sort_values(
            ["inline", "xline"]
        ).reset_index(drop=True)

    def _write_horizon_pair_violation_log(
        self,
        top_name: str,
        bottom_name: str,
        violated: pd.DataFrame,
    ) -> Optional[Path]:
        """将层位顺序违例点导出到调试目录。"""
        if self.debug_dir is None or violated.empty:
            return None

        self.debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_top = _sanitize_debug_token(top_name)
        safe_bottom = _sanitize_debug_token(bottom_name)
        log_path = self.debug_dir / f"horizon_order_violation__{safe_top}__lt__{safe_bottom}__{timestamp}.csv"

        log_df = violated.copy()
        log_df.insert(0, "bottom_name", bottom_name)
        log_df.insert(0, "top_name", top_name)
        log_df["interpretation_delta"] = log_df["interpretation_top"] - log_df["interpretation_bottom"]
        log_df = log_df[
            [
                "top_name",
                "bottom_name",
                "inline",
                "xline",
                "interpretation_top",
                "interpretation_bottom",
                "interpretation_delta",
            ]
        ]
        log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
        return log_path.resolve()

    def _assert_horizon_sequence_is_strictly_increasing(self) -> None:
        """校验所有相邻层位在共址样点上的顺序。"""
        for top_name, bottom_name in self.iter_zones():
            self._assert_horizon_pair_is_strictly_increasing(top_name, bottom_name)

    def _build_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        il_axis = _build_axis(
            float(self.geometry["inline_min"]),
            float(self.geometry["inline_max"]),
            float(self.geometry["inline_step"]),
            "inline",
        )
        xl_axis = _build_axis(
            float(self.geometry["xline_min"]),
            float(self.geometry["xline_max"]),
            float(self.geometry["xline_step"]),
            "xline",
        )
        sample_axis = _build_axis(
            float(self.geometry["sample_min"]),
            float(self.geometry["sample_max"]),
            float(self.geometry["sample_step"]),
            "sample",
        )
        return il_axis, xl_axis, sample_axis

    @property
    def ilines(self) -> np.ndarray:
        """返回规则 inline 轴副本。"""
        return self._il_axis.copy()

    @property
    def xlines(self) -> np.ndarray:
        """返回规则 xline 轴副本。"""
        return self._xl_axis.copy()

    @property
    def samples(self) -> np.ndarray:
        """返回规则采样轴副本。"""
        return self._sample_axis.copy()

    def get_horizon_interpretation_at_location(
        self,
        horizon_name: str,
        il_float: float,
        xl_float: float,
    ) -> float:
        """返回单个层位在浮点位置上的双线性插值解释值。

        Parameters
        ----------
        horizon_name : str
            目标层位名称。
        il_float : float
            查询点的 inline 坐标，可位于规则轴的相邻节点之间。
        xl_float : float
            查询点的 xline 坐标，可位于规则轴的相邻节点之间。

        Returns
        -------
        float
            指定层位在该位置上的插值解释值。

        Raises
        ------
        ValueError
            当层位名称不存在、查询坐标越界，或双线性插值所需网格节点存在 NaN 时抛出。
        """
        if horizon_name not in self._horizon_grids:
            raise ValueError(f"horizon_name '{horizon_name}' is not in interpolated_horizon_dfs.")

        return _bilinear_interpolate_surface_at_location(
            surface=self._horizon_grids[horizon_name],
            il_axis=self._il_axis,
            xl_axis=self._xl_axis,
            il_float=il_float,
            xl_float=xl_float,
        )

    def get_interpretation_values_at_location(
        self,
        il_float: float,
        xl_float: float,
    ) -> Dict[str, float]:
        """返回所有层位在浮点位置上的双线性插值解释值。

        Parameters
        ----------
        il_float : float
            查询点的 inline 坐标。
        xl_float : float
            查询点的 xline 坐标。

        Returns
        -------
        Dict[str, float]
            层位名到插值解释值的映射，顺序与 ``horizon_names`` 一致。

        Raises
        ------
        ValueError
            当任一层位在该位置无法完成插值时抛出。
        """
        return {
            horizon_name: self.get_horizon_interpretation_at_location(horizon_name, il_float, xl_float)
            for horizon_name in self.horizon_names
        }

    def _build_boundary_extension_horizon_df(
        self,
        source_horizon_name: str,
        *,
        offset: float,
        sample_min: float,
        sample_max: float,
        extension_name: str,
    ) -> pd.DataFrame:
        """基于已有层位构建一个按采样轴裁剪的边界扩展层位。"""
        source_df = self._normalized_horizon_dfs[source_horizon_name]
        out_df = source_df[["inline", "xline", "interpretation"]].copy()
        interpretation = out_df["interpretation"].to_numpy(dtype=float, copy=True)
        finite_mask = np.isfinite(interpretation)
        interpretation[finite_mask] = np.clip(interpretation[finite_mask] + offset, sample_min, sample_max)
        out_df["interpretation"] = interpretation
        out_df.attrs["boundary_extension"] = {
            "name": extension_name,
            "source_horizon": source_horizon_name,
            "offset": float(offset),
            "sample_min": float(sample_min),
            "sample_max": float(sample_max),
        }
        return out_df

    def with_boundary_extension(
        self,
        extension_samples: int,
        *,
        top_extension_name: str = "top_extension",
        bottom_extension_name: str = "bottom_extension",
    ) -> "TargetLayer":
        """返回在顶部上方、底部下方各新增一个合成层位的目的层副本。

        Parameters
        ----------
        extension_samples : int
            扩展的采样点数。顶部层位会向上移动
            ``extension_samples * sample_step``，底部层位会向下移动同样距离。
            结果会裁剪到 ``[sample_min, sample_max]``。
        top_extension_name, bottom_extension_name : str
            新增顶部/底部扩展层位名称。

        Returns
        -------
        TargetLayer
            一个新的 ``TargetLayer``，其层位顺序为
            ``top_extension_name``、原始层位序列、``bottom_extension_name``。
        """
        if extension_samples < 0:
            raise ValueError(f"extension_samples must be >= 0, got {extension_samples}.")
        if extension_samples == 0:
            return TargetLayer(
                interpolated_horizon_dfs=self._normalized_horizon_dfs,
                geometry=self.geometry,
                horizon_names=self.horizon_names,
                debug_dir=self.debug_dir,
            )

        if top_extension_name == bottom_extension_name:
            raise ValueError("top_extension_name and bottom_extension_name must be different.")
        duplicate_extension_names = {
            name for name in (top_extension_name, bottom_extension_name) if name in self.interpolated_horizon_dfs
        }
        if duplicate_extension_names:
            raise ValueError(f"extension horizon names already exist: {sorted(duplicate_extension_names)}")

        sample_min = float(self._sample_axis[0])
        sample_max = float(self._sample_axis[-1])
        sample_step = float(self.geometry["sample_step"])

        offset = float(extension_samples) * sample_step
        top_source_name = self.horizon_names[0]
        bottom_source_name = self.horizon_names[-1]
        extended_horizon_dfs = {name: df.copy() for name, df in self._normalized_horizon_dfs.items()}
        extended_horizon_dfs[top_extension_name] = self._build_boundary_extension_horizon_df(
            top_source_name,
            offset=-offset,
            sample_min=sample_min,
            sample_max=sample_max,
            extension_name=top_extension_name,
        )
        extended_horizon_dfs[bottom_extension_name] = self._build_boundary_extension_horizon_df(
            bottom_source_name,
            offset=offset,
            sample_min=sample_min,
            sample_max=sample_max,
            extension_name=bottom_extension_name,
        )

        return TargetLayer(
            interpolated_horizon_dfs=extended_horizon_dfs,
            geometry=self.geometry,
            horizon_names=[top_extension_name, *self.horizon_names, bottom_extension_name],
            debug_dir=self.debug_dir,
        )

    def _resolve_zone(self, zone: Optional[tuple[str, str]]) -> tuple[str, str]:
        if zone is None:
            raise ValueError("zone must be provided for zone-specific operations.")
        top_name, bottom_name = zone
        if top_name not in self.horizon_names or bottom_name not in self.horizon_names:
            raise ValueError(f"zone contains unknown horizons: {zone}")
        top_idx = self.horizon_names.index(top_name)
        bottom_idx = self.horizon_names.index(bottom_name)
        if bottom_idx != top_idx + 1:
            raise ValueError(f"zone must contain adjacent horizons, got {zone}")
        return top_name, bottom_name

    def convert_horizon_to_relative_sample_index(
        self,
        horizon_name: str,
    ) -> pd.DataFrame:
        """将单个层位的绝对解释值转换为相对采样索引。

        Parameters
        ----------
        horizon_name : str
            需要转换的层位名称。

        Returns
        -------
        pd.DataFrame
            包含 ``inline``、``xline``、``interpretation`` 和 ``sample_index`` 四列
            的新 DataFrame。其中 ``sample_index`` 以
            ``(interpretation - sample_min) / sample_step`` 计算。

        Raises
        ------
        ValueError
            当层位缺列或层位值超出采样范围时抛出。
        """
        if horizon_name not in self.interpolated_horizon_dfs:
            raise ValueError(f"horizon_name '{horizon_name}' is not in interpolated_horizon_dfs.")

        sample_min = float(self._sample_axis[0])
        sample_max = float(self._sample_axis[-1])
        sample_step = float(self.geometry["sample_step"])

        horizon_df = self.interpolated_horizon_dfs[horizon_name].copy()
        required_cols = {"inline", "xline", "interpretation"}
        missing_cols = required_cols - set(horizon_df.columns)
        if missing_cols:
            raise ValueError(
                f"interpolated_horizon_dfs['{horizon_name}'] is missing required columns: {sorted(missing_cols)}"
            )

        normalized_horizon_df = self._normalized_horizon_dfs[horizon_name].copy()

        interpretation = normalized_horizon_df["interpretation"].to_numpy(dtype=float, copy=False)
        sample_index = np.full(interpretation.shape, np.nan, dtype=float)
        finite_mask = np.isfinite(interpretation)
        sample_index[finite_mask] = (interpretation[finite_mask] - sample_min) / sample_step

        if np.any(finite_mask):
            tol = 1e-6
            out_of_range = finite_mask & ((interpretation < sample_min - tol) | (interpretation > sample_max + tol))
            if np.any(out_of_range):
                bad_rows = normalized_horizon_df.loc[out_of_range, ["inline", "xline", "interpretation"]].head(5)
                raise ValueError(
                    "Horizon values are out of sample range. "
                    f"Expected within [{sample_min}, {sample_max}], "
                    f"found {int(np.count_nonzero(out_of_range))} out-of-range points. "
                    f"Examples: {bad_rows.to_dict(orient='records')}"
                )

        out_df = normalized_horizon_df[["inline", "xline", "interpretation"]].copy()
        out_df["sample_index"] = sample_index
        return out_df

    def _get_horizon_sample_index_grid(self, horizon_name: str) -> np.ndarray:
        """返回单个层位在规则网格上的相对采样索引面。"""
        sample_idx_df = self.convert_horizon_to_relative_sample_index(horizon_name)
        grid_df = sample_idx_df[["inline", "xline", "sample_index"]].rename(columns={"sample_index": "interpretation"})
        return _to_surface_grid(grid_df, self._il_axis, self._xl_axis)

    def get_zone_sample_index_grids(self, zone: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
        """返回相邻层段顶底界面的采样索引网格。"""
        top_name, bottom_name = self._resolve_zone(zone)
        return self._get_horizon_sample_index_grid(top_name), self._get_horizon_sample_index_grid(bottom_name)

    def to_mask(
        self,
        zone: Optional[tuple[str, str]] = None,
    ) -> np.ndarray:
        """根据相邻层段生成三维布尔掩码。

        Parameters
        ----------
        zone : Optional[tuple[str, str]], default=None
            指定单个目标层段 ``(top_name, bottom_name)``。为 ``None`` 时，
            返回所有相邻层段的并集掩码。

        Returns
        -------
        np.ndarray
            形状为 ``(n_inline, n_xline, n_sample)`` 的布尔数组。True 表示对应体素
            落在目标层段内。

        Raises
        ------
        ValueError
            当 geometry 中的 inline/xline/sample 维度与规则轴不一致时抛出。

        Notes
        -----
        当前实现使用四舍五入后的顶底采样索引，并将底界面按切片右端点纳入掩码范围。
        因此相邻层段在共享 horizon 处可能出现单个 sample 的重叠归属。
        """
        il_axis, xl_axis = self._il_axis, self._xl_axis

        n_il = int(self.geometry["n_il"])
        n_xl = int(self.geometry["n_xl"])
        if n_il != il_axis.size:
            raise ValueError(f"geometry n_il={n_il} does not match axis size {il_axis.size}.")
        if n_xl != xl_axis.size:
            raise ValueError(f"geometry n_xl={n_xl} does not match axis size {xl_axis.size}.")

        n_sample = int(self.geometry["n_sample"])
        if n_sample != self._sample_axis.size:
            raise ValueError(f"geometry n_sample={n_sample} does not match axis size {self._sample_axis.size}.")

        mask = np.zeros((n_il, n_xl, n_sample), dtype=bool)

        zones = [self._resolve_zone(zone)] if zone is not None else self.iter_zones()
        for zone_top, zone_bottom in zones:
            top_grid, bot_grid = self.get_zone_sample_index_grids((zone_top, zone_bottom))
            for i in range(n_il):
                for j in range(n_xl):
                    t_top = top_grid[i, j]
                    t_bot = bot_grid[i, j]
                    if np.isfinite(t_top) and np.isfinite(t_bot):
                        idx_top = max(0, int(np.round(t_top)))
                        idx_bot = min(n_sample, int(np.round(t_bot)) + 1)
                        if idx_top < idx_bot:
                            mask[i, j, idx_top:idx_bot] = True

        return mask


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
