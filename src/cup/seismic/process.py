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


def _validate_geometry_keys(geometry: Dict[str, Any]) -> None:
    """校验几何信息是否包含插值必需键。"""
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


def _interpolate_interpretation_surface_grid(
    interpretation_df: pd.DataFrame,
    il_axis: np.ndarray,
    xl_axis: np.ndarray,
    outlier_threshold: float,
    min_neighbor_count: int = 2,
) -> np.ndarray:
    """对单个层位执行去孤立点与全域插值，返回 2D 网格。"""
    surface = _to_surface_grid(interpretation_df, il_axis, xl_axis)
    surface_despiked = _remove_isolated_outliers(
        surface,
        threshold=outlier_threshold,
        min_neighbor_count=min_neighbor_count,
    )
    return _interpolate_whole_mesh_linear_then_nearest(surface_despiked)


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


class TargetLayer:
    """目的层对象：持有已插值层位并做顶底约束校验。"""

    def __init__(
        self,
        interpolated_horizon_dfs: Dict[str, pd.DataFrame],
        geometry: Dict[str, Any],
        top_name: str,
        bottom_name: str,
    ) -> None:
        """初始化目的层并校验已插值顶底层位关系。"""
        if len(interpolated_horizon_dfs) < 2:
            raise ValueError("interpolated_horizon_dfs must contain at least two horizons.")
        if top_name not in interpolated_horizon_dfs:
            raise ValueError(f"top_name '{top_name}' is not in interpolated_horizon_dfs.")
        if bottom_name not in interpolated_horizon_dfs:
            raise ValueError(f"bottom_name '{bottom_name}' is not in interpolated_horizon_dfs.")

        _validate_geometry_keys(geometry)

        self.geometry = dict(geometry)
        self.interpolated_horizon_dfs = {name: df.copy() for name, df in interpolated_horizon_dfs.items()}
        self.top_name = top_name
        self.bottom_name = bottom_name

        self._assert_top_strictly_smaller_than_bottom()

    def _assert_top_strictly_smaller_than_bottom(self) -> None:
        """在共址样点上断言顶层位解释值严格小于底层位解释值。"""
        required_cols = ["inline", "xline", "interpretation"]
        top_df = self.interpolated_horizon_dfs[self.top_name][required_cols].copy()
        bot_df = self.interpolated_horizon_dfs[self.bottom_name][required_cols].copy()

        top_df = top_df[np.isfinite(top_df["interpretation"])].rename(columns={"interpretation": "interpretation_top"})
        bot_df = bot_df[np.isfinite(bot_df["interpretation"])].rename(
            columns={"interpretation": "interpretation_bottom"}
        )

        aligned = pd.merge(top_df, bot_df, on=["inline", "xline"], how="inner")
        if aligned.empty:
            raise AssertionError(
                "No overlapping (inline, xline) samples were found between top and bottom horizons; "
                "cannot assert top < bottom."
            )

        violated = aligned[aligned["interpretation_top"] >= aligned["interpretation_bottom"]]
        assert violated.empty, (
            f"Top horizon must be strictly smaller than bottom horizon on overlapping samples. "
            f"Found {len(violated)} violation(s)."
        )

    def convert_horizon_to_relative_sample_index(
        self,
        horizon_name: str,
    ) -> pd.DataFrame:
        """将绝对层位值转换为相对采样索引，并强制校验采样范围。"""
        if horizon_name not in self.interpolated_horizon_dfs:
            raise ValueError(f"horizon_name '{horizon_name}' is not in interpolated_horizon_dfs.")

        required_geometry_keys = {"sample_min", "sample_max", "sample_step"}
        missing_geometry_keys = required_geometry_keys - set(self.geometry)
        if missing_geometry_keys:
            raise ValueError(f"geometry is missing required keys for sample indexing: {sorted(missing_geometry_keys)}")

        sample_min = float(self.geometry["sample_min"])
        sample_max = float(self.geometry["sample_max"])
        sample_step = float(self.geometry["sample_step"])
        if sample_step <= 0:
            raise ValueError(f"sample_step must be positive, got {sample_step}.")

        horizon_df = self.interpolated_horizon_dfs[horizon_name].copy()
        required_cols = {"inline", "xline", "interpretation"}
        missing_cols = required_cols - set(horizon_df.columns)
        if missing_cols:
            raise ValueError(
                f"interpolated_horizon_dfs['{horizon_name}'] is missing required columns: {sorted(missing_cols)}"
            )

        normalized_horizon_df = _normalize_interpretation_unit_for_geometry(horizon_df, self.geometry)

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

    def to_mask(
        self,
        top_name: Optional[str] = None,
        bottom_name: Optional[str] = None,
    ) -> np.ndarray:
        """根据顶/底层位生成三维布尔掩码。"""
        top_name = self.top_name if top_name is None else top_name
        bottom_name = self.bottom_name if bottom_name is None else bottom_name

        il_axis = _build_line_axis(
            int(self.geometry["inline_min"]),
            int(self.geometry["inline_max"]),
            int(self.geometry["inline_step"]),
        )
        xl_axis = _build_line_axis(
            int(self.geometry["xline_min"]),
            int(self.geometry["xline_max"]),
            int(self.geometry["xline_step"]),
        )

        n_il = int(self.geometry.get("n_il", il_axis.size))
        n_xl = int(self.geometry.get("n_xl", xl_axis.size))
        if n_il != il_axis.size:
            raise ValueError(f"geometry n_il={n_il} does not match axis size {il_axis.size}.")
        if n_xl != xl_axis.size:
            raise ValueError(f"geometry n_xl={n_xl} does not match axis size {xl_axis.size}.")

        if "n_sample" in self.geometry:
            n_sample = int(self.geometry["n_sample"])
        else:
            required_keys = {"sample_min", "sample_max", "sample_step"}
            missing = required_keys - set(self.geometry)
            if missing:
                raise ValueError(f"geometry is missing required keys for n_sample: {sorted(missing)}")
            sample_max = float(self.geometry["sample_max"])
            sample_min = float(self.geometry["sample_min"])
            sample_step = float(self.geometry["sample_step"])
            n_sample = int(round((sample_max - sample_min) / sample_step)) + 1

        top_idx_df = self.convert_horizon_to_relative_sample_index(top_name)
        bot_idx_df = self.convert_horizon_to_relative_sample_index(bottom_name)

        top_grid_df = top_idx_df[["inline", "xline", "sample_index"]].rename(columns={"sample_index": "interpretation"})
        bot_grid_df = bot_idx_df[["inline", "xline", "sample_index"]].rename(columns={"sample_index": "interpretation"})

        top_grid = _to_surface_grid(top_grid_df, il_axis, xl_axis)
        bot_grid = _to_surface_grid(bot_grid_df, il_axis, xl_axis)

        mask = np.zeros((n_il, n_xl, n_sample), dtype=bool)
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
            单位与 `interpretation` 一致（例如 s 或 m）。
    min_neighbor_count : int, default=2
            执行孤立点判断所需的最小有效邻居数（十字邻域，不含中心点）。
    keep_nan : bool, default=True
            为 True 时返回完整网格；
            为 False 时仅返回有效解释点。

    Notes
    -----
    当 geometry 指定为 ``sample_domain='time'`` 且 ``sample_unit='s'`` 时，
    若 interpretation 数值量级明显超过秒域常见范围（>10），会自动按 ms 输入除以 1000 转为 s。

    Returns
    -------
    pd.DataFrame
            列为 `inline`, `xline`, `interpretation`。
    """
    _validate_geometry_keys(geometry)

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

    normalized_interpretation_df = _normalize_interpretation_unit_for_geometry(interpretation_df, geometry)

    surface_filled = _interpolate_interpretation_surface_grid(
        interpretation_df=normalized_interpretation_df,
        il_axis=il_axis,
        xl_axis=xl_axis,
        outlier_threshold=outlier_threshold,
        min_neighbor_count=min_neighbor_count,
    )

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
