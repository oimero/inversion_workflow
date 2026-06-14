"""cup.seismic.horizon: 单层位解释面构建与采样。

本模块把“单个层位面”的规则网格、插值支撑和井位采样从目标层段
工作流中抽出来。它不负责多层位排序、厚度修复或三维掩码构建。

边界说明
--------
- 负责单个层位解释点到规则网格面的构建、异常点处理和采样审计。
- 不负责多层位排序、层间厚度修复或三维目标窗构建。
- 不读取 Petrel 文件；调用方先用 ``cup.petrel.load`` 得到 DataFrame。

核心公开对象
------------
1. HorizonSurface: 单个层位解释面。
2. HorizonSample: 单点采样结果和审计信息。
3. build_horizon_surface: 从原始解释表构建层位面。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import generic_filter
from scipy.spatial import QhullError, cKDTree  # type: ignore

OUTLIER_REMOVAL_WARNING_RATIO = 0.05


@dataclass(frozen=True)
class HorizonSample:
    """层位面在单个位置的采样结果。"""

    value: float
    inline_float: float
    xline_float: float
    method: str
    nearest_inline: float
    nearest_xline: float
    nearest_line_distance: float
    support_status: str

    def to_dict(self) -> dict[str, float | str]:
        """转换为脚本 CSV/JSON 友好的字典。"""
        return {
            "value": float(self.value),
            "inline_float": float(self.inline_float),
            "xline_float": float(self.xline_float),
            "method": self.method,
            "nearest_inline": float(self.nearest_inline),
            "nearest_xline": float(self.nearest_xline),
            "nearest_line_distance": float(self.nearest_line_distance),
            "support_status": self.support_status,
        }

    def __getitem__(self, key: str) -> float | str:
        """提供旧 dict 风格读取，便于脚本逐步迁移。"""
        return self.to_dict()[key]


@dataclass
class SurfaceInterpolation:
    """单层位从原始控制点到插值面的中间结果。"""

    raw_grid: np.ndarray
    despiked_grid: np.ndarray
    linear_grid: np.ndarray
    nearest_grid: np.ndarray
    linear_support_mask: np.ndarray
    raw_mask: np.ndarray
    nearest_distance_grid: np.ndarray
    outlier_stats: dict[str, Any]


def build_axis(axis_min: float, axis_max: float, axis_step: float, axis_name: str) -> np.ndarray:
    """构建等步长坐标轴。"""
    if axis_step <= 0:
        raise ValueError(f"{axis_name}_step must be positive, got {axis_step}.")
    return np.arange(float(axis_min), float(axis_max) + float(axis_step), float(axis_step), dtype=float)


def normalize_interpretation_unit_for_geometry(
    interpretation_df: pd.DataFrame,
    geometry: dict[str, Any],
) -> pd.DataFrame:
    """按地震几何采样单位规范解释值。"""
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


def require_interpretation_columns(df: pd.DataFrame, name: str) -> None:
    """检查解释表所需列。"""
    required = {"inline", "xline", "interpretation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"horizon '{name}' is missing required columns: {sorted(missing)}")


def grid_raw_interpretation(
    interpretation_df: pd.DataFrame,
    il_axis: np.ndarray,
    xl_axis: np.ndarray,
) -> np.ndarray:
    """将原始解释点落到规则 inline/xline 网格。"""
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


def remove_isolated_outliers_with_stats(
    surface: np.ndarray,
    threshold: Optional[float],
    min_neighbor_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """移除孤立异常点并返回统计。"""
    if min_neighbor_count < 1:
        raise ValueError(f"outlier_min_neighbor_count must be >= 1, got {min_neighbor_count}.")

    valid_mask = np.isfinite(surface)
    total_count = int(np.count_nonzero(valid_mask))
    stats: dict[str, Any] = {
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
    footprint = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    local_median = generic_filter(out, function=_nanmedian, footprint=footprint, mode="constant", cval=np.nan)
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


def linear_then_nearest_from_grid(
    control_grid: np.ndarray,
    *,
    nearest_distance_limit: Optional[float],
    raw_grid: Optional[np.ndarray] = None,
    outlier_stats: Optional[dict[str, Any]] = None,
) -> SurfaceInterpolation:
    """先线性插值再最近邻补全单层位网格。"""
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

    return SurfaceInterpolation(
        raw_grid=original_grid,
        despiked_grid=despiked_grid,
        linear_grid=linear_grid,
        nearest_grid=nearest_grid,
        linear_support_mask=linear_support_mask,
        raw_mask=raw_mask,
        nearest_distance_grid=nearest_distance_grid,
        outlier_stats={} if outlier_stats is None else dict(outlier_stats),
    )


def build_surface_interpolation(
    interpretation_df: pd.DataFrame,
    il_axis: np.ndarray,
    xl_axis: np.ndarray,
    *,
    nearest_distance_limit: Optional[float],
    outlier_threshold: Optional[float],
    outlier_min_neighbor_count: int,
) -> SurfaceInterpolation:
    """从原始解释表构建单层位插值结果。"""
    raw_grid = grid_raw_interpretation(interpretation_df, il_axis, xl_axis)
    despiked_grid, outlier_stats = remove_isolated_outliers_with_stats(
        raw_grid,
        threshold=outlier_threshold,
        min_neighbor_count=outlier_min_neighbor_count,
    )
    return linear_then_nearest_from_grid(
        despiked_grid,
        nearest_distance_limit=nearest_distance_limit,
        raw_grid=raw_grid,
        outlier_stats=outlier_stats,
    )


def resolve_axis_interpolation_window(axis: np.ndarray, coord_float: float, axis_name: str) -> tuple[int, int, float]:
    """解析轴向插值窗口索引与权重。"""
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


def bilinear_interpolate_surface_at_location(
    surface: np.ndarray,
    il_axis: np.ndarray,
    xl_axis: np.ndarray,
    il_float: float,
    xl_float: float,
) -> float:
    """对单个规则网格层位面进行双线性插值。"""
    il0, il1, wi = resolve_axis_interpolation_window(il_axis, il_float, "inline")
    xl0, xl1, wj = resolve_axis_interpolation_window(xl_axis, xl_float, "xline")
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


@dataclass(frozen=True)
class HorizonSurface:
    """单个规则网格层位解释面。"""

    name: str
    inline_axis: np.ndarray
    xline_axis: np.ndarray
    values: np.ndarray
    value_domain: str = ""
    value_unit: str = ""
    support_mask: np.ndarray | None = None
    source_mask: np.ndarray | None = None
    nearest_distance_grid: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        il = np.asarray(self.inline_axis, dtype=np.float64)
        xl = np.asarray(self.xline_axis, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (il.size, xl.size):
            raise ValueError("Horizon surface shape does not match inline/xline axes.")
        if il.size < 2 or xl.size < 2:
            raise ValueError("Horizon surface requires at least 2 inline and 2 xline samples.")
        object.__setattr__(self, "inline_axis", il)
        object.__setattr__(self, "xline_axis", xl)
        object.__setattr__(self, "values", values)
        if self.support_mask is not None:
            support_mask = np.asarray(self.support_mask, dtype=bool)
            if support_mask.shape != values.shape:
                raise ValueError("support_mask shape does not match horizon values.")
            object.__setattr__(self, "support_mask", support_mask)
        if self.source_mask is not None:
            source_mask = np.asarray(self.source_mask, dtype=bool)
            if source_mask.shape != values.shape:
                raise ValueError("source_mask shape does not match horizon values.")
            object.__setattr__(self, "source_mask", source_mask)
        if self.nearest_distance_grid is not None:
            nearest_distance_grid = np.asarray(self.nearest_distance_grid, dtype=np.float64)
            if nearest_distance_grid.shape != values.shape:
                raise ValueError("nearest_distance_grid shape does not match horizon values.")
            object.__setattr__(self, "nearest_distance_grid", nearest_distance_grid)
        if not np.any(np.isfinite(values)):
            raise ValueError("Horizon surface contains no finite interpretation values.")

    @classmethod
    def from_grid(
        cls,
        *,
        name: str,
        inline_axis: np.ndarray,
        xline_axis: np.ndarray,
        values: np.ndarray,
        value_domain: str = "",
        value_unit: str = "",
        support_mask: np.ndarray | None = None,
        source_mask: np.ndarray | None = None,
        nearest_distance_grid: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "HorizonSurface":
        """从已有规则网格构建层位面。"""
        return cls(
            name=name,
            inline_axis=inline_axis,
            xline_axis=xline_axis,
            values=values,
            value_domain=value_domain,
            value_unit=value_unit,
            support_mask=support_mask,
            source_mask=source_mask,
            nearest_distance_grid=nearest_distance_grid,
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def from_petrel_dataframe(cls, df: pd.DataFrame, *, name: str = "") -> "HorizonSurface":
        """从已规则化的 Petrel 解释 DataFrame 构建轻量层位面。"""
        require_interpretation_columns(df, name)
        inline_axis = np.sort(pd.to_numeric(df["inline"], errors="coerce").dropna().unique().astype(np.float64))
        xline_axis = np.sort(pd.to_numeric(df["xline"], errors="coerce").dropna().unique().astype(np.float64))
        pivot = df.pivot_table(index="inline", columns="xline", values="interpretation", aggfunc="first")
        pivot = pivot.reindex(index=inline_axis, columns=xline_axis)
        values = pivot.to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        return cls(
            name=name,
            inline_axis=inline_axis,
            xline_axis=xline_axis,
            values=values,
            support_mask=finite,
            source_mask=finite,
        )

    def sample_at_line(
        self,
        inline_float: float,
        xline_float: float,
        *,
        nearest_fallback_max_line_distance: float = 5.0,
    ) -> HorizonSample:
        """在浮点线号坐标处采样层位面，并返回审计信息。"""
        il = float(inline_float)
        xl = float(xline_float)
        if not (self.inline_axis[0] <= il <= self.inline_axis[-1]):
            raise ValueError(f"Inline {il} is outside horizon {self.name!r} range.")
        if not (self.xline_axis[0] <= xl <= self.xline_axis[-1]):
            raise ValueError(f"Xline {xl} is outside horizon {self.name!r} range.")

        i1 = int(np.searchsorted(self.inline_axis, il, side="right"))
        j1 = int(np.searchsorted(self.xline_axis, xl, side="right"))
        i0 = max(0, min(i1 - 1, self.inline_axis.size - 2))
        j0 = max(0, min(j1 - 1, self.xline_axis.size - 2))
        i1 = i0 + 1
        j1 = j0 + 1
        il0, il1 = self.inline_axis[i0], self.inline_axis[i1]
        xl0, xl1 = self.xline_axis[j0], self.xline_axis[j1]
        values = np.array(
            [[self.values[i0, j0], self.values[i0, j1]], [self.values[i1, j0], self.values[i1, j1]]],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(values)):
            il_grid, xl_grid = np.meshgrid(self.inline_axis, self.xline_axis, indexing="ij")
            finite = np.isfinite(self.values)
            distances = np.hypot(il_grid[finite] - il, xl_grid[finite] - xl)
            nearest_index = int(np.nanargmin(distances))
            nearest_distance = float(distances[nearest_index])
            finite_inline = il_grid[finite]
            finite_xline = xl_grid[finite]
            finite_values = self.values[finite]
            if nearest_distance <= float(nearest_fallback_max_line_distance):
                return HorizonSample(
                    value=float(finite_values[nearest_index]),
                    inline_float=il,
                    xline_float=xl,
                    method="nearest_valid_fallback",
                    nearest_inline=float(finite_inline[nearest_index]),
                    nearest_xline=float(finite_xline[nearest_index]),
                    nearest_line_distance=nearest_distance,
                    support_status="nearest_fallback",
                )
            raise ValueError(f"Horizon {self.name!r} has missing support around inline/xline {il}, {xl}.")

        wi = 0.0 if il1 == il0 else (il - il0) / (il1 - il0)
        wj = 0.0 if xl1 == xl0 else (xl - xl0) / (xl1 - xl0)
        value = float(
            values[0, 0] * (1.0 - wi) * (1.0 - wj)
            + values[1, 0] * wi * (1.0 - wj)
            + values[0, 1] * (1.0 - wi) * wj
            + values[1, 1] * wi * wj
        )
        exact = np.isclose(wi, 0.0, atol=1e-12) and np.isclose(wj, 0.0, atol=1e-12)
        return HorizonSample(
            value=value,
            inline_float=il,
            xline_float=xl,
            method="exact" if exact else "bilinear",
            nearest_inline=il,
            nearest_xline=xl,
            nearest_line_distance=0.0,
            support_status="supported",
        )

    def value_at_line(
        self,
        inline_float: float,
        xline_float: float,
        *,
        nearest_fallback_max_line_distance: float = 5.0,
    ) -> float:
        """返回浮点线号坐标处的层位值。"""
        return float(
            self.sample_at_line(
                inline_float,
                xline_float,
                nearest_fallback_max_line_distance=nearest_fallback_max_line_distance,
            ).value
        )

    def to_dataframe(self) -> pd.DataFrame:
        """将层位面展开为 DataFrame。"""
        il_grid, xl_grid = np.meshgrid(self.inline_axis, self.xline_axis, indexing="ij")
        return pd.DataFrame({"inline": il_grid.ravel(), "xline": xl_grid.ravel(), "interpretation": self.values.ravel()})


def build_horizon_surface(
    interpretation_df: pd.DataFrame,
    inline_axis: np.ndarray,
    xline_axis: np.ndarray,
    *,
    name: str = "",
    nearest_distance_limit: Optional[float] = None,
    outlier_threshold: Optional[float] = None,
    outlier_min_neighbor_count: int = 2,
    value_domain: str = "",
    value_unit: str = "",
) -> tuple[HorizonSurface, SurfaceInterpolation]:
    """从原始解释表构建 ``HorizonSurface`` 和插值审计结果。"""
    interpolation = build_surface_interpolation(
        interpretation_df,
        inline_axis,
        xline_axis,
        nearest_distance_limit=nearest_distance_limit,
        outlier_threshold=outlier_threshold,
        outlier_min_neighbor_count=outlier_min_neighbor_count,
    )
    surface = HorizonSurface.from_grid(
        name=name,
        inline_axis=inline_axis,
        xline_axis=xline_axis,
        values=interpolation.nearest_grid,
        value_domain=value_domain,
        value_unit=value_unit,
        support_mask=interpolation.linear_support_mask,
        source_mask=interpolation.raw_mask,
        nearest_distance_grid=interpolation.nearest_distance_grid,
        metadata={"outlier_stats": dict(interpolation.outlier_stats)},
    )
    return surface, interpolation
