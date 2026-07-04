"""cup.seismic.geometry: 规则地震工区几何与采样轴。

本模块封装 inline/xline 规则工区的坐标轴、XY 变换、工区 footprint、
道中心 XY 网格和采样窗口索引。它不读取 SEG-Y/ZGY 文件，也不处理地震道数据。

边界说明
--------
- ``LineAxis.step`` 是线号步长，不是米制距离。
- 所有米制距离都通过 ``SurveyLineGeometry.line_to_coord`` 或 XY 网格计算。
- 文件格式差异由 ``cup.seismic.survey`` 的 Adapter 处理。

核心公开对象
------------
1. LineAxis: 规则线号轴，负责线号/index 互转和最近线号吸附。
2. SampleAxis: 时间或深度采样轴，负责采样窗口裁剪。
3. SurveyLineGeometry: 规则工区几何，负责 XY 与 inline/xline 互转。
4. nominal_bin_spacing_m / xy_distance_grid / xy_circle_mask: 纯 XY 网格距离计算。
5. resolve_well_line_position: 从井对象解析 inline/xline。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class LineAxis:
    """规则 inline 或 xline 线号轴。"""

    minimum: float
    step: float
    count: int
    name: str = "line"

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError(f"{self.name} axis count must be positive, got {self.count}.")
        if self.count > 1 and self.step == 0.0:
            raise ValueError(f"{self.name} axis step must be non-zero when count > 1.")

    @property
    def maximum(self) -> float:
        """返回轴最大线号。"""
        return float(self.minimum + (self.count - 1) * self.step)

    def values(self) -> np.ndarray:
        """返回完整线号轴。"""
        return self.minimum + np.arange(self.count, dtype=np.float64) * self.step

    def index_of_line(self, line_no: float) -> float:
        """将线号转换为浮点 index，并检查是否位于轴范围内。"""
        if self.count == 1:
            if not np.isclose(float(line_no), self.minimum):
                raise ValueError(f"{self.name} is outside survey range: {line_no}")
            return 0.0
        index = (float(line_no) - self.minimum) / self.step
        if not (0.0 <= index <= self.count - 1):
            raise ValueError(f"{self.name} is outside survey range: {line_no}")
        return float(index)

    def line_at_index(self, index: float) -> float:
        """将浮点 index 转换为线号，并检查是否位于轴范围内。"""
        index_float = float(index)
        if not (0.0 <= index_float <= self.count - 1):
            raise ValueError(f"{self.name} index is outside survey range: {index}")
        return float(self.minimum + index_float * self.step)

    def snap(self, line_float: float) -> float:
        """将浮点线号吸附到规则线号轴上的最近有效线号。"""
        if self.count == 1 or self.step == 0.0:
            return float(self.minimum)
        line_index = round((float(line_float) - self.minimum) / self.step)
        line_index = min(self.count - 1, max(0, line_index))
        return float(self.minimum + line_index * self.step)

    def describe(self, prefix: str) -> dict[str, Any]:
        """按历史几何字典字段描述轴。"""
        return {
            f"n_{prefix}": int(self.count),
            f"{prefix}_min": float(self.minimum),
            f"{prefix}_max": float(self.maximum),
            f"{prefix}_step": float(self.step),
        }


@dataclass(frozen=True)
class SampleAxis:
    """时间或深度采样轴。"""

    values: np.ndarray
    domain: str
    unit: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if (
            values.ndim != 1
            or values.size == 0
            or np.any(~np.isfinite(values))
            or (values.size > 1 and np.any(np.diff(values) <= 0.0))
        ):
            raise ValueError(
                "SampleAxis values must be a finite, strictly increasing 1D array."
            )
        expected_unit = "s" if self.domain == "time" else "m" if self.domain == "depth" else None
        if expected_unit is None or self.unit != expected_unit:
            raise ValueError(
                f"Unsupported SampleAxis domain/unit: {self.domain!r}/{self.unit!r}."
            )
        object.__setattr__(self, "values", values)

    @property
    def step(self) -> float:
        """返回采样步长；单样点轴返回 0。"""
        if self.values.size == 1:
            return 0.0
        return float(self.values[1] - self.values[0])

    def describe(self) -> dict[str, Any]:
        """按历史几何字典字段描述采样轴。"""
        return {
            "n_sample": int(self.values.size),
            "sample_min": float(self.values[0]),
            "sample_max": float(self.values[-1]),
            "sample_step": float(self.step),
            "sample_domain": self.domain,
            "sample_unit": self.unit,
        }

    def window_indices(self, start: float | None, end: float | None) -> tuple[int, int]:
        """将采样窗口映射为半开索引区间。"""
        sample_start = float(self.values[0]) if start is None else float(start)
        sample_end = float(self.values[-1]) if end is None else float(end)
        if sample_start >= sample_end:
            raise ValueError(f"Invalid window: start={sample_start}, end={sample_end}")

        idx_start = int(np.searchsorted(self.values, sample_start, side="left"))
        idx_end = int(np.searchsorted(self.values, sample_end, side="right"))
        idx_start = max(0, idx_start)
        idx_end = min(self.values.size, idx_end)
        if idx_start >= idx_end:
            raise ValueError("Selected sample window is empty.")
        return idx_start, idx_end


def nearest_sample_indices(sample_axis: np.ndarray, sample_values: np.ndarray) -> np.ndarray:
    """Map physical sample coordinates to nearest indices on one explicit axis."""
    axis = np.asarray(sample_axis, dtype=np.float64).reshape(-1)
    values = np.asarray(sample_values, dtype=np.float64).reshape(-1)
    if axis.size == 0 or np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
        raise ValueError("sample_axis must be a finite, strictly increasing 1D array.")
    if np.any(~np.isfinite(values)):
        raise ValueError("sample_values must contain only finite coordinates.")

    right = np.searchsorted(axis, values, side="left")
    right = np.clip(right, 0, axis.size - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(axis[left] - values) < np.abs(axis[right] - values)
    return np.where(choose_left, left, right).astype(np.int64)


def validate_sample_indices(
    sample_axis: np.ndarray,
    sample_values: np.ndarray,
    sample_indices: np.ndarray,
    *,
    field_name: str,
    max_index_delta: int = 1,
) -> np.ndarray:
    """Validate a derived index column against its physical sample coordinates."""
    expected = nearest_sample_indices(sample_axis, sample_values)
    actual_float = np.asarray(sample_indices, dtype=np.float64).reshape(-1)
    if actual_float.size != expected.size:
        raise ValueError(
            f"{field_name} size {actual_float.size} does not match sample coordinate size {expected.size}."
        )
    if np.any(~np.isfinite(actual_float)) or np.any(np.abs(actual_float - np.round(actual_float)) > 1e-6):
        raise ValueError(f"{field_name} must contain only finite integer values.")

    actual = np.round(actual_float).astype(np.int64)
    if np.any((actual < 0) | (actual >= np.asarray(sample_axis).size)):
        raise ValueError(f"{field_name} contains indices outside the current sample axis.")
    mismatch = np.abs(actual - expected) > int(max_index_delta)
    if np.any(mismatch):
        examples = [
            {
                "sample_value": float(value),
                "csv_index": int(csv_index),
                "expected_index": int(expected_index),
            }
            for value, csv_index, expected_index in zip(
                np.asarray(sample_values, dtype=np.float64).reshape(-1)[mismatch][:5],
                actual[mismatch][:5],
                expected[mismatch][:5],
            )
        ]
        raise ValueError(
            f"{field_name} does not match the current sample axis for {int(np.count_nonzero(mismatch))} rows. "
            f"Examples: {examples}"
        )
    return expected


def _point_segment_distance_m(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom == 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, segment) / denom)
    t = min(1.0, max(0.0, t))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))


def distance_to_footprint_boundary_m(point_xy: tuple[float, float], footprint_xy: np.ndarray) -> float:
    """计算 XY 点到四边形 footprint 边界的最近距离。"""
    footprint = np.asarray(footprint_xy, dtype=np.float64)
    if footprint.shape != (4, 2):
        raise ValueError(f"Expected footprint shape (4, 2), got {footprint.shape}.")
    point = np.asarray(point_xy, dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"Invalid XY point: {point_xy}")
    return min(
        _point_segment_distance_m(point, footprint[index], footprint[(index + 1) % footprint.shape[0]])
        for index in range(footprint.shape[0])
    )


def nominal_bin_spacing_m(x_grid: np.ndarray, y_grid: np.ndarray) -> float:
    """返回稳健的名义道中心米制间距。"""
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
    """计算每个道中心到指定中心点的 XY 欧氏距离。"""
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
    """基于真实 XY 圆形半径返回 ``(mask, distance_m)``。"""
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


@dataclass(frozen=True)
class SurveyLineGeometry:
    """规则地震工区的 inline/xline 与 XY 变换。"""

    inline_axis: LineAxis
    xline_axis: LineAxis
    x0: float
    y0: float
    dx_inline: float
    dy_inline: float
    dx_xline: float
    dy_xline: float

    def describe(self, sample_axis: SampleAxis | None = None) -> dict[str, Any]:
        """返回历史几何字典格式，必要时附加采样轴字段。"""
        geometry: dict[str, Any] = {}
        geometry.update(self.inline_axis.describe("il"))
        geometry.update(
            {
                "inline_min": geometry.pop("il_min"),
                "inline_max": geometry.pop("il_max"),
                "inline_step": geometry.pop("il_step"),
            }
        )
        geometry.update(self.xline_axis.describe("xl"))
        geometry.update(
            {
                "xline_min": geometry.pop("xl_min"),
                "xline_max": geometry.pop("xl_max"),
                "xline_step": geometry.pop("xl_step"),
            }
        )
        if sample_axis is not None:
            geometry.update(sample_axis.describe())
        return geometry

    def coord_to_index(self, x: float, y: float) -> tuple[float, float]:
        """将 XY 坐标转换为浮点 inline/xline index。"""
        det = self.dx_inline * self.dy_xline - self.dy_inline * self.dx_xline
        if abs(det) < 1e-10:
            raise ValueError("Coordinate system is degenerate (determinant is zero).")

        dx = float(x) - self.x0
        dy = float(y) - self.y0
        i = (dx * self.dy_xline - dy * self.dx_xline) / det
        j = (dy * self.dx_inline - dx * self.dy_inline) / det

        i_max = float(self.inline_axis.count - 1)
        j_max = float(self.xline_axis.count - 1)
        tolerance = 64.0 * np.finfo(np.float64).eps * max(abs(i), abs(j), i_max, j_max, 1.0)
        if -tolerance <= i <= i_max + tolerance:
            i = min(i_max, max(0.0, i))
        if -tolerance <= j <= j_max + tolerance:
            j = min(j_max, max(0.0, j))

        if not (0.0 <= i <= i_max):
            raise ValueError(f"Point is outside survey inline range: {i}")
        if not (0.0 <= j <= j_max):
            raise ValueError(f"Point is outside survey xline range: {j}")
        return float(i), float(j)

    def coord_to_line(self, x: float, y: float) -> tuple[float, float]:
        """将 XY 坐标转换为浮点 inline/xline 线号。"""
        i, j = self.coord_to_index(x, y)
        return self.inline_axis.line_at_index(i), self.xline_axis.line_at_index(j)

    def line_to_index(self, inline_no: float, xline_no: float) -> tuple[float, float]:
        """将 inline/xline 线号转换为浮点 index。"""
        return self.inline_axis.index_of_line(inline_no), self.xline_axis.index_of_line(xline_no)

    def line_to_coord(self, inline_no: float, xline_no: float) -> tuple[float, float]:
        """将 inline/xline 线号转换为 XY 坐标。"""
        i, j = self.line_to_index(inline_no, xline_no)
        x = self.x0 + i * self.dx_inline + j * self.dx_xline
        y = self.y0 + i * self.dy_inline + j * self.dy_xline
        return float(x), float(y)

    def snap_inline(self, inline_float: float) -> float:
        """将浮点 inline 线号吸附到最近有效线号。"""
        return self.inline_axis.snap(inline_float)

    def snap_xline(self, xline_float: float) -> float:
        """将浮点 xline 线号吸附到最近有效线号。"""
        return self.xline_axis.snap(xline_float)

    def footprint_xy(self) -> np.ndarray:
        """返回工区四角 XY footprint。"""
        il_min = self.inline_axis.minimum
        il_max = self.inline_axis.maximum
        xl_min = self.xline_axis.minimum
        xl_max = self.xline_axis.maximum
        return np.asarray(
            [
                self.line_to_coord(il_min, xl_min),
                self.line_to_coord(il_max, xl_min),
                self.line_to_coord(il_max, xl_max),
                self.line_to_coord(il_min, xl_max),
            ],
            dtype=np.float64,
        )

    def distance_to_footprint_m(self, x: float, y: float) -> float:
        """返回 XY 点到工区 footprint 边界的最近距离。"""
        return distance_to_footprint_boundary_m((float(x), float(y)), self.footprint_xy())

    def trace_xy_grids(self, ilines: np.ndarray, xlines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """根据线号轴构建道中心 XY 网格。"""
        ilines = np.asarray(ilines, dtype=np.float64)
        xlines = np.asarray(xlines, dtype=np.float64)
        if ilines.ndim != 1 or xlines.ndim != 1 or ilines.size == 0 or xlines.size == 0:
            raise ValueError("ilines and xlines must be non-empty 1D arrays.")

        inline_indices = np.asarray([self.inline_axis.index_of_line(float(il)) for il in ilines], dtype=np.float64)
        xline_indices = np.asarray([self.xline_axis.index_of_line(float(xl)) for xl in xlines], dtype=np.float64)
        i_grid = inline_indices[:, None]
        j_grid = xline_indices[None, :]
        x_grid = self.x0 + i_grid * self.dx_inline + j_grid * self.dx_xline
        y_grid = self.y0 + i_grid * self.dy_inline + j_grid * self.dy_xline
        return x_grid.astype(np.float64), y_grid.astype(np.float64)

    def bin_spacing_m(self) -> dict[str, float]:
        """返回 inline/xline 相邻道中心米制间距。"""
        spacings: list[float] = []
        inline_spacing_m = 0.0
        xline_spacing_m = 0.0
        if self.inline_axis.count > 1:
            inline_spacing_m = float(np.hypot(self.dx_inline, self.dy_inline))
            if np.isfinite(inline_spacing_m) and inline_spacing_m > 0.0:
                spacings.append(inline_spacing_m)
        if self.xline_axis.count > 1:
            xline_spacing_m = float(np.hypot(self.dx_xline, self.dy_xline))
            if np.isfinite(xline_spacing_m) and xline_spacing_m > 0.0:
                spacings.append(xline_spacing_m)

        nominal = float(np.median(np.asarray(spacings, dtype=np.float64))) if spacings else 0.0
        return {
            "inline_spacing_m": inline_spacing_m,
            "xline_spacing_m": xline_spacing_m,
            "nominal_bin_spacing_m": nominal,
        }


class SurveyLineGeometryLike(Protocol):
    """解析井位线号所需的最小几何接口。"""

    def coord_to_line(self, x: float, y: float) -> tuple[float, float]: ...


class WellPositionLike(Protocol):
    """解析井位线号所需的最小井对象接口。"""

    well_name: str
    inline: float | None
    xline: float | None
    x: float | None
    y: float | None


def resolve_well_line_position(
    well: WellPositionLike,
    geometry: SurveyLineGeometryLike | None,
) -> tuple[float, float]:
    """解析井位 inline/xline，优先使用已有线号，其次用 XY 通过工区几何转换。"""
    if well.inline is not None and well.xline is not None:
        inline = float(well.inline)
        xline = float(well.xline)
        if not np.isfinite(inline) or not np.isfinite(xline):
            raise ValueError(f"well '{well.well_name}' must provide finite inline/xline coordinates.")
        return inline, xline

    if well.x is not None and well.y is not None:
        if geometry is None:
            raise ValueError(
                f"well '{well.well_name}' provides x/y but no survey geometry was supplied for coord_to_line."
            )
        inline, xline = geometry.coord_to_line(float(well.x), float(well.y))
        return float(inline), float(xline)

    raise ValueError(
        f"well '{well.well_name}' must provide either inline/xline or x/y coordinates for location resolution."
    )
