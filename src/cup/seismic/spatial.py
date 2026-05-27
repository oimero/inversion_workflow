"""cup.seismic.spatial: 纯 XY 空间距离计算。

本模块只处理已经给定的 XY 数组，不读取地震体文件，也不执行
inline/xline 与 XY 的坐标转换。工区几何转换请使用 ``cup.seismic.geometry``。

边界说明
--------
- 输入的 ``x_grid`` / ``y_grid`` 必须已经是真实 XY 道中心坐标。
- 不能直接用 ``inline_step`` / ``xline_step`` 当米制距离。

核心公开对象
------------
1. nominal_bin_spacing_m: 从 XY 道中心网格估计名义米制道间距。
2. xy_distance_grid: 计算 XY 欧氏距离网格。
3. xy_circle_mask: 基于真实 XY 半径生成圆形掩码。
"""

from __future__ import annotations

import numpy as np


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
