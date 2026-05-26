"""cup.utils.raw_trace: 原始一维/二维道集预处理。

本模块提供道集级别的 z-score 标准化等基础处理。

边界说明
--------
- 本模块仅处理数值数组，不涉及 ``grid.BaseTrace`` 等高级对象。
- NaN 插值委托给 ``wtie.processing.logs.interpolate_nans``。

核心公开对象
------------
1. zscore_trace: 单道 z-score 标准化。
2. zscore_traces_axis: 按行独立 z-score 标准化。
"""

from __future__ import annotations

import numpy as np


def zscore_trace(values: np.ndarray, *, nan_method: str = "linear") -> np.ndarray:
    """插值缺失样点，并对完整一维道做 z-score 标准化。"""
    from wtie.processing.logs import interpolate_nans

    x = interpolate_nans(values, method=nan_method)
    mean = float(np.mean(x))
    scale = float(np.std(x))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Trace has non-positive standard deviation.")
    return (x - mean) / scale


def zscore_traces_axis(values: np.ndarray) -> np.ndarray:
    """对二维数组的每一行独立执行 z-score 标准化。

    NaN 位置不会参与均值和标准差计算。
    """
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    counts = finite.sum(axis=1, keepdims=True)
    if np.any(counts <= 0):
        raise ValueError("At least one trace has no finite samples.")
    means = np.where(finite, values, 0.0).sum(axis=1, keepdims=True) / counts
    centered = np.where(finite, values - means, 0.0)
    stds = np.sqrt((centered**2).sum(axis=1, keepdims=True) / counts)
    if np.any(~np.isfinite(stds) | (stds <= 0.0)):
        raise ValueError("At least one trace has non-positive standard deviation.")
    return centered / stds


# ── Centered moving window ──


def _window_radius(window: int) -> tuple[int, int]:
    w = int(window)
    if w < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    left = w // 2
    right = w - 1 - left
    return left, right


def centered_moving_sum(values: np.ndarray, window: int) -> np.ndarray:
    """一维居中滑动求和，边界使用零填充。"""
    values = np.asarray(values, dtype=float)
    left, right = _window_radius(window)
    padded = np.pad(values, (left, right), mode="constant", constant_values=0.0)
    cumsum = np.cumsum(np.insert(padded, 0, 0.0))
    return cumsum[window:] - cumsum[:-window]


def centered_moving_sum_axis(values: np.ndarray, window: int) -> np.ndarray:
    """沿 axis=1 的二维居中滑动求和，适用于批量一维道。"""
    values = np.asarray(values, dtype=np.float32)
    left, right = _window_radius(window)
    padded = np.pad(values, ((0, 0), (left, right)), mode="constant", constant_values=0.0)
    cumsum = np.cumsum(
        np.concatenate([np.zeros((values.shape[0], 1), dtype=np.float64), padded], axis=1),
        axis=1,
        dtype=np.float64,
    )
    return cumsum[:, window:] - cumsum[:, :-window]


def centered_moving_rms(values: np.ndarray, window: int) -> np.ndarray:
    """一维居中滑动 RMS，将 NaN 视为缺失值。"""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    numerator = centered_moving_sum(np.where(valid, values**2, 0.0), window)
    denominator = centered_moving_sum(valid.astype(float), window)
    out = np.full(values.shape, np.nan, dtype=float)
    positive = denominator > 0.0
    out[positive] = np.sqrt(numerator[positive] / denominator[positive])
    return out


def centered_moving_rms_axis(values: np.ndarray, window: int) -> np.ndarray:
    """沿 axis=1 的二维居中滑动 RMS，将 NaN 视为缺失值。"""
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    numerator = centered_moving_sum_axis(np.where(valid, values**2, 0.0), window)
    denominator = centered_moving_sum_axis(valid.astype(np.float32), window)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    positive = denominator > 0.0
    out[positive] = np.sqrt(numerator[positive] / denominator[positive]).astype(np.float32)
    return out


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """一维居中滑动平均，边界延拓并保持样点数不变。"""
    if window <= 1:
        return values.astype(np.float32, copy=False)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.full((window,), 1.0 / float(window), dtype=np.float32)
    smoothed = np.convolve(padded, kernel, mode="valid")
    if smoothed.size != values.size:
        raise RuntimeError(f"Moving average changed sample count: {values.size} -> {smoothed.size}.")
    if not np.all(np.isfinite(smoothed)):
        raise ValueError("Moving average produced non-finite values.")
    return smoothed.astype(np.float32, copy=False)


# ── Unit conversion ──


def meters_to_odd_samples(
    window_m: float,
    sample_step_m: float,
    *,
    min_samples: int = 3,
) -> int:
    """将米制窗口长度换算为最接近的奇数样点数。"""
    n = int(round(float(window_m) / float(sample_step_m)))
    n = max(n, int(min_samples))
    if n % 2 == 0:
        n += 1
    return n
