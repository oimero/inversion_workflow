"""cup.well.wavelet: 小波加载与生成辅助工具。

本模块提供 Ricker 小波生成、CSV 波形加载与采样间隔校验等功能。

边界说明
--------
- 本模块不处理地震文件读取或可视化绘图。
- CSV 必须包含 ``time_s`` 与 ``amplitude`` 两列。

核心公开对象
------------
1. make_wavelet: 生成时间域 Ricker 小波。
2. load_wavelet_csv: 从 CSV 读取小波。
3. infer_wavelet_dt: 推断规则采样间隔。
4. compute_wavelet_active_half_support_s: 估计有效半支撑。
5. validate_wavelet_dt: 校验小波采样间隔。

Examples
--------
>>> from cup.well.wavelet import make_wavelet, infer_wavelet_dt
>>> time_s, amp = make_wavelet("ricker", freq=30.0, dt=0.001, length=128)
>>> _ = infer_wavelet_dt(time_s)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ACTIVE_SUPPORT_THRESHOLD = 0.05


def make_wavelet(
    wavelet_type: str,
    freq: float,
    dt: float,
    length: int,
    gain: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成时间域小波。

    Parameters
    ----------
    wavelet_type : str
        小波类型，仅支持 ``"ricker"``。
    freq : float
        主频，单位 Hz。
    dt : float
        采样间隔，单位 s。
    length : int
        采样点数。
    gain : float, default=1.0
        振幅增益。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(time_s, amplitude)``，单位为秒与振幅。

    Raises
    ------
    ValueError
        当输入参数非法或小波类型不支持时。
    """
    if wavelet_type != "ricker":
        raise ValueError(f"Unsupported wavelet_type: {wavelet_type}")
    if freq <= 0.0:
        raise ValueError(f"wavelet_freq must be positive, got {freq}.")
    if dt <= 0.0:
        raise ValueError(f"wavelet_dt must be positive, got {dt}.")
    if length < 2:
        raise ValueError(f"wavelet_length must be at least 2, got {length}.")

    from wtie.modeling.wavelet import ricker

    time_s, amplitude = ricker(freq, dt, length)
    return np.asarray(time_s, dtype=np.float64), (np.asarray(amplitude, dtype=np.float64) * float(gain))


def load_wavelet_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """从 CSV 读取时间域小波。

    Parameters
    ----------
    path : str or Path
        CSV 路径，需包含 ``time_s`` 与 ``amplitude`` 列。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(time_s, amplitude)``，时间轴严格递增。

    Raises
    ------
    ValueError
        当缺失必需列或样本不足/非法时。
    """
    path = Path(path)
    df = pd.read_csv(path)
    required = {"time_s", "amplitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"wavelet CSV is missing columns: {sorted(missing)}")

    time_s = df["time_s"].to_numpy(dtype=np.float64)
    amplitude = df["amplitude"].to_numpy(dtype=np.float64)
    finite = np.isfinite(time_s) & np.isfinite(amplitude)
    if np.count_nonzero(finite) < 2:
        raise ValueError(f"wavelet CSV does not contain enough finite samples: {path}")

    time_s = time_s[finite]
    amplitude = amplitude[finite]
    order = np.argsort(time_s)
    time_s = time_s[order]
    amplitude = amplitude[order]
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("wavelet time_s samples must be strictly increasing after sorting.")
    return time_s, amplitude


def infer_wavelet_dt(time_s: np.ndarray) -> float:
    """推断小波时间轴的规则采样间隔。

    Parameters
    ----------
    time_s : np.ndarray
        时间轴数组，单位 s。

    Returns
    -------
    float
        采样间隔，单位 s。

    Raises
    ------
    ValueError
        当时间轴不严格递增或非规则采样时。
    """
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if time_s.size < 2:
        raise ValueError("wavelet time_s must contain at least two samples.")
    deltas = np.diff(time_s)
    if np.any(deltas <= 0.0):
        raise ValueError("wavelet time_s samples must be strictly increasing.")

    dt = float(np.median(deltas))
    if not np.allclose(deltas, dt, rtol=1e-5, atol=1e-9):
        raise ValueError("wavelet time_s samples must be regularly sampled.")
    return dt


def compute_wavelet_active_half_support_s(
    wavelet_time_s: np.ndarray,
    wavelet: np.ndarray,
    *,
    active_threshold: float = DEFAULT_ACTIVE_SUPPORT_THRESHOLD,
) -> float:
    """估计小波有效半支撑（秒）。

    有效支撑定义为 ``abs(wavelet)`` 不小于峰值的
    ``active_threshold`` 倍；返回峰值处到有效支撑边界的最大时间偏移。

    Parameters
    ----------
    wavelet_time_s : np.ndarray
        时间轴数组，单位 s。
    wavelet : np.ndarray
        小波振幅数组。
    active_threshold : float, default=DEFAULT_ACTIVE_SUPPORT_THRESHOLD
        有效支撑阈值，相对于峰值的比例。

    Returns
    -------
    float
        有效半支撑时间，单位 s。

    Raises
    ------
    ValueError
        当输入为空、形状不一致或阈值非法时。
    """
    if not 0.0 < active_threshold <= 1.0:
        raise ValueError(f"active_threshold must be within (0, 1], got {active_threshold}.")

    wavelet_time_s = np.asarray(wavelet_time_s, dtype=np.float64).reshape(-1)
    wavelet = np.asarray(wavelet, dtype=np.float64).reshape(-1)
    if wavelet_time_s.shape != wavelet.shape:
        raise ValueError(f"wavelet_time_s shape {wavelet_time_s.shape} does not match wavelet shape {wavelet.shape}.")
    if wavelet.size == 0:
        raise ValueError("Cannot compute active half-support from an empty wavelet.")

    abs_wavelet = np.abs(wavelet)
    peak = float(abs_wavelet.max())
    if peak <= 0.0:
        raise ValueError("Cannot compute active half-support because wavelet peak amplitude is zero.")

    peak_index = int(abs_wavelet.argmax())
    active = abs_wavelet >= peak * float(active_threshold)
    return float(np.abs(wavelet_time_s[active] - wavelet_time_s[peak_index]).max())


def validate_wavelet_dt(time_s: np.ndarray, expected_dt_s: float) -> float:
    """校验小波采样间隔是否匹配预期 dt。

    Parameters
    ----------
    time_s : np.ndarray
        小波时间轴，单位 s。
    expected_dt_s : float
        预期采样间隔，单位 s。

    Returns
    -------
    float
        推断得到的采样间隔。

    Raises
    ------
    ValueError
        当预期 dt 非法或与小波 dt 不一致时。
    """
    expected_dt = float(expected_dt_s)
    if expected_dt <= 0.0:
        raise ValueError(f"expected_dt_s must be positive, got {expected_dt_s}.")

    wavelet_dt = infer_wavelet_dt(time_s)
    if not np.isclose(wavelet_dt, expected_dt, rtol=1e-5, atol=1e-9):
        raise ValueError(
            "precomputed wavelet dt does not match seismic sample interval: "
            f"wavelet_dt={wavelet_dt}, seismic_dt={expected_dt}."
        )
    return wavelet_dt
