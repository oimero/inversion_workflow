"""cup.seismic.wavelet: 地震子波加载、生成与频谱辅助工具。

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
5. crop_wavelet_center_energy_normalize: 居中裁剪并做 L2 能量归一化。
6. validate_wavelet_dt: 校验小波采样间隔。
7. validate_wavelet_normalization: 校验子波中心、有限性和 L2 能量。
8. wavelet_l2_normalize / wavelet_roughness / wavelet_spectrum_features: 子波属性计算。
9. wavelet_half_amplitude_frequencies: 计算振幅谱峰值及左右半峰值频率。

Examples
--------
>>> from cup.seismic.wavelet import make_wavelet, infer_wavelet_dt
>>> time_s, amp = make_wavelet("ricker", freq=30.0, dt=0.001, length=128)
>>> _ = infer_wavelet_dt(time_s)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ACTIVE_SUPPORT_THRESHOLD = 0.05


@dataclass(frozen=True)
class WaveletNormalizationQC:
    """QC result for a workflow wavelet."""

    status: str
    l2_energy: float
    center_time_s: float
    center_index: int
    n_samples: int
    renormalized: bool = False
    reasons: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WaveletSpectrumFeatures:
    """Basic spectrum features for one regularly sampled wavelet."""

    dominant_frequency_hz: float
    spectral_centroid_hz: float
    bandwidth_hz: float
    low_frequency_hz: float
    high_frequency_hz: float
    side_lobe_ratio: float

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


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


def wavelet_l2_normalize(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a L2-normalized wavelet and its pre-normalization energy."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot normalize an empty wavelet.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Cannot normalize a wavelet with non-finite samples.")
    energy = float(np.sqrt(np.sum(values * values)))
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("Cannot normalize a zero-energy wavelet.")
    return values / energy, energy


def validate_wavelet_normalization(
    time_s: np.ndarray,
    amplitude: np.ndarray,
    *,
    expected_l2_energy: float = 1.0,
    l2_energy_tolerance: float = 1e-5,
    max_center_abs_time_s: float = 1e-9,
    allow_small_renormalization: bool = False,
) -> tuple[np.ndarray, WaveletNormalizationQC]:
    """Validate workflow wavelet centering, finite values, and L2 energy."""
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    amplitude = np.asarray(amplitude, dtype=np.float64).reshape(-1)
    if time_s.shape != amplitude.shape:
        raise ValueError(f"time_s shape {time_s.shape} does not match amplitude shape {amplitude.shape}.")
    if time_s.size == 0:
        raise ValueError("wavelet must contain at least one sample.")

    reasons: list[str] = []
    if not np.all(np.isfinite(time_s)) or not np.all(np.isfinite(amplitude)):
        reasons.append("non_finite_samples")
    if time_s.size > 1 and np.any(np.diff(time_s) <= 0.0):
        reasons.append("time_not_strictly_increasing")

    center_index = int(np.argmin(np.abs(time_s)))
    center_time = float(time_s[center_index])
    if abs(center_time) > float(max_center_abs_time_s):
        reasons.append("center_not_zero")

    l2_energy = float(np.sqrt(np.sum(amplitude * amplitude))) if np.all(np.isfinite(amplitude)) else float("nan")
    expected = float(expected_l2_energy)
    tolerance = float(l2_energy_tolerance)
    output = amplitude.copy()
    renormalized = False
    if not np.isfinite(l2_energy) or l2_energy <= 0.0:
        reasons.append("zero_or_invalid_l2_energy")
    else:
        energy_delta = abs(l2_energy - expected)
        if energy_delta > tolerance:
            reasons.append("l2_energy_out_of_tolerance")
        elif energy_delta > 0.0 and allow_small_renormalization:
            output, l2_energy = wavelet_l2_normalize(output)
            renormalized = True

    return output, WaveletNormalizationQC(
        status="ok" if not reasons else "failed",
        l2_energy=l2_energy,
        center_time_s=center_time,
        center_index=center_index,
        n_samples=int(time_s.size),
        renormalized=renormalized,
        reasons=";".join(dict.fromkeys(reasons)),
    )


def wavelet_roughness(values: np.ndarray) -> float:
    """Second-difference roughness normalized by wavelet energy."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 3:
        return 0.0
    if not np.all(np.isfinite(values)):
        return float("nan")
    denom = float(np.sum(values * values))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")
    second = np.diff(values, n=2)
    return float(np.sum(second * second) / denom)


def wavelet_spectrum_features(
    time_s: np.ndarray,
    amplitude: np.ndarray,
    *,
    energy_percentile: float = 0.95,
) -> WaveletSpectrumFeatures:
    """Compute simple amplitude-spectrum features for a regularly sampled wavelet."""
    if not 0.0 < float(energy_percentile) <= 1.0:
        raise ValueError(f"energy_percentile must be within (0, 1], got {energy_percentile}.")
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    amplitude = np.asarray(amplitude, dtype=np.float64).reshape(-1)
    if time_s.shape != amplitude.shape:
        raise ValueError(f"time_s shape {time_s.shape} does not match amplitude shape {amplitude.shape}.")
    if amplitude.size < 2:
        raise ValueError("wavelet must contain at least two samples for spectrum features.")
    if not np.all(np.isfinite(amplitude)):
        raise ValueError("wavelet amplitude contains non-finite samples.")

    dt = infer_wavelet_dt(time_s)
    frequencies = np.fft.rfftfreq(amplitude.size, d=dt)
    spectrum = np.abs(np.fft.rfft(amplitude))
    power = spectrum * spectrum
    total_power = float(np.sum(power))
    if total_power <= 0.0 or not np.isfinite(total_power):
        raise ValueError("Cannot compute spectrum features for a zero-energy wavelet.")

    dominant_index = int(np.argmax(spectrum))
    centroid = float(np.sum(frequencies * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * power) / total_power))
    cumulative = np.cumsum(power) / total_power
    tail = (1.0 - float(energy_percentile)) / 2.0
    low = float(frequencies[int(np.searchsorted(cumulative, tail, side="left"))])
    high_index = int(np.searchsorted(cumulative, 1.0 - tail, side="left"))
    high_index = min(high_index, frequencies.size - 1)

    abs_amp = np.abs(amplitude)
    peak_index = int(np.argmax(abs_amp))
    peak = float(abs_amp[peak_index])
    left_lobe = float(np.max(abs_amp[:peak_index])) if peak_index > 0 else 0.0
    right_lobe = float(np.max(abs_amp[peak_index + 1 :])) if peak_index + 1 < abs_amp.size else 0.0
    side_lobe_ratio = float(max(left_lobe, right_lobe) / peak) if peak > 0.0 else float("nan")

    return WaveletSpectrumFeatures(
        dominant_frequency_hz=float(frequencies[dominant_index]),
        spectral_centroid_hz=centroid,
        bandwidth_hz=bandwidth,
        low_frequency_hz=low,
        high_frequency_hz=float(frequencies[high_index]),
        side_lobe_ratio=side_lobe_ratio,
    )


def wavelet_half_amplitude_frequencies(
    time_s: np.ndarray,
    amplitude: np.ndarray,
) -> tuple[float, float, float]:
    """Return peak, left, and right normalized-amplitude 0.5 frequencies."""
    time = np.asarray(time_s, dtype=np.float64).reshape(-1)
    values = np.asarray(amplitude, dtype=np.float64).reshape(-1)
    if time.shape != values.shape or time.size < 4:
        raise ValueError("Wavelet time and amplitude must have matching length >= 4.")
    diffs = np.diff(time)
    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(values)) or np.any(diffs <= 0.0):
        raise ValueError("Wavelet time must be finite and strictly increasing; amplitude must be finite.")
    dt_s = float(np.median(diffs))
    if not np.allclose(diffs, dt_s, rtol=1e-4, atol=1e-9):
        raise ValueError("Wavelet time axis must be regularly sampled.")
    n_fft = max(4096, 1 << int(np.ceil(np.log2(values.size * 16))))
    frequency = np.fft.rfftfreq(n_fft, d=dt_s)
    spectrum = np.abs(np.fft.rfft(values, n=n_fft))
    peak_index = int(np.argmax(spectrum))
    peak = float(spectrum[peak_index])
    if peak <= 0.0:
        raise ValueError("Wavelet spectrum has zero peak amplitude.")
    normalized = spectrum / peak

    def crossing(indices: range) -> float:
        for index in indices:
            a = float(normalized[index])
            b = float(normalized[index + 1])
            if (a - 0.5) * (b - 0.5) <= 0.0 and a != b:
                fraction = (0.5 - a) / (b - a)
                return float(frequency[index] + fraction * (frequency[index + 1] - frequency[index]))
        raise ValueError("Wavelet spectrum has no amplitude-0.5 crossing around its peak.")

    left_candidates = [
        index
        for index in range(0, peak_index)
        if (float(normalized[index]) - 0.5) * (float(normalized[index + 1]) - 0.5) <= 0.0
        and float(normalized[index]) != float(normalized[index + 1])
    ]
    if not left_candidates:
        raise ValueError("Wavelet spectrum has no left amplitude-0.5 crossing.")
    left = crossing(range(left_candidates[-1], left_candidates[-1] + 1))
    right = crossing(range(peak_index, frequency.size - 1))
    return float(frequency[peak_index]), left, right


def crop_wavelet_center_energy_normalize(
    wavelet: Any,
    target_ms: float,
) -> tuple[Any, dict[str, Any]]:
    """将小波居中裁剪到目标长度并做 L2 能量归一化。

    Parameters
    ----------
    wavelet : Any
        输入小波对象（``wtie.processing.grid.Wavelet``）。
    target_ms : float
        目标裁剪长度，单位 ms。

    Returns
    -------
    tuple[Any, dict[str, Any]]
        ``(cropped_wavelet, crop_info)``，``cropped_wavelet`` 为裁剪并归一化后的小波。
    """
    from wtie.processing.grid import Wavelet

    dt = float(wavelet.sampling_rate)
    target_s = float(target_ms) / 1000.0
    n_target = int(round(target_s / dt))
    if n_target % 2 == 0:
        n_target += 1
    n_target = min(n_target, int(wavelet.size))
    if n_target % 2 == 0:
        n_target -= 1
    center_idx = int(np.argmin(np.abs(wavelet.basis)))
    half = n_target // 2
    start = max(0, center_idx - half)
    end = start + n_target
    if end > int(wavelet.size):
        end = int(wavelet.size)
        start = end - n_target
    values = np.asarray(wavelet.values[start:end], dtype=np.float64).copy()
    basis = np.asarray(wavelet.basis[start:end], dtype=np.float64).copy()
    normalized, energy = wavelet_l2_normalize(values)
    cropped = Wavelet(normalized, basis, name="Auto well tie wavelet cropped energy-normalized")
    return cropped, {
        "target_ms": float(target_ms),
        "dt_s": dt,
        "original_samples": int(wavelet.size),
        "cropped_samples": int(cropped.size),
        "pre_normalization_l2_energy": energy,
    }


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
