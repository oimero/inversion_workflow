"""cup.seismic.lfm_time: 时间域层位约束低频模型构建。

本模块提供时间域控制点低频模型构建能力。控制点必须已经携带
inline/xline、TWT、层段名与层段内比例坐标；本模块只做时间域约束校验、
控制点适配和结果体封装。

边界说明
--------
- 本模块不负责层位解释文件读取、测井提取或地震体加载。
- 本模块仅支持 ``TargetZone.geometry['sample_domain'] == 'time'`` 的场景。
- 克里金建模由 ``cup.seismic.modeling`` 的点级控制入口负责。

核心公开对象
------------
1. LfmTimeControlPoint: 时间域点级低频模型控制样本。
2. LfmTimePointModelResult: 低频模型体、方差体与覆盖统计的结果封装。
3. lowpass_twt_log: 对 TWT 域单条曲线执行低通滤波。
4. build_lfm_time_model_from_points: 基于控制点构建时间域低频模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from cup.seismic.modeling import (
    LayerControlPoint,
    build_point_constrained_model,
)
from cup.seismic.target_zone import TargetZone
from wtie.processing import grid
from wtie.processing.spectral import apply_butter_lowpass_filter

DEFAULT_FILTER_BUFFER_CYCLES = 2.0
VALID_FILTER_BUFFER_MODES = {"reflect", "edge", "none"}


@dataclass(frozen=True)
class LfmTimeControlPoint:
    """时间域点级低频模型控制样本。

    一个对象表示目标层段中的一个物理样本。斜井沿 TWT 改变 inline/xline
    时，应拆成多条控制点，而不是压缩成单个井位控制。
    """

    well_name: str
    route: str
    twt_s: float
    md_m: float
    x_m: float
    y_m: float
    inline_float: float
    xline_float: float
    zone_name: str
    u_in_zone: float
    ai: float
    weight: float = 1.0
    source: str = ""
    flat_idx: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LfmTimePointModelResult:
    """时间域点级控制 LFM 构建结果。"""

    volume: np.ndarray
    variance_volume: np.ndarray
    geometry: Dict[str, Any]
    ilines: np.ndarray
    xlines: np.ndarray
    samples: np.ndarray
    metadata: Dict[str, Any]
    control_points: list[LfmTimeControlPoint]
    coverage_stats: Dict[str, Any]


def _finite_time_control_point(point: LfmTimeControlPoint) -> bool:
    values = [
        point.twt_s,
        point.inline_float,
        point.xline_float,
        point.u_in_zone,
        point.ai,
        point.weight,
    ]
    return bool(np.all(np.isfinite(np.asarray(values, dtype=float))) and point.weight > 0.0)


def _time_control_point_to_layer_control(point: LfmTimeControlPoint) -> LayerControlPoint:
    metadata = {} if point.metadata is None else dict(point.metadata)
    metadata.update(
        {
            "route": str(point.route),
            "twt_s": float(point.twt_s),
            "md_m": float(point.md_m),
            "x_m": float(point.x_m),
            "y_m": float(point.y_m),
            "source": str(point.source),
            "flat_idx": None if point.flat_idx is None else int(point.flat_idx),
        }
    )
    return LayerControlPoint(
        control_id=f"{point.well_name}:{point.zone_name}:{point.twt_s:.9g}",
        property_name="AI",
        inline=float(point.inline_float),
        xline=float(point.xline_float),
        zone_name=str(point.zone_name),
        u_in_zone=float(point.u_in_zone),
        value=float(point.ai),
        weight=float(point.weight),
        group_name=str(point.well_name),
        metadata=metadata,
    )


def lowpass_twt_log(
    log: grid.Log,
    cutoff_hz: float = 10.0,
    order: int = 6,
    buffer_seconds: Optional[float] = None,
    buffer_mode: str = "reflect",
) -> grid.Log:
    """对 TWT 域单条曲线执行 Butterworth 低通滤波。

    Parameters
    ----------
    log : grid.Log
        输入曲线，必须位于 TWT 域且采样间隔为常数。
    cutoff_hz : float, default=10.0
        低通截止频率，单位 Hz。
    order : int, default=6
        滤波器阶数。启用零相位滤波时，内部会按当前实现调用双向滤波。
    buffer_seconds : float, optional
        滤波前在曲线两端附加的缓冲时长，单位 s。若未提供，则按
        ``DEFAULT_FILTER_BUFFER_CYCLES / cutoff_hz`` 自动估算，并限制在
        曲线长度允许的范围内。
    buffer_mode : {"reflect", "edge", "none"}, default="reflect"
        生成缓冲样本的方式。``reflect`` 适合减弱端点突变带来的边界效应，
        ``edge`` 则使用端点常值外延。``none`` 表示不做外部边界延拓。

    Returns
    -------
    grid.Log
        与输入曲线共享 basis、名称与单位信息的滤波后新曲线。

    Raises
    ------
    ValueError
        当输入曲线不在 TWT 域，或 ``cutoff_hz`` 不在有效 Nyquist 范围内时抛出。
    """
    if not log.is_twt:
        raise ValueError("lowpass_twt_log only supports TWT-domain logs.")
    if buffer_mode not in VALID_FILTER_BUFFER_MODES:
        raise ValueError(f"buffer_mode must be one of {sorted(VALID_FILTER_BUFFER_MODES)}, got {buffer_mode!r}.")

    fs = 1.0 / log.sampling_rate
    nyquist = 0.5 * fs
    if cutoff_hz <= 0.0 or cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz must be within (0, {nyquist}), got {cutoff_hz}.")

    values = log.values.astype(np.float64)
    if values.size <= 1:
        return grid.Log(
            values.copy(),
            log.basis.copy(),
            "twt",
            name=log.name,
            unit=getattr(log, "unit", None),
        )

    pad_samples = 0
    values_to_filter = values
    if buffer_mode != "none":
        dt = float(log.sampling_rate)
        resolved_buffer_seconds = (
            max(DEFAULT_FILTER_BUFFER_CYCLES / cutoff_hz, 3.0 * order * dt)
            if buffer_seconds is None
            else float(buffer_seconds)
        )
        if resolved_buffer_seconds < 0.0:
            raise ValueError(f"buffer_seconds must be non-negative when provided, got {buffer_seconds}.")

        pad_samples = int(np.ceil(resolved_buffer_seconds / dt))
        pad_samples = min(max(pad_samples, 0), values.size - 1)
        if pad_samples > 0:
            values_to_filter = np.pad(values, (pad_samples, pad_samples), mode=buffer_mode)  # type: ignore

    filtered = apply_butter_lowpass_filter(
        values_to_filter,
        highcut=cutoff_hz,
        fs=fs,
        order=order,
        zero_phase=True,
    )
    if pad_samples > 0:
        filtered = filtered[pad_samples : pad_samples + values.size]
    return grid.Log(
        filtered.astype(np.float64),
        log.basis.copy(),
        "twt",
        name=log.name,
        unit=log.unit,
        allow_nan=log.allow_nan,
    )


def build_lfm_time_model_from_points(
    target_layer: TargetZone,
    control_points: list[LfmTimeControlPoint],
    *,
    boundary_extension_samples: int = 50,
    n_slices: int = 32,
    variogram: str = "spherical",
    exact: bool = True,
    nugget: float = 0.0,
) -> LfmTimePointModelResult:
    """基于时间域点级控制构建 AI 低频模型。

    每个输入点都携带自己的 inline/xline/TWT 和 ``u_in_zone``。克里金、
    比例切片和边界外延由 ``cup.seismic.modeling`` 统一完成。
    """
    if not control_points:
        raise ValueError("control_points must contain at least one LfmTimeControlPoint.")
    if n_slices < 2:
        raise ValueError(f"n_slices must be >= 2, got {n_slices}.")
    if boundary_extension_samples < 0:
        raise ValueError(f"boundary_extension_samples must be >= 0, got {boundary_extension_samples}.")

    sample_domain = str(target_layer.geometry.get("sample_domain", "")).lower()
    sample_unit = str(target_layer.geometry.get("sample_unit", "")).lower()
    if sample_domain != "time":
        raise ValueError("Point-control time LFM only supports TargetZone geometry in time domain.")
    if sample_unit not in {"s", "sec", "second", "seconds"}:
        raise ValueError("Point-control time LFM expects TargetZone sample_unit in seconds.")

    finite_points = [point for point in control_points if _finite_time_control_point(point)]
    if not finite_points:
        raise ValueError("control_points contain no finite positive-weight samples.")

    model_result = build_point_constrained_model(
        target_layer,
        [_time_control_point_to_layer_control(point) for point in finite_points],
        property_name="AI",
        boundary_extension_samples=int(boundary_extension_samples),
        n_slices=int(n_slices),
        variogram=variogram,
        exact=exact,
        nugget=nugget,
    )

    coverage_stats = dict(model_result.coverage_stats)
    coverage_stats["well_count"] = int(len({point.well_name for point in finite_points}))
    coverage_stats["wells"] = {}
    for well_name in sorted({point.well_name for point in finite_points}):
        well_points = [point for point in finite_points if point.well_name == well_name]
        flat_values = [point.flat_idx for point in well_points if point.flat_idx is not None]
        coverage_stats["wells"][well_name] = {
            "control_point_count": int(len(well_points)),
            "unique_trace_count": int(len(set(flat_values))) if flat_values else None,
            "source_modes": sorted({str(point.source) for point in well_points}),
        }

    metadata = {
        **model_result.metadata,
        "sample_domain": "time",
        "sample_unit": "s",
        "well_names": sorted({point.well_name for point in finite_points}),
    }
    return LfmTimePointModelResult(
        volume=model_result.volume,
        variance_volume=model_result.variance_volume,
        geometry=dict(model_result.geometry),
        ilines=model_result.ilines,
        xlines=model_result.xlines,
        samples=model_result.samples,
        metadata=metadata,
        control_points=list(finite_points),
        coverage_stats=coverage_stats,
    )


