"""cup.seismic.lfm_time: 时间域层位约束低频模型构建。

本模块提供基于层位解释、井曲线与时深关系的时间域低频模型构建能力，
包括井曲线预处理、按层段比例切片采样、二维切片插值以及结果体封装。

边界说明
--------
- 本模块不负责层位解释文件读取、测井提取或地震体加载。
- 本模块仅支持 ``TargetLayer.geometry['sample_domain'] == 'time'`` 的场景。
- 井曲线若为 MD 域，可借助时深表与井轨迹转换到 TWT 域；TVDSS 域曲线当前不支持直接输入。

核心公开对象
------------
1. LfmTimeWell: 单井低频模型输入描述。
2. LfmTimeModelResult: 低频模型体、方差体与覆盖统计的结果封装。
3. lowpass_twt_log: 对 TWT 域单条曲线执行低通滤波。
4. build_lfm_time_model: 基于层位约束与井点控制构建时间域低频模型。

Examples
--------
>>> import numpy as np
>>> from cup.seismic.lfm_time import LfmTimeWell, lowpass_twt_log
>>> from wtie.processing import grid
>>> twt_log = grid.Log(np.array([10., 20., 30.]), np.array([1.0, 1.5, 2.0]), "twt", name="AI")
>>> filtered = lowpass_twt_log(twt_log, cutoff_hz=0.4, order=6)
>>> filtered.is_twt
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from cup.seismic.modeling import WellControl, build_layer_constrained_model
from cup.seismic.survey import SurveyContext
from cup.seismic.target_layer import TargetLayer
from wtie.processing import grid
from wtie.processing.spectral import apply_butter_lowpass_filter

DEFAULT_FILTER_BUFFER_CYCLES = 2.0
VALID_FILTER_BUFFER_MODES = {"reflect", "edge", "none"}


@dataclass
class LfmTimeWell:
    """单井时间域低频模型输入。

    Parameters
    ----------
    well_name : str
        井名或井标识。
    property_name : str
        当前井参与建模的属性名称，例如 ``"AI"``、``"Vp"``。
    property_log : grid.Log
        属性曲线，允许为 TWT 域或 MD 域。
    time_depth_table : grid.TimeDepthTable
        当前井对应的时深关系表，支持 MD 域或 TVDSS 域。
    inline, xline : float, optional
        井位所在的 inline/xline 坐标。若提供，则优先使用。
    x, y : float, optional
        井口或目标点的平面坐标。当未提供 ``inline``/``xline`` 时，可结合
        ``survey`` 上下文转换到道号坐标。
    trajectory : grid.WellPath, optional
        当属性曲线为 MD 域且时深表为 TVDSS 域时，用于辅助坐标域转换的井轨迹。
    metadata : dict, optional
        附加元信息，会在结果中原样保留一份副本。
    """

    well_name: str
    property_name: str
    property_log: grid.Log
    time_depth_table: grid.TimeDepthTable
    inline: Optional[float] = None
    xline: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    trajectory: Optional[grid.WellPath] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LfmTimeModelResult:
    """时间域低频模型构建结果。

    Attributes
    ----------
    volume : np.ndarray
        低频模型体，shape 为 ``(n_inline, n_xline, n_sample)``。
    variance_volume : np.ndarray
        与 ``volume`` 同 shape 的切片插值方差体。
    geometry : Dict[str, Any]
        参与建模的几何描述，通常直接来自 ``TargetLayer.geometry``。
    ilines, xlines, samples : np.ndarray
        模型体三个维度对应的规则坐标轴。
    metadata : Dict[str, Any]
        建模参数与处理流程摘要。
    wells : list[LfmTimeWell]
        进入建模流程后的井输入副本；其中曲线会以处理后的 TWT 域版本返回。
    coverage_stats : Dict[str, Any]
        井级与层段级覆盖统计，例如切片控制点数量、切片插值模式等。
    """

    volume: np.ndarray
    variance_volume: np.ndarray
    geometry: Dict[str, Any]
    ilines: np.ndarray
    xlines: np.ndarray
    samples: np.ndarray
    metadata: Dict[str, Any]
    wells: list[LfmTimeWell]
    coverage_stats: Dict[str, Any]


@dataclass
class _PreparedLfmTimeWell:
    """内部标准化后的单井输入。"""

    control: WellControl
    time_depth_table: grid.TimeDepthTable
    x: Optional[float]
    y: Optional[float]
    trajectory: Optional[grid.WellPath]
    original_basis_type: str
    twt_conversion_mode: str
    td_table_extended: bool
    input_log_filtered: bool


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


def _resolve_well_position(well: LfmTimeWell, survey: Optional[SurveyContext]) -> tuple[float, float]:
    if well.inline is not None and well.xline is not None:
        inline = float(well.inline)
        xline = float(well.xline)
        if not np.isfinite(inline) or not np.isfinite(xline):
            raise ValueError(f"well '{well.well_name}' must provide finite inline/xline coordinates.")
        return inline, xline

    if well.x is not None and well.y is not None:
        if survey is None:
            raise ValueError(
                f"well '{well.well_name}' provides x/y but no survey context was supplied for coord_to_line."
            )
        inline, xline = survey.coord_to_line(float(well.x), float(well.y))
        return float(inline), float(xline)

    raise ValueError(
        f"well '{well.well_name}' must provide either inline/xline or x/y coordinates for location resolution."
    )


def _maybe_extend_time_depth_table(
    table: grid.TimeDepthTable,
    twt_min: float,
    twt_max: float,
    dt: float,
) -> tuple[grid.TimeDepthTable, bool]:
    extended = table.extend_to_twt_range(twt_min=twt_min, twt_max=twt_max, dt=dt, fit_points=3)
    was_extended = bool(
        extended.size != table.size
        or not np.isclose(extended.twt[0], table.twt[0])
        or not np.isclose(extended.twt[-1], table.twt[-1])
    )
    return extended, was_extended


def _convert_property_log_to_twt(
    log: grid.Log,
    table: grid.TimeDepthTable,
    trajectory: Optional[grid.WellPath],
    dt: float,
) -> tuple[grid.Log, str]:
    if log.is_twt:
        return log, "already_twt"

    if log.is_tvdss:
        raise ValueError("TVDSS-domain property logs are not supported in lfm_time.py.")

    if not log.is_md:
        raise ValueError(f"Unsupported property log basis type: {log.basis_type}.")

    if table.is_md_domain:
        return grid.convert_log_from_md_to_twt(log, table, None, dt), "md_to_twt_via_md_table"  # type: ignore

    if trajectory is None:
        raise ValueError(
            "MD-domain property logs require a WellPath trajectory when the time-depth table is in TVDSS domain."
        )
    return (
        grid.convert_log_from_md_to_twt(log, table, trajectory, dt),
        "md_to_twt_via_tvdss_table_and_trajectory",
    )


def _prepare_well(
    well: LfmTimeWell,
    target_layer: TargetLayer,
    survey: Optional[SurveyContext],
    dt: float,
    filter_cutoff_hz: float,
    filter_order: int,
    filter_buffer_seconds: Optional[float],
    filter_buffer_mode: str,
) -> _PreparedLfmTimeWell:
    inline, xline = _resolve_well_position(well, survey)
    horizon_times = target_layer.get_interpretation_values_at_location(inline, xline)
    sample_min = float(target_layer.geometry["sample_min"])
    sample_max = float(target_layer.geometry["sample_max"])
    required_twt_min = max(sample_min, float(horizon_times[target_layer.horizon_names[0]]))
    required_twt_max = min(sample_max, float(horizon_times[target_layer.horizon_names[-1]]))
    if not np.isfinite(required_twt_min) or not np.isfinite(required_twt_max) or required_twt_max <= required_twt_min:
        raise ValueError(f"well '{well.well_name}' has invalid horizon time coverage for target layer.")

    td_table, was_extended = _maybe_extend_time_depth_table(
        well.time_depth_table, required_twt_min, required_twt_max, dt
    )
    property_log_twt, conversion_mode = _convert_property_log_to_twt(
        well.property_log,
        td_table,
        well.trajectory,
        dt,
    )
    property_log_twt = lowpass_twt_log(
        property_log_twt,
        cutoff_hz=filter_cutoff_hz,
        order=filter_order,
        buffer_seconds=filter_buffer_seconds,
        buffer_mode=filter_buffer_mode,
    )
    metadata = {} if well.metadata is None else dict(well.metadata)
    return _PreparedLfmTimeWell(
        control=WellControl(
            well_name=well.well_name,
            property_name=str(well.property_name),
            property_log=property_log_twt,
            inline=inline,
            xline=xline,
            horizon_values={name: float(value) for name, value in horizon_times.items()},
            metadata=metadata,
        ),
        time_depth_table=td_table,
        x=None if well.x is None else float(well.x),
        y=None if well.y is None else float(well.y),
        trajectory=well.trajectory,
        original_basis_type=well.property_log.basis_type,
        twt_conversion_mode=conversion_mode,
        td_table_extended=was_extended,
        input_log_filtered=True,
    )


def build_lfm_time_model(
    target_layer: TargetLayer,
    wells: list[LfmTimeWell],
    *,
    survey: Optional[SurveyContext] = None,
    boundary_extension_samples: int = 50,
    n_slices: int = 32,
    variogram: str = "spherical",
    exact: bool = True,
    nugget: float = 0.0,
    filter_cutoff_hz: float = 10.0,
    filter_order: int = 6,
    filter_buffer_seconds: Optional[float] = None,
    filter_buffer_mode: str = "reflect",
    post_slice_smoothing: bool = False,
) -> LfmTimeModelResult:
    """构建时间域层位约束低频模型。

    Parameters
    ----------
    target_layer : TargetLayer
        目的层对象，需提供时间域几何信息、层位顺序与层段采样索引面。
    wells : list[LfmTimeWell]
        参与建模的井列表，至少包含一口井，且所有井的 ``property_name`` 必须一致。
    survey : SurveyContext, optional
        地震工区上下文。当井输入只提供 ``x``/``y`` 坐标时，用于换算到
        ``inline``/``xline``。
    n_slices : int, default=32
        每个层段沿顶底界面之间按比例离散的切片数量。
    boundary_extension_samples : int, default=50
        在目标层最顶、最底边界外额外扩展的采样点数。扩展区会纳入正式建模；
        若扩展区整体没有直接控制值，则退化为相邻原始层段边界 slice 的兜底填充。
    variogram : {"spherical", "exponential", "gaussian"}, default="spherical"
        多井控制时二维切片插值所使用的变差模型名称。
    exact : bool, default=True
        是否启用严格通过控制点的普通克里金插值。
    nugget : float, default=0.0
        克里金模型 nugget 参数。
    filter_cutoff_hz : float, default=10.0
        井曲线转换到 TWT 域后的低通滤波截止频率。
    filter_order : int, default=5
        井曲线低通滤波阶数。
    filter_buffer_seconds : float, optional
        滤波前在井曲线上下两端附加的缓冲时长，单位 s。默认按截止频率自动估算。
    filter_buffer_mode : {"reflect", "edge", "none"}, default="reflect"
        曲线两端缓冲样本的生成方式。``none`` 表示不做外部边界延拓。
    post_slice_smoothing : bool, default=False
        是否对各层段的比例切片结果沿 slice 维做轻度平滑。

    Returns
    -------
    LfmTimeModelResult
        包含低频模型体、方差体、结果井列表与覆盖统计的结果对象。

    Raises
    ------
    ValueError
        当井列表为空、采样域不是时间域、几何参数非法、井位置无法解析，
        或某个原始目的层段在所有切片上都没有可用控制值时抛出。

    Notes
    -----
    建模流程大致为：

    1. 解析井位并为每口井提取各层位在井点处的时间解释值；
    2. 视需要扩展时深表，并将输入属性曲线统一转换到 TWT 域后低通滤波；
    3. 对每个层段沿顶底界面做比例切片采样；
    4. 对每张切片按井点控制值执行常数填充或二维克里金插值；
    5. 将切片结果重新映射回三维时间采样体，并对建模层段之外做首尾常值延拓。
    """
    if not wells:
        raise ValueError("wells must contain at least one LfmTimeWell.")
    if n_slices < 2:
        raise ValueError(f"n_slices must be >= 2, got {n_slices}.")
    if boundary_extension_samples < 0:
        raise ValueError(f"boundary_extension_samples must be >= 0, got {boundary_extension_samples}.")
    extension_samples = int(boundary_extension_samples)

    sample_domain = str(target_layer.geometry.get("sample_domain", "")).lower()
    if sample_domain != "time":
        raise ValueError("lfm_time.py only supports TargetLayer geometry in time domain.")

    dt = float(target_layer.geometry["sample_step"])
    if dt <= 0.0:
        raise ValueError(f"target_layer.geometry['sample_step'] must be positive, got {dt}.")

    modeling_target_layer = (
        target_layer.with_boundary_extension(extension_samples) if extension_samples > 0 else target_layer
    )
    prepared_wells = [
        _prepare_well(
            well,
            modeling_target_layer,
            survey,
            dt,
            filter_cutoff_hz=float(filter_cutoff_hz),
            filter_order=int(filter_order),
            filter_buffer_seconds=filter_buffer_seconds,
            filter_buffer_mode=str(filter_buffer_mode),
        )
        for well in wells
    ]
    model_result = build_layer_constrained_model(
        target_layer,
        [well.control for well in prepared_wells],
        boundary_extension_samples=extension_samples,
        n_slices=int(n_slices),
        variogram=variogram,
        exact=exact,
        nugget=nugget,
        post_slice_smoothing=post_slice_smoothing,
    )

    coverage_stats = model_result.coverage_stats
    for well in prepared_wells:
        control = well.control
        coverage_stats["wells"][control.well_name].update(
            {
                "original_basis_type": well.original_basis_type,
                "twt_conversion_mode": well.twt_conversion_mode,
                "td_table_extended": bool(well.td_table_extended),
                "input_log_filtered": bool(well.input_log_filtered),
            }
        )

    metadata = {
        **model_result.metadata,
        "filter_cutoff_hz": float(filter_cutoff_hz),
        "filter_order": int(filter_order),
        "filter_buffer_seconds": None if filter_buffer_seconds is None else float(filter_buffer_seconds),
        "filter_buffer_mode": str(filter_buffer_mode),
    }

    result_wells = [
        LfmTimeWell(
            well_name=well.control.well_name,
            property_name=well.control.property_name,
            property_log=well.control.property_log,
            time_depth_table=well.time_depth_table,
            inline=well.control.inline,
            xline=well.control.xline,
            x=well.x,
            y=well.y,
            trajectory=well.trajectory,
            metadata={} if well.control.metadata is None else dict(well.control.metadata),
        )
        for well in prepared_wells
    ]

    return LfmTimeModelResult(
        volume=model_result.volume,
        variance_volume=model_result.variance_volume,
        geometry=dict(model_result.geometry),
        ilines=model_result.ilines,
        xlines=model_result.xlines,
        samples=model_result.samples,
        metadata=metadata,
        wells=result_wells,
        coverage_stats=coverage_stats,
    )
