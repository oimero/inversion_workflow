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
>>> filtered = lowpass_twt_log(twt_log, cutoff_hz=0.4, order=3)
>>> filtered.is_twt
True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gstools as gs
import numpy as np

from cup.seismic.process import TargetLayer
from cup.seismic.survey import SurveyContext
from wtie.processing import grid
from wtie.processing.spectral import apply_butter_lowpass_filter

POST_SLICE_SMOOTHING_KERNEL = np.asarray([0.1, 0.2, 0.4, 0.2, 0.1], dtype=float)
DEFAULT_FILTER_BUFFER_CYCLES = 2.0
VALID_FILTER_BUFFER_MODES = {"reflect", "edge"}


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

    well_name: str
    property_name: str
    property_log_twt: grid.Log
    time_depth_table: grid.TimeDepthTable
    inline: float
    xline: float
    horizon_times: Dict[str, float]
    x: Optional[float]
    y: Optional[float]
    trajectory: Optional[grid.WellPath]
    metadata: Dict[str, Any]
    original_basis_type: str
    twt_conversion_mode: str
    td_table_extended: bool
    input_log_filtered: bool


@dataclass
class _ZoneSliceModel:
    """单个层段的比例切片建模结果。"""

    top_label: str
    bottom_label: str
    top_grid: np.ndarray
    bottom_grid: np.ndarray
    slice_values: np.ndarray
    slice_variance: np.ndarray
    slice_control_counts: np.ndarray
    slice_modes: list[str]
    valid_slice_mask: np.ndarray
    fallback_source_zone: Optional[str] = None
    fallback_source_slice_index: Optional[int] = None


def _build_line_axis(line_min: float, line_max: float, line_step: float) -> np.ndarray:
    if line_step <= 0:
        raise ValueError(f"line_step must be positive, got {line_step}.")
    return np.arange(line_min, line_max + line_step, line_step, dtype=float)


def _build_sample_axis(geometry: Dict[str, Any]) -> np.ndarray:
    required_keys = {"sample_min", "sample_max", "sample_step"}
    missing_keys = required_keys - set(geometry)
    if missing_keys:
        raise ValueError(f"geometry is missing required sample keys: {sorted(missing_keys)}")

    sample_min = float(geometry["sample_min"])
    sample_max = float(geometry["sample_max"])
    sample_step = float(geometry["sample_step"])
    if sample_step <= 0:
        raise ValueError(f"sample_step must be positive, got {sample_step}.")
    return np.arange(sample_min, sample_max + sample_step, sample_step, dtype=float)


def _normalize_line_coordinates(coords: np.ndarray, line_min: float, line_step: float) -> np.ndarray:
    if line_step <= 0:
        raise ValueError(f"line_step must be positive, got {line_step}.")
    return (np.asarray(coords, dtype=float) - float(line_min)) / float(line_step)


def _nearest_neighbor_range(inlines: np.ndarray, xlines: np.ndarray) -> float:
    if inlines.size <= 1:
        return 1.0
    coords = np.column_stack([inlines, xlines]).astype(float, copy=False)
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    nearest = np.min(dists, axis=1)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        return 1.0
    return float(max(np.median(finite), 1.0))


def _krige_slice_on_line_domain(
    control_inlines: np.ndarray,
    control_xlines: np.ndarray,
    control_values: np.ndarray,
    ilines: np.ndarray,
    xlines: np.ndarray,
    *,
    range_hint: float,
    variogram: str,
    exact: bool,
    nugget: float,
) -> tuple[np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(control_inlines) & np.isfinite(control_xlines) & np.isfinite(control_values)
    control_inlines = control_inlines[finite_mask]
    control_xlines = control_xlines[finite_mask]
    control_values = control_values[finite_mask]
    if control_values.size == 0:
        raise ValueError("No finite control points were provided for kriging.")

    if control_values.size == 1 or np.allclose(control_values, control_values[0]):
        constant_grid = np.full((ilines.size, xlines.size), float(control_values[0]), dtype=float)
        zero_var = np.zeros_like(constant_grid)
        return constant_grid, zero_var

    model_name = variogram.lower()
    model_cls_map = {
        "spherical": gs.Spherical,
        "exponential": gs.Exponential,
        "gaussian": gs.Gaussian,
    }
    if model_name not in model_cls_map:
        raise ValueError(f"Unsupported variogram model: {variogram}")

    sill = float(np.var(control_values))
    if sill <= 0.0:
        sill = 1.0
    range_value = float(max(range_hint, 1.0))
    cov_model = model_cls_map[model_name](dim=2, var=sill, len_scale=range_value, nugget=float(max(nugget, 0.0)))
    krige = gs.krige.Ordinary(
        cov_model,
        cond_pos=[control_inlines, control_xlines],
        cond_val=control_values,
        exact=exact,
    )
    field, variance = krige((ilines, xlines), mesh_type="structured", return_var=True)
    return np.asarray(field, dtype=float), np.asarray(variance, dtype=float)


def _interpolate_log_at_time(log: grid.Log, twt_s: float) -> float:
    basis = np.asarray(log.basis, dtype=float)
    values = np.asarray(log.values, dtype=float)
    finite_mask = np.isfinite(basis) & np.isfinite(values)
    if not np.any(finite_mask):
        return float("nan")

    basis = basis[finite_mask]
    values = values[finite_mask]
    order = np.argsort(basis)
    basis = basis[order]
    values = values[order]

    if basis.size == 1:
        return float(values[0]) if np.isclose(float(twt_s), float(basis[0]), atol=1e-8) else float("nan")

    if twt_s < basis[0] or twt_s > basis[-1]:
        return float("nan")

    return float(np.interp(twt_s, basis, values))


def lowpass_twt_log(
    log: grid.Log,
    cutoff_hz: float = 10.0,
    order: int = 5,
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
    order : int, default=5
        滤波器阶数。启用零相位滤波时，内部会按当前实现调用双向滤波。
    buffer_seconds : float, optional
        滤波前在曲线两端附加的缓冲时长，单位 s。若未提供，则按
        ``DEFAULT_FILTER_BUFFER_CYCLES / cutoff_hz`` 自动估算，并限制在
        曲线长度允许的范围内。
    buffer_mode : {"reflect", "edge"}, default="reflect"
        生成缓冲样本的方式。``reflect`` 适合减弱端点突变带来的边界效应，
        ``edge`` 则使用端点常值外延。

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
    values_to_filter = np.pad(values, (pad_samples, pad_samples), mode=buffer_mode) if pad_samples > 0 else values  # type: ignore

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
    boundary_extension_samples: int,
    filter_cutoff_hz: float,
    filter_order: int,
    filter_buffer_seconds: Optional[float],
    filter_buffer_mode: str,
) -> _PreparedLfmTimeWell:
    inline, xline = _resolve_well_position(well, survey)
    horizon_times = target_layer.get_interpretation_values_at_location(inline, xline)
    sample_min = float(target_layer.geometry["sample_min"])
    sample_max = float(target_layer.geometry["sample_max"])
    extension_seconds = float(max(boundary_extension_samples, 0)) * dt
    required_twt_min = max(sample_min, float(horizon_times[target_layer.horizon_names[0]]) - extension_seconds)
    required_twt_max = min(sample_max, float(horizon_times[target_layer.horizon_names[-1]]) + extension_seconds)
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
        well_name=well.well_name,
        property_name=str(well.property_name),
        property_log_twt=property_log_twt,
        time_depth_table=td_table,
        inline=inline,
        xline=xline,
        horizon_times={name: float(value) for name, value in horizon_times.items()},
        x=None if well.x is None else float(well.x),
        y=None if well.y is None else float(well.y),
        trajectory=well.trajectory,
        metadata=metadata,
        original_basis_type=well.property_log.basis_type,
        twt_conversion_mode=conversion_mode,
        td_table_extended=was_extended,
        input_log_filtered=True,
    )


def _fill_missing_slices_with_neighbors(
    slice_u: np.ndarray,
    slice_values: np.ndarray,
    slice_variance: np.ndarray,
    slice_modes: list[str],
    valid_slice_mask: np.ndarray,
) -> None:
    valid_indices = np.flatnonzero(valid_slice_mask)
    if valid_indices.size == 0:
        return

    for slice_idx in range(slice_u.size):
        if valid_slice_mask[slice_idx]:
            continue

        prev_candidates = valid_indices[valid_indices < slice_idx]
        next_candidates = valid_indices[valid_indices > slice_idx]
        prev_idx = int(prev_candidates[-1]) if prev_candidates.size else None
        next_idx = int(next_candidates[0]) if next_candidates.size else None

        if prev_idx is None and next_idx is None:
            continue
        if prev_idx is None:
            slice_values[slice_idx] = slice_values[next_idx]
            slice_variance[slice_idx] = slice_variance[next_idx]
        elif next_idx is None:
            slice_values[slice_idx] = slice_values[prev_idx]
            slice_variance[slice_idx] = slice_variance[prev_idx]
        else:
            u_prev = slice_u[prev_idx]
            u_next = slice_u[next_idx]
            weight = float((slice_u[slice_idx] - u_prev) / (u_next - u_prev))
            slice_values[slice_idx] = (1.0 - weight) * slice_values[prev_idx] + weight * slice_values[next_idx]
            slice_variance[slice_idx] = (1.0 - weight) * slice_variance[prev_idx] + weight * slice_variance[next_idx]

        slice_modes[slice_idx] = "neighbor_slice_fill"


def _build_zone_slice_model(
    prepared_wells: list[_PreparedLfmTimeWell],
    ilines: np.ndarray,
    xlines: np.ndarray,
    kriging_ilines: np.ndarray,
    kriging_xlines: np.ndarray,
    slice_u: np.ndarray,
    *,
    inline_min: float,
    inline_step: float,
    xline_min: float,
    xline_step: float,
    top_label: str,
    bottom_label: str,
    top_grid: np.ndarray,
    bottom_grid: np.ndarray,
    well_top_times: np.ndarray,
    well_bottom_times: np.ndarray,
    variogram: str,
    exact: bool,
    nugget: float,
) -> _ZoneSliceModel:
    n_slices = slice_u.size
    n_il = ilines.size
    n_xl = xlines.size
    slice_values = np.full((n_slices, n_il, n_xl), np.nan, dtype=float)
    slice_variance = np.full((n_slices, n_il, n_xl), np.nan, dtype=float)
    slice_control_counts = np.zeros(n_slices, dtype=int)
    slice_modes = [""] * n_slices
    valid_slice_mask = np.zeros(n_slices, dtype=bool)

    for slice_idx, u in enumerate(slice_u):
        control_inlines = []
        control_xlines = []
        control_values = []

        for well_idx, well in enumerate(prepared_wells):
            t_top = float(well_top_times[well_idx])
            t_bottom = float(well_bottom_times[well_idx])
            if not np.isfinite(t_top) or not np.isfinite(t_bottom) or t_bottom <= t_top:
                continue

            t_slice = (1.0 - u) * t_top + u * t_bottom
            value = _interpolate_log_at_time(well.property_log_twt, t_slice)
            if not np.isfinite(value):
                continue

            control_inlines.append(well.inline)
            control_xlines.append(well.xline)
            control_values.append(value)

        slice_control_counts[slice_idx] = len(control_values)
        if len(control_values) == 0:
            continue

        valid_slice_mask[slice_idx] = True
        control_inlines_arr = np.asarray(control_inlines, dtype=float)
        control_xlines_arr = np.asarray(control_xlines, dtype=float)
        control_values_arr = np.asarray(control_values, dtype=float)

        if len(control_values) == 1:
            slice_values[slice_idx] = np.full((n_il, n_xl), float(control_values_arr[0]), dtype=float)
            slice_variance[slice_idx] = np.zeros((n_il, n_xl), dtype=float)
            slice_modes[slice_idx] = "single_well_constant"
        else:
            control_inlines_kriging = _normalize_line_coordinates(
                control_inlines_arr,
                line_min=inline_min,
                line_step=inline_step,
            )
            control_xlines_kriging = _normalize_line_coordinates(
                control_xlines_arr,
                line_min=xline_min,
                line_step=xline_step,
            )
            range_hint = _nearest_neighbor_range(control_inlines_kriging, control_xlines_kriging)
            field, variance = _krige_slice_on_line_domain(
                control_inlines_kriging,
                control_xlines_kriging,
                control_values_arr,
                kriging_ilines,
                kriging_xlines,
                range_hint=range_hint,
                variogram=variogram,
                exact=exact,
                nugget=nugget,
            )
            slice_values[slice_idx] = field
            slice_variance[slice_idx] = variance
            slice_modes[slice_idx] = "kriging"

    return _ZoneSliceModel(
        top_label=top_label,
        bottom_label=bottom_label,
        top_grid=top_grid,
        bottom_grid=bottom_grid,
        slice_values=slice_values,
        slice_variance=slice_variance,
        slice_control_counts=slice_control_counts,
        slice_modes=slice_modes,
        valid_slice_mask=valid_slice_mask,
    )


def _fill_zone_with_adjacent_boundary(
    zone_model: _ZoneSliceModel,
    *,
    source_zone_key: str,
    source_slice_index: int,
    source_values: np.ndarray,
    source_variance: np.ndarray,
) -> None:
    zone_model.slice_values[:] = source_values
    zone_model.slice_variance[:] = source_variance
    zone_model.slice_modes[:] = ["adjacent_zone_boundary_fill"] * zone_model.slice_values.shape[0]
    zone_model.fallback_source_zone = source_zone_key
    zone_model.fallback_source_slice_index = int(source_slice_index)


def _build_extension_zone(
    prepared_wells: list[_PreparedLfmTimeWell],
    ilines: np.ndarray,
    xlines: np.ndarray,
    kriging_ilines: np.ndarray,
    kriging_xlines: np.ndarray,
    slice_u: np.ndarray,
    *,
    inline_min: float,
    inline_step: float,
    xline_min: float,
    xline_step: float,
    reference_zone: _ZoneSliceModel,
    direction: str,
    extension_samples: int,
    n_sample: int,
    dt: float,
    sample_min: float,
    sample_max: float,
    variogram: str,
    exact: bool,
    nugget: float,
    post_slice_smoothing: bool,
) -> _ZoneSliceModel:
    """基于相邻原始层段构建顶/底扩展区。"""
    if direction not in {"top", "bottom"}:
        raise ValueError(f"direction must be 'top' or 'bottom', got {direction!r}.")

    if direction == "top":
        zone_model = _build_zone_slice_model(
            prepared_wells,
            ilines,
            xlines,
            kriging_ilines,
            kriging_xlines,
            slice_u,
            inline_min=inline_min,
            inline_step=inline_step,
            xline_min=xline_min,
            xline_step=xline_step,
            top_label="top_extension",
            bottom_label=reference_zone.top_label,
            top_grid=np.clip(reference_zone.top_grid - extension_samples, 0.0, float(n_sample - 1)),
            bottom_grid=reference_zone.top_grid.copy(),
            well_top_times=np.asarray(
                [max(sample_min, well.horizon_times[reference_zone.top_label] - extension_samples * dt) for well in prepared_wells],
                dtype=float,
            ),
            well_bottom_times=np.asarray(
                [well.horizon_times[reference_zone.top_label] for well in prepared_wells],
                dtype=float,
            ),
            variogram=variogram,
            exact=exact,
            nugget=nugget,
        )
        fallback_source_zone = f"{reference_zone.top_label}->{reference_zone.bottom_label}"
        fallback_source_slice_index = 0
        fallback_source_values = reference_zone.slice_values[0]
        fallback_source_variance = reference_zone.slice_variance[0]
    else:
        zone_model = _build_zone_slice_model(
            prepared_wells,
            ilines,
            xlines,
            kriging_ilines,
            kriging_xlines,
            slice_u,
            inline_min=inline_min,
            inline_step=inline_step,
            xline_min=xline_min,
            xline_step=xline_step,
            top_label=reference_zone.bottom_label,
            bottom_label="bottom_extension",
            top_grid=reference_zone.bottom_grid.copy(),
            bottom_grid=np.clip(reference_zone.bottom_grid + extension_samples, 0.0, float(n_sample - 1)),
            well_top_times=np.asarray(
                [well.horizon_times[reference_zone.bottom_label] for well in prepared_wells],
                dtype=float,
            ),
            well_bottom_times=np.asarray(
                [min(sample_max, well.horizon_times[reference_zone.bottom_label] + extension_samples * dt) for well in prepared_wells],
                dtype=float,
            ),
            variogram=variogram,
            exact=exact,
            nugget=nugget,
        )
        fallback_source_zone = f"{reference_zone.top_label}->{reference_zone.bottom_label}"
        fallback_source_slice_index = slice_u.size - 1
        fallback_source_values = reference_zone.slice_values[-1]
        fallback_source_variance = reference_zone.slice_variance[-1]

    if np.any(zone_model.valid_slice_mask):
        _fill_missing_slices_with_neighbors(
            slice_u=slice_u,
            slice_values=zone_model.slice_values,
            slice_variance=zone_model.slice_variance,
            slice_modes=zone_model.slice_modes,
            valid_slice_mask=zone_model.valid_slice_mask,
        )
    else:
        _fill_zone_with_adjacent_boundary(
            zone_model,
            source_zone_key=fallback_source_zone,
            source_slice_index=fallback_source_slice_index,
            source_values=fallback_source_values,
            source_variance=fallback_source_variance,
        )

    if post_slice_smoothing:
        zone_model.slice_values, zone_model.slice_variance = _apply_post_slice_smoothing(
            slice_values=zone_model.slice_values,
            slice_variance=zone_model.slice_variance,
        )

    return zone_model


def _write_zone_model_to_volume(
    zone_model: _ZoneSliceModel,
    slice_u: np.ndarray,
    volume: np.ndarray,
    variance_volume: np.ndarray,
) -> None:
    """将单个层段的比例切片结果批量写回三维体。"""
    top_grid = zone_model.top_grid
    bottom_grid = zone_model.bottom_grid
    valid_bounds = np.isfinite(top_grid) & np.isfinite(bottom_grid) & (bottom_grid > top_grid)
    if not np.any(valid_bounds):
        return

    rounded_top = np.where(valid_bounds, np.rint(top_grid), 0.0).astype(np.int64)
    rounded_bottom = np.where(valid_bounds, np.rint(bottom_grid), 0.0).astype(np.int64)
    rounded_top = np.clip(rounded_top, 0, volume.shape[2] - 1)
    rounded_bottom = np.clip(rounded_bottom, 0, volume.shape[2] - 1)
    valid_bounds &= rounded_bottom >= rounded_top
    if not np.any(valid_bounds):
        return

    denom = bottom_grid - top_grid
    n_slices = slice_u.size

    for sample_idx in range(volume.shape[2]):
        sample_mask = valid_bounds & (sample_idx >= rounded_top) & (sample_idx <= rounded_bottom)
        if not np.any(sample_mask):
            continue

        u_local = np.zeros_like(top_grid, dtype=float)
        u_local[sample_mask] = np.clip((float(sample_idx) - top_grid[sample_mask]) / denom[sample_mask], 0.0, 1.0)
        slice_pos = u_local * float(n_slices - 1)
        lower_idx = np.floor(slice_pos).astype(np.int64)
        upper_idx = np.ceil(slice_pos).astype(np.int64)
        weights = slice_pos - lower_idx

        values_lower = np.take_along_axis(zone_model.slice_values, lower_idx[None, :, :], axis=0)[0]
        values_upper = np.take_along_axis(zone_model.slice_values, upper_idx[None, :, :], axis=0)[0]
        local_values = (1.0 - weights) * values_lower + weights * values_upper

        variance_lower = np.take_along_axis(zone_model.slice_variance, lower_idx[None, :, :], axis=0)[0]
        variance_upper = np.take_along_axis(zone_model.slice_variance, upper_idx[None, :, :], axis=0)[0]
        local_variance = (1.0 - weights) * variance_lower + weights * variance_upper

        volume_slice = volume[:, :, sample_idx]
        variance_slice = variance_volume[:, :, sample_idx]
        volume_slice[sample_mask] = local_values[sample_mask].astype(np.float32)
        variance_slice[sample_mask] = local_variance[sample_mask].astype(np.float32)


def _extend_volume_constant_outside_modeled_range(
    volume: np.ndarray,
    variance_volume: np.ndarray,
) -> None:
    """对每条 trace 的已建模范围外做首尾常值延拓。"""
    finite_mask = np.isfinite(volume)
    has_any = np.any(finite_mask, axis=2)
    if not np.any(has_any):
        return

    first_valid = np.argmax(finite_mask, axis=2)
    last_valid = volume.shape[2] - 1 - np.argmax(finite_mask[:, :, ::-1], axis=2)
    sample_indices = np.arange(volume.shape[2], dtype=np.int64)[None, None, :]

    first_values = np.take_along_axis(volume, first_valid[:, :, None], axis=2)
    last_values = np.take_along_axis(volume, last_valid[:, :, None], axis=2)
    first_variance = np.take_along_axis(variance_volume, first_valid[:, :, None], axis=2)
    last_variance = np.take_along_axis(variance_volume, last_valid[:, :, None], axis=2)

    top_fill_mask = has_any[:, :, None] & (sample_indices < first_valid[:, :, None])
    bottom_fill_mask = has_any[:, :, None] & (sample_indices > last_valid[:, :, None])

    volume[:] = np.where(top_fill_mask, first_values, volume)
    variance_volume[:] = np.where(top_fill_mask, first_variance, variance_volume)
    volume[:] = np.where(bottom_fill_mask, last_values, volume)
    variance_volume[:] = np.where(bottom_fill_mask, last_variance, variance_volume)


def _apply_post_slice_smoothing(
    slice_values: np.ndarray,
    slice_variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """沿 zone 的 slice 维做轻度平滑，减弱相邻 slice 面的突变。"""
    if slice_values.shape != slice_variance.shape:
        raise ValueError(
            "slice_values and slice_variance must share the same shape, "
            f"got {slice_values.shape} and {slice_variance.shape}."
        )
    if slice_values.ndim != 3:
        raise ValueError(f"Expected 3D slice arrays, got ndim={slice_values.ndim}.")

    kernel = POST_SLICE_SMOOTHING_KERNEL
    radius = kernel.size // 2
    n_slices = slice_values.shape[0]

    smoothed_values = np.zeros_like(slice_values, dtype=float)
    smoothed_variance = np.zeros_like(slice_variance, dtype=float)

    for slice_idx in range(n_slices):
        weight_sum = 0.0
        for kernel_idx, weight in enumerate(kernel):
            neighbor_idx = slice_idx + kernel_idx - radius
            if not (0 <= neighbor_idx < n_slices):
                continue

            smoothed_values[slice_idx] += float(weight) * slice_values[neighbor_idx]
            smoothed_variance[slice_idx] += float(weight) * slice_variance[neighbor_idx]
            weight_sum += float(weight)

        if weight_sum <= 0.0:
            raise ValueError("Post-slice smoothing kernel has no valid support for current slice index.")

        smoothed_values[slice_idx] /= weight_sum
        smoothed_variance[slice_idx] /= weight_sum

    return smoothed_values, smoothed_variance


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
    filter_order: int = 5,
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
    filter_buffer_mode : {"reflect", "edge"}, default="reflect"
        曲线两端缓冲样本的生成方式。
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
        或某个层段在所有切片上都没有可用控制值时抛出。

    Notes
    -----
    建模流程大致为：

    1. 解析井位并为每口井提取各层位在井点处的时间解释值；
    2. 视需要扩展时深表，并将输入属性曲线统一转换到 TWT 域后低通滤波；
    3. 对每个层段沿顶底界面做比例切片采样；
    4. 对每张切片按井点控制值执行常数填充或二维克里金插值；
    5. 将切片结果重新映射回三维时间采样体，并向层段外上下延拓。
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

    prepared_wells = [
        _prepare_well(
            well,
            target_layer,
            survey,
            dt,
            boundary_extension_samples=extension_samples,
            filter_cutoff_hz=float(filter_cutoff_hz),
            filter_order=int(filter_order),
            filter_buffer_seconds=filter_buffer_seconds,
            filter_buffer_mode=str(filter_buffer_mode),
        )
        for well in wells
    ]
    property_names = {well.property_name for well in prepared_wells}
    if len(property_names) != 1:
        raise ValueError(f"All wells must share the same property_name, got {sorted(property_names)}.")
    property_name = next(iter(property_names))

    ilines = _build_line_axis(
        float(target_layer.geometry["inline_min"]),
        float(target_layer.geometry["inline_max"]),
        float(target_layer.geometry["inline_step"]),
    )
    xlines = _build_line_axis(
        float(target_layer.geometry["xline_min"]),
        float(target_layer.geometry["xline_max"]),
        float(target_layer.geometry["xline_step"]),
    )
    inline_min = float(target_layer.geometry["inline_min"])
    inline_step = float(target_layer.geometry["inline_step"])
    xline_min = float(target_layer.geometry["xline_min"])
    xline_step = float(target_layer.geometry["xline_step"])
    kriging_ilines = _normalize_line_coordinates(ilines, line_min=inline_min, line_step=inline_step)
    kriging_xlines = _normalize_line_coordinates(xlines, line_min=xline_min, line_step=xline_step)
    samples = _build_sample_axis(target_layer.geometry)
    n_il, n_xl, n_sample = ilines.size, xlines.size, samples.size
    sample_min = float(target_layer.geometry["sample_min"])
    sample_max = float(target_layer.geometry["sample_max"])

    volume = np.full((n_il, n_xl, n_sample), np.nan, dtype=np.float32)
    variance_volume = np.full((n_il, n_xl, n_sample), np.nan, dtype=np.float32)
    slice_u = np.linspace(0.0, 1.0, n_slices, dtype=float)
    coverage_stats: Dict[str, Any] = {
        "property_name": property_name,
        "wells": {},
        "zones": {},
    }

    for well in prepared_wells:
        coverage_stats["wells"][well.well_name] = {
            "inline": float(well.inline),
            "xline": float(well.xline),
            "original_basis_type": well.original_basis_type,
            "twt_conversion_mode": well.twt_conversion_mode,
            "td_table_extended": bool(well.td_table_extended),
            "input_log_filtered": bool(well.input_log_filtered),
            "horizon_times": dict(well.horizon_times),
            "modeled_twt_min": max(
                sample_min,
                float(well.horizon_times[target_layer.horizon_names[0]]) - extension_samples * dt,
            ),
            "modeled_twt_max": min(
                sample_max,
                float(well.horizon_times[target_layer.horizon_names[-1]]) + extension_samples * dt,
            ),
        }

    base_zone_models: list[_ZoneSliceModel] = []
    base_zones = target_layer.iter_zones()
    for top_name, bottom_name in base_zones:
        top_grid, bottom_grid = target_layer.get_zone_sample_index_grids((top_name, bottom_name))
        zone_model = _build_zone_slice_model(
            prepared_wells,
            ilines,
            xlines,
            kriging_ilines,
            kriging_xlines,
            slice_u,
            inline_min=inline_min,
            inline_step=inline_step,
            xline_min=xline_min,
            xline_step=xline_step,
            top_label=top_name,
            bottom_label=bottom_name,
            top_grid=top_grid,
            bottom_grid=bottom_grid,
            well_top_times=np.asarray([well.horizon_times[top_name] for well in prepared_wells], dtype=float),
            well_bottom_times=np.asarray([well.horizon_times[bottom_name] for well in prepared_wells], dtype=float),
            variogram=variogram,
            exact=exact,
            nugget=nugget,
        )
        if not np.any(zone_model.valid_slice_mask):
            raise ValueError(f"Zone '{top_name}' -> '{bottom_name}' has no valid control values on any slice.")

        _fill_missing_slices_with_neighbors(
            slice_u=slice_u,
            slice_values=zone_model.slice_values,
            slice_variance=zone_model.slice_variance,
            slice_modes=zone_model.slice_modes,
            valid_slice_mask=zone_model.valid_slice_mask,
        )
        if post_slice_smoothing:
            zone_model.slice_values, zone_model.slice_variance = _apply_post_slice_smoothing(
                slice_values=zone_model.slice_values,
                slice_variance=zone_model.slice_variance,
            )
        base_zone_models.append(zone_model)

    modeled_zones: list[_ZoneSliceModel] = []
    if extension_samples > 0:
        modeled_zones.append(
            _build_extension_zone(
                prepared_wells,
                ilines,
                xlines,
                kriging_ilines,
                kriging_xlines,
                slice_u,
                inline_min=inline_min,
                inline_step=inline_step,
                xline_min=xline_min,
                xline_step=xline_step,
                reference_zone=base_zone_models[0],
                direction="top",
                extension_samples=extension_samples,
                n_sample=n_sample,
                dt=dt,
                sample_min=sample_min,
                sample_max=sample_max,
                variogram=variogram,
                exact=exact,
                nugget=nugget,
                post_slice_smoothing=post_slice_smoothing,
            )
        )

    modeled_zones.extend(base_zone_models)

    if extension_samples > 0:
        modeled_zones.append(
            _build_extension_zone(
                prepared_wells,
                ilines,
                xlines,
                kriging_ilines,
                kriging_xlines,
                slice_u,
                inline_min=inline_min,
                inline_step=inline_step,
                xline_min=xline_min,
                xline_step=xline_step,
                reference_zone=base_zone_models[-1],
                direction="bottom",
                extension_samples=extension_samples,
                n_sample=n_sample,
                dt=dt,
                sample_min=sample_min,
                sample_max=sample_max,
                variogram=variogram,
                exact=exact,
                nugget=nugget,
                post_slice_smoothing=post_slice_smoothing,
            )
        )

    for zone_model in modeled_zones:
        zone_key = f"{zone_model.top_label}->{zone_model.bottom_label}"
        coverage_stats["zones"][zone_key] = {
            "slice_control_counts": zone_model.slice_control_counts.tolist(),
            "slice_modes": zone_model.slice_modes,
        }
        if zone_model.fallback_source_zone is not None:
            coverage_stats["zones"][zone_key]["fallback_source_zone"] = zone_model.fallback_source_zone
            coverage_stats["zones"][zone_key]["fallback_source_slice_index"] = zone_model.fallback_source_slice_index

        _write_zone_model_to_volume(
            zone_model=zone_model,
            slice_u=slice_u,
            volume=volume,
            variance_volume=variance_volume,
        )

    _extend_volume_constant_outside_modeled_range(volume=volume, variance_volume=variance_volume)

    metadata = {
        "backend": "gstools",
        "slice_mode": "proportional",
        "property_name": property_name,
        "variogram": variogram,
        "exact": bool(exact),
        "nugget": float(nugget),
        "n_slices": int(n_slices),
        "boundary_extension_samples": extension_samples,
        "filter_cutoff_hz": float(filter_cutoff_hz),
        "filter_order": int(filter_order),
        "filter_buffer_seconds": None if filter_buffer_seconds is None else float(filter_buffer_seconds),
        "filter_buffer_mode": str(filter_buffer_mode),
        "coord_system": "inline_xline",
        "kriging_coord_system": "inline_xline_normalized_by_step",
        "kriging_inline_step": inline_step,
        "kriging_xline_step": xline_step,
        "horizon_names": list(target_layer.horizon_names),
        "zone_names": [[zone.top_label, zone.bottom_label] for zone in modeled_zones],
        "well_names": [well.well_name for well in prepared_wells],
        "post_slice_smoothing": bool(post_slice_smoothing),
        "post_slice_smoothing_kernel": POST_SLICE_SMOOTHING_KERNEL.tolist(),
        "variance_volume_included": True,
    }

    result_wells = [
        LfmTimeWell(
            well_name=well.well_name,
            property_name=well.property_name,
            property_log=well.property_log_twt,
            time_depth_table=well.time_depth_table,
            inline=well.inline,
            xline=well.xline,
            x=well.x,
            y=well.y,
            trajectory=well.trajectory,
            metadata=dict(well.metadata),
        )
        for well in prepared_wells
    ]

    return LfmTimeModelResult(
        volume=volume,
        variance_volume=variance_volume,
        geometry=dict(target_layer.geometry),
        ilines=ilines,
        xlines=xlines,
        samples=samples,
        metadata=metadata,
        wells=result_wells,
        coverage_stats=coverage_stats,
    )
