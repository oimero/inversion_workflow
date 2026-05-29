"""cup.seismic.lfm_time: 时间域层位约束低频模型构建。

本模块提供基于层位解释、井曲线与时深关系的时间域低频模型构建能力，
包括井曲线预处理、按层段比例切片采样、二维切片插值以及结果体封装。

边界说明
--------
- 本模块不负责层位解释文件读取、测井提取或地震体加载。
- 本模块仅支持 ``TargetZone.geometry['sample_domain'] == 'time'`` 的场景。
- 井曲线若为 MD 域，可借助时深表与井轨迹转换到 TWT 域；TVDSS 域曲线当前不支持直接输入。

核心公开对象
------------
1. LfmTimeWell: 单井低频模型输入描述。
2. LfmTimeModelResult: 低频模型体、方差体与覆盖统计的结果封装。
3. lowpass_twt_log: 对 TWT 域单条曲线执行低通滤波。
4. build_lfm_time_model: 基于层位约束与井点控制构建时间域低频模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from cup.seismic.geometry import resolve_well_line_position
from cup.seismic.modeling import (
    WellControl,
    ZoneSliceModel,
    _apply_post_slice_smoothing,
    _extend_volume_constant_outside_modeled_range,
    _fill_missing_slices_with_neighbors,
    _fill_zone_with_adjacent_boundary,
    _krige_slice_on_line_domain,
    _nearest_neighbor_range,
    _normalize_line_coordinates,
    _write_zone_model_to_volume,
    build_layer_constrained_model,
)
from cup.seismic.survey import SurveyContext
from cup.seismic.target_zone import TargetZone
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
        参与建模的几何描述，通常直接来自 ``TargetZone.geometry``。
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


@dataclass(frozen=True)
class LfmTimeControlPoint:
    """时间域点级低频模型控制样本。

    Unlike :class:`LfmTimeWell`, this object represents one physical sample in
    the target interval.  It is the natural input for deviated wells whose
    inline/xline position changes with TWT.
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
    sample_index: Optional[int] = None
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


def _zone_key(top_name: str, bottom_name: str) -> str:
    return f"{top_name}->{bottom_name}"


def _point_zone_matches(point: LfmTimeControlPoint, top_name: str, bottom_name: str) -> bool:
    zone = str(point.zone_name)
    return zone == _zone_key(top_name, bottom_name) or zone == f"{top_name}_{bottom_name}"


def _finite_point(point: LfmTimeControlPoint) -> bool:
    values = [
        point.twt_s,
        point.inline_float,
        point.xline_float,
        point.u_in_zone,
        point.ai,
        point.weight,
    ]
    return bool(np.all(np.isfinite(np.asarray(values, dtype=float))) and point.weight > 0.0)


def _aggregate_duplicate_slice_points(
    inlines: np.ndarray,
    xlines: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Collapse exact duplicate inline/xline controls within one slice.

    Dense well grids can put multiple controls on the same floating trace
    coordinate.  Kriging backends generally require unique condition
    coordinates, so this only aggregates exact coordinate duplicates inside a
    single slice.  Broader conflict-resolution policies stay outside this first
    point-control builder.
    """
    if values.size <= 1:
        return inlines, xlines, values, 0

    frame = np.column_stack([inlines, xlines, values, weights]).astype(float, copy=False)
    keys: dict[tuple[float, float], list[int]] = {}
    for index, row in enumerate(frame):
        keys.setdefault((float(row[0]), float(row[1])), []).append(index)
    duplicate_count = sum(max(0, len(indices) - 1) for indices in keys.values())
    if duplicate_count == 0:
        return inlines, xlines, values, 0

    out_inline = []
    out_xline = []
    out_value = []
    for (inline, xline), indices in keys.items():
        idx = np.asarray(indices, dtype=np.int64)
        local_weights = np.asarray(weights[idx], dtype=float)
        local_values = np.asarray(values[idx], dtype=float)
        if np.sum(local_weights) <= 0.0:
            averaged = float(np.mean(local_values))
        else:
            averaged = float(np.average(local_values, weights=local_weights))
        out_inline.append(inline)
        out_xline.append(xline)
        out_value.append(averaged)
    return (
        np.asarray(out_inline, dtype=float),
        np.asarray(out_xline, dtype=float),
        np.asarray(out_value, dtype=float),
        int(duplicate_count),
    )


def _build_zone_slice_model_from_points(
    control_points: list[LfmTimeControlPoint],
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
    variogram: str,
    exact: bool,
    nugget: float,
) -> tuple[ZoneSliceModel, dict[str, Any]]:
    """Build one proportional zone model from point-level controls."""
    n_slices = slice_u.size
    n_il = ilines.size
    n_xl = xlines.size
    slice_values = np.full((n_slices, n_il, n_xl), np.nan, dtype=float)
    slice_variance = np.full((n_slices, n_il, n_xl), np.nan, dtype=float)
    slice_control_counts = np.zeros(n_slices, dtype=int)
    slice_modes = [""] * n_slices
    valid_slice_mask = np.zeros(n_slices, dtype=bool)
    duplicate_controls_aggregated = 0

    finite_points = [point for point in control_points if _finite_point(point)]
    for slice_idx, _u in enumerate(slice_u):
        if slice_idx == 0:
            lower = -np.inf
        else:
            lower = 0.5 * (float(slice_u[slice_idx - 1]) + float(slice_u[slice_idx]))
        if slice_idx == n_slices - 1:
            upper = np.inf
        else:
            upper = 0.5 * (float(slice_u[slice_idx]) + float(slice_u[slice_idx + 1]))

        points = [point for point in finite_points if lower <= float(point.u_in_zone) < upper]
        slice_control_counts[slice_idx] = len(points)
        if not points:
            continue

        control_inlines = np.asarray([point.inline_float for point in points], dtype=float)
        control_xlines = np.asarray([point.xline_float for point in points], dtype=float)
        control_values = np.asarray([point.ai for point in points], dtype=float)
        control_weights = np.asarray([point.weight for point in points], dtype=float)
        control_inlines, control_xlines, control_values, duplicates = _aggregate_duplicate_slice_points(
            control_inlines,
            control_xlines,
            control_values,
            control_weights,
        )
        duplicate_controls_aggregated += duplicates

        valid_slice_mask[slice_idx] = True
        if control_values.size == 1:
            slice_values[slice_idx] = np.full((n_il, n_xl), float(control_values[0]), dtype=float)
            slice_variance[slice_idx] = np.zeros((n_il, n_xl), dtype=float)
            slice_modes[slice_idx] = "single_point_constant"
            continue

        control_inlines_kriging = _normalize_line_coordinates(
            control_inlines,
            line_min=inline_min,
            line_step=inline_step,
        )
        control_xlines_kriging = _normalize_line_coordinates(
            control_xlines,
            line_min=xline_min,
            line_step=xline_step,
        )
        range_hint = _nearest_neighbor_range(control_inlines_kriging, control_xlines_kriging)
        field, variance = _krige_slice_on_line_domain(
            control_inlines_kriging,
            control_xlines_kriging,
            control_values,
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

    zone_model = ZoneSliceModel(
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
    stats = {
        "point_count": int(len(finite_points)),
        "duplicate_controls_aggregated": int(duplicate_controls_aggregated),
    }
    return zone_model, stats


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
    post_slice_smoothing: bool = False,
) -> LfmTimePointModelResult:
    """Build a time-domain AI LFM from point-level layer controls.

    This is the point-control counterpart to :func:`build_lfm_time_model`.
    Every input point carries its own inline/xline/TWT and normalized
    ``u_in_zone`` coordinate, which allows deviated well samples to influence
    different traces along the well path.
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

    finite_points = [point for point in control_points if _finite_point(point)]
    if not finite_points:
        raise ValueError("control_points contain no finite positive-weight samples.")

    extension_samples = int(boundary_extension_samples)
    modeling_target_layer = (
        target_layer.with_boundary_extension(extension_samples) if extension_samples > 0 else target_layer
    )

    ilines = target_layer.ilines.astype(float, copy=False)
    xlines = target_layer.xlines.astype(float, copy=False)
    samples = target_layer.samples.astype(float, copy=False)
    inline_min = float(target_layer.geometry["inline_min"])
    inline_step = float(target_layer.geometry["inline_step"])
    xline_min = float(target_layer.geometry["xline_min"])
    xline_step = float(target_layer.geometry["xline_step"])
    kriging_ilines = _normalize_line_coordinates(ilines, line_min=inline_min, line_step=inline_step)
    kriging_xlines = _normalize_line_coordinates(xlines, line_min=xline_min, line_step=xline_step)

    volume = np.full((ilines.size, xlines.size, samples.size), np.nan, dtype=np.float32)
    variance_volume = np.full_like(volume, np.nan, dtype=np.float32)
    slice_u = np.linspace(0.0, 1.0, int(n_slices), dtype=float)
    coverage_stats: Dict[str, Any] = {
        "property_name": "AI",
        "control_point_count": int(len(finite_points)),
        "well_count": int(len({point.well_name for point in finite_points})),
        "wells": {},
        "zones": {},
    }
    for well_name in sorted({point.well_name for point in finite_points}):
        well_points = [point for point in finite_points if point.well_name == well_name]
        flat_values = [point.flat_idx for point in well_points if point.flat_idx is not None]
        coverage_stats["wells"][well_name] = {
            "control_point_count": int(len(well_points)),
            "unique_trace_count": int(len(set(flat_values))) if flat_values else None,
            "source_modes": sorted({str(point.source) for point in well_points}),
        }

    base_zone_models: list[ZoneSliceModel] = []
    for top_name, bottom_name in target_layer.iter_zones():
        zone_points = [
            point
            for point in finite_points
            if _point_zone_matches(point, top_name, bottom_name)
        ]
        top_grid, bottom_grid = modeling_target_layer.get_zone_sample_index_grids((top_name, bottom_name))
        zone_model, point_stats = _build_zone_slice_model_from_points(
            zone_points,
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
            variogram=variogram,
            exact=exact,
            nugget=nugget,
        )
        if not np.any(zone_model.valid_slice_mask):
            raise ValueError(f"Zone '{top_name}' -> '{bottom_name}' has no point controls on any slice.")
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
        zone_key = _zone_key(top_name, bottom_name)
        coverage_stats["zones"][zone_key] = {
            **point_stats,
            "slice_control_counts": zone_model.slice_control_counts.tolist(),
            "slice_modes": zone_model.slice_modes,
        }
        base_zone_models.append(zone_model)

    modeled_zones: list[ZoneSliceModel] = []
    if extension_samples > 0:
        top_extension_zone = (modeling_target_layer.horizon_names[0], target_layer.horizon_names[0])
        top_grid, bottom_grid = modeling_target_layer.get_zone_sample_index_grids(top_extension_zone)
        top_extension_model = ZoneSliceModel(
            top_label=top_extension_zone[0],
            bottom_label=top_extension_zone[1],
            top_grid=top_grid,
            bottom_grid=bottom_grid,
            slice_values=np.full((slice_u.size, ilines.size, xlines.size), np.nan, dtype=float),
            slice_variance=np.full((slice_u.size, ilines.size, xlines.size), np.nan, dtype=float),
            slice_control_counts=np.zeros(slice_u.size, dtype=int),
            slice_modes=[""] * slice_u.size,
            valid_slice_mask=np.zeros(slice_u.size, dtype=bool),
        )
        reference_zone = base_zone_models[0]
        _fill_zone_with_adjacent_boundary(
            top_extension_model,
            source_zone_key=_zone_key(reference_zone.top_label, reference_zone.bottom_label),
            source_slice_index=0,
            source_values=reference_zone.slice_values[0],
            source_variance=reference_zone.slice_variance[0],
        )
        modeled_zones.append(top_extension_model)

    modeled_zones.extend(base_zone_models)

    if extension_samples > 0:
        bottom_extension_zone = (target_layer.horizon_names[-1], modeling_target_layer.horizon_names[-1])
        top_grid, bottom_grid = modeling_target_layer.get_zone_sample_index_grids(bottom_extension_zone)
        bottom_extension_model = ZoneSliceModel(
            top_label=bottom_extension_zone[0],
            bottom_label=bottom_extension_zone[1],
            top_grid=top_grid,
            bottom_grid=bottom_grid,
            slice_values=np.full((slice_u.size, ilines.size, xlines.size), np.nan, dtype=float),
            slice_variance=np.full((slice_u.size, ilines.size, xlines.size), np.nan, dtype=float),
            slice_control_counts=np.zeros(slice_u.size, dtype=int),
            slice_modes=[""] * slice_u.size,
            valid_slice_mask=np.zeros(slice_u.size, dtype=bool),
        )
        reference_zone = base_zone_models[-1]
        _fill_zone_with_adjacent_boundary(
            bottom_extension_model,
            source_zone_key=_zone_key(reference_zone.top_label, reference_zone.bottom_label),
            source_slice_index=slice_u.size - 1,
            source_values=reference_zone.slice_values[-1],
            source_variance=reference_zone.slice_variance[-1],
        )
        modeled_zones.append(bottom_extension_model)

    for zone_model in modeled_zones:
        zone_key = _zone_key(zone_model.top_label, zone_model.bottom_label)
        coverage_stats["zones"].setdefault(zone_key, {})
        coverage_stats["zones"][zone_key].update(
            {
                "slice_control_counts": zone_model.slice_control_counts.tolist(),
                "slice_modes": zone_model.slice_modes,
            }
        )
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
        "slice_mode": "point_control_proportional",
        "property_name": "AI",
        "sample_domain": "time",
        "sample_unit": "s",
        "variogram": variogram,
        "exact": bool(exact),
        "nugget": float(nugget),
        "n_slices": int(n_slices),
        "boundary_extension_samples": extension_samples,
        "coord_system": "inline_xline",
        "kriging_coord_system": "grid_index_normalized",
        "kriging_inline_step": inline_step,
        "kriging_xline_step": xline_step,
        "horizon_names": list(target_layer.horizon_names),
        "zone_names": [[zone.top_label, zone.bottom_label] for zone in modeled_zones],
        "well_names": sorted({point.well_name for point in finite_points}),
        "post_slice_smoothing": bool(post_slice_smoothing),
        "variance_volume_included": True,
    }
    return LfmTimePointModelResult(
        volume=volume,
        variance_volume=variance_volume,
        geometry=dict(target_layer.geometry),
        ilines=ilines,
        xlines=xlines,
        samples=samples,
        metadata=metadata,
        control_points=list(finite_points),
        coverage_stats=coverage_stats,
    )


def _maybe_extend_time_depth_table(
    table: grid.TimeDepthTable,
    twt_min: float,
    twt_max: float,
    dt: float,
) -> tuple[grid.TimeDepthTable, bool]:
    """按目标时间范围扩展时深表。"""
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
    """将属性曲线转换到 TWT 域并返回转换模式。"""
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
    target_layer: TargetZone,
    survey: Optional[SurveyContext],
    dt: float,
    filter_cutoff_hz: float,
    filter_order: int,
    filter_buffer_seconds: Optional[float],
    filter_buffer_mode: str,
) -> _PreparedLfmTimeWell:
    """标准化井输入并生成建模控制点。"""
    inline, xline = resolve_well_line_position(well, survey.line_geometry if survey is not None else None)
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
    target_layer: TargetZone,
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
    target_layer : TargetZone
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
        raise ValueError("lfm_time.py only supports TargetZone geometry in time domain.")

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
