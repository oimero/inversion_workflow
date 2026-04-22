"""cup.seismic.lfm_depth: 深度域层位约束低频模型构建。

本模块提供 TVDSS 深度域低频模型构建能力，负责将井曲线统一到 TVDSS
深度域、执行沿深度轴的空间低通滤波，并复用通用层位约束建模核心。

边界说明
--------
- 本模块不负责层位解释文件读取、测井提取或地震体加载。
- 本模块仅支持 ``TargetLayer.geometry['sample_domain'] == 'depth'`` 且
  ``sample_unit == 'm'`` 的场景。
- 深度域建模以 TVDSS 为目标深度口径。MD 域井曲线必须提供井轨迹，
  或提供 ``kb`` 以按直井近似生成井轨迹。

核心公开对象
------------
1. LfmDepthWell: 单井深度域低频模型输入描述。
2. LfmDepthModelResult: 低频模型体、方差体与覆盖统计的结果封装。
3. lowpass_depth_log: 对 TVDSS 域单条曲线执行空间低通滤波。
4. build_lfm_depth_model: 基于层位约束与井点控制构建 TVDSS 深度域低频模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from cup.seismic.modeling import WellControl, build_layer_constrained_model
from cup.seismic.target_layer import TargetLayer
from cup.seismic.survey import SurveyContext
from wtie.processing import grid
from wtie.processing.spectral import apply_butter_lowpass_filter

DEFAULT_FILTER_BUFFER_WAVELENGTHS = 2.0
VALID_FILTER_BUFFER_MODES = {"reflect", "edge"}


@dataclass
class LfmDepthWell:
    """单井 TVDSS 深度域低频模型输入。

    Parameters
    ----------
    well_name : str
        井名或井标识。
    property_name : str
        当前井参与建模的属性名称，例如 ``"AI"``、``"Vp"``。
    property_log : grid.Log
        属性曲线，允许为 TVDSS 域或 MD 域。
    inline, xline : float, optional
        井位所在的 inline/xline 坐标。若提供，则优先使用。
    x, y : float, optional
        井口或目标点的平面坐标。当未提供 ``inline``/``xline`` 时，可结合
        ``survey`` 上下文转换到道号坐标。
    trajectory : grid.WellPath, optional
        当属性曲线为 MD 域时，用于辅助转换到 TVDSS 域。
    kb : float, optional
        Kelly Bushing 高程。MD 域属性曲线未提供 ``trajectory`` 时，若提供
        ``kb``，会按直井近似创建 ``grid.WellPath``。
    metadata : dict, optional
        附加元信息，会在结果中原样保留一份副本。
    """

    well_name: str
    property_name: str
    property_log: grid.Log
    inline: Optional[float] = None
    xline: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    trajectory: Optional[grid.WellPath] = None
    kb: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LfmDepthModelResult:
    """深度域低频模型构建结果。"""

    volume: np.ndarray
    variance_volume: np.ndarray
    geometry: Dict[str, Any]
    ilines: np.ndarray
    xlines: np.ndarray
    samples: np.ndarray
    metadata: Dict[str, Any]
    wells: list[LfmDepthWell]
    coverage_stats: Dict[str, Any]


@dataclass
class _PreparedLfmDepthWell:
    """内部标准化后的单井输入。"""

    control: WellControl
    x: Optional[float]
    y: Optional[float]
    trajectory: Optional[grid.WellPath]
    kb: Optional[float]
    original_basis_type: str
    depth_conversion_mode: str
    vertical_assumption_used: bool
    input_log_filtered: bool


def lowpass_depth_log(
    log: grid.Log,
    cutoff_wavelength_m: float = 150.0,
    order: int = 5,
    buffer_meters: Optional[float] = None,
    buffer_mode: str = "reflect",
) -> grid.Log:
    """对 TVDSS 域单条曲线执行 Butterworth 空间低通滤波。

    Parameters
    ----------
    log : grid.Log
        输入曲线，必须位于 TVDSS 域且采样间隔为常数。
    cutoff_wavelength_m : float, default=150.0
        截止波长，单位 m。内部会转换为 ``1 / cutoff_wavelength_m`` cycles/m。
    order : int, default=5
        滤波器阶数。
    buffer_meters : float, optional
        滤波前在曲线两端附加的缓冲长度，单位 m。若未提供，则按
        ``2 * cutoff_wavelength_m`` 与 ``3 * order * dz`` 的较大值估算。
    buffer_mode : {"reflect", "edge"}, default="reflect"
        生成缓冲样本的方式。
    """
    if not log.is_tvdss:
        raise ValueError("lowpass_depth_log only supports TVDSS-domain logs.")
    if buffer_mode not in VALID_FILTER_BUFFER_MODES:
        raise ValueError(f"buffer_mode must be one of {sorted(VALID_FILTER_BUFFER_MODES)}, got {buffer_mode!r}.")

    cutoff_wavelength = float(cutoff_wavelength_m)
    if cutoff_wavelength <= 0.0:
        raise ValueError(f"cutoff_wavelength_m must be positive, got {cutoff_wavelength_m}.")

    dz = float(log.sampling_rate)
    if dz <= 0.0:
        raise ValueError(f"log sampling_rate must be positive, got {dz}.")

    cutoff_cycles_per_m = 1.0 / cutoff_wavelength
    fs = 1.0 / dz
    nyquist = 0.5 * fs
    if cutoff_cycles_per_m <= 0.0 or cutoff_cycles_per_m >= nyquist:
        raise ValueError(
            "cutoff_wavelength_m is too short for the log sampling interval. "
            f"cutoff_cycles_per_m must be within (0, {nyquist}), got {cutoff_cycles_per_m}."
        )

    values = log.values.astype(np.float64)
    resolved_buffer_meters = (
        max(DEFAULT_FILTER_BUFFER_WAVELENGTHS * cutoff_wavelength, 3.0 * int(order) * dz)
        if buffer_meters is None
        else float(buffer_meters)
    )
    if resolved_buffer_meters < 0.0:
        raise ValueError(f"buffer_meters must be non-negative when provided, got {buffer_meters}.")

    pad_samples = int(np.ceil(resolved_buffer_meters / dz))
    pad_samples = min(max(pad_samples, 0), values.size - 1)
    values_to_filter = np.pad(values, (pad_samples, pad_samples), mode=buffer_mode) if pad_samples > 0 else values  # type: ignore

    filtered = apply_butter_lowpass_filter(
        values_to_filter,
        highcut=cutoff_cycles_per_m,
        fs=fs,
        order=int(order),
        zero_phase=True,
    )
    if pad_samples > 0:
        filtered = filtered[pad_samples : pad_samples + values.size]

    return grid.Log(
        filtered.astype(np.float64),
        log.basis.copy(),
        "tvdss",
        name=log.name,
        unit=log.unit,
        allow_nan=log.allow_nan,
    )


def _resolve_well_position(well: LfmDepthWell, survey: Optional[SurveyContext]) -> tuple[float, float]:
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


def _normalize_depth_basis_and_values(log: grid.Log) -> tuple[np.ndarray, np.ndarray]:
    basis = np.asarray(log.basis, dtype=float)
    values = np.asarray(log.values, dtype=float)
    finite_mask = np.isfinite(basis) & np.isfinite(values)
    if not np.any(finite_mask):
        raise ValueError(f"log '{log.name}' does not contain any finite depth/value samples.")

    basis = basis[finite_mask]
    values = values[finite_mask]
    order = np.argsort(basis)
    basis = basis[order]
    values = values[order]

    unique_basis, unique_indices = np.unique(basis, return_index=True)
    unique_values = values[unique_indices]
    if unique_basis.size < 2:
        raise ValueError(f"log '{log.name}' must contain at least two unique finite depth samples.")
    return unique_basis, unique_values


def _convert_md_log_to_tvdss(
    log: grid.Log,
    trajectory: grid.WellPath,
    dz: float,
) -> grid.Log:
    log_md, log_values = _normalize_depth_basis_and_values(log)
    trajectory_md = np.asarray(trajectory.md, dtype=float)
    trajectory_tvdss = np.asarray(trajectory.tvdss, dtype=float)
    finite_traj = np.isfinite(trajectory_md) & np.isfinite(trajectory_tvdss)
    trajectory_md = trajectory_md[finite_traj]
    trajectory_tvdss = trajectory_tvdss[finite_traj]
    order = np.argsort(trajectory_md)
    trajectory_md = trajectory_md[order]
    trajectory_tvdss = trajectory_tvdss[order]

    if trajectory_md.size < 2:
        raise ValueError("trajectory must contain at least two finite MD/TVDSS samples.")
    if np.any(np.diff(trajectory_md) <= 0.0):
        raise ValueError("trajectory MD samples must be strictly increasing.")
    if np.any(np.diff(trajectory_tvdss) < 0.0):
        raise ValueError("trajectory TVDSS samples must be non-decreasing for TVDSS-domain modeling.")

    overlap_min = max(float(log_md[0]), float(trajectory_md[0]))
    overlap_max = min(float(log_md[-1]), float(trajectory_md[-1]))
    if overlap_max <= overlap_min:
        raise ValueError("Log MD range and well trajectory MD range do not overlap.")

    overlap_mask = (log_md >= overlap_min) & (log_md <= overlap_max)
    if np.count_nonzero(overlap_mask) < 2:
        raise ValueError("Insufficient overlapping MD samples to convert log from MD to TVDSS.")

    overlap_md = log_md[overlap_mask]
    overlap_values = log_values[overlap_mask]
    tvdss_at_log = np.interp(overlap_md, trajectory_md, trajectory_tvdss)
    valid_order = np.argsort(tvdss_at_log)
    tvdss_at_log = tvdss_at_log[valid_order]
    overlap_values = overlap_values[valid_order]

    unique_tvdss, unique_indices = np.unique(tvdss_at_log, return_index=True)
    unique_values = overlap_values[unique_indices]
    if unique_tvdss.size < 2:
        raise ValueError("Converted TVDSS log must contain at least two unique depth samples.")

    linear_tvdss = np.arange(float(unique_tvdss[0]), float(unique_tvdss[-1]) + float(dz), float(dz))
    values_at_dz = np.interp(linear_tvdss, unique_tvdss, unique_values)

    return grid.Log(
        values_at_dz.astype(np.float64),
        linear_tvdss.astype(np.float64),
        "tvdss",
        name=log.name,
        unit=log.unit,
        allow_nan=log.allow_nan,
    )


def _convert_property_log_to_tvdss(
    well: LfmDepthWell,
    dz: float,
) -> tuple[grid.Log, str, Optional[grid.WellPath], bool]:
    log = well.property_log
    if log.is_tvdss:
        return log, "already_tvdss", well.trajectory, False

    if not log.is_md:
        raise ValueError(f"Unsupported property log basis type for depth modeling: {log.basis_type}.")

    if well.trajectory is not None:
        return (
            _convert_md_log_to_tvdss(log, well.trajectory, dz=dz),
            "md_to_tvdss_via_trajectory",
            well.trajectory,
            False,
        )

    if well.kb is None:
        raise ValueError(
            f"MD-domain property log for well '{well.well_name}' requires either a WellPath trajectory or kb."
        )

    trajectory = grid.WellPath(md=np.asarray(log.basis, dtype=float), kb=float(well.kb))
    return (
        _convert_md_log_to_tvdss(log, trajectory, dz=dz),
        "md_to_tvdss_via_vertical_trajectory",
        trajectory,
        True,
    )


def _prepare_well(
    well: LfmDepthWell,
    target_layer: TargetLayer,
    survey: Optional[SurveyContext],
    dz: float,
    filter_cutoff_wavelength_m: float,
    filter_order: int,
    filter_buffer_meters: Optional[float],
    filter_buffer_mode: str,
) -> _PreparedLfmDepthWell:
    inline, xline = _resolve_well_position(well, survey)
    horizon_depths = target_layer.get_interpretation_values_at_location(inline, xline)
    required_depth_min = float(horizon_depths[target_layer.horizon_names[0]])
    required_depth_max = float(horizon_depths[target_layer.horizon_names[-1]])
    if (
        not np.isfinite(required_depth_min)
        or not np.isfinite(required_depth_max)
        or required_depth_max <= required_depth_min
    ):
        raise ValueError(f"well '{well.well_name}' has invalid horizon depth coverage for target layer.")

    property_log_tvdss, conversion_mode, trajectory, vertical_assumption_used = _convert_property_log_to_tvdss(
        well,
        dz=dz,
    )
    property_log_tvdss = lowpass_depth_log(
        property_log_tvdss,
        cutoff_wavelength_m=filter_cutoff_wavelength_m,
        order=filter_order,
        buffer_meters=filter_buffer_meters,
        buffer_mode=filter_buffer_mode,
    )
    metadata = {} if well.metadata is None else dict(well.metadata)
    return _PreparedLfmDepthWell(
        control=WellControl(
            well_name=well.well_name,
            property_name=str(well.property_name),
            property_log=property_log_tvdss,
            inline=inline,
            xline=xline,
            horizon_values={name: float(value) for name, value in horizon_depths.items()},
            metadata=metadata,
        ),
        x=None if well.x is None else float(well.x),
        y=None if well.y is None else float(well.y),
        trajectory=trajectory,
        kb=None if well.kb is None else float(well.kb),
        original_basis_type=well.property_log.basis_type,
        depth_conversion_mode=conversion_mode,
        vertical_assumption_used=vertical_assumption_used,
        input_log_filtered=True,
    )


def build_lfm_depth_model(
    target_layer: TargetLayer,
    wells: list[LfmDepthWell],
    *,
    survey: Optional[SurveyContext] = None,
    boundary_extension_samples: int = 50,
    n_slices: int = 32,
    variogram: str = "spherical",
    exact: bool = True,
    nugget: float = 0.0,
    filter_cutoff_wavelength_m: float = 150.0,
    filter_order: int = 5,
    filter_buffer_meters: Optional[float] = None,
    filter_buffer_mode: str = "reflect",
    post_slice_smoothing: bool = False,
) -> LfmDepthModelResult:
    """构建 TVDSS 深度域层位约束低频模型。"""
    if not wells:
        raise ValueError("wells must contain at least one LfmDepthWell.")
    if n_slices < 2:
        raise ValueError(f"n_slices must be >= 2, got {n_slices}.")
    if boundary_extension_samples < 0:
        raise ValueError(f"boundary_extension_samples must be >= 0, got {boundary_extension_samples}.")
    extension_samples = int(boundary_extension_samples)

    sample_domain = str(target_layer.geometry.get("sample_domain", "")).lower()
    sample_unit = str(target_layer.geometry.get("sample_unit", "")).lower()
    if sample_domain != "depth":
        raise ValueError("lfm_depth.py only supports TargetLayer geometry in depth domain.")
    if sample_unit != "m":
        raise ValueError("lfm_depth.py only supports depth TargetLayer geometry with sample_unit='m'.")

    dz = float(target_layer.geometry["sample_step"])
    if dz <= 0.0:
        raise ValueError(f"target_layer.geometry['sample_step'] must be positive, got {dz}.")

    modeling_target_layer = (
        target_layer.with_boundary_extension(extension_samples) if extension_samples > 0 else target_layer
    )
    prepared_wells = [
        _prepare_well(
            well,
            modeling_target_layer,
            survey,
            dz,
            filter_cutoff_wavelength_m=float(filter_cutoff_wavelength_m),
            filter_order=int(filter_order),
            filter_buffer_meters=filter_buffer_meters,
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
                "depth_conversion_mode": well.depth_conversion_mode,
                "vertical_assumption_used": bool(well.vertical_assumption_used),
                "input_log_filtered": bool(well.input_log_filtered),
            }
        )

    cutoff_wavelength = float(filter_cutoff_wavelength_m)
    metadata = {
        **model_result.metadata,
        "depth_domain": "tvdss",
        "filter_cutoff_wavelength_m": cutoff_wavelength,
        "filter_cutoff_cycles_per_m": 1.0 / cutoff_wavelength,
        "filter_order": int(filter_order),
        "filter_buffer_meters": None if filter_buffer_meters is None else float(filter_buffer_meters),
        "filter_buffer_mode": str(filter_buffer_mode),
    }

    result_wells = [
        LfmDepthWell(
            well_name=well.control.well_name,
            property_name=well.control.property_name,
            property_log=well.control.property_log,
            inline=well.control.inline,
            xline=well.control.xline,
            x=well.x,
            y=well.y,
            trajectory=well.trajectory,
            kb=well.kb,
            metadata={} if well.control.metadata is None else dict(well.control.metadata),
        )
        for well in prepared_wells
    ]

    return LfmDepthModelResult(
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
