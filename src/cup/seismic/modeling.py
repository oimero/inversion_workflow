"""cup.seismic.modeling: 采样域无关的层位约束建模核心。

本模块基于层位约束与井点控制构建三维体，支持层段比例切片、
Kriging 插值、边界外延与方差输出。

边界说明
--------
- 不负责时深转换、测井预处理或井轨迹解析。
- 不包含可视化或反演训练流程。

核心公开对象
------------
1. WellControl: 单井控制点描述。
2. ZoneSliceModel: 单层段比例切片建模结果。
3. LayerConstrainedModelResult: 建模输出与覆盖统计。
4. build_layer_constrained_model: 层位约束建模入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import gstools as gs
import numpy as np

from cup.seismic.target_layer import TargetLayer
from wtie.processing import grid

POST_SLICE_SMOOTHING_KERNEL = np.asarray([0.1, 0.2, 0.4, 0.2, 0.1], dtype=float)


@dataclass
class WellControl:
    """参与层位约束建模的单井控制点。

    ``property_log`` 必须已经位于目标采样域；本模块不负责时深转换、
    滤波、survey 坐标解析或原始井输入适配。
    """

    well_name: str
    property_name: str
    property_log: grid.Log
    inline: float
    xline: float
    horizon_values: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ZoneSliceModel:
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


@dataclass
class LayerConstrainedModelResult:
    """层位约束建模通用结果。"""

    volume: np.ndarray
    variance_volume: np.ndarray
    geometry: Dict[str, Any]
    ilines: np.ndarray
    xlines: np.ndarray
    samples: np.ndarray
    metadata: Dict[str, Any]
    well_controls: list[WellControl]
    zone_models: list[ZoneSliceModel]
    coverage_stats: Dict[str, Any]


def _normalize_line_coordinates(coords: np.ndarray, line_min: float, line_step: float) -> np.ndarray:
    """归一化 inline/xline 坐标到步长尺度。"""
    if line_step <= 0:
        raise ValueError(f"line_step must be positive, got {line_step}.")
    return (np.asarray(coords, dtype=float) - float(line_min)) / float(line_step)


def _nearest_neighbor_range(inlines: np.ndarray, xlines: np.ndarray) -> float:
    """估计控制点最近邻距离的中位数。"""
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
    """在规范化坐标上执行单切片 Kriging。"""
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


def _interpolate_log_at_sample(log: grid.Log, sample_value: float) -> float:
    """在测井曲线中插值采样位置值。"""
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
        return float(values[0]) if np.isclose(float(sample_value), float(basis[0]), atol=1e-8) else float("nan")

    if sample_value < basis[0] or sample_value > basis[-1]:
        return float("nan")

    return float(np.interp(sample_value, basis, values))


def _normalize_well_control(well: WellControl) -> WellControl:
    """规范化井控制点数值类型与元信息。"""
    horizon_values = {name: float(value) for name, value in well.horizon_values.items()}
    return WellControl(
        well_name=well.well_name,
        property_name=str(well.property_name),
        property_log=well.property_log,
        inline=float(well.inline),
        xline=float(well.xline),
        horizon_values=horizon_values,
        metadata={} if well.metadata is None else dict(well.metadata),
    )


def _validate_required_horizon_values(well_controls: list[WellControl], target_layer: TargetLayer) -> None:
    """校验井控制点包含所需层位值。"""
    for well in well_controls:
        missing_horizon_names = [name for name in target_layer.horizon_names if name not in well.horizon_values]
        if missing_horizon_names:
            raise ValueError(f"well '{well.well_name}' is missing horizon_values for: {missing_horizon_names}.")

        for horizon_name in target_layer.horizon_names:
            horizon_value = float(well.horizon_values[horizon_name])
            if not np.isfinite(horizon_value):
                raise ValueError(f"well '{well.well_name}' has non-finite horizon value for '{horizon_name}'.")


def _add_boundary_extension_horizon_values(
    well_controls: list[WellControl],
    modeling_target_layer: TargetLayer,
) -> list[WellControl]:
    """补全边界外延层位值。"""
    completed_controls = []
    for well in well_controls:
        horizon_values = dict(well.horizon_values)
        missing_horizon_names = [name for name in modeling_target_layer.horizon_names if name not in horizon_values]
        if missing_horizon_names:
            resolved_values = modeling_target_layer.get_interpretation_values_at_location(
                float(well.inline),
                float(well.xline),
            )
            for name in missing_horizon_names:
                horizon_values[name] = float(resolved_values[name])

        completed_controls.append(
            WellControl(
                well_name=well.well_name,
                property_name=well.property_name,
                property_log=well.property_log,
                inline=well.inline,
                xline=well.xline,
                horizon_values=horizon_values,
                metadata={} if well.metadata is None else dict(well.metadata),
            )
        )

    return completed_controls


def _validate_well_controls(
    well_controls: list[WellControl],
    target_layer: TargetLayer,
    modeling_target_layer: TargetLayer,
) -> list[WellControl]:
    """校验并规范化井控制点集合。"""
    if not well_controls:
        raise ValueError("well_controls must contain at least one WellControl.")

    prepared_controls = [_normalize_well_control(well) for well in well_controls]
    property_names = {well.property_name for well in prepared_controls}
    if len(property_names) != 1:
        raise ValueError(f"All well controls must share the same property_name, got {sorted(property_names)}.")

    for well in prepared_controls:
        if not np.isfinite(well.inline) or not np.isfinite(well.xline):
            raise ValueError(f"well '{well.well_name}' must provide finite inline/xline coordinates.")

    _validate_required_horizon_values(prepared_controls, target_layer)
    prepared_controls = _add_boundary_extension_horizon_values(prepared_controls, modeling_target_layer)
    _validate_required_horizon_values(prepared_controls, modeling_target_layer)
    return prepared_controls


def _fill_missing_slices_with_neighbors(
    slice_u: np.ndarray,
    slice_values: np.ndarray,
    slice_variance: np.ndarray,
    slice_modes: list[str],
    valid_slice_mask: np.ndarray,
) -> None:
    """用相邻切片补全缺失切片。"""
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
    well_controls: list[WellControl],
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
    well_top_values: np.ndarray,
    well_bottom_values: np.ndarray,
    variogram: str,
    exact: bool,
    nugget: float,
) -> ZoneSliceModel:
    """构建单层段比例切片模型。"""
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

        for well_idx, well in enumerate(well_controls):
            top_value = float(well_top_values[well_idx])
            bottom_value = float(well_bottom_values[well_idx])
            if not np.isfinite(top_value) or not np.isfinite(bottom_value) or bottom_value <= top_value:
                continue

            sample_value = (1.0 - u) * top_value + u * bottom_value
            value = _interpolate_log_at_sample(well.property_log, sample_value)
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

    return ZoneSliceModel(
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
    zone_model: ZoneSliceModel,
    *,
    source_zone_key: str,
    source_slice_index: int,
    source_values: np.ndarray,
    source_variance: np.ndarray,
) -> None:
    """用相邻层段边界切片填充空层段。"""
    zone_model.slice_values[:] = source_values
    zone_model.slice_variance[:] = source_variance
    zone_model.slice_modes[:] = ["adjacent_zone_boundary_fill"] * zone_model.slice_values.shape[0]
    zone_model.fallback_source_zone = source_zone_key
    zone_model.fallback_source_slice_index = int(source_slice_index)


def _write_zone_model_to_volume(
    zone_model: ZoneSliceModel,
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


def build_layer_constrained_model(
    target_layer: TargetLayer,
    well_controls: list[WellControl],
    *,
    boundary_extension_samples: int = 0,
    n_slices: int = 32,
    variogram: str = "spherical",
    exact: bool = True,
    nugget: float = 0.0,
    post_slice_smoothing: bool = False,
) -> LayerConstrainedModelResult:
    """基于层位约束与井点控制构建三维采样体。

    Parameters
    ----------
    target_layer : TargetLayer
        目标层位对象。
    well_controls : list[WellControl]
        井控制点列表，property_log 已位于目标采样域。
    boundary_extension_samples : int, default=0
        上下边界外延的样点数。
    n_slices : int, default=32
        每个层段的比例切片数量。
    variogram : str, default="spherical"
        变差函数模型名称："spherical"、"exponential" 或 "gaussian"。
    exact : bool, default=True
        Kriging 是否精确通过控制点。
    nugget : float, default=0.0
        块金效应参数。
    post_slice_smoothing : bool, default=False
        是否对切片结果进行轻度平滑。

    Returns
    -------
    LayerConstrainedModelResult
        建模输出与覆盖统计。

    Raises
    ------
    ValueError
        当输入参数或井控制点非法时。

    Notes
    -----
    Kriging 在 grid-index normalized 坐标上执行，输出方差体与建模元信息；
    这里的变差距离不是 XY 米制距离。
    """
    if n_slices < 2:
        raise ValueError(f"n_slices must be >= 2, got {n_slices}.")
    if boundary_extension_samples < 0:
        raise ValueError(f"boundary_extension_samples must be >= 0, got {boundary_extension_samples}.")
    extension_samples = int(boundary_extension_samples)

    modeling_target_layer = (
        target_layer.with_boundary_extension(extension_samples) if extension_samples > 0 else target_layer
    )
    prepared_controls = _validate_well_controls(
        list(well_controls),
        target_layer=target_layer,
        modeling_target_layer=modeling_target_layer,
    )
    property_name = prepared_controls[0].property_name

    ilines = target_layer.ilines.astype(float, copy=False)
    xlines = target_layer.xlines.astype(float, copy=False)
    inline_min = float(target_layer.geometry["inline_min"])
    inline_step = float(target_layer.geometry["inline_step"])
    xline_min = float(target_layer.geometry["xline_min"])
    xline_step = float(target_layer.geometry["xline_step"])
    kriging_ilines = _normalize_line_coordinates(ilines, line_min=inline_min, line_step=inline_step)
    kriging_xlines = _normalize_line_coordinates(xlines, line_min=xline_min, line_step=xline_step)
    samples = target_layer.samples
    n_il, n_xl, n_sample = ilines.size, xlines.size, samples.size

    volume = np.full((n_il, n_xl, n_sample), np.nan, dtype=np.float32)
    variance_volume = np.full((n_il, n_xl, n_sample), np.nan, dtype=np.float32)
    slice_u = np.linspace(0.0, 1.0, n_slices, dtype=float)
    coverage_stats: Dict[str, Any] = {
        "property_name": property_name,
        "wells": {},
        "zones": {},
    }

    for well in prepared_controls:
        coverage_stats["wells"][well.well_name] = {
            "inline": float(well.inline),
            "xline": float(well.xline),
            "horizon_values": dict(well.horizon_values),
            "modeled_sample_min": float(well.horizon_values[modeling_target_layer.horizon_names[0]]),
            "modeled_sample_max": float(well.horizon_values[modeling_target_layer.horizon_names[-1]]),
        }

    base_zone_models: list[ZoneSliceModel] = []
    for top_name, bottom_name in target_layer.iter_zones():
        top_grid, bottom_grid = modeling_target_layer.get_zone_sample_index_grids((top_name, bottom_name))
        zone_model = _build_zone_slice_model(
            prepared_controls,
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
            well_top_values=np.asarray([well.horizon_values[top_name] for well in prepared_controls], dtype=float),
            well_bottom_values=np.asarray(
                [well.horizon_values[bottom_name] for well in prepared_controls], dtype=float
            ),
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

    modeled_zones: list[ZoneSliceModel] = []
    if extension_samples > 0:
        top_extension_zone = (modeling_target_layer.horizon_names[0], target_layer.horizon_names[0])
        top_grid, bottom_grid = modeling_target_layer.get_zone_sample_index_grids(top_extension_zone)
        top_extension_model = _build_zone_slice_model(
            prepared_controls,
            ilines,
            xlines,
            kriging_ilines,
            kriging_xlines,
            slice_u,
            inline_min=inline_min,
            inline_step=inline_step,
            xline_min=xline_min,
            xline_step=xline_step,
            top_label=top_extension_zone[0],
            bottom_label=top_extension_zone[1],
            top_grid=top_grid,
            bottom_grid=bottom_grid,
            well_top_values=np.asarray(
                [well.horizon_values[top_extension_zone[0]] for well in prepared_controls], dtype=float
            ),
            well_bottom_values=np.asarray(
                [well.horizon_values[top_extension_zone[1]] for well in prepared_controls], dtype=float
            ),
            variogram=variogram,
            exact=exact,
            nugget=nugget,
        )
        if np.any(top_extension_model.valid_slice_mask):
            _fill_missing_slices_with_neighbors(
                slice_u=slice_u,
                slice_values=top_extension_model.slice_values,
                slice_variance=top_extension_model.slice_variance,
                slice_modes=top_extension_model.slice_modes,
                valid_slice_mask=top_extension_model.valid_slice_mask,
            )
        else:
            reference_zone = base_zone_models[0]
            _fill_zone_with_adjacent_boundary(
                top_extension_model,
                source_zone_key=f"{reference_zone.top_label}->{reference_zone.bottom_label}",
                source_slice_index=0,
                source_values=reference_zone.slice_values[0],
                source_variance=reference_zone.slice_variance[0],
            )
        if post_slice_smoothing:
            top_extension_model.slice_values, top_extension_model.slice_variance = _apply_post_slice_smoothing(
                slice_values=top_extension_model.slice_values,
                slice_variance=top_extension_model.slice_variance,
            )
        modeled_zones.append(top_extension_model)

    modeled_zones.extend(base_zone_models)

    if extension_samples > 0:
        bottom_extension_zone = (target_layer.horizon_names[-1], modeling_target_layer.horizon_names[-1])
        top_grid, bottom_grid = modeling_target_layer.get_zone_sample_index_grids(bottom_extension_zone)
        bottom_extension_model = _build_zone_slice_model(
            prepared_controls,
            ilines,
            xlines,
            kriging_ilines,
            kriging_xlines,
            slice_u,
            inline_min=inline_min,
            inline_step=inline_step,
            xline_min=xline_min,
            xline_step=xline_step,
            top_label=bottom_extension_zone[0],
            bottom_label=bottom_extension_zone[1],
            top_grid=top_grid,
            bottom_grid=bottom_grid,
            well_top_values=np.asarray(
                [well.horizon_values[bottom_extension_zone[0]] for well in prepared_controls], dtype=float
            ),
            well_bottom_values=np.asarray(
                [well.horizon_values[bottom_extension_zone[1]] for well in prepared_controls], dtype=float
            ),
            variogram=variogram,
            exact=exact,
            nugget=nugget,
        )
        if np.any(bottom_extension_model.valid_slice_mask):
            _fill_missing_slices_with_neighbors(
                slice_u=slice_u,
                slice_values=bottom_extension_model.slice_values,
                slice_variance=bottom_extension_model.slice_variance,
                slice_modes=bottom_extension_model.slice_modes,
                valid_slice_mask=bottom_extension_model.valid_slice_mask,
            )
        else:
            reference_zone = base_zone_models[-1]
            _fill_zone_with_adjacent_boundary(
                bottom_extension_model,
                source_zone_key=f"{reference_zone.top_label}->{reference_zone.bottom_label}",
                source_slice_index=slice_u.size - 1,
                source_values=reference_zone.slice_values[-1],
                source_variance=reference_zone.slice_variance[-1],
            )
        if post_slice_smoothing:
            bottom_extension_model.slice_values, bottom_extension_model.slice_variance = _apply_post_slice_smoothing(
                slice_values=bottom_extension_model.slice_values,
                slice_variance=bottom_extension_model.slice_variance,
            )
        modeled_zones.append(bottom_extension_model)

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
        "coord_system": "inline_xline",
        "kriging_coord_system": "grid_index_normalized",
        "kriging_inline_step": inline_step,
        "kriging_xline_step": xline_step,
        "horizon_names": list(target_layer.horizon_names),
        "zone_names": [[zone.top_label, zone.bottom_label] for zone in modeled_zones],
        "well_names": [well.well_name for well in prepared_controls],
        "post_slice_smoothing": bool(post_slice_smoothing),
        "post_slice_smoothing_kernel": POST_SLICE_SMOOTHING_KERNEL.tolist(),
        "variance_volume_included": True,
    }

    return LayerConstrainedModelResult(
        volume=volume,
        variance_volume=variance_volume,
        geometry=dict(target_layer.geometry),
        ilines=ilines,
        xlines=xlines,
        samples=samples,
        metadata=metadata,
        well_controls=prepared_controls,
        zone_models=modeled_zones,
        coverage_stats=coverage_stats,
    )
