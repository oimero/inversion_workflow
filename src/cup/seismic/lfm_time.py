"""时间域层位约束低频模型构建。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from cup.seismic.process import TargetLayer
from cup.seismic.survey import SurveyContext
from wtie.processing import grid
from wtie.processing.spectral import apply_butter_lowpass_filter

POST_SLICE_SMOOTHING_KERNEL = np.asarray([0.1, 0.2, 0.4, 0.2, 0.1], dtype=float)


@dataclass
class LfmTimeWell:
    """单井时间域低频模型输入。"""

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
    """时间域低频模型构建结果。"""

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

    try:
        import gstools as gs
    except ImportError as exc:
        raise ImportError(
            "GSTools is required for kriging-based low-frequency model building. "
            "Please install 'gstools' from environment.yml."
        ) from exc

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
) -> grid.Log:
    """对 TWT 域单条曲线做低通滤波。"""
    if not log.is_twt:
        raise ValueError("lowpass_twt_log only supports TWT-domain logs.")

    fs = 1.0 / log.sampling_rate
    nyquist = 0.5 * fs
    if cutoff_hz <= 0.0 or cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz must be within (0, {nyquist}), got {cutoff_hz}.")

    filtered = apply_butter_lowpass_filter(
        log.values.astype(np.float64),
        highcut=cutoff_hz,
        fs=fs,
        order=order,
        zero_phase=True,
    )
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
        return grid.convert_log_from_md_to_twt(log, table, None, dt), "md_to_twt_via_md_table"

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
) -> _PreparedLfmTimeWell:
    inline, xline = _resolve_well_position(well, survey)
    horizon_times = target_layer.get_interpretation_values_at_location(inline, xline)
    required_twt_min = float(horizon_times[target_layer.horizon_names[0]])
    required_twt_max = float(horizon_times[target_layer.horizon_names[-1]])
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
    n_slices: int = 32,
    variogram: str = "spherical",
    exact: bool = True,
    nugget: float = 0.0,
    filter_cutoff_hz: float = 10.0,
    filter_order: int = 5,
    post_slice_smoothing: bool = True,
) -> LfmTimeModelResult:
    """构建时间域层位约束低频模型。"""
    if not wells:
        raise ValueError("wells must contain at least one LfmTimeWell.")
    if n_slices < 2:
        raise ValueError(f"n_slices must be >= 2, got {n_slices}.")

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
            filter_cutoff_hz=float(filter_cutoff_hz),
            filter_order=int(filter_order),
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
    samples = _build_sample_axis(target_layer.geometry)
    n_il, n_xl, n_sample = ilines.size, xlines.size, samples.size

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
        }

    for top_name, bottom_name in target_layer.iter_zones():
        top_grid, bottom_grid = target_layer.get_zone_sample_index_grids((top_name, bottom_name))
        zone_slice_values = np.full((n_slices, n_il, n_xl), np.nan, dtype=float)
        zone_slice_variance = np.full((n_slices, n_il, n_xl), np.nan, dtype=float)
        slice_control_counts = np.zeros(n_slices, dtype=int)
        slice_modes = [""] * n_slices
        valid_slice_mask = np.zeros(n_slices, dtype=bool)

        for slice_idx, u in enumerate(slice_u):
            control_inlines = []
            control_xlines = []
            control_values = []

            for well in prepared_wells:
                t_top = well.horizon_times[top_name]
                t_bottom = well.horizon_times[bottom_name]
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
                zone_slice_values[slice_idx] = np.full((n_il, n_xl), float(control_values_arr[0]), dtype=float)
                zone_slice_variance[slice_idx] = np.zeros((n_il, n_xl), dtype=float)
                slice_modes[slice_idx] = "single_well_constant"
            else:
                range_hint = _nearest_neighbor_range(control_inlines_arr, control_xlines_arr)
                field, variance = _krige_slice_on_line_domain(
                    control_inlines_arr,
                    control_xlines_arr,
                    control_values_arr,
                    ilines,
                    xlines,
                    range_hint=range_hint,
                    variogram=variogram,
                    exact=exact,
                    nugget=nugget,
                )
                zone_slice_values[slice_idx] = field
                zone_slice_variance[slice_idx] = variance
                slice_modes[slice_idx] = "kriging"

        if not np.any(valid_slice_mask):
            raise ValueError(f"Zone '{top_name}' -> '{bottom_name}' has no valid control values on any slice.")

        _fill_missing_slices_with_neighbors(
            slice_u=slice_u,
            slice_values=zone_slice_values,
            slice_variance=zone_slice_variance,
            slice_modes=slice_modes,
            valid_slice_mask=valid_slice_mask,
        )
        if post_slice_smoothing:
            zone_slice_values, zone_slice_variance = _apply_post_slice_smoothing(
                slice_values=zone_slice_values,
                slice_variance=zone_slice_variance,
            )

        zone_key = f"{top_name}->{bottom_name}"
        coverage_stats["zones"][zone_key] = {
            "slice_control_counts": slice_control_counts.tolist(),
            "slice_modes": slice_modes,
        }

        for i_il in range(n_il):
            for i_xl in range(n_xl):
                t_top = top_grid[i_il, i_xl]
                t_bottom = bottom_grid[i_il, i_xl]
                if not np.isfinite(t_top) or not np.isfinite(t_bottom) or t_bottom <= t_top:
                    continue

                idx_top = max(0, int(np.round(t_top)))
                idx_bottom = min(n_sample - 1, int(np.round(t_bottom)))
                if idx_bottom < idx_top:
                    continue

                local_indices = np.arange(idx_top, idx_bottom + 1, dtype=float)
                denom = float(t_bottom - t_top)
                if denom <= 0.0:
                    continue

                u_local = np.clip((local_indices - float(t_top)) / denom, 0.0, 1.0)
                local_values = np.interp(u_local, slice_u, zone_slice_values[:, i_il, i_xl])
                local_variance = np.interp(u_local, slice_u, zone_slice_variance[:, i_il, i_xl])
                volume[i_il, i_xl, idx_top : idx_bottom + 1] = local_values.astype(np.float32)
                variance_volume[i_il, i_xl, idx_top : idx_bottom + 1] = local_variance.astype(np.float32)

    for i_il in range(n_il):
        for i_xl in range(n_xl):
            trace = volume[i_il, i_xl]
            finite = np.isfinite(trace)
            if not np.any(finite):
                continue
            valid_indices = np.flatnonzero(finite)
            first_idx = int(valid_indices[0])
            last_idx = int(valid_indices[-1])
            if first_idx > 0:
                trace[:first_idx] = trace[first_idx]
                variance_volume[i_il, i_xl, :first_idx] = variance_volume[i_il, i_xl, first_idx]
            if last_idx < n_sample - 1:
                trace[last_idx + 1 :] = trace[last_idx]
                variance_volume[i_il, i_xl, last_idx + 1 :] = variance_volume[i_il, i_xl, last_idx]

    metadata = {
        "backend": "gstools",
        "slice_mode": "proportional",
        "property_name": property_name,
        "variogram": variogram,
        "exact": bool(exact),
        "nugget": float(nugget),
        "n_slices": int(n_slices),
        "filter_cutoff_hz": float(filter_cutoff_hz),
        "filter_order": int(filter_order),
        "coord_system": "inline_xline",
        "horizon_names": list(target_layer.horizon_names),
        "zone_names": [list(zone) for zone in target_layer.iter_zones()],
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
