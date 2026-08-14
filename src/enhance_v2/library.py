"""Build a finite-run, paired body/residual dictionary from well controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cup.seismic.geometry import SampleAxis
from cup.well.scale_separation import gaussian_smooth_numpy

from .contracts import DictionaryAtom, ResidualTextureLibrary, ScaleContract
from .keys import BodyKeyEncoder, compute_feature_scales, compute_zone_temperature_bases


@dataclass(frozen=True)
class _NativeControlView:
    well_name: str
    coordinates: np.ndarray
    values: np.ndarray
    valid_mask: np.ndarray
    sample_domain: str
    sample_unit: str
    depth_basis: str | None


def _as_native_view(control: Any, *, fallback_name: str | None = None) -> _NativeControlView:
    native = getattr(control, "native", None)
    if native is None and isinstance(control, Mapping):
        native = control.get("native", control)
    source = native if native is not None else control

    def get_value(*names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(source, Mapping) and name in source:
                return source[name]
            if hasattr(source, name):
                return getattr(source, name)
        return default

    name = get_value("well_name", "name", default=fallback_name)
    if name is None:
        raise ValueError("Each residual dictionary control requires a well_name.")
    coordinates = get_value("coordinates", "native_coordinates", "samples", "axis")
    values = get_value("full_log_ai", "native_full_log_ai", "log_ai", "values")
    valid = get_value("valid_mask", "native_valid_mask")
    if coordinates is None or values is None:
        raise ValueError(f"{name}: native control requires coordinates and full_log_ai values.")
    if hasattr(values, "values") and not isinstance(values, np.ndarray):
        values = values.values
    coordinates = np.asarray(coordinates, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if valid is None:
        valid_array = np.isfinite(values)
    else:
        valid_array = np.asarray(valid, dtype=bool)
    if coordinates.ndim != 1 or values.shape != coordinates.shape or valid_array.shape != coordinates.shape:
        raise ValueError(f"{name}: native coordinates, values and valid_mask must be matching 1-D arrays.")
    if coordinates.size < 2 or np.any(~np.isfinite(coordinates)) or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError(f"{name}: native coordinates must be finite and strictly increasing.")
    if not np.array_equal(valid_array, np.isfinite(values)):
        raise ValueError(f"{name}: valid_mask must exactly describe finite native values.")

    sample_axis = getattr(control, "sample_axis", None)
    sample_domain = get_value("sample_domain", default=getattr(sample_axis, "domain", None))
    sample_unit = get_value("sample_unit", default=getattr(sample_axis, "unit", None))
    depth_basis = get_value("depth_basis", default=getattr(sample_axis, "depth_basis", None))
    if sample_domain is None:
        sample_domain = "depth"
    if sample_unit is None:
        sample_unit = "m" if str(sample_domain).casefold() == "depth" else "s"
    return _NativeControlView(
        well_name=str(name),
        coordinates=coordinates,
        values=values,
        valid_mask=valid_array,
        sample_domain=str(sample_domain).casefold(),
        sample_unit=str(sample_unit).casefold(),
        depth_basis=None if depth_basis in (None, "") else str(depth_basis).casefold(),
    )


def _iter_controls(well_controls: Any) -> tuple[_NativeControlView, ...]:
    mapping_names: list[str | None] = []
    if hasattr(well_controls, "controls"):
        raw_controls = getattr(well_controls, "controls")
    elif isinstance(well_controls, Mapping):
        if "controls" in well_controls:
            raw_controls = well_controls["controls"]
        else:
            raw_controls = tuple(well_controls.values())
            mapping_names = [str(key) for key in well_controls]
    else:
        raw_controls = well_controls
    if raw_controls is None:
        raise TypeError("well_controls must provide an iterable of controls.")
    views: list[_NativeControlView] = []
    for index, control in enumerate(raw_controls):
        fallback = mapping_names[index] if index < len(mapping_names) else None
        views.append(_as_native_view(control, fallback_name=fallback))
    if not views:
        raise ValueError("well_controls contains no controls.")
    names = [item.well_name.casefold() for item in views]
    if len(names) != len(set(names)):
        raise ValueError("Residual dictionary well names must be unique case-insensitively.")
    return tuple(views)


def _sample_axis_from_controls(well_controls: Any, views: tuple[_NativeControlView, ...]) -> SampleAxis:
    axis = getattr(well_controls, "sample_axis", None)
    if axis is not None:
        if not isinstance(axis, SampleAxis):
            raise TypeError("well_controls.sample_axis must be SampleAxis.")
        return axis
    domain = views[0].sample_domain
    unit = views[0].sample_unit
    depth_basis = views[0].depth_basis
    if domain == "depth":
        depth_basis = "tvdss" if depth_basis is None else depth_basis
    return SampleAxis(views[0].coordinates, domain, unit, depth_basis)


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    finite = np.asarray(mask, dtype=bool)
    padded = np.r_[False, finite, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in changes.reshape(-1, 2)]


def gaussian_smooth_finite_run(values: np.ndarray, coordinates: np.ndarray, *, fwhm_m: float) -> np.ndarray:
    """Apply the shared body-scale Gaussian to one already finite run."""

    values = np.asarray(values, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 1 or coordinates.shape != values.shape or values.size < 2:
        raise ValueError("gaussian_smooth_finite_run requires matching 1-D arrays with at least two samples.")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(coordinates)) or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("gaussian_smooth_finite_run inputs must be finite and increasing.")
    return np.asarray(
        gaussian_smooth_numpy(values, coordinates, fwhm_m=float(fwhm_m)),
        dtype=np.float64,
    )


def _centers(start: float, end: float, *, half_width: float, spacing: float) -> np.ndarray:
    if not np.isfinite(start) or not np.isfinite(end) or not start < end:
        raise ValueError("A residual window run must have finite start < end.")
    if half_width <= 0.0 or spacing <= 0.0:
        raise ValueError("Residual window half width and spacing must be positive.")
    values = [float(start)]
    current = float(start) + float(spacing)
    while current < float(end) - 1.0e-10:
        values.append(float(current))
        current += float(spacing)
    values.append(float(end))
    return np.asarray(values, dtype=np.float64)


def _zone_intervals(
    views: tuple[_NativeControlView, ...],
    scale_contract: ScaleContract,
) -> dict[str, dict[str, tuple[float, float]]]:
    if scale_contract.zone_intervals_by_well:
        configured = {
            well_name.casefold(): dict(intervals)
            for well_name, intervals in scale_contract.zone_intervals_by_well.items()
        }
        missing = [view.well_name for view in views if view.well_name.casefold() not in configured]
        if missing:
            raise ValueError(
                "Missing per-well zone intervals for residual dictionary controls: "
                + ", ".join(sorted(missing))
            )
        zone_sets = [set(configured[view.well_name.casefold()]) for view in views]
        if any(zone_set != zone_sets[0] for zone_set in zone_sets[1:]):
            raise ValueError("All residual dictionary wells must publish the same zone ids.")
        return {
            view.well_name: configured[view.well_name.casefold()]
            for view in views
        }
    if scale_contract.zone_intervals:
        return {view.well_name: dict(scale_contract.zone_intervals) for view in views}
    finite_coordinates = np.concatenate(
        [view.coordinates[view.valid_mask] for view in views if np.any(view.valid_mask)]
    )
    if finite_coordinates.size == 0:
        raise ValueError("No finite native well samples are available for residual dictionary construction.")
    interval = {"global": (float(np.min(finite_coordinates)), float(np.max(finite_coordinates)))}
    return {view.well_name: dict(interval) for view in views}


def build_residual_library(
    well_controls: Any,
    scale_contract: ScaleContract | Mapping[str, Any] | None = None,
) -> ResidualTextureLibrary:
    """Build a paired finite-run residual dictionary.

    ``well_controls`` is normally a ``cup.well.real_field_controls.WellControlSet``.
    A mapping or iterable of native records with ``coordinates`` and
    ``full_log_ai`` fields is also accepted for small deterministic smoke runs.
    Native filtered values are used directly; the model-axis log is never
    silently substituted when a native layer is available.
    """

    contract = ScaleContract.from_any(scale_contract)
    views = _iter_controls(well_controls)
    sample_axis = _sample_axis_from_controls(well_controls, views)
    if contract.sample_axis is not None and not np.array_equal(contract.sample_axis.values, sample_axis.values):
        raise ValueError("ScaleContract SampleAxis differs from well_controls SampleAxis.")
    if contract.domain is not None and contract.domain != sample_axis.domain:
        raise ValueError("ScaleContract domain differs from well_controls SampleAxis domain.")
    if sample_axis.domain == "depth" and sample_axis.depth_basis != "tvdss":
        raise ValueError("Residual texture transfer depth controls require a TVDSS SampleAxis.")
    expected_unit = "m" if sample_axis.domain == "depth" else "s"
    if sample_axis.unit != expected_unit:
        raise ValueError("Residual texture transfer SampleAxis unit is inconsistent with its domain.")
    for view in views:
        if view.sample_domain != sample_axis.domain or view.sample_unit != sample_axis.unit:
            raise ValueError(f"{view.well_name}: native control domain/unit differs from the common SampleAxis.")
        if sample_axis.domain == "depth" and view.depth_basis not in (None, "tvdss"):
            raise ValueError(f"{view.well_name}: native depth basis must be tvdss.")

    step = float(sample_axis.step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("Residual texture transfer requires a regular SampleAxis with positive step.")
    half_width = contract.resolved_window_half_width_m(step)
    spacing = contract.resolved_window_center_spacing_m(step)
    encoder = BodyKeyEncoder(
        window_half_width_m=half_width,
        profile_samples=contract.normalized_profile_samples,
        denominator_floor=contract.denominator_floor,
    )
    intervals_by_well = _zone_intervals(views, contract)
    atoms: list[DictionaryAtom] = []
    residual_by_well: dict[str, list[np.ndarray]] = {view.well_name: [] for view in views}

    for view in views:
        intervals = intervals_by_well[view.well_name]
        for zone_id, (zone_top, zone_bottom) in intervals.items():
            selected = view.valid_mask & (view.coordinates >= zone_top) & (view.coordinates <= zone_bottom)
            for run_start, run_stop in _finite_runs(selected):
                run_coordinates = view.coordinates[run_start:run_stop]
                run_values = view.values[run_start:run_stop]
                if run_coordinates.size < contract.min_window_samples:
                    continue
                body = gaussian_smooth_finite_run(
                    run_values,
                    run_coordinates,
                    fwhm_m=contract.body_smoothing_fwhm_m,
                )
                residual = run_values - body
                residual_by_well[view.well_name].append(residual)
                centers = _centers(
                    float(run_coordinates[0]),
                    float(run_coordinates[-1]),
                    half_width=half_width,
                    spacing=spacing,
                )
                run_start_coordinate = float(run_coordinates[0])
                run_end_coordinate = float(run_coordinates[-1])
                for center in centers:
                    window_mask = (
                        (run_coordinates >= center - half_width - 1.0e-10)
                        & (run_coordinates <= center + half_width + 1.0e-10)
                    )
                    if np.count_nonzero(window_mask) < contract.min_window_samples:
                        continue
                    window_axis = run_coordinates[window_mask]
                    window_body = body[window_mask]
                    window_residual = residual[window_mask]
                    normalized_position = (center - zone_top) / max(zone_bottom - zone_top, contract.denominator_floor)
                    normalized_position = float(np.clip(normalized_position, 0.0, 1.0))
                    key = encoder.encode(
                        window_body,
                        window_axis,
                        center_m=float(center),
                        zone_id=zone_id,
                        normalized_zone_position=normalized_position,
                    )
                    support = np.isfinite(window_residual)
                    residual_value = np.asarray(window_residual, dtype=np.float64)
                    residual_value[~support] = np.nan
                    source_interval = (
                        max(run_start_coordinate, zone_top),
                        min(run_end_coordinate, zone_bottom),
                    )
                    atoms.append(
                        DictionaryAtom(
                            body_key=key,
                            body_value=window_body,
                            residual_value=residual_value,
                            physical_axis=window_axis,
                            zone_id=zone_id,
                            normalized_zone_position=normalized_position,
                            valid_support=support,
                            source_well=view.well_name,
                            source_interval=source_interval,
                        )
                    )

    if not atoms:
        raise ValueError("No residual dictionary atoms could be built from finite native runs.")
    feature_scales = compute_feature_scales(atoms, scale_floor=contract.feature_scale_floor)
    zone_stats = compute_zone_temperature_bases(atoms, feature_scales)
    for zone_id, stats in zone_stats.items():
        interval_by_well = {
            well_name: list(intervals[zone_id])
            for well_name, intervals in intervals_by_well.items()
        }
        tops = [float(interval[0]) for interval in interval_by_well.values()]
        bottoms = [float(interval[1]) for interval in interval_by_well.values()]
        stats["interval_by_well"] = interval_by_well
        stats["interval_envelope"] = [min(tops), max(bottoms)]
        stats["temperature_multiplier_default"] = 1.0
        stats["temperature_default"] = float(stats["temperature_base"])
    source_rms: dict[str, float] = {}
    for well_name, arrays in residual_by_well.items():
        if arrays:
            values = np.concatenate(arrays)
            source_rms[well_name] = float(np.sqrt(np.mean(np.square(values))))
        else:
            source_rms[well_name] = 0.0
    return ResidualTextureLibrary(
        sample_axis=sample_axis,
        atoms=tuple(atoms),
        feature_scales=feature_scales,
        zone_stats=zone_stats,
        scale_contract=contract,
        profile_axis=encoder.profile_axis,
        source_well_residual_rms=source_rms,
    )


__all__ = ["build_residual_library", "gaussian_smooth_finite_run"]
