"""Domain-neutral trend and proportional-kriging LFM builders."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cup.lfm.math import (
    apply_lfm_lowpass,
    ordinary_krige_xy,
    parse_lowpass_spec,
    values_are_constant,
)
from cup.lfm.types import LfmContext, LfmVariantResult
from cup.well.real_field_controls import WellControl, WellControlSet


def _required_mapping(config: Mapping[str, Any], key: str, *, path: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}.{key} must be a mapping.")
    return dict(value)


def _spatial_config(config: Mapping[str, Any]) -> dict[str, Any]:
    spatial = _required_mapping(config, "spatial", path="baseline")
    required = {"variogram", "exact", "nugget"}
    missing = sorted(required - set(spatial))
    if missing:
        raise ValueError(f"baseline.spatial is missing keys: {missing}")
    if set(spatial) != required:
        raise ValueError("baseline.spatial must contain exactly variogram/exact/nugget.")
    if not isinstance(spatial["exact"], bool):
        raise ValueError("baseline.spatial.exact must be a YAML boolean.")
    return spatial


def _filtered_controls(controls: WellControlSet, config: Mapping[str, Any]) -> tuple[WellControl, ...]:
    spec = parse_lowpass_spec(_required_mapping(config, "filter", path="baseline"), controls.sample_axis)
    out = []
    for control in controls.controls:
        filtered = apply_lfm_lowpass(control.log_ai, spec)
        valid = control.valid_mask & np.isfinite(filtered.values)
        values = np.asarray(filtered.values, dtype=np.float64).copy()
        positions = [
            np.asarray(value, dtype=np.float64).copy()
            for value in (
                control.inline_by_sample,
                control.xline_by_sample,
                control.x_m_by_sample,
                control.y_m_by_sample,
            )
        ]
        for value in [values, *positions]:
            value[~valid] = np.nan
        filtered.series.iloc[:] = values
        out.append(
            replace(
                control,
                log_ai=filtered,
                inline_by_sample=positions[0],
                xline_by_sample=positions[1],
                x_m_by_sample=positions[2],
                y_m_by_sample=positions[3],
                valid_mask=valid,
            )
        )
    return tuple(out)


def _surface_for_output(context: LfmContext, horizon_name: str) -> np.ndarray:
    surface = context.target_zone.get_horizon_surface(horizon_name)
    output = context.output_geometry
    if output.is_section:
        return np.asarray(
            [surface.sample_at_line(float(il), float(xl)).value for il, xl in zip(output.ilines, output.xlines)],
            dtype=np.float64,
        )
    il_indices = []
    for il in output.ilines:
        index = context.line_geometry.inline_axis.index_of_line(float(il))
        if not np.isclose(index, round(index), rtol=0.0, atol=1e-8):
            raise ValueError(f"Output inline is not on the survey axis: {il}")
        il_indices.append(int(round(index)))
    xl_indices = []
    for xl in output.xlines:
        index = context.line_geometry.xline_axis.index_of_line(float(xl))
        if not np.isclose(index, round(index), rtol=0.0, atol=1e-8):
            raise ValueError(f"Output xline is not on the survey axis: {xl}")
        xl_indices.append(int(round(index)))
    return surface.values[np.ix_(np.asarray(il_indices), np.asarray(xl_indices))].astype(np.float64)


def _surface_along_control(context: LfmContext, horizon_name: str, control: WellControl) -> np.ndarray:
    surface = context.target_zone.get_horizon_surface(horizon_name)
    out = np.full(control.valid_mask.shape, np.nan, dtype=np.float64)
    for index in np.flatnonzero(
        control.valid_mask & np.isfinite(control.inline_by_sample) & np.isfinite(control.xline_by_sample)
    ):
        try:
            out[index] = surface.sample_at_line(
                float(control.inline_by_sample[index]), float(control.xline_by_sample[index])
            ).value
        except ValueError:
            continue
    return out


def _target_mask(context: LfmContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = context.target_zone.horizon_names
    top = _surface_for_output(context, names[0])
    bottom = _surface_for_output(context, names[-1])
    trace_valid = _target_trace_mask(context)
    samples = context.output_geometry.samples
    if context.output_geometry.is_section:
        u = (samples[None, :] - top[:, None]) / (bottom[:, None] - top[:, None])
        valid = trace_valid[:, None] & np.isfinite(u)
    else:
        u = (samples[None, None, :] - top[:, :, None]) / (bottom[:, :, None] - top[:, :, None])
        valid = trace_valid[:, :, None] & np.isfinite(u)
    return u, valid & (u >= 0.0) & (u <= 1.0), bottom - top


def _target_trace_mask(context: LfmContext) -> np.ndarray:
    """Require every ordered TargetZone horizon on an output trace."""

    surfaces = [
        _surface_for_output(context, horizon_name)
        for horizon_name in context.target_zone.horizon_names
    ]
    valid = np.ones(context.output_geometry.lateral_shape, dtype=bool)
    for top, bottom in zip(surfaces[:-1], surfaces[1:]):
        valid &= np.isfinite(top) & np.isfinite(bottom) & (bottom > top)
    return valid


def _output_xy(context: LfmContext) -> tuple[np.ndarray, np.ndarray]:
    return context.output_geometry.x_m, context.output_geometry.y_m


def _distance_to_controls(context: LfmContext, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out_x, out_y = _output_xy(context)
    distance = np.full(out_x.shape, np.inf, dtype=np.float64)
    for cx, cy in zip(x, y):
        distance = np.minimum(distance, np.hypot(out_x - float(cx), out_y - float(cy)))
    return distance


def _huber_fit(x: np.ndarray, y: np.ndarray, *, f_scale: float) -> tuple[float, float]:
    from scipy.optimize import least_squares

    if not np.isfinite(f_scale) or f_scale <= 0.0:
        raise ValueError("huber_f_scale_log_ai must be finite and positive.")
    initial = np.asarray([np.median(y), 0.0], dtype=np.float64)
    fit = least_squares(lambda parameters: parameters[0] + parameters[1] * x - y, initial, loss="huber", f_scale=f_scale)
    if not fit.success or np.any(~np.isfinite(fit.x)):
        raise ValueError(f"Huber trend fit failed: {fit.message}")
    return float(fit.x[0]), float(fit.x[1])


def _field_from_controls(
    *,
    context: LfmContext,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    spatial: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if values.size == 1:
        shape = context.output_geometry.lateral_shape
        return (
            np.full(shape, float(values[0]), dtype=np.float64),
            np.zeros(shape, dtype=np.float64),
            {"mode": "single_control_constant", "n_controls": 1, "sill": 0.0, "range_m": None},
        )
    if values_are_constant(values):
        raise ValueError("Two-or-more controls form a degenerate constant field.")
    out_x, out_y = _output_xy(context)
    field, variance, metadata = ordinary_krige_xy(
        control_x_m=x,
        control_y_m=y,
        control_values=values,
        output_x_m=out_x,
        output_y_m=out_y,
        nominal_bin_spacing_m=context.line_geometry.bin_spacing_m()["nominal_bin_spacing_m"],
        variogram=str(spatial["variogram"]),
        exact=bool(spatial["exact"]),
        nugget=float(spatial["nugget"]),
    )
    metadata["mode"] = "kriging"
    return field, variance, metadata


class TrendBuilder:
    method = "trend"

    def build(
        self,
        *,
        baseline_id: str,
        config: Mapping[str, Any],
        controls: WellControlSet,
        context: LfmContext,
    ) -> LfmVariantResult:
        if str(config.get("method") or "") != self.method:
            raise ValueError("TrendBuilder received a non-trend baseline config.")
        if set(config) != {"method", "filter", "fit", "spatial"}:
            raise ValueError("trend baseline must contain exactly method/filter/fit/spatial.")
        fit_config = _required_mapping(config, "fit", path=f"baselines.{baseline_id}")
        if set(fit_config) != {"min_valid_samples_per_well", "huber_f_scale_log_ai"}:
            raise ValueError("trend.fit must explicitly contain only min_valid_samples_per_well and huber_f_scale_log_ai.")
        min_valid = int(fit_config["min_valid_samples_per_well"])
        if min_valid < 2:
            raise ValueError("min_valid_samples_per_well must be at least 2 for an a/b trend fit.")
        filtered = _filtered_controls(controls, config)
        names = context.target_zone.horizon_names
        fit_rows: list[dict[str, Any]] = []
        for control in filtered:
            top = _surface_along_control(context, names[0], control)
            bottom = _surface_along_control(context, names[-1], control)
            u = (context.sample_axis.values - top) / (bottom - top)
            valid = control.valid_mask & np.isfinite(u) & (bottom > top) & (u >= 0.0) & (u <= 1.0)
            n_valid = int(np.count_nonzero(valid))
            if n_valid < min_valid:
                fit_rows.append(
                    {
                        "well_name": control.well_name,
                        "status": "rejected",
                        "reason": f"insufficient_valid_samples:{n_valid}<{min_valid}",
                        "n_valid_samples": n_valid,
                        "sampling_mode": control.sampling_mode,
                    }
                )
                continue
            x = 2.0 * u[valid] - 1.0
            y = np.asarray(control.log_ai.values, dtype=np.float64)[valid]
            try:
                a, b = _huber_fit(x, y, f_scale=float(fit_config["huber_f_scale_log_ai"]))
            except ValueError as exc:
                fit_rows.append(
                    {
                        "well_name": control.well_name,
                        "status": "rejected",
                        "reason": f"{type(exc).__name__}:{exc}",
                        "n_valid_samples": n_valid,
                        "sampling_mode": control.sampling_mode,
                    }
                )
                continue
            residual = y - (a + b * x)
            fit_rows.append(
                {
                    "well_name": control.well_name,
                    "status": "ok",
                    "reason": "",
                    "n_valid_samples": n_valid,
                    "a": a,
                    "b": b,
                    "representative_x_m": float(np.mean(control.x_m_by_sample[valid])),
                    "representative_y_m": float(np.mean(control.y_m_by_sample[valid])),
                    "residual_rms": float(np.sqrt(np.mean(residual**2))),
                    "sampling_mode": control.sampling_mode,
                }
            )
        fit_qc = pd.DataFrame.from_records(fit_rows)
        fit_frame = fit_qc[fit_qc["status"].eq("ok")].copy() if not fit_qc.empty else fit_qc
        if fit_frame.empty:
            raise ValueError("Trend baseline has no well satisfying its fit controls.")
        control_x = fit_frame["representative_x_m"].to_numpy(dtype=np.float64)
        control_y = fit_frame["representative_y_m"].to_numpy(dtype=np.float64)
        spatial = _spatial_config(config)
        method_fields: dict[str, np.ndarray] = {}
        parameter_rows = []
        for parameter in ("a", "b"):
            values = fit_frame[parameter].to_numpy(dtype=np.float64)
            field, variance, metadata = _field_from_controls(
                context=context, x=control_x, y=control_y, values=values, spatial=spatial
            )
            method_fields[f"{parameter}_field"] = field.astype(np.float32)
            method_fields[f"{parameter}_variance"] = variance.astype(np.float32)
            parameter_rows.append({"parameter": parameter, **metadata})
        distance = _distance_to_controls(context, control_x, control_y)
        method_fields["distance_to_control_m"] = distance.astype(np.float32)
        u, mask, _thickness = _target_mask(context)
        a_field = method_fields["a_field"].astype(np.float64)
        b_field = method_fields["b_field"].astype(np.float64)
        if context.output_geometry.is_section:
            log_ai = a_field[:, None] + b_field[:, None] * (2.0 * u - 1.0)
        else:
            log_ai = a_field[:, :, None] + b_field[:, :, None] * (2.0 * u - 1.0)
        log_ai = np.where(mask, log_ai, np.nan)
        result = LfmVariantResult(
            log_ai=log_ai,
            valid_mask_model=mask,
            baseline_id=baseline_id,
            baseline_method=self.method,
            method_fields=method_fields,
            qc_tables={
                "trend_well_fit": fit_qc,
                "trend_parameter_model": pd.DataFrame.from_records(parameter_rows),
            },
            metadata={
                "filter": dict(config["filter"]),
                "n_candidate_wells": int(len(fit_qc)),
                "n_fit_wells": int(len(fit_frame)),
                "n_rejected_wells": int(len(fit_qc) - len(fit_frame)),
            },
        )
        result.validate(context)
        return result


def _control_zone_coordinates(
    *, control: WellControl, context: LfmContext, top_name: str, bottom_name: str
) -> tuple[np.ndarray, np.ndarray]:
    top = _surface_along_control(context, top_name, control)
    bottom = _surface_along_control(context, bottom_name, control)
    u = (context.sample_axis.values - top) / (bottom - top)
    valid = control.valid_mask & np.isfinite(u) & (bottom > top) & (u >= 0.0) & (u <= 1.0)
    indices = np.flatnonzero(valid)
    if indices.size < 2:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    u_values = u[indices]
    if np.any(np.diff(u_values) <= 0.0):
        raise ValueError(f"{control.well_name}: relative stratigraphic coordinate is not strictly increasing.")
    return indices, u_values


class ProportionalKrigingBuilder:
    method = "proportional_kriging"

    def build(
        self,
        *,
        baseline_id: str,
        config: Mapping[str, Any],
        controls: WellControlSet,
        context: LfmContext,
    ) -> LfmVariantResult:
        if str(config.get("method") or "") != self.method:
            raise ValueError("ProportionalKrigingBuilder received the wrong method config.")
        if set(config) != {"method", "filter", "n_slices", "spatial"}:
            raise ValueError("proportional_kriging baseline must contain exactly method/filter/n_slices/spatial.")
        n_slices = int(config.get("n_slices"))
        if n_slices < 2:
            raise ValueError("proportional_kriging.n_slices must be >= 2.")
        filtered = _filtered_controls(controls, config)
        spatial = _spatial_config(config)
        slice_u = np.linspace(0.0, 1.0, n_slices, dtype=np.float64)
        output_shape = context.output_geometry.volume_shape
        volume = np.full(output_shape, np.nan, dtype=np.float64)
        variance_volume = np.full(output_shape, np.nan, dtype=np.float64)
        valid_mask = np.zeros(output_shape, dtype=bool)
        target_trace_valid = _target_trace_mask(context)
        slice_rows: list[dict[str, Any]] = []
        names = context.target_zone.horizon_names

        for zone_index, (top_name, bottom_name) in enumerate(zip(names[:-1], names[1:])):
            slice_fields = np.full((n_slices,) + context.output_geometry.lateral_shape, np.nan, dtype=np.float64)
            slice_variance = np.full_like(slice_fields, np.nan)
            modes = ["" for _ in range(n_slices)]
            valid_slices = np.zeros(n_slices, dtype=bool)
            row_by_slice: dict[int, dict[str, Any]] = {}
            for slice_index, target_u in enumerate(slice_u):
                values: list[float] = []
                x_values: list[float] = []
                y_values: list[float] = []
                well_names: list[str] = []
                for control in filtered:
                    indices, u_values = _control_zone_coordinates(
                        control=control, context=context, top_name=top_name, bottom_name=bottom_name
                    )
                    if indices.size == 0 or target_u < u_values[0] or target_u > u_values[-1]:
                        continue
                    values.append(float(np.interp(target_u, u_values, control.log_ai.values[indices])))
                    x_values.append(float(np.interp(target_u, u_values, control.x_m_by_sample[indices])))
                    y_values.append(float(np.interp(target_u, u_values, control.y_m_by_sample[indices])))
                    well_names.append(control.well_name)
                count = len(values)
                row = {
                    "zone_index": zone_index,
                    "top_horizon": top_name,
                    "bottom_horizon": bottom_name,
                    "slice_index": slice_index,
                    "slice_u": float(target_u),
                    "original_control_count": count,
                    "control_wells": ";".join(well_names),
                    "original_mode": "missing" if count == 0 else "single_control_constant" if count == 1 else "kriging",
                    "final_mode": "",
                    "lower_source_slice": "",
                    "upper_source_slice": "",
                    "upper_weight": "",
                }
                if count == 0:
                    row_by_slice[slice_index] = row
                    continue
                array_values = np.asarray(values, dtype=np.float64)
                if count == 1:
                    field = np.full(context.output_geometry.lateral_shape, array_values[0], dtype=np.float64)
                    variance = np.zeros_like(field)
                    metadata = {"mode": "single_control_constant", "sill": 0.0, "range_m": None}
                else:
                    if values_are_constant(array_values):
                        raise ValueError(
                            f"Zone {top_name}->{bottom_name} slice {slice_index} has degenerate constant controls."
                        )
                    out_x, out_y = _output_xy(context)
                    field, variance, metadata = ordinary_krige_xy(
                        control_x_m=np.asarray(x_values),
                        control_y_m=np.asarray(y_values),
                        control_values=array_values,
                        output_x_m=out_x,
                        output_y_m=out_y,
                        nominal_bin_spacing_m=context.line_geometry.bin_spacing_m()["nominal_bin_spacing_m"],
                        variogram=str(spatial["variogram"]),
                        exact=bool(spatial["exact"]),
                        nugget=float(spatial["nugget"]),
                    )
                    metadata["mode"] = "kriging"
                slice_fields[slice_index] = field
                slice_variance[slice_index] = variance
                modes[slice_index] = str(metadata["mode"])
                valid_slices[slice_index] = True
                row.update({"final_mode": modes[slice_index], **{f"kriging_{k}": v for k, v in metadata.items() if k != "mode"}})
                row_by_slice[slice_index] = row
            valid_indices = np.flatnonzero(valid_slices)
            if valid_indices.size == 0:
                raise ValueError(f"Zone {top_name}->{bottom_name} has no valid proportional slices.")
            for slice_index in np.flatnonzero(~valid_slices):
                lower_candidates = valid_indices[valid_indices < slice_index]
                upper_candidates = valid_indices[valid_indices > slice_index]
                lower = int(lower_candidates[-1]) if lower_candidates.size else None
                upper = int(upper_candidates[0]) if upper_candidates.size else None
                if lower is None:
                    lower = upper
                if upper is None:
                    upper = lower
                assert lower is not None and upper is not None
                weight = 0.0 if lower == upper else float((slice_index - lower) / (upper - lower))
                slice_fields[slice_index] = (1.0 - weight) * slice_fields[lower] + weight * slice_fields[upper]
                slice_variance[slice_index] = (1.0 - weight) * slice_variance[lower] + weight * slice_variance[upper]
                modes[slice_index] = "neighbor_slice_fill"
                row_by_slice[slice_index].update(
                    {
                        "final_mode": "neighbor_slice_fill",
                        "lower_source_slice": lower,
                        "upper_source_slice": upper,
                        "upper_weight": weight,
                    }
                )
            slice_rows.extend(row_by_slice[index] for index in range(n_slices))

            top = _surface_for_output(context, top_name)
            bottom = _surface_for_output(context, bottom_name)
            samples = context.output_geometry.samples
            if context.output_geometry.is_section:
                u_grid = (samples[None, :] - top[:, None]) / (bottom[:, None] - top[:, None])
                zone_valid = target_trace_valid[:, None] & np.isfinite(u_grid) & (u_grid >= 0.0) & (u_grid <= 1.0)
            else:
                u_grid = (samples[None, None, :] - top[:, :, None]) / (bottom[:, :, None] - top[:, :, None])
                zone_valid = target_trace_valid[:, :, None] & np.isfinite(u_grid) & (u_grid >= 0.0) & (u_grid <= 1.0)
            # Invalid horizon traces remain outside the authoritative mask, but
            # their NaN relative coordinates must never become array indices.
            position = np.where(zone_valid, np.clip(u_grid, 0.0, 1.0) * (n_slices - 1), 0.0)
            lower = np.floor(position).astype(np.int64)
            upper = np.ceil(position).astype(np.int64)
            weight = position - lower
            if context.output_geometry.is_section:
                trace_index = np.arange(context.output_geometry.ilines.size)[:, None]
                lower_values = slice_fields[lower, trace_index]
                upper_values = slice_fields[upper, trace_index]
                lower_variance = slice_variance[lower, trace_index]
                upper_variance = slice_variance[upper, trace_index]
            else:
                il_index = np.arange(context.output_geometry.ilines.size)[:, None, None]
                xl_index = np.arange(context.output_geometry.xlines.size)[None, :, None]
                lower_values = slice_fields[lower, il_index, xl_index]
                upper_values = slice_fields[upper, il_index, xl_index]
                lower_variance = slice_variance[lower, il_index, xl_index]
                upper_variance = slice_variance[upper, il_index, xl_index]
            zone_values = (1.0 - weight) * lower_values + weight * upper_values
            zone_variance = (1.0 - weight) * lower_variance + weight * upper_variance
            volume[zone_valid] = zone_values[zone_valid]
            variance_volume[zone_valid] = zone_variance[zone_valid]
            valid_mask[zone_valid] = True

        volume[~valid_mask] = np.nan
        variance_volume[~valid_mask] = np.nan
        result = LfmVariantResult(
            log_ai=volume,
            valid_mask_model=valid_mask,
            baseline_id=baseline_id,
            baseline_method=self.method,
            method_fields={
                "kriging_variance": variance_volume.astype(np.float32),
                "slice_u": slice_u.astype(np.float64),
            },
            qc_tables={"proportional_slice_qc": pd.DataFrame.from_records(slice_rows)},
            metadata={"filter": dict(config["filter"]), "n_slices": n_slices},
        )
        result.validate(context)
        return result


BUILDERS = {"trend": TrendBuilder(), "proportional_kriging": ProportionalKrigingBuilder()}


__all__ = ["BUILDERS", "ProportionalKrigingBuilder", "TrendBuilder"]
