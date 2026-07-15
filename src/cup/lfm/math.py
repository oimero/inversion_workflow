"""Shared LFM low-pass and real-XY ordinary kriging primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from cup.seismic.geometry import SampleAxis
from wtie.processing import grid


CONSTANT_RTOL = 1e-7
CONSTANT_ATOL = 1e-10


@dataclass(frozen=True)
class LowpassSpec:
    enabled: bool
    cutoff_cycles_per_axis_unit: float | None = None
    order: int | None = None
    buffer_mode: str | None = None
    buffer_axis_units: float | None = None
    source_parameter: str | None = None


def parse_lowpass_spec(config: Mapping[str, Any] | None, sample_axis: SampleAxis) -> LowpassSpec:
    raw = dict(config or {})
    if "enabled" not in raw or not isinstance(raw["enabled"], bool):
        raise ValueError("filter.enabled must be an explicit YAML boolean.")
    enabled = bool(raw["enabled"])
    time_key = "cutoff_hz"
    depth_key = "cutoff_wavelength_m"
    active_key = time_key if sample_axis.domain == "time" else depth_key
    wrong_key = depth_key if sample_axis.domain == "time" else time_key
    parameter_keys = {time_key, depth_key, "order", "buffer_mode", "buffer_axis_units"}
    if not enabled:
        present = sorted(parameter_keys & set(raw))
        if present:
            raise ValueError(f"Disabled filter must not include parameters: {present}")
        return LowpassSpec(enabled=False)
    if wrong_key in raw or active_key not in raw:
        raise ValueError(f"{sample_axis.domain} filter requires {active_key} and rejects {wrong_key}.")
    unknown = sorted(set(raw) - ({"enabled"} | parameter_keys))
    if unknown:
        raise ValueError(f"Unknown filter keys: {unknown}")
    order = int(raw.get("order"))
    if order <= 0:
        raise ValueError("filter.order must be positive.")
    mode = str(raw.get("buffer_mode") or "").casefold()
    if mode not in {"reflect", "edge", "none"}:
        raise ValueError("filter.buffer_mode must be reflect, edge, or none.")
    buffer_units = float(raw.get("buffer_axis_units"))
    if not np.isfinite(buffer_units) or buffer_units < 0.0:
        raise ValueError("filter.buffer_axis_units must be finite and non-negative.")
    if mode == "none" and buffer_units != 0.0:
        raise ValueError("buffer_mode='none' requires buffer_axis_units=0.")
    raw_cutoff = float(raw[active_key])
    if not np.isfinite(raw_cutoff) or raw_cutoff <= 0.0:
        raise ValueError(f"filter.{active_key} must be finite and positive.")
    cycles = raw_cutoff if sample_axis.domain == "time" else 1.0 / raw_cutoff
    nyquist = 0.5 / float(sample_axis.step)
    if not cycles < nyquist:
        raise ValueError(f"Low-pass cutoff {cycles} must be below Nyquist {nyquist}.")
    return LowpassSpec(
        enabled=True,
        cutoff_cycles_per_axis_unit=cycles,
        order=order,
        buffer_mode=mode,
        buffer_axis_units=buffer_units,
        source_parameter=active_key,
    )


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in changes.reshape(-1, 2)]


def apply_lfm_lowpass(log: grid.Log, spec: LowpassSpec) -> grid.Log:
    """Filter every finite run independently without crossing NaN gaps."""

    values = np.asarray(log.values, dtype=np.float64)
    if not spec.enabled:
        return grid.Log(values.copy(), log.basis.copy(), "twt" if log.is_twt else "tvdss", name=log.name, unit=log.unit)
    from scipy.signal import butter, sosfiltfilt

    step = float(log.sampling_rate)
    sos = butter(
        int(spec.order),
        float(spec.cutoff_cycles_per_axis_unit),
        btype="lowpass",
        fs=1.0 / step,
        output="sos",
    )
    output = np.full(values.shape, np.nan, dtype=np.float64)
    pad_samples = int(np.ceil(float(spec.buffer_axis_units) / step))
    min_length = 3 * (2 * sos.shape[0] + 1)
    for start, stop in _true_runs(np.isfinite(values)):
        segment = values[start:stop]
        if segment.size < min_length:
            raise ValueError(
                f"Finite log run [{start}:{stop}] has {segment.size} samples; filter requires at least {min_length}."
            )
        if spec.buffer_mode == "none" or pad_samples == 0:
            padded = segment
            crop = slice(None)
        else:
            mode = "reflect" if spec.buffer_mode == "reflect" else "edge"
            if mode == "reflect" and pad_samples >= segment.size:
                raise ValueError("Reflect buffer is not smaller than the finite log run.")
            padded = np.pad(segment, pad_samples, mode=mode)
            crop = slice(pad_samples, pad_samples + segment.size)
        output[start:stop] = sosfiltfilt(sos, padded, padtype=None)[crop]
    return grid.Log(output, log.basis.copy(), "twt" if log.is_twt else "tvdss", name=log.name, unit=log.unit)


def values_are_constant(values: np.ndarray) -> bool:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return bool(finite.size > 0 and np.allclose(finite, finite[0], rtol=CONSTANT_RTOL, atol=CONSTANT_ATOL))


def _nearest_neighbor_distance(x: np.ndarray, y: np.ndarray) -> float:
    coords = np.column_stack([x, y])
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    if np.any(~np.isfinite(nearest)) or np.any(nearest <= 0.0):
        raise ValueError("XY kriging controls contain duplicate or degenerate positions.")
    return float(np.median(nearest))


def ordinary_krige_xy(
    *,
    control_x_m: np.ndarray,
    control_y_m: np.ndarray,
    control_values: np.ndarray,
    output_x_m: np.ndarray,
    output_y_m: np.ndarray,
    nominal_bin_spacing_m: float,
    variogram: str,
    exact: bool,
    nugget: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Ordinary kriging in physical XY metres with fully audited parameters."""

    x = np.asarray(control_x_m, dtype=np.float64)
    y = np.asarray(control_y_m, dtype=np.float64)
    values = np.asarray(control_values, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x, y, values = x[finite], y[finite], values[finite]
    if values.size < 2:
        raise ValueError("XY ordinary kriging requires at least two finite controls.")
    if values_are_constant(values):
        raise ValueError("XY ordinary kriging rejects two-or-more near-constant controls.")
    sill = float(np.var(values))
    if not np.isfinite(sill) or sill <= 0.0:
        raise ValueError("XY ordinary kriging sill is non-positive.")
    nearest = _nearest_neighbor_distance(x, y)
    nominal = float(nominal_bin_spacing_m)
    if not np.isfinite(nominal) or nominal <= 0.0:
        raise ValueError("nominal_bin_spacing_m must be finite and positive.")
    range_m = float(max(nearest, nominal))
    model_names = {"spherical": "Spherical", "exponential": "Exponential", "gaussian": "Gaussian"}
    key = str(variogram).casefold()
    if key not in model_names:
        raise ValueError(f"Unsupported variogram: {variogram!r}.")
    nugget_value = float(nugget)
    if not np.isfinite(nugget_value) or nugget_value < 0.0:
        raise ValueError("Kriging nugget must be finite and non-negative.")
    import gstools as gs

    model = getattr(gs, model_names[key])(dim=2, var=sill, len_scale=range_m, nugget=nugget_value)
    krige = gs.krige.Ordinary(model, cond_pos=[x, y], cond_val=values, exact=bool(exact))
    out_x = np.asarray(output_x_m, dtype=np.float64)
    out_y = np.asarray(output_y_m, dtype=np.float64)
    if out_x.shape != out_y.shape or np.any(~np.isfinite(out_x)) or np.any(~np.isfinite(out_y)):
        raise ValueError("Output XY coordinates must be matching and finite.")
    field, variance = krige([out_x.ravel(), out_y.ravel()], mesh_type="unstructured", return_var=True)
    metadata = {
        "variogram": key,
        "exact": bool(exact),
        "nugget": nugget_value,
        "sill": sill,
        "range_m": range_m,
        "nearest_neighbor_distance_median_m": nearest,
        "nominal_bin_spacing_m": nominal,
        "constant_rtol": CONSTANT_RTOL,
        "constant_atol": CONSTANT_ATOL,
        "n_controls": int(values.size),
    }
    return (
        np.asarray(field, dtype=np.float64).reshape(out_x.shape),
        np.asarray(variance, dtype=np.float64).reshape(out_x.shape),
        metadata,
    )


__all__ = [
    "CONSTANT_ATOL",
    "CONSTANT_RTOL",
    "LowpassSpec",
    "apply_lfm_lowpass",
    "ordinary_krige_xy",
    "parse_lowpass_spec",
    "values_are_constant",
]
