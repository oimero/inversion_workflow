"""Raw Marmousi2 model readers and depth-to-TWT conversion.

The Marmousi2 model files used by this workflow are SEG-Y containers for
spatially sampled model values.  Their binary sample interval and trace
coordinates are not a time axis or a usable geometry contract.  This module
therefore reads only the SEG-Y sample count/format, applies the explicit
spatial strides supplied by the caller, and builds TWT from Vp by integrating
two-way slowness over depth.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
from typing import Any, Mapping

import numpy as np


_TEXT_HEADER_BYTES = 3200
_BINARY_HEADER_BYTES = 400
_TRACE_HEADER_BYTES = 240
_DATA_OFFSET = _TEXT_HEADER_BYTES + _BINARY_HEADER_BYTES
_IBM_FORMAT_CODE = 1
_IEEE_FORMAT_CODE = 5
_IBM_FRACTION_SCALE = float(1 << 24)


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _unit_key(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an explicit unit string.")
    return value.strip().lower()


def _velocity_factor(unit: Any) -> tuple[str, float]:
    key = _unit_key(unit, name="velocity_unit")
    if key == "km/s":
        return key, 1000.0
    if key == "m/s":
        return key, 1.0
    raise ValueError("velocity_unit must be 'km/s' or 'm/s'.")


def _density_factor(unit: Any) -> tuple[str, float]:
    key = _unit_key(unit, name="density_unit")
    if key == "g/cm3":
        return key, 1.0
    if key == "kg/m3":
        return key, 1.0 / 1000.0
    raise ValueError("density_unit must be 'g/cm3' or 'kg/m3'.")


def _path(value: Any, *, name: str) -> Path:
    if isinstance(value, (str, os.PathLike)):
        result = Path(value)
    else:
        raise TypeError(f"{name} must be a path-like value.")
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


@dataclass(frozen=True)
class MarmousiModels:
    """Spatially sampled Marmousi2 velocity and density models.

    Arrays use ``[lateral, depth]`` order.  ``vp_mps`` is in metres/second,
    ``rho_gcc`` is in grams/cm³, ``x_m`` is the explicit lateral coordinate,
    and ``depth_m`` is the explicit depth coordinate.  The TWT origin is not
    stored here; conversion records that it is relative to ``depth_m[0]``.
    """

    vp_mps: np.ndarray
    rho_gcc: np.ndarray
    x_m: np.ndarray
    depth_m: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        vp = np.asarray(self.vp_mps, dtype=np.float64)
        rho = np.asarray(self.rho_gcc, dtype=np.float64)
        x = np.asarray(self.x_m, dtype=np.float64)
        depth = np.asarray(self.depth_m, dtype=np.float64)
        if vp.ndim != 2 or rho.ndim != 2:
            raise ValueError("vp_mps and rho_gcc must be two-dimensional [lateral, depth] arrays.")
        if vp.shape != rho.shape:
            raise ValueError(f"vp_mps and rho_gcc shapes must match, got {vp.shape} and {rho.shape}.")
        if vp.shape[0] < 1 or vp.shape[1] < 2:
            raise ValueError("Models require at least one lateral sample and two depth samples.")
        if x.ndim != 1 or x.size != vp.shape[0]:
            raise ValueError("x_m length must match the lateral model dimension.")
        if depth.ndim != 1 or depth.size != vp.shape[1]:
            raise ValueError("depth_m length must match the depth model dimension.")
        for name, values in (("vp_mps", vp), ("rho_gcc", rho), ("x_m", x), ("depth_m", depth)):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values.")
        if np.any(vp <= 0.0) or np.any(rho <= 0.0):
            raise ValueError("vp_mps and rho_gcc must be strictly positive.")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("x_m must be strictly increasing.")
        if np.any(np.diff(depth) <= 0.0):
            raise ValueError("depth_m must be strictly increasing.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "vp_mps", vp)
        object.__setattr__(self, "rho_gcc", rho)
        object.__setattr__(self, "x_m", x)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TimeImpedanceModel:
    """Marmousi2 log acoustic impedance on a common TWT grid."""

    log_ai: np.ndarray
    twt_s: np.ndarray
    native_twt_s: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        log_ai = np.asarray(self.log_ai, dtype=np.float64)
        twt = np.asarray(self.twt_s, dtype=np.float64)
        native = np.asarray(self.native_twt_s, dtype=np.float64)
        if log_ai.ndim != 2:
            raise ValueError("log_ai must be a two-dimensional [lateral, time] array.")
        if twt.ndim != 1 or twt.size != log_ai.shape[1]:
            raise ValueError("twt_s length must match the time dimension of log_ai.")
        if native.ndim != 2 or native.shape[0] != log_ai.shape[0]:
            raise ValueError("native_twt_s must be two-dimensional with matching lateral support.")
        if not np.all(np.isfinite(log_ai)) or not np.all(np.isfinite(twt)) or not np.all(np.isfinite(native)):
            raise ValueError("TimeImpedanceModel arrays must contain only finite values.")
        if twt.size == 0 or np.any(np.diff(twt) <= 0.0):
            if twt.size != 1:
                raise ValueError("twt_s must be strictly increasing.")
        if native.shape[1] < 2 or np.any(np.diff(native, axis=1) <= 0.0):
            raise ValueError("native_twt_s must be strictly increasing along depth for every trace.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "log_ai", log_ai)
        object.__setattr__(self, "twt_s", twt)
        object.__setattr__(self, "native_twt_s", native)
        object.__setattr__(self, "metadata", dict(self.metadata))


def _read_segy_header(path: Path) -> tuple[int, int, int, int]:
    file_size = path.stat().st_size
    if file_size < _DATA_OFFSET:
        raise ValueError(f"{path} is shorter than the SEG-Y text and binary headers.")
    with path.open("rb") as handle:
        handle.seek(_TEXT_HEADER_BYTES)
        binary = handle.read(_BINARY_HEADER_BYTES)
    if len(binary) != _BINARY_HEADER_BYTES:
        raise ValueError(f"{path} has an incomplete SEG-Y binary header.")
    # SEG-Y binary header offsets are relative to byte 3201 (one-based):
    # sample interval 3217, samples/trace 3221, format code 3225.
    binary_dt_us = struct.unpack_from(">H", binary, 16)[0]
    ns = struct.unpack_from(">H", binary, 20)[0]
    format_code = struct.unpack_from(">H", binary, 24)[0]
    if ns < 1:
        raise ValueError(f"{path} declares an invalid SEG-Y sample count: {ns}.")
    if format_code not in {_IBM_FORMAT_CODE, _IEEE_FORMAT_CODE}:
        raise ValueError(
            f"{path} uses unsupported SEG-Y sample format {format_code}; only IBM(1) and IEEE(5) are supported."
        )
    record_bytes = _TRACE_HEADER_BYTES + 4 * ns
    payload_bytes = file_size - _DATA_OFFSET
    if payload_bytes % record_bytes:
        raise ValueError(
            f"{path} has {payload_bytes} payload bytes, not an integer number of {record_bytes}-byte traces."
        )
    trace_count = payload_bytes // record_bytes
    if trace_count < 1:
        raise ValueError(f"{path} contains no complete SEG-Y traces.")
    return int(ns), int(format_code), int(binary_dt_us), int(trace_count)


def _decode_ibm32(raw_words: np.ndarray) -> np.ndarray:
    words = np.asarray(raw_words, dtype=np.dtype(">u4")).astype(np.uint32, copy=False)
    fraction = (words & np.uint32(0x00FFFFFF)).astype(np.float64)
    exponent = ((words >> np.uint32(24)) & np.uint32(0x7F)).astype(np.int16) - np.int16(64)
    sign = np.where((words & np.uint32(0x80000000)) != 0, -1.0, 1.0)
    result = sign * (fraction / _IBM_FRACTION_SCALE) * np.power(16.0, exponent.astype(np.float64))
    result[fraction == 0.0] = 0.0
    return result


def _decode_selected_samples(raw: np.ndarray, *, format_code: int) -> np.ndarray:
    if format_code == _IBM_FORMAT_CODE:
        return _decode_ibm32(raw)
    if format_code == _IEEE_FORMAT_CODE:
        return np.asarray(raw, dtype=np.dtype(">f4")).astype(np.float64)
    raise ValueError(f"Unsupported SEG-Y format code: {format_code}.")


def _read_selected_segy(
    path: Path,
    *,
    trace_stride: int,
    depth_stride: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    ns, format_code, binary_dt_us, trace_count = _read_segy_header(path)
    trace_indices = np.arange(0, trace_count, trace_stride, dtype=np.int64)
    depth_indices = np.arange(0, ns, depth_stride, dtype=np.int64)
    sample_dtype = np.dtype(">u4") if format_code == _IBM_FORMAT_CODE else np.dtype(">f4")
    record_dtype = np.dtype([("header", "V240"), ("samples", sample_dtype, (ns,))])
    mapped = np.memmap(
        path,
        mode="r",
        offset=_DATA_OFFSET,
        dtype=record_dtype,
        shape=(trace_count,),
    )
    try:
        # Index both axes before decoding.  The complete file remains a
        # memmap and only the requested trace/depth subset is materialized.
        selected_words = np.asarray(mapped["samples"][trace_indices][:, depth_indices])
    finally:
        del mapped
    values = _decode_selected_samples(selected_words, format_code=format_code)
    if values.ndim != 2 or values.shape != (trace_indices.size, depth_indices.size):
        raise ValueError(f"Decoded {path} does not have the expected selected 2-D shape.")
    metadata = {
        "path": str(path.resolve()),
        "format_code": format_code,
        "format_name": "IBM32" if format_code == _IBM_FORMAT_CODE else "IEEE32",
        "binary_dt_us": binary_dt_us,
        "binary_dt_used": False,
        "binary_ns": ns,
        "trace_count": trace_count,
        "selected_trace_count": int(trace_indices.size),
        "selected_depth_count": int(depth_indices.size),
        "record_bytes": int(_TRACE_HEADER_BYTES + 4 * ns),
        "trace_indices_first_last": [int(trace_indices[0]), int(trace_indices[-1])],
        "depth_indices_first_last": [int(depth_indices[0]), int(depth_indices[-1])],
    }
    return values, metadata


def _validate_selected_values(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must decode to a two-dimensional [lateral, depth] array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite samples.")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} contains non-positive samples.")
    return array


def read_marmousi_models(
    vp_path: str | os.PathLike[str],
    density_path: str | os.PathLike[str],
    *,
    trace_stride: int = 8,
    depth_stride: int = 1,
    dx_m: float = 1.249,
    dz_m: float = 1.249,
    x_start_m: float = 0.0,
    z_start_m: float = 0.0,
    velocity_unit: str = "km/s",
    density_unit: str = "g/cm3",
) -> MarmousiModels:
    """Read explicit spatial Marmousi2 Vp and density SEG-Y model files.

    SEG-Y binary ``dt`` and trace coordinate headers are recorded for
    provenance only.  Spatial coordinates are always generated from the
    explicit ``dx_m``, ``dz_m``, ``x_start_m`` and ``z_start_m`` arguments.
    """

    trace_stride = _positive_int(trace_stride, name="trace_stride")
    depth_stride = _positive_int(depth_stride, name="depth_stride")
    dx_m = _positive_float(dx_m, name="dx_m")
    dz_m = _positive_float(dz_m, name="dz_m")
    x_start_m = _finite_float(x_start_m, name="x_start_m")
    z_start_m = _finite_float(z_start_m, name="z_start_m")
    velocity_key, velocity_factor = _velocity_factor(velocity_unit)
    density_key, density_factor = _density_factor(density_unit)
    vp_source = _path(vp_path, name="vp_path")
    density_source = _path(density_path, name="density_path")

    vp_native, vp_info = _read_selected_segy(
        vp_source,
        trace_stride=trace_stride,
        depth_stride=depth_stride,
    )
    rho_native, rho_info = _read_selected_segy(
        density_source,
        trace_stride=trace_stride,
        depth_stride=depth_stride,
    )
    if vp_info["trace_count"] != rho_info["trace_count"] or vp_info["binary_ns"] != rho_info["binary_ns"]:
        raise ValueError("Vp and density SEG-Y files must have matching trace counts and binary sample counts.")
    if vp_native.shape != rho_native.shape:
        raise ValueError(f"Vp and density selected shapes must match, got {vp_native.shape} and {rho_native.shape}.")

    vp_mps = _validate_selected_values(vp_native * velocity_factor, name="vp_mps")
    rho_gcc = _validate_selected_values(rho_native * density_factor, name="rho_gcc")
    trace_indices = np.arange(0, vp_info["trace_count"], trace_stride, dtype=np.int64)
    depth_indices = np.arange(0, vp_info["binary_ns"], depth_stride, dtype=np.int64)
    x_m = x_start_m + trace_indices.astype(np.float64) * dx_m
    depth_m = z_start_m + depth_indices.astype(np.float64) * dz_m

    metadata: dict[str, Any] = {
        "schema": "marmousi2_spatial_models_v1",
        "source_paths": {
            "vp": str(vp_source.resolve()),
            "density": str(density_source.resolve()),
        },
        "source_files": {"vp": vp_info, "density": rho_info},
        "native_units": {"velocity": velocity_key, "density": density_key},
        "output_units": {"velocity": "m/s", "density": "g/cm3"},
        "conversion_factors": {"velocity_to_mps": velocity_factor, "density_to_gcc": density_factor},
        "shape": [int(vp_mps.shape[0]), int(vp_mps.shape[1])],
        "strides": {"trace_stride": trace_stride, "depth_stride": depth_stride},
        "spatial_sampling": {
            "dx_m": dx_m,
            "dz_m": dz_m,
            "x_start_m": x_start_m,
            "z_start_m": z_start_m,
            "coordinate_source": "explicit_function_arguments",
            "segy_trace_coordinates_used": False,
            "segy_binary_dt_used": False,
        },
        "time_axis": {
            "source": "not_segy_binary_dt",
            "binary_dt_us": {"vp": vp_info["binary_dt_us"], "density": rho_info["binary_dt_us"]},
        },
        "twt_reference": {
            "reference": "relative_to_depth_m[0]",
            "depth_m[0]": float(depth_m[0]),
            "overburden_added_s": 0.0,
            "nonzero_depth_start_is_relative": bool(not np.isclose(depth_m[0], 0.0)),
        },
    }
    return MarmousiModels(vp_mps, rho_gcc, x_m, depth_m, metadata)


def _native_twt_from_vp(vp_mps: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    dz = np.diff(depth_m)
    interval_slowness = 0.5 * (1.0 / vp_mps[:, :-1] + 1.0 / vp_mps[:, 1:])
    return np.concatenate(
        (np.zeros((vp_mps.shape[0], 1), dtype=np.float64), 2.0 * np.cumsum(dz[None, :] * interval_slowness, axis=1)),
        axis=1,
    )


def resample_models_to_time(
    models: MarmousiModels,
    *,
    dt_s: float = 0.004,
    time_start_s: float = 0.0,
    time_end_s: float | None = None,
) -> TimeImpedanceModel:
    """Integrate native TWT and interpolate log-AI without smoothing.

    Every trace is required to support the complete returned common time grid.
    When ``time_end_s`` is omitted, the common shortest bottom TWT is floored
    to ``dt_s``.  No extrapolation is performed.
    """

    if not isinstance(models, MarmousiModels):
        raise TypeError("models must be a MarmousiModels instance.")
    dt_s = _positive_float(dt_s, name="dt_s")
    time_start_s = _finite_float(time_start_s, name="time_start_s")
    requested_end = None if time_end_s is None else _finite_float(time_end_s, name="time_end_s")
    native_twt = _native_twt_from_vp(models.vp_mps, models.depth_m)
    common_start = float(np.max(native_twt[:, 0]))
    common_bottom = float(np.min(native_twt[:, -1]))
    if time_start_s < common_start - max(1e-12, dt_s * 1e-10):
        raise ValueError("time_start_s is before the common native TWT support; extrapolation is forbidden.")
    if requested_end is None:
        actual_end = float(np.floor((common_bottom + dt_s * 1e-10) / dt_s) * dt_s)
        end_policy = "common_shortest_bottom_floor_to_dt"
    else:
        if requested_end < time_start_s:
            raise ValueError("time_end_s must be greater than or equal to time_start_s.")
        actual_end = requested_end
        end_policy = "explicit_upper_bound"
    if actual_end > common_bottom + max(1e-12, dt_s * 1e-10):
        raise ValueError("time_end_s exceeds the common native TWT support; extrapolation is forbidden.")
    if actual_end < time_start_s:
        raise ValueError("The requested time interval contains no supported samples.")

    sample_count = int(np.floor((actual_end - time_start_s) / dt_s + 1e-10)) + 1
    if sample_count < 1:
        raise ValueError("The requested time interval contains no samples.")
    twt_s = time_start_s + dt_s * np.arange(sample_count, dtype=np.float64)
    if twt_s[-1] > common_bottom + max(1e-12, dt_s * 1e-10):
        raise ValueError("The generated time grid exceeds the common native TWT support.")
    ai = models.vp_mps * models.rho_gcc
    log_ai = np.empty((models.vp_mps.shape[0], twt_s.size), dtype=np.float64)
    for row in range(models.vp_mps.shape[0]):
        # np.interp is deliberately used directly: no Gaussian or other
        # smoothing is applied while moving the native model onto TWT.
        log_ai[row] = np.log(np.interp(twt_s, native_twt[row], ai[row]))
    metadata = dict(models.metadata)
    metadata.update(
        {
            "schema": "marmousi2_time_impedance_v1",
            "time_grid": {
                "dt_s": dt_s,
                "time_start_s": time_start_s,
                "requested_time_end_s": requested_end,
                "time_end_s": float(twt_s[-1]),
                "end_policy": end_policy,
                "extrapolation": False,
            },
            "native_twt": {
                "integration": "two_way_slowness_trapezoid",
                "formula": "TWT=2*integral(dz/vp)",
                "reference": "relative_to_depth_m[0]",
                "overburden_added_s": 0.0,
            },
            "impedance": {
                "definition": "log(vp_mps * rho_gcc)",
                "log_base": "natural",
                "interpolation": "numpy.interp",
                "smoothing": "none",
            },
        }
    )
    return TimeImpedanceModel(log_ai, twt_s, native_twt, metadata)


__all__ = ["MarmousiModels", "TimeImpedanceModel", "read_marmousi_models", "resample_models_to_time"]
