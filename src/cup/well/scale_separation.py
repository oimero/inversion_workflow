"""Physical-distance scale separation for well log-AI curves."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


BODY_SMOOTHING_FWHM_M = 15.0
_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
_GAUSSIAN_TRUNCATE_SIGMA = 8.0


def _validate_numpy(
    values: Any,
    coordinates_m: Any,
    fwhm_m: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    array = np.asarray(values)
    coordinates = np.asarray(coordinates_m)
    if not np.issubdtype(array.dtype, np.floating) or array.ndim < 1:
        raise TypeError("values must be a floating array with shape [..., samples].")
    if not np.issubdtype(coordinates.dtype, np.floating):
        raise TypeError("coordinates_m must have a floating dtype.")
    shared_coordinates = coordinates.ndim == 1
    if shared_coordinates:
        coordinates = np.broadcast_to(coordinates, array.shape)
    if coordinates.shape != array.shape:
        raise ValueError("coordinates_m must have shape (samples,) or match values.")
    if np.any(~np.isfinite(array)) or np.any(~np.isfinite(coordinates)):
        raise ValueError("values and coordinates_m must contain only finite values.")
    flat_coordinates = coordinates.reshape((-1, coordinates.shape[-1]))
    if np.any(np.diff(flat_coordinates, axis=-1) <= 0.0):
        raise ValueError("coordinates_m must be strictly increasing along every trace.")
    width = float(fwhm_m)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("fwhm_m must be finite and positive.")
    return array, coordinates, width, shared_coordinates


def gaussian_smooth_numpy(
    values: Any,
    coordinates_m: Any,
    *,
    fwhm_m: float = BODY_SMOOTHING_FWHM_M,
) -> np.ndarray:
    """Apply a normalized Gaussian defined by physical coordinate distance."""

    array, coordinates, width, shared_coordinates = _validate_numpy(
        values,
        coordinates_m,
        fwhm_m,
    )
    sigma = width * _FWHM_TO_SIGMA
    flat_values = array.reshape((-1, array.shape[-1]))
    flat_coordinates = coordinates.reshape((-1, coordinates.shape[-1]))
    output = np.empty(flat_values.shape, dtype=np.result_type(array.dtype, np.float64))
    radius = _GAUSSIAN_TRUNCATE_SIGMA * sigma
    if shared_coordinates:
        axis = flat_coordinates[0]
        starts = np.searchsorted(axis, axis - radius, side="left")
        stops = np.searchsorted(axis, axis + radius, side="right")
        for index, (start, stop) in enumerate(zip(starts, stops)):
            weights = np.exp(-0.5 * np.square((axis[start:stop] - axis[index]) / sigma))
            weights /= np.sum(weights)
            output[:, index] = flat_values[:, start:stop] @ weights
    else:
        for trace_index, (trace_values, axis) in enumerate(zip(flat_values, flat_coordinates)):
            starts = np.searchsorted(axis, axis - radius, side="left")
            stops = np.searchsorted(axis, axis + radius, side="right")
            for sample_index, (start, stop) in enumerate(zip(starts, stops)):
                weights = np.exp(
                    -0.5 * np.square((axis[start:stop] - axis[sample_index]) / sigma)
                )
                weights /= np.sum(weights)
                output[trace_index, sample_index] = trace_values[start:stop] @ weights
    return output.reshape(array.shape).astype(array.dtype, copy=False)


def gaussian_smooth_finite_runs_numpy(
    values: Any,
    coordinates_m: Any,
    *,
    fwhm_m: float = BODY_SMOOTHING_FWHM_M,
) -> np.ndarray:
    """Smooth contiguous finite runs without crossing missing-log gaps."""

    array = np.asarray(values)
    coordinates = np.asarray(coordinates_m)
    if array.ndim != 1 or coordinates.shape != array.shape:
        raise ValueError("finite-run smoothing requires matching one-dimensional arrays.")
    if not np.issubdtype(array.dtype, np.floating) or not np.issubdtype(
        coordinates.dtype,
        np.floating,
    ):
        raise TypeError("values and coordinates_m must have floating dtypes.")
    if np.any(~np.isfinite(coordinates)) or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("coordinates_m must be finite and strictly increasing.")
    output = np.full(array.shape, np.nan, dtype=array.dtype)
    padded = np.r_[False, np.isfinite(array), False]
    for start, stop in np.flatnonzero(padded[1:] != padded[:-1]).reshape((-1, 2)):
        output[start:stop] = gaussian_smooth_numpy(
            array[start:stop],
            coordinates[start:stop],
            fwhm_m=fwhm_m,
        )
    return output


__all__ = [
    "BODY_SMOOTHING_FWHM_M",
    "gaussian_smooth_finite_runs_numpy",
    "gaussian_smooth_numpy",
]
