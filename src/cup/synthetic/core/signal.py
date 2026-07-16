"""Shared finite-support numerical primitives for Synthoseis science v2."""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin


TAPS_PER_FACTOR = 32
CUTOFF_OUTPUT_NYQUIST_FRACTION = 0.9
KAISER_BETA = 8.6


def finite_support_fir(factor: int) -> np.ndarray:
    factor = int(factor)
    if factor < 1:
        raise ValueError("oversampling factor must be positive")
    return firwin(
        TAPS_PER_FACTOR * factor + 1,
        CUTOFF_OUTPUT_NYQUIST_FRACTION / factor,
        window=("kaiser", KAISER_BETA),
        scale=True,
    ).astype(np.float64)


def fir_half_width(factor: int, highres_sample_interval: float) -> float:
    return (finite_support_fir(factor).size // 2) * float(highres_sample_interval)


def valid_filter_decimate(
    values: np.ndarray, *, factor: int, taps: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=np.float64)
    kernel = finite_support_fir(factor) if taps is None else np.asarray(taps, dtype=np.float64)
    n_model = (data.shape[-1] - 1) // int(factor) + 1
    output = data[..., :: int(factor)].copy()
    valid = np.zeros(n_model, dtype=bool)
    half = kernel.size // 2
    centers_high = np.arange(n_model, dtype=np.int64) * int(factor)
    supported = (centers_high >= half) & (centers_high < data.shape[-1] - half)
    centers_valid = centers_high[supported] - half
    flat_output = output.reshape((-1, n_model))
    for row_index, row in enumerate(data.reshape((-1, data.shape[-1]))):
        flat_output[row_index, supported] = np.convolve(row, kernel, mode="valid")[centers_valid]
    valid[supported] = True
    return output, valid


def required_context(
    *, projection_fir_half_width: float, forward_input_halo: float,
    observed_decimation_fir_half_width: float, domain_extra_halo: float = 0.0
) -> float:
    return float(max(
        projection_fir_half_width,
        forward_input_halo + observed_decimation_fir_half_width,
        domain_extra_halo,
    ))


__all__ = [
    "CUTOFF_OUTPUT_NYQUIST_FRACTION", "KAISER_BETA", "TAPS_PER_FACTOR",
    "finite_support_fir", "fir_half_width", "required_context", "valid_filter_decimate",
]
