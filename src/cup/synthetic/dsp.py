"""DSP helpers shared by synthoseis-lite generators and QC."""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, resample_poly


def antialias_taps(
    factor: int,
    *,
    taps_per_factor: int = 32,
    cutoff_output_nyquist_fraction: float = 0.9,
    kaiser_beta: float = 8.6,
) -> np.ndarray:
    if factor < 1:
        raise ValueError("factor must be positive.")
    return firwin(
        taps_per_factor * factor + 1,
        cutoff_output_nyquist_fraction / factor,
        window=("kaiser", kaiser_beta),
        scale=True,
    ).astype(np.float64)


def downsample_continuous(values: np.ndarray, factor: int, taps: np.ndarray) -> np.ndarray:
    return np.asarray(
        resample_poly(values, up=1, down=factor, axis=-1, window=taps, padtype="line"),
        dtype=np.float64,
    )
