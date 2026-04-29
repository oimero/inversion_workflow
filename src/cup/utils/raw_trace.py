"""Basic preprocessing helpers for raw 1D traces."""

from __future__ import annotations

import numpy as np


def zscore_trace(values: np.ndarray, *, nan_method: str = "linear") -> np.ndarray:
    """Interpolate missing samples and z-score a full 1D trace."""
    from wtie.processing.logs import interpolate_nans

    x = interpolate_nans(values, method=nan_method)
    x = x - np.mean(x)
    scale = np.std(x)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Trace has non-positive standard deviation.")
    return x / scale
