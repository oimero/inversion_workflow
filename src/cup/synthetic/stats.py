"""Small statistical helpers shared across synthoseis-lite modules."""

from __future__ import annotations

import numpy as np


def centered_rms(values: np.ndarray, mask: np.ndarray, *, min_count: int = 1) -> float:
    finite = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if np.count_nonzero(finite) < int(min_count):
        return float("nan")
    selected = np.asarray(values, dtype=np.float64)[finite]
    centered = selected - float(np.mean(selected))
    return float(np.sqrt(np.mean(centered * centered)))
