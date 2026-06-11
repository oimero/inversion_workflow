"""Boolean-mask utilities shared across workflow domains."""

from __future__ import annotations

import numpy as np


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open ``[start, stop)`` runs where a 1D mask is true."""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0 or not np.any(values):
        return []
    padded = np.concatenate(([False], values, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(0, edges.size, 2)]
