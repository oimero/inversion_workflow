"""Small numerical helpers that are still shared by production workflows."""

from __future__ import annotations

import numpy as np


def centered_rms(values: np.ndarray, mask: np.ndarray, *, min_count: int = 1) -> float:
    """Root mean square after removing the masked finite mean."""
    finite = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if np.count_nonzero(finite) < int(min_count):
        return float("nan")
    selected = np.asarray(values, dtype=np.float64)[finite]
    centered = selected - float(np.mean(selected))
    return float(np.sqrt(np.mean(centered * centered)))


def radius_connected_components(points_xy: np.ndarray, radius: float) -> np.ndarray:
    """Label connected components formed by point pairs within ``radius``.

    Non-finite points are kept as singleton clusters so dense valid wells do not
    accidentally inherit missing-coordinate rows.
    """
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_xy must have shape [n, 2], got {points.shape}.")
    radius = float(radius)
    if radius < 0.0 or not np.isfinite(radius):
        raise ValueError(f"radius must be a finite non-negative number, got {radius}.")

    n_points = int(points.shape[0])
    labels = np.full(n_points, -1, dtype=np.int64)
    finite = np.isfinite(points).all(axis=1)
    finite_indices = np.flatnonzero(finite)
    radius_sq = radius * radius
    next_label = 0

    for start in range(n_points):
        if labels[start] >= 0:
            continue
        labels[start] = next_label
        if not finite[start]:
            next_label += 1
            continue
        stack = [start]
        while stack:
            current = stack.pop()
            candidate_indices = finite_indices[labels[finite_indices] < 0]
            if candidate_indices.size == 0:
                continue
            deltas = points[candidate_indices] - points[current]
            distances_sq = np.sum(deltas * deltas, axis=1)
            neighbors = candidate_indices[distances_sq <= radius_sq]
            for neighbor in neighbors:
                labels[neighbor] = next_label
                stack.append(int(neighbor))
        next_label += 1

    return labels


__all__ = ["centered_rms", "radius_connected_components"]
