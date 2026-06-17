"""Grid aggregation helpers for synthoseis-lite truth products."""

from __future__ import annotations

import numpy as np


def categorical_model_grids(
    state_highres: np.ndarray,
    object_highres: np.ndarray,
    zone_highres: np.ndarray,
    boundary_highres: np.ndarray,
    factor: int,
    n_model: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_lateral, n_highres = state_highres.shape
    fractions = np.zeros((n_lateral, n_model, 3), dtype=np.float32)
    dominant = np.full((n_lateral, n_model), -1, dtype=np.int32)
    zone_model = np.full((n_lateral, n_model), -1, dtype=np.int16)
    boundary_fraction = np.zeros((n_lateral, n_model), dtype=np.float32)
    valid = np.zeros((n_lateral, n_model), dtype=bool)
    for model_index in range(n_model):
        center = model_index * factor
        start = max(0, center - factor // 2)
        end = min(n_highres, center + factor // 2 + 1)
        for lateral_index in range(n_lateral):
            states = state_highres[lateral_index, start:end]
            objects = object_highres[lateral_index, start:end]
            zones = zone_highres[lateral_index, start:end]
            finite = states >= 0
            if not np.any(finite):
                continue
            valid[lateral_index, model_index] = True
            for state in range(3):
                fractions[lateral_index, model_index, state] = np.mean(states[finite] == state)
            object_values, object_counts = np.unique(objects[finite], return_counts=True)
            dominant[lateral_index, model_index] = int(object_values[np.argmax(object_counts)])
            zone_values, zone_counts = np.unique(zones[finite], return_counts=True)
            zone_model[lateral_index, model_index] = int(zone_values[np.argmax(zone_counts)])
            boundary_fraction[lateral_index, model_index] = float(
                np.mean(boundary_highres[lateral_index, start:end])
            )
    return fractions, dominant, zone_model, boundary_fraction, valid
