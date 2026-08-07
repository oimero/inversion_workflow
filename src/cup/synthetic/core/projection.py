"""Domain-neutral finite-support truth projection for science v2."""

from __future__ import annotations

import numpy as np
from cup.synthetic.core.records import ProjectedTruth, SampleAxis
from cup.synthetic.core.rejections import ProjectionRejected
from cup.synthetic.core.signal import finite_support_fir, valid_filter_decimate
from cup.synthetic.core.truth import SyntheticTruth


def _categorical_model_grids(
    truth: SyntheticTruth,
    *,
    factor: int,
    n_model: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_highres = truth.state_id_highres
    object_highres = truth.object_id_highres
    zone_highres = truth.zone_id_highres
    boundary_highres = truth.boundary_mask_highres
    n_lateral, n_highres = state_highres.shape
    fractions = np.zeros((n_lateral, n_model, 3), dtype=np.float32)
    dominant = np.full((n_lateral, n_model), -1, dtype=np.int32)
    zone_model = np.full((n_lateral, n_model), -1, dtype=np.int16)
    boundary_fraction = np.zeros((n_lateral, n_model), dtype=np.float32)
    valid = np.zeros((n_lateral, n_model), dtype=bool)
    for model_index in range(n_model):
        center = model_index * factor
        left = factor // 2
        right = factor - left
        start = max(0, center - left)
        end = min(n_highres, center + right + 1)
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


def _projection_factor(truth: SyntheticTruth, model_axis: SampleAxis) -> int:
    if truth.sample_domain != model_axis.sample_domain or truth.axis_unit != model_axis.unit:
        raise ProjectionRejected(
            ["projection_domain_mismatch"],
            diagnostics={},
            details=[{"reason": "projection_domain_mismatch"}],
        )
    ratio = model_axis.sample_interval / truth.highres_sample_interval
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1e-12):
        raise ProjectionRejected(
            ["projection_axis_not_nested"],
            diagnostics={"interval_ratio": ratio},
            details=[{"reason": "projection_axis_not_nested", "interval_ratio": ratio}],
        )
    nested = truth.highres_axis[::factor]
    if nested.shape != model_axis.coordinates.shape or not np.allclose(
        nested, model_axis.coordinates, rtol=1e-10, atol=1e-12
    ):
        raise ProjectionRejected(
            ["projection_axis_not_nested"],
            diagnostics={"factor": factor},
            details=[{"reason": "projection_axis_not_nested", "factor": factor}],
        )
    return factor


def project_truth_to_model_grid(
    truth: SyntheticTruth,
    axis: SampleAxis,
) -> ProjectedTruth:
    """Project truth with the single finite-support science-v2 operator."""
    factor = _projection_factor(truth, axis)
    taps = finite_support_fir(factor)
    n_model = axis.coordinates.size
    n_lateral = truth.lateral_m.size

    model_log_ai, model_support_1d = valid_filter_decimate(
            truth.log_ai_highres, factor=factor, taps=taps
        )
    rgt_model, rgt_support = valid_filter_decimate(
            truth.rgt_highres, factor=factor, taps=taps
        )
    if not np.array_equal(model_support_1d, rgt_support):
        raise ProjectionRejected(["projection_support_mismatch"], diagnostics={}, details=[{"reason": "projection_support_mismatch"}])
    half = taps.size // 2
    high_support_1d = np.zeros(truth.highres_axis.size, dtype=bool)
    high_support_1d[half : truth.highres_axis.size - half] = True

    fractions, dominant, zones, boundary_fraction, categorical_valid = (
        _categorical_model_grids(truth, factor=factor, n_model=n_model)
    )
    geometric_valid = truth.state_id_highres[:, ::factor] >= 0
    return ProjectedTruth(
        model_axis=axis,
        model_target_log_ai=model_log_ai,
        rgt_model=rgt_model,
        state_fraction_model=fractions,
        dominant_object_id_model=dominant,
        zone_id_model=zones,
        boundary_fraction_model=boundary_fraction,
        boundary_mask_model=boundary_fraction > 0.0,
        geometric_valid_mask_model=geometric_valid,
        categorical_valid_mask_model=categorical_valid,
        projection_support_highres=np.broadcast_to(
            high_support_1d, (n_lateral, high_support_1d.size)
        ),
        projection_support_model=np.broadcast_to(
            model_support_1d, (n_lateral, model_support_1d.size)
        ),
    )


__all__ = [
    "project_truth_to_model_grid",
]
