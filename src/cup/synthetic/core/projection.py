"""Domain-neutral truth projection with frozen time and depth policies."""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, resample_poly

from cup.synthetic.core.records import ProjectedTruth, ProjectionPolicy, SampleAxis
from cup.synthetic.core.rejections import ProjectionRejected
from cup.synthetic.core.truth import SyntheticTruth


def time_projection_policy() -> ProjectionPolicy:
    return ProjectionPolicy(
        continuous_method="scipy_resample_poly",
        edge_mode="line",
        support_mode="full",
        antialias_taps_per_factor=32,
        cutoff_output_nyquist_fraction=0.9,
        kaiser_beta=8.6,
        geometric_valid_mode="categorical_window_any",
    )


def depth_projection_policy() -> ProjectionPolicy:
    return ProjectionPolicy(
        continuous_method="valid_fir_decimate",
        edge_mode="finite_support",
        support_mode="valid_fir",
        antialias_taps_per_factor=32,
        cutoff_output_nyquist_fraction=0.9,
        kaiser_beta=8.6,
        geometric_valid_mode="point_sample_highres",
    )


def _antialias_taps(factor: int, policy: ProjectionPolicy) -> np.ndarray:
    count = int(policy.antialias_taps_per_factor) * factor + 1
    if count % 2 == 0:
        raise ProjectionRejected(
            ["invalid_antialias_filter"],
            diagnostics={"factor": factor, "tap_count": count},
            details=[{"reason": "invalid_antialias_filter", "tap_count": count}],
        )
    return firwin(
        count,
        float(policy.cutoff_output_nyquist_fraction) / factor,
        window=("kaiser", float(policy.kaiser_beta)),
        scale=True,
    ).astype(np.float64)


def _valid_filter_decimate(
    values: np.ndarray,
    *,
    factor: int,
    taps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=np.float64)
    n_model = (data.shape[-1] - 1) // factor + 1
    output = data[..., ::factor].copy()
    valid = np.zeros(n_model, dtype=bool)
    half = taps.size // 2
    model_high_indices = np.arange(n_model, dtype=np.int64) * factor
    supported = (model_high_indices >= half) & (
        model_high_indices < data.shape[-1] - half
    )
    flat_input = data.reshape((-1, data.shape[-1]))
    flat_output = output.reshape((-1, n_model))
    for row_index, row in enumerate(flat_input):
        filtered = np.convolve(row, taps, mode="valid")
        centers = model_high_indices[supported] - half
        flat_output[row_index, supported] = filtered[centers]
    valid[supported] = True
    return output, valid


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
    policy: ProjectionPolicy,
) -> ProjectedTruth:
    """Project high-resolution scientific truth using one frozen domain policy."""
    factor = _projection_factor(truth, axis)
    taps = _antialias_taps(factor, policy)
    n_model = axis.coordinates.size
    n_lateral = truth.lateral_m.size

    if policy.continuous_method == "scipy_resample_poly":
        model_log_ai = np.asarray(
            resample_poly(
                truth.log_ai_highres,
                up=1,
                down=factor,
                axis=-1,
                window=taps,
                padtype="line",
            ),
            dtype=np.float64,
        )[..., :n_model]
        rgt_model = np.asarray(
            resample_poly(
                truth.rgt_highres,
                up=1,
                down=factor,
                axis=-1,
                window=taps,
                padtype="line",
            ),
            dtype=np.float64,
        )[..., :n_model]
        model_support_1d = np.ones(n_model, dtype=bool)
        high_support_1d = np.ones(truth.highres_axis.size, dtype=bool)
    elif policy.continuous_method == "valid_fir_decimate":
        model_log_ai, model_support_1d = _valid_filter_decimate(
            truth.log_ai_highres, factor=factor, taps=taps
        )
        rgt_model, rgt_support = _valid_filter_decimate(
            truth.rgt_highres, factor=factor, taps=taps
        )
        if not np.array_equal(model_support_1d, rgt_support):
            raise ProjectionRejected(
                ["projection_support_mismatch"],
                diagnostics={},
                details=[{"reason": "projection_support_mismatch"}],
            )
        half = taps.size // 2
        high_support_1d = np.zeros(truth.highres_axis.size, dtype=bool)
        high_support_1d[half : truth.highres_axis.size - half] = True
    else:  # guarded by ProjectionPolicy
        raise AssertionError(policy.continuous_method)

    fractions, dominant, zones, boundary_fraction, categorical_valid = (
        _categorical_model_grids(truth, factor=factor, n_model=n_model)
    )
    geometric_valid = (
        categorical_valid
        if policy.geometric_valid_mode == "categorical_window_any"
        else truth.state_id_highres[:, ::factor] >= 0
    )
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
    "depth_projection_policy",
    "project_truth_to_model_grid",
    "time_projection_policy",
]
