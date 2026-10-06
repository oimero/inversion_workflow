"""RGT-coordinate low-pass slicing LFM construction for Marmousi2.

The builder keeps every impedance operation in natural-log AI space.  Training
well curves are low-passed once on their native TWT axis, then carried through
RGT as a monotonic tau coordinate and interpolated between adjacent wells on
every slice.  Because the transfer uses tau, the low-pass well values survive
exactly at the training-well traces.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cup.lfm.math import LowpassSpec, apply_lfm_lowpass, ordinary_krige_xy
from wtie.processing.grid import Log

from .lfm_models import LfmArray


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name} must have a non-empty [row, time] shape.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _strict_axis(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing and unique.")
    return array


def _regular_twt(value: Any, *, expected_size: int) -> tuple[np.ndarray, float]:
    twt = _strict_axis(value, name="twt_s")
    if twt.size != expected_size:
        raise ValueError(f"twt_s length {twt.size} does not match the time dimension {expected_size}.")
    if twt.size < 2:
        raise ValueError("twt_s requires at least two samples.")
    differences = np.diff(twt)
    step = float(np.median(differences))
    if not np.allclose(differences, step, rtol=1e-8, atol=1e-10):
        raise ValueError("twt_s must be regularly sampled for the Step-7 low-pass.")
    return twt, step


def _validate_lowpass_spec(spec: LowpassSpec, *, sample_step_s: float) -> None:
    if not isinstance(spec, LowpassSpec):
        raise TypeError("lowpass_spec must be a cup.lfm.math.LowpassSpec instance.")
    if not spec.enabled:
        raise ValueError("lowpass_spec.enabled must be True for RGT-guided construction.")
    cutoff = spec.cutoff_cycles_per_axis_unit
    if cutoff is None or not np.isfinite(float(cutoff)) or float(cutoff) <= 0.0:
        raise ValueError("lowpass_spec cutoff must be finite and positive.")
    if float(cutoff) >= 0.5 / sample_step_s:
        raise ValueError("lowpass_spec cutoff must be below the TWT Nyquist frequency.")
    if spec.order is None or isinstance(spec.order, bool) or int(spec.order) != spec.order or int(spec.order) <= 0:
        raise ValueError("lowpass_spec.order must be a positive integer.")
    if spec.buffer_mode not in {"reflect", "edge", "none"}:
        raise ValueError("lowpass_spec.buffer_mode must be reflect, edge, or none.")
    if spec.buffer_axis_units is None or not np.isfinite(float(spec.buffer_axis_units)):
        raise ValueError("lowpass_spec.buffer_axis_units must be finite.")
    if float(spec.buffer_axis_units) < 0.0:
        raise ValueError("lowpass_spec.buffer_axis_units must be non-negative.")


def _pava_non_decreasing(values: np.ndarray) -> np.ndarray:
    """Equal-weight least-squares projection onto non-decreasing sequences."""

    levels: list[float] = []
    weights: list[int] = []
    counts: list[int] = []
    for value in np.asarray(values, dtype=np.float64):
        levels.append(float(value))
        weights.append(1)
        counts.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total_weight = weights[-2] + weights[-1]
            merged = (weights[-2] * levels[-2] + weights[-1] * levels[-1]) / total_weight
            levels[-2] = merged
            weights[-2] = total_weight
            counts[-2] += counts[-1]
            levels.pop()
            weights.pop()
            counts.pop()
    return np.repeat(np.asarray(levels, dtype=np.float64), np.asarray(counts, dtype=np.int64))


def _repair_rgt(rgt_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    profiles, samples = rgt_raw.shape
    if samples < 2:
        raise ValueError("rgt requires at least two TWT samples.")
    indices = np.arange(samples, dtype=np.float64)
    centered = indices - float(np.mean(indices))
    denominator = float(np.dot(centered, centered))
    slopes = ((rgt_raw - np.mean(rgt_raw, axis=1, keepdims=True)) @ centered) / denominator
    spans = np.ptp(rgt_raw, axis=1)
    if np.any(~np.isfinite(spans)) or np.any(spans <= 0.0):
        raise ValueError("Each RGT profile must have a finite positive span.")
    if np.any(~np.isfinite(slopes)) or np.any(slopes <= 0.0):
        raise ValueError("Each RGT profile must have a positive main trend.")
    global_range = float(np.ptp(rgt_raw))
    if not np.isfinite(global_range) or global_range <= 0.0:
        raise ValueError("RGT has no finite positive global span.")
    epsilon = global_range * 1e-6 / float(samples - 1)
    pava = np.empty_like(rgt_raw)
    used = np.empty_like(rgt_raw)
    for profile in range(profiles):
        # Subtracting the ramp before PAVA and adding it back afterwards gives
        # a strict minimum slope while keeping the PAVA correction tiny.
        pava[profile] = _pava_non_decreasing(rgt_raw[profile] - epsilon * indices)
        used[profile] = pava[profile] + epsilon * indices
    if np.any(np.diff(used, axis=1) <= 0.0):
        raise ValueError("RGT monotonic repair did not produce strict profiles.")
    correction = used - rgt_raw
    stats = {
        "global_range": global_range,
        "ramp_total": global_range * 1e-6,
        "ramp_epsilon_per_sample": epsilon,
        "raw_profile_span_min": float(np.min(spans)),
        "raw_profile_span_max": float(np.max(spans)),
        "main_trend_slope_min_per_sample": float(np.min(slopes)),
        "main_trend_slope_max_per_sample": float(np.max(slopes)),
        "correction_max_abs": float(np.max(np.abs(correction))),
        "correction_rms": float(np.sqrt(np.mean(np.square(correction)))),
    }
    return pava, used, stats


def _horizontal_brackets(source_x: np.ndarray, target_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return left/right indices, weights, and endpoint masks for interpolation."""

    if source_x.size == 1:
        if np.any(~np.isclose(target_x, source_x[0], rtol=0.0, atol=1e-8)):
            raise ValueError("A single source profile can only serve the matching target coordinate.")
        zeros = np.zeros(target_x.size, dtype=np.int64)
        return zeros, zeros, np.zeros(target_x.size), np.zeros(target_x.size, dtype=bool), np.zeros(target_x.size, dtype=bool)
    left = np.searchsorted(source_x, target_x, side="right") - 1
    below = target_x < source_x[0]
    above = target_x > source_x[-1]
    left = np.clip(left, 0, source_x.size - 2)
    right = left + 1
    denominator = source_x[right] - source_x[left]
    weight = (target_x - source_x[left]) / denominator
    weight[below] = 0.0
    weight[above] = 1.0
    return left, right, weight, below, above


def _sample_profiles(source_values: np.ndarray, source_x: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    left, right, weight, below, above = _horizontal_brackets(source_x, target_x)
    sampled = source_values[left, :] * (1.0 - weight[:, None]) + source_values[right, :] * weight[:, None]
    if np.any(below):
        sampled[below] = source_values[0]
    if np.any(above):
        sampled[above] = source_values[-1]
    return sampled


def _lowpass_rows(values: np.ndarray, twt_s: np.ndarray, spec: LowpassSpec) -> np.ndarray:
    output = np.empty_like(values)
    for row, values_row in enumerate(values):
        log = Log(values_row.copy(), twt_s.copy(), "twt", name=f"rgt_lfm_row_{row}", unit="logAI", allow_nan=False)
        output[row] = apply_lfm_lowpass(log, spec).values
    return output


def _default_range_m(well_x_m: np.ndarray, output_x_m: np.ndarray) -> float:
    """Default length scale: the median nearest-neighbour well distance.

    This mirrors the rule the workflow's kriging helper uses, so a direct call
    without an explicit range behaves like the rest of the LFM module.
    """

    if well_x_m.size > 1:
        spacing = float(np.median(np.diff(np.sort(well_x_m))))
    else:
        spacing = 0.0
    nominal = float(np.min(np.diff(output_x_m))) if output_x_m.size > 1 else 0.0
    resolved = max(spacing, nominal)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("Cannot derive a length scale from the well and output coordinates.")
    return resolved


def _kriging_weights(
    well_x_m: np.ndarray,
    output_x_m: np.ndarray,
    *,
    variogram: str = "exponential",
    nugget: float = 0.0,
    range_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Ordinary-kriging weight of every training well at every output trace.

    This is the weighting the workflow's proportional-kriging LFM applies to each
    slice, evaluated through the same kriging helper.  Control positions are fixed
    in x, and ordinary-kriging weights depend on the variogram and the geometry
    rather than on the data, so one solution serves every vertical slice.  Exact
    interpolation makes the weight vector a unit vector on the training-well
    traces.  Each well is recovered with one identity-basis run, because the
    operator is linear in the data values.
    """

    if well_x_m.size == 1:
        return np.ones((output_x_m.size, 1), dtype=np.float64), {
            "mode": "single_control_constant", "n_controls": 1,
            "variogram": str(variogram).casefold(), "nugget": float(nugget),
            "range_m": float(range_m), "exact": True,
        }
    nominal = float(np.min(np.diff(output_x_m))) if output_x_m.size > 1 else float(range_m)
    control_y = np.zeros_like(well_x_m)
    output_y = np.zeros_like(output_x_m)
    weights = np.empty((output_x_m.size, well_x_m.size), dtype=np.float64)
    metadata: dict[str, Any] = {}
    for index in range(well_x_m.size):
        basis = np.zeros(well_x_m.size, dtype=np.float64)
        basis[index] = 1.0
        field, _variance, metadata = ordinary_krige_xy(
            control_x_m=well_x_m, control_y_m=control_y, control_values=basis,
            output_x_m=output_x_m, output_y_m=output_y, nominal_bin_spacing_m=nominal,
            variogram=variogram, exact=True, nugget=nugget, range_m=range_m,
        )
        weights[:, index] = np.ravel(field)
    metadata = {
        "mode": "weight_basis_kriging", "n_controls": int(well_x_m.size),
        **{key: metadata[key] for key in ("variogram", "exact", "nugget", "range_m", "range_m_source")},
        "sill_note": "the identity-basis sill is meaningless here; ordinary-kriging weights do not depend on it",
        "weight_min": float(np.min(weights)), "weight_max": float(np.max(weights)),
    }
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError("Kriging weights must sum to one on every output trace.")
    return weights, metadata


def _tau_map_structure(
    training_log_ai: np.ndarray,
    training_tau: np.ndarray,
    used_rgt: np.ndarray,
    horizontal_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map each well curve through tau, then combine the wells with the weights."""

    profiles, samples = used_rgt.shape
    wells = training_log_ai.shape[0]
    if horizontal_weights.shape != (profiles, wells):
        raise ValueError("horizontal weights must be shaped [output profile, training well].")
    if not np.all(np.isfinite(horizontal_weights)):
        raise ValueError("horizontal weights must be finite.")
    structure = np.empty_like(used_rgt)
    extrapolation_counts = np.zeros(profiles, dtype=np.int64)
    active_extrapolation_counts = np.zeros(profiles, dtype=np.int64)
    weighted_extrapolation_counts = np.zeros(profiles, dtype=np.float64)
    block_size = 64
    for start in range(0, profiles, block_size):
        stop = min(start + block_size, profiles)
        target_flat = used_rgt[start:stop].reshape(-1)
        block_count = stop - start
        mapped_wells = np.empty((wells, block_count, samples), dtype=np.float64)
        active_extrapolated = np.zeros((block_count, samples), dtype=bool)
        weighted_extrapolated = np.zeros((block_count, samples), dtype=np.float64)
        for well in range(wells):
            tau = training_tau[well]
            extrapolated = ((target_flat < tau[0]) | (target_flat > tau[-1])).reshape(block_count, samples)
            extrapolation_counts[start:stop] += extrapolated.sum(axis=1)
            well_weight = horizontal_weights[start:stop, well][:, None]
            active_extrapolated |= extrapolated & (well_weight > 0.0)
            weighted_extrapolated += extrapolated * well_weight
            mapped_wells[well] = np.interp(
                target_flat,
                tau,
                training_log_ai[well],
                left=float(training_log_ai[well, 0]),
                right=float(training_log_ai[well, -1]),
            ).reshape(block_count, samples)
        active_extrapolation_counts[start:stop] += active_extrapolated.sum(axis=1)
        weighted_extrapolation_counts[start:stop] += weighted_extrapolated.sum(axis=1)
        structure[start:stop] = np.einsum(
            "iw,wis->is", horizontal_weights[start:stop], mapped_wells
        )
    return (
        structure,
        extrapolation_counts / float(samples * wells),
        active_extrapolation_counts / float(samples),
        weighted_extrapolation_counts / float(samples),
    )


def _rgt_inputs(
    training_log_ai: np.ndarray,
    well_x_m: np.ndarray,
    output_x_m: np.ndarray,
    twt_s: np.ndarray,
    rgt: np.ndarray,
) -> dict[str, Any]:
    """Validate a 2-D RGT build and repair the tau field once."""

    training = _finite_matrix(training_log_ai, name="training_log_ai")
    well_x = _strict_axis(well_x_m, name="well_x_m")
    output_x = _strict_axis(output_x_m, name="output_x_m")
    twt, sample_step_s = _regular_twt(twt_s, expected_size=training.shape[1])
    raw_rgt = _finite_matrix(rgt, name="rgt")
    if raw_rgt.shape != (output_x.size, twt.size):
        raise ValueError(
            f"rgt shape {raw_rgt.shape} must equal [output_x_m, twt_s] {(output_x.size, twt.size)}."
        )
    if well_x.size != training.shape[0]:
        raise ValueError(f"well_x_m length {well_x.size} does not match well count {training.shape[0]}.")
    tolerance = max(1e-8, 1e-10 * max(1.0, float(np.ptp(output_x))))
    if well_x[0] < output_x[0] - tolerance or well_x[-1] > output_x[-1] + tolerance:
        raise ValueError("All training wells must lie inside the RGT output_x_m support.")
    pava_rgt, used_rgt, repair_stats = _repair_rgt(raw_rgt)
    rgt_at_wells = _sample_profiles(used_rgt, output_x, well_x)
    if np.any(np.diff(rgt_at_wells, axis=1) <= 0.0):
        raise ValueError("RGT sampled at training wells must be strictly increasing in TWT.")
    return {
        "training": training, "well_x": well_x, "output_x": output_x, "twt": twt,
        "sample_step_s": sample_step_s, "raw_rgt": raw_rgt, "pava_rgt": pava_rgt,
        "used_rgt": used_rgt, "rgt_at_wells": rgt_at_wells, "repair_stats": repair_stats,
    }


def build_rgt_lowpass_slices(
    training_log_ai: np.ndarray,
    well_x_m: np.ndarray,
    output_x_m: np.ndarray,
    twt_s: np.ndarray,
    rgt: np.ndarray,
    lowpass_spec: LowpassSpec,
    *,
    variogram: str = "exponential",
    nugget: float = 0.0,
    kriging_range_m: float | None = None,
) -> LfmArray:
    """Low-pass the training wells first, then slice them along the RGT coordinate.

    The order matches the workflow's low-pass slicing LFM: the well curves are
    filtered once on their native TWT axis, and only afterwards are they carried
    through tau and combined across the section.  Every output sample is treated
    as one slice, so no extra interpolation between a coarse slice grid is added.
    The across-section weighting is ordinary kriging in real x, the same operator
    the workflow's proportional-kriging LFM applies to each slice, which also
    gives this variant a length scale.  Because tau is the transfer coordinate and
    the kriging is exact, the low-pass well values are preserved at the
    training-well traces.
    """

    inputs = _rgt_inputs(training_log_ai, well_x_m, output_x_m, twt_s, rgt)
    training, well_x, output_x = inputs["training"], inputs["well_x"], inputs["output_x"]
    twt, used_rgt = inputs["twt"], inputs["used_rgt"]
    _validate_lowpass_spec(lowpass_spec, sample_step_s=float(inputs["sample_step_s"]))
    resolved_range_m = (
        float(kriging_range_m) if kriging_range_m is not None else _default_range_m(well_x, output_x)
    )
    horizontal_weights, weight_metadata = _kriging_weights(
        well_x, output_x, variogram=variogram, nugget=nugget, range_m=resolved_range_m
    )

    filtered_wells = _lowpass_rows(training, twt, lowpass_spec)
    structure, tau_extrap_by_profile, active_tau_extrap_by_profile, weighted_tau_extrap_by_profile = (
        _tau_map_structure(filtered_wells, inputs["rgt_at_wells"], used_rgt, horizontal_weights)
    )
    well_indices = [int(np.argmin(np.abs(output_x - position))) for position in well_x]
    matched = [bool(np.isclose(output_x[index], position, rtol=0.0, atol=1e-8))
               for index, position in zip(well_indices, well_x)]
    preserved = (
        float(np.max(np.abs(structure[well_indices] - filtered_wells)))
        if all(matched) else None
    )
    metadata = {
        "schema": "marmousi2_lfm_rgt_lowpass_slices_v1",
        "method": "lowpass_training_wells_then_rgt_slice_kriging",
        "fit_space": "logAI",
        "source": "training_wells_only",
        "label_leakage": "validation_and_test inputs are not accepted",
        "training_well_count": int(training.shape[0]),
        "output_profile_count": int(output_x.size),
        "time_count": int(twt.size),
        "well_x_m": [float(value) for value in well_x],
        "twt_start_s": float(twt[0]),
        "twt_end_s": float(twt[-1]),
        "sample_step_s": float(inputs["sample_step_s"]),
        "sequence": [
            "Step7_lowpass_on_training_well_TWT",
            "tau_mapping_to_every_output_sample",
            "x_ordinary_kriging_on_each_slice",
        ],
        "slice_definition": "one output TWT sample is one slice; no coarse slice grid",
        "rgt": {
            "input_shape": [int(inputs["raw_rgt"].shape[0]), int(inputs["raw_rgt"].shape[1])],
            "sampling_at_training_wells": "linear_in_output_x_m",
            "used_field": "rgt_monotonic",
            "projection": "equal_weight_least_squares_PAVA_non_decreasing",
            **inputs["repair_stats"],
        },
        "horizontal_interpolation": {
            "coordinate": "actual_x_m",
            "weights": "ordinary_kriging",
            "helper": "cup.lfm.math.ordinary_krige_xy",
            "range_m_argument": None if kriging_range_m is None else float(kriging_range_m),
            "outside_policy": "ordinary_kriging_stationary_extrapolation",
            **weight_metadata,
        },
        "training_well_preservation_max_abs_error_log_ai": preserved,
        "tau_mapping": {
            "source_curve": "lowpassed_training_log_ai",
            "interpolation": "linear_in_tau",
            "outside_policy": "nearest_endpoint_constant",
            "extrapolation_fraction": float(np.mean(tau_extrap_by_profile)),
            "active_extrapolation_fraction": float(np.mean(active_tau_extrap_by_profile)),
            "weighted_extrapolation_fraction": float(np.mean(weighted_tau_extrap_by_profile)),
        },
        "lowpass": {
            "applied_to": "training_well_curves_on_native_TWT_before_rgt_mapping",
            "output_lowpass_applied": False,
            "double_lowpass": False,
            "order": int(lowpass_spec.order),
            "cutoff_cycles_per_s": float(lowpass_spec.cutoff_cycles_per_axis_unit),
            "buffer_mode": str(lowpass_spec.buffer_mode),
            "buffer_axis_units": float(lowpass_spec.buffer_axis_units),
        },
        "gaussian_smoothing": False,
        "units": {"log_ai": "natural_log_of_m/s*g/cc", "x": "m", "twt": "s"},
    }
    fields = {
        "rgt_raw": inputs["raw_rgt"].copy(),
        "rgt_pava": inputs["pava_rgt"],
        "rgt_monotonic": used_rgt.copy(),
        "rgt_at_training_wells": inputs["rgt_at_wells"],
        "filtered_training_well_log_ai": filtered_wells,
        "horizontal_weight_by_well": horizontal_weights.T.copy(),
        "tau_extrapolation_fraction_by_profile": tau_extrap_by_profile,
        "active_tau_extrapolation_fraction_by_profile": active_tau_extrap_by_profile,
        "weighted_tau_extrapolation_fraction_by_profile": weighted_tau_extrap_by_profile,
    }
    return LfmArray(structure, metadata, fields=fields)


__all__ = ["build_rgt_lowpass_slices"]
