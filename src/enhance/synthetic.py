"""Reusable synthetic helpers for resolution-enhancement experiments."""

from __future__ import annotations

import numpy as np

from ginn.well_prior import (
    WellResolutionPriorBundle,
    edge_taper,
    fit_delta_to_base_ai_bounds,
    fit_residual_to_lfm_bounds,
    highpass_log_ai_residual,
    moving_average,
    random_reflectivity,
    random_reflectivity_in_taper,
    reflectivity_to_log_ai,
    true_runs,
)

__all__ = [
    "WellResolutionPriorBundle",
    "downsample_highres_to_samples",
    "edge_taper",
    "fit_delta_to_base_ai_bounds",
    "fit_residual_to_lfm_bounds",
    "highpass_log_ai_residual",
    "make_highres_axis",
    "markov_thin_bed_packet",
    "moving_average",
    "random_reflectivity",
    "random_reflectivity_in_taper",
    "reflectivity_to_log_ai",
    "sample_well_patch_window",
    "summary_or_percentile",
    "true_runs",
    "valid_prior_rows",
]


def make_highres_axis(axis: np.ndarray, factor: int) -> np.ndarray:
    """Upsample a strictly increasing 1D sample axis by an integer factor."""
    values = np.asarray(axis, dtype=np.float32).reshape(-1)
    if factor <= 1 or values.size < 2:
        return values.copy()
    segments = [
        np.linspace(float(values[i]), float(values[i + 1]), factor + 1, dtype=np.float32)[:-1]
        for i in range(values.size - 1)
    ]
    return np.concatenate([*segments, values[-1:].astype(np.float32)])


def downsample_highres_to_samples(values: np.ndarray, factor: int, n_sample: int) -> np.ndarray:
    """Average high-resolution samples back to the coarse sample centers."""
    highres = np.asarray(values, dtype=np.float32).reshape(-1)
    if factor <= 1:
        return highres[:n_sample].astype(np.float32, copy=True)
    coarse = np.zeros((n_sample,), dtype=np.float32)
    for idx in range(n_sample):
        center = idx * factor
        start = max(0, center - factor // 2)
        stop = min(highres.size, center + (factor + 1) // 2 + 1)
        if stop <= start:
            coarse[idx] = highres[min(center, highres.size - 1)]
        else:
            coarse[idx] = float(np.mean(highres[start:stop]))
    return coarse


def markov_thin_bed_packet(width: int, *, n_events: int, amp_scale: float, sign0: float) -> np.ndarray:
    """Generate a bounded alternating thin-bed packet with approximately n_events transitions."""
    width = int(max(1, width))
    target_transitions = max(1, int(n_events))
    sand_fraction = float(np.random.uniform(0.35, 0.65))
    avg_run = max(1.0, width / float(target_transitions + 1))
    beta = float(np.clip(1.0 / avg_run, 0.05, 0.95))
    alpha = float(np.clip(sand_fraction / (avg_run * max(1e-3, 1.0 - sand_fraction)), 0.05, 0.95))
    transition = np.array([[1.0 - alpha, alpha], [beta, 1.0 - beta]], dtype=np.float64)

    best_states = None
    best_score = float("inf")
    for _ in range(16):
        states = np.empty((width,), dtype=np.int8)
        states[0] = int(np.random.random() < sand_fraction)
        for idx in range(1, width):
            states[idx] = int(np.random.choice([0, 1], p=transition[int(states[idx - 1])]))
        transitions = int(np.count_nonzero(states[1:] != states[:-1]))
        score = abs(transitions - target_transitions)
        if score < best_score:
            best_score = score
            best_states = states
        if score == 0:
            break

    assert best_states is not None
    levels = np.where(best_states > 0, 1.0, -1.0).astype(np.float32)
    levels -= float(np.mean(levels))
    peak = float(np.max(np.abs(levels)))
    if peak > 0.0:
        levels /= peak
    return (float(sign0) * float(amp_scale) * levels).astype(np.float32)


def valid_prior_rows(prior: WellResolutionPriorBundle) -> list[int]:
    """Return well rows with enough valid samples for patch extraction."""
    return [int(row) for row in range(prior.n_wells) if int(np.asarray(prior.well_mask[row]).sum()) >= 2]


def summary_or_percentile(prior: WellResolutionPriorBundle, key: str, percentile: float) -> float:
    """Read a residual summary statistic, falling back to a percentile of prior values."""
    summary_value = prior.summary.get("residual", {}).get(key)
    if summary_value is not None and np.isfinite(float(summary_value)):
        return float(summary_value)
    values = prior.residual_log_ai[prior.well_mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-3
    return float(np.percentile(np.abs(values), percentile))


def sample_well_patch_window(
    prior: WellResolutionPriorBundle,
    well_rows: list[int],
    n_sample: int,
    taper: np.ndarray,
    support_mask: np.ndarray | None = None,
    *,
    min_len_samples: int,
) -> tuple[np.ndarray, int] | None:
    """Sample a residual patch from a valid well row and place it in active support."""
    active = np.asarray(taper) > 0.0
    if support_mask is not None:
        support = np.asarray(support_mask, dtype=bool).reshape(-1)
        if support.shape == active.shape and np.any(active & support):
            active = active & support
    active_runs = true_runs(active)
    if not active_runs or not well_rows:
        return None

    row_idx = int(np.random.choice(well_rows))
    well_mask = np.asarray(prior.well_mask[row_idx], dtype=bool)
    well_residual = np.asarray(prior.residual_log_ai[row_idx], dtype=np.float32)
    well_runs = true_runs(well_mask)
    if not well_runs:
        return None

    src_start, src_stop = well_runs[int(np.random.randint(0, len(well_runs)))]
    dst_start, dst_stop = active_runs[int(np.random.randint(0, len(active_runs)))]
    max_len = min(src_stop - src_start, dst_stop - dst_start, n_sample)
    if max_len <= 0:
        return None
    min_len = min(max_len, max(4, int(min_len_samples)))
    length = int(np.random.randint(min_len, max_len + 1)) if max_len > min_len else int(max_len)
    src0 = int(np.random.randint(src_start, src_stop - length + 1))
    dst0 = int(np.random.randint(dst_start, dst_stop - length + 1))
    patch = well_residual[src0 : src0 + length].astype(np.float32, copy=True)
    return patch, dst0
