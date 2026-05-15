"""Reusable synthetic helpers for resolution-enhancement experiments."""

from __future__ import annotations

import numpy as np

from enhance.prior import (
    WellResolutionPriorBundle,
    edge_taper,
    fit_delta_to_base_ai_bounds,
    fit_residual_to_lfm_bounds,
    highpass_log_ai_residual,
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
    "random_reflectivity",
    "random_reflectivity_in_taper",
    "reflectivity_to_log_ai",
    "sample_highres_prior_patch",
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
    return [
        int(row)
        for row in range(prior.n_wells)
        if int(np.asarray(prior.well_mask[row]).sum()) >= 2 and int(np.asarray(prior.highres_well_mask[row]).sum()) >= 2
    ]


def summary_or_percentile(prior: WellResolutionPriorBundle, key: str, percentile: float) -> float:
    """Read a residual summary statistic, falling back to a percentile of prior values."""
    highres_summary_value = prior.summary.get("highres_well_high_log_ai", {}).get(key)
    if highres_summary_value is not None and np.isfinite(float(highres_summary_value)):
        return float(highres_summary_value)
    summary_value = prior.summary.get("well_high_log_ai", {}).get(key)
    if summary_value is not None and np.isfinite(float(summary_value)):
        return float(summary_value)
    values = prior.highres_well_high_log_ai[prior.highres_well_mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = prior.well_high_log_ai[prior.well_mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-3
    return float(np.percentile(np.abs(values), percentile))


def sample_highres_prior_patch(
    prior: WellResolutionPriorBundle,
    well_rows: list[int],
    target_depth: np.ndarray,
    taper: np.ndarray,
    support_mask: np.ndarray | None = None,
    *,
    min_len_samples: int,
    max_attempts: int = 32,
    values_key: str = "highres_well_high_log_ai",
) -> tuple[np.ndarray, int, int, np.ndarray] | None:
    """Sample a native high-resolution prior patch into a target high-res axis."""
    target_depth_1d = np.asarray(target_depth, dtype=np.float32).reshape(-1)
    active = np.asarray(taper, dtype=np.float32).reshape(-1) > 0.0
    if active.shape != target_depth_1d.shape:
        raise ValueError(f"taper shape {active.shape} does not match target_depth shape {target_depth_1d.shape}.")
    if support_mask is not None:
        support = np.asarray(support_mask, dtype=bool).reshape(-1)
        if support.shape == active.shape and np.any(active & support):
            active = active & support
    active_runs = true_runs(active)
    if not active_runs or not well_rows:
        return None

    candidate_rows = [
        int(row)
        for row in well_rows
        if int(np.asarray(prior.highres_well_mask[row], dtype=bool).sum()) >= max(2, int(min_len_samples))
    ]
    if not candidate_rows:
        candidate_rows = [
            int(row) for row in well_rows if int(np.asarray(prior.highres_well_mask[row], dtype=bool).sum()) >= 2
        ]
    if not candidate_rows:
        return None

    min_len_floor = max(2, int(min_len_samples))
    for _ in range(max(1, int(max_attempts))):
        dst_start, dst_stop = active_runs[int(np.random.randint(0, len(active_runs)))]
        max_len = max(0, dst_stop - dst_start)
        if max_len < 2:
            continue
        min_len = min(max_len, min_len_floor)
        length = int(np.random.randint(min_len, max_len + 1)) if max_len > min_len else int(max_len)
        dst0 = int(np.random.randint(dst_start, dst_stop - length + 1))
        dst1 = dst0 + length
        local_depth = target_depth_1d[dst0:dst1]
        if local_depth.size < 2 or not np.all(np.isfinite(local_depth)) or np.any(np.diff(local_depth) <= 0.0):
            continue
        span = float(local_depth[-1] - local_depth[0])
        if span <= 0.0:
            continue

        row_idx = int(np.random.choice(candidate_rows))
        row_mask = np.asarray(prior.highres_well_mask[row_idx], dtype=bool)
        row_depth = np.asarray(prior.highres_depth[row_idx], dtype=np.float32)
        row_residual = np.asarray(getattr(prior, values_key)[row_idx], dtype=np.float32)
        valid = row_mask & np.isfinite(row_depth) & np.isfinite(row_residual)
        for src_start, src_stop in np.random.permutation(true_runs(valid)).tolist():
            src_depth = row_depth[src_start:src_stop]
            src_residual = row_residual[src_start:src_stop]
            if src_depth.size < 2 or float(src_depth[-1] - src_depth[0]) < span:
                continue
            src_min = float(src_depth[0])
            src_max_start = float(src_depth[-1] - span)
            src0_depth = src_min if src_max_start <= src_min else float(np.random.uniform(src_min, src_max_start))
            query_depth = src0_depth + (local_depth - float(local_depth[0]))
            local_patch = np.interp(query_depth, src_depth, src_residual).astype(np.float32)
            placed = np.zeros_like(target_depth_1d, dtype=np.float32)
            placed[dst0:dst1] = local_patch
            return placed, dst0, dst1, local_patch
    return None
