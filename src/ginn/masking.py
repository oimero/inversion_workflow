"""Shared target-layer mask utilities for GINN datasets."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


def resolve_mask_bounds(mask_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve per-trace contiguous mask bounds from a flattened mask array."""
    if mask_flat.ndim != 2:
        raise ValueError(f"mask_flat must be a 2D array, got shape={mask_flat.shape}.")

    n_trace, n_sample = mask_flat.shape
    has_valid = mask_flat.any(axis=1)
    start = np.zeros(n_trace, dtype=np.int64)
    end = np.zeros(n_trace, dtype=np.int64)

    if np.any(has_valid):
        valid_mask = mask_flat[has_valid]
        start[has_valid] = np.argmax(valid_mask, axis=1)
        end[has_valid] = n_sample - np.argmax(valid_mask[:, ::-1], axis=1)

    return start, end, has_valid  # type: ignore


def build_eroded_loss_mask(mask_flat: np.ndarray, erosion_samples: int) -> np.ndarray:
    """Build an inward-eroded waveform loss mask from a core target-layer mask."""
    start, end, has_valid = resolve_mask_bounds(mask_flat)
    n_sample = mask_flat.shape[1]
    sample_index = np.arange(n_sample, dtype=np.int64)[np.newaxis, :]
    lengths = end - start

    erosion = np.minimum(int(erosion_samples), np.maximum((lengths - 1) // 2, 0))
    loss_start = start + erosion
    loss_end = end - erosion

    return (
        has_valid[:, np.newaxis]
        & (sample_index >= loss_start[:, np.newaxis])
        & (sample_index < loss_end[:, np.newaxis])
    )


def build_residual_taper(mask_flat: np.ndarray, halo_samples: int) -> np.ndarray:
    """Build a core plus halo taper for residual support."""
    start, end, has_valid = resolve_mask_bounds(mask_flat)
    n_sample = mask_flat.shape[1]
    sample_index = np.arange(n_sample, dtype=np.int64)[np.newaxis, :]
    halo = int(halo_samples)

    support_start = np.maximum(start - halo, 0)
    support_end = np.minimum(end + halo, n_sample)

    core_region = (
        has_valid[:, np.newaxis] & (sample_index >= start[:, np.newaxis]) & (sample_index < end[:, np.newaxis])
    )
    left_region = (
        has_valid[:, np.newaxis]
        & (sample_index >= support_start[:, np.newaxis])
        & (sample_index < start[:, np.newaxis])
    )
    right_region = (
        has_valid[:, np.newaxis] & (sample_index >= end[:, np.newaxis]) & (sample_index < support_end[:, np.newaxis])
    )

    taper = np.zeros(mask_flat.shape, dtype=np.float32)
    taper[core_region] = 1.0

    if halo > 0:
        left_denom = (start - support_start + 1).astype(np.float32)[:, np.newaxis]
        right_denom = (support_end - end + 1).astype(np.float32)[:, np.newaxis]

        left_weight = (sample_index - support_start[:, np.newaxis] + 1).astype(np.float32) / left_denom
        right_weight = (support_end[:, np.newaxis] - sample_index).astype(np.float32) / right_denom

        taper[left_region] = left_weight[left_region]
        taper[right_region] = right_weight[right_region]

    return taper


def get_valid_trace_indices(mask_flat: np.ndarray) -> np.ndarray:
    """Return flattened trace indices that contain at least one valid sample."""
    return np.flatnonzero(mask_flat.any(axis=1))


def select_spatial_validation_split(
    valid_indices: np.ndarray,
    *,
    n_il: int,
    n_xl: int,
    validation_fraction: float,
    gap_traces: int,
    anchor: str,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Select a contiguous inline/xline validation block with a surrounding gap."""
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    if valid_indices.size == 0:
        raise ValueError("Cannot build a validation split because no valid traces were found.")
    if validation_fraction <= 0.0:
        metadata = {
            "mode": "none",
            "train_trace_count": int(valid_indices.size),
            "val_trace_count": 0,
            "gap_trace_count": 0,
        }
        return valid_indices.copy(), np.empty(0, dtype=np.int64), metadata

    il_coords = valid_indices // n_xl
    xl_coords = valid_indices % n_xl
    il_min = int(il_coords.min())
    il_max = int(il_coords.max())
    xl_min = int(xl_coords.min())
    xl_max = int(xl_coords.max())
    il_extent = il_max - il_min + 1
    xl_extent = xl_max - xl_min + 1

    side_fraction = math.sqrt(validation_fraction)
    block_il = max(1, min(il_extent, int(math.ceil(il_extent * side_fraction))))
    block_xl = max(1, min(xl_extent, int(math.ceil(xl_extent * side_fraction))))

    if anchor == "maxmax":
        il_start = il_max - block_il + 1
        xl_start = xl_max - block_xl + 1
    elif anchor == "maxmin":
        il_start = il_max - block_il + 1
        xl_start = xl_min
    elif anchor == "minmax":
        il_start = il_min
        xl_start = xl_max - block_xl + 1
    elif anchor == "minmin":
        il_start = il_min
        xl_start = xl_min
    elif anchor == "center":
        il_start = il_min + max((il_extent - block_il) // 2, 0)
        xl_start = xl_min + max((xl_extent - block_xl) // 2, 0)
    else:
        raise ValueError(f"Unsupported validation block anchor: {anchor!r}")

    il_end = il_start + block_il
    xl_end = xl_start + block_xl

    val_mask = (il_coords >= il_start) & (il_coords < il_end) & (xl_coords >= xl_start) & (xl_coords < xl_end)
    val_indices = valid_indices[val_mask]
    if val_indices.size == 0:
        raise ValueError(
            "Validation block did not capture any valid traces. "
            f"Try a different validation_block_anchor or a larger validation_fraction (current={validation_fraction})."
        )

    gap = int(gap_traces)
    gap_il_start = max(il_start - gap, 0)
    gap_il_end = min(il_end + gap, n_il)
    gap_xl_start = max(xl_start - gap, 0)
    gap_xl_end = min(xl_end + gap, n_xl)

    exclusion_mask = (
        (il_coords >= gap_il_start) & (il_coords < gap_il_end) & (xl_coords >= gap_xl_start) & (xl_coords < gap_xl_end)
    )
    train_indices = valid_indices[~exclusion_mask]
    gap_only_mask = exclusion_mask & ~val_mask
    if train_indices.size == 0:
        raise ValueError(
            "Validation block plus gap removed all training traces. "
            f"Try a smaller validation_fraction or validation_gap_traces (current gap={gap})."
        )

    metadata = {
        "mode": "spatial_block",
        "anchor": anchor,
        "requested_validation_fraction": float(validation_fraction),
        "actual_validation_fraction": float(val_indices.size / valid_indices.size),
        "gap_traces": gap,
        "train_trace_count": int(train_indices.size),
        "val_trace_count": int(val_indices.size),
        "gap_trace_count": int(gap_only_mask.sum()),
        "block_il_start": int(il_start),
        "block_il_end": int(il_end),
        "block_xl_start": int(xl_start),
        "block_xl_end": int(xl_end),
    }
    return train_indices, val_indices, metadata
