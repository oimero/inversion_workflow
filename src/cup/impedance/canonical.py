"""Canonical log(AI) low-pass decomposition."""

from __future__ import annotations

from typing import Any
import math

import numpy as np
from scipy.signal import butter, sosfiltfilt

from cup.impedance.contracts import (
    CanonicalIncrementContract,
    validate_increment_contract,
    validate_sample_axis,
)


def _finite_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def canonical_lowpass(
    values: np.ndarray,
    sample_axis: np.ndarray,
    contract: CanonicalIncrementContract | dict[str, Any],
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the fixed SOS forward/backward filter per finite segment."""
    resolved = validate_increment_contract(contract)
    axis = validate_sample_axis(sample_axis, resolved)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != axis.size:
        raise ValueError(
            f"values last axis {array.shape[-1] if array.ndim else None} "
            f"does not match sample_axis length {axis.size}."
        )
    finite = np.isfinite(array)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != array.shape:
            raise ValueError("valid_mask must match values shape.")
        finite &= mask
    flat = array.reshape(-1, axis.size)
    flat_mask = finite.reshape(-1, axis.size)
    result = np.full_like(flat, np.nan, dtype=np.float64)
    sos = butter(
        resolved.design_order,
        resolved.cutoff_cycles_per_unit,
        btype="lowpass",
        fs=1.0 / resolved.sample_interval,
        output="sos",
    )
    pad = resolved.pad_samples
    for row_index, row_mask in enumerate(flat_mask):
        for start, stop in _finite_segments(row_mask):
            segment = flat[row_index, start:stop]
            if segment.size < resolved.minimum_segment_samples:
                raise ValueError(
                    "finite segment is shorter than canonical minimum "
                    f"({segment.size} < {resolved.minimum_segment_samples})."
                )
            padded = np.pad(segment, pad, mode="reflect")
            filtered = sosfiltfilt(sos, padded, padtype=None)
            result[row_index, start:stop] = filtered[pad : pad + segment.size]
    return result.reshape(array.shape)


def decompose_log_ai(
    target_log_ai: np.ndarray,
    sample_axis: np.ndarray,
    contract: CanonicalIncrementContract | dict[str, Any],
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complete-trace ``(canonical_background, target_increment)``."""
    target = np.asarray(target_log_ai, dtype=np.float64)
    background = canonical_lowpass(
        target,
        sample_axis,
        contract,
        valid_mask=valid_mask,
    )
    increment = target - background
    finite = np.isfinite(target) & np.isfinite(background)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)
    return np.where(finite, background, np.nan), np.where(finite, increment, np.nan)


def generation_contract(sample_domain: str, sample_interval: float) -> CanonicalIncrementContract:
    """Build the fixed producer contract for a generated time/depth axis."""
    domain = str(sample_domain).strip().lower()
    if domain == "time":
        unit = "s"
        basis = None
        cutoff_key = "cutoff_hz"
        cutoff = 15.0
        buffer_axis_units = 0.4
    elif domain == "depth":
        unit = "m"
        basis = "tvdss"
        cutoff_key = "cutoff_wavelength_m"
        cutoff = 400.0
        buffer_axis_units = 400.0
    else:
        raise ValueError(f"Unsupported sample_domain: {sample_domain!r}")
    return CanonicalIncrementContract.from_mapping(
        {
            "contract_version": "canonical_increment_v1",
            "semantics": "canonical_complement_log_ai",
            "sample_domain": domain,
            "sample_unit": unit,
            "sample_interval": float(sample_interval),
            "sample_axis_uniform": True,
            "sample_axis_dtype": "float64",
            "sample_interval_relative_tolerance": 1.0e-6,
            "sample_interval_absolute_tolerance": 1.0e-9,
            "depth_basis": basis,
            "value_domain": "log(AI)",
            "log_base": "natural",
            "ai_unit_convention": "m/s*g/cm3",
            "lowpass": {
                "implementation": "scipy_butter_sosfiltfilt",
                "design_order": 6,
                "effective_zero_phase_order": 12,
                "cutoff_definition": "single_pass_minus_3db_final_minus_6db",
                "buffer_mode": "reflect",
                "buffer_axis_units": buffer_axis_units,
                cutoff_key: cutoff,
            },
        }
    )


__all__ = ["canonical_lowpass", "decompose_log_ai", "generation_contract"]
