"""Shared canonical log-AI decomposition for time and depth domains.

The implementation is deliberately independent of training and synthetic
generation.  Producers use it to create complete-trace labels and consumers
use the same contract to validate the sampling axis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class CanonicalIncrementContract:
    """Resolved numerical contract for one regular sampling axis."""

    sample_domain: str
    sample_unit: str
    sample_interval: float
    depth_basis: str | None
    cutoff: float
    cutoff_kind: str
    buffer_axis_units: float
    axis_absolute_tolerance: float = 1.0e-9
    design_order: int = 6
    effective_zero_phase_order: int = 12
    implementation: str = "scipy_butter_sosfiltfilt"
    buffer_mode: str = "reflect"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CanonicalIncrementContract":
        raw = dict(value)
        domain = str(raw.get("sample_domain") or "").strip().lower()
        if domain not in {"time", "depth"}:
            raise ValueError("increment_contract.sample_domain must be time or depth.")
        expected_unit = "s" if domain == "time" else "m"
        if str(raw.get("sample_unit") or "") != expected_unit:
            raise ValueError(
                f"increment_contract.sample_unit must be {expected_unit!r} for {domain}."
            )
        sample_interval = _positive_float(
            raw.get("sample_interval"), "increment_contract.sample_interval"
        )
        if raw.get("sample_axis_uniform") is not True:
            raise ValueError("increment_contract.sample_axis_uniform must be true.")
        if str(raw.get("sample_axis_dtype") or "") != "float64":
            raise ValueError("increment_contract.sample_axis_dtype must be float64.")
        depth_basis = raw.get("depth_basis")
        if domain == "depth" and str(depth_basis or "").lower() != "tvdss":
            raise ValueError("Depth increment contracts require depth_basis=tvdss.")
        if domain == "time" and depth_basis not in (None, ""):
            raise ValueError("Time increment contracts must not declare depth_basis.")
        lowpass = _mapping(raw.get("lowpass"), "increment_contract.lowpass")
        if str(lowpass.get("implementation") or "") != cls.implementation:
            raise ValueError(
                "increment_contract.lowpass.implementation must be "
                "scipy_butter_sosfiltfilt."
            )
        if int(lowpass.get("design_order", 0)) != 6:
            raise ValueError("Canonical lowpass design_order must be 6.")
        if int(lowpass.get("effective_zero_phase_order", 0)) != 12:
            raise ValueError("Canonical lowpass effective_zero_phase_order must be 12.")
        if str(lowpass.get("cutoff_definition") or "") != (
            "single_pass_minus_3db_final_minus_6db"
        ):
            raise ValueError("Unsupported canonical lowpass cutoff_definition.")
        if str(lowpass.get("buffer_mode") or "") != "reflect":
            raise ValueError("Canonical lowpass buffer_mode must be reflect.")
        cutoff_kind = "cutoff_hz" if domain == "time" else "cutoff_wavelength_m"
        cutoff = _positive_float(
            lowpass.get(cutoff_kind), f"increment_contract.lowpass.{cutoff_kind}"
        )
        buffer_axis_units = _positive_float(
            lowpass.get("buffer_axis_units"),
            "increment_contract.lowpass.buffer_axis_units",
        )
        expected_buffer = 0.4 if domain == "time" else 400.0
        if not math.isclose(buffer_axis_units, expected_buffer, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{domain} canonical buffer_axis_units must be {expected_buffer}."
            )
        tolerance = _positive_float(
            raw.get("sample_axis_absolute_tolerance", 1.0e-9),
            "increment_contract.sample_axis_absolute_tolerance",
        )
        return cls(
            sample_domain=domain,
            sample_unit=expected_unit,
            sample_interval=sample_interval,
            depth_basis="tvdss" if domain == "depth" else None,
            cutoff=cutoff,
            cutoff_kind=cutoff_kind,
            buffer_axis_units=buffer_axis_units,
            axis_absolute_tolerance=tolerance,
        )

    def as_dict(self) -> dict[str, Any]:
        lowpass: dict[str, Any] = {
            "implementation": self.implementation,
            "design_order": self.design_order,
            "effective_zero_phase_order": self.effective_zero_phase_order,
            "cutoff_definition": "single_pass_minus_3db_final_minus_6db",
            "buffer_mode": self.buffer_mode,
            "buffer_axis_units": self.buffer_axis_units,
            self.cutoff_kind: self.cutoff,
        }
        result: dict[str, Any] = {
            "semantics": "canonical_complement_log_ai",
            "sample_domain": self.sample_domain,
            "sample_unit": self.sample_unit,
            "sample_interval": self.sample_interval,
            "sample_axis_uniform": True,
            "sample_axis_dtype": "float64",
            "sample_axis_absolute_tolerance": self.axis_absolute_tolerance,
            "value_domain": "log(AI)",
            "log_base": "natural",
            "ai_unit_convention": "m/s*g/cm3",
            "lowpass": lowpass,
        }
        if self.depth_basis is not None:
            result["depth_basis"] = self.depth_basis
        return result

    @property
    def cutoff_cycles_per_unit(self) -> float:
        return self.cutoff if self.sample_domain == "time" else 1.0 / self.cutoff

    @property
    def pad_samples(self) -> int:
        return int(math.ceil(self.buffer_axis_units / self.sample_interval))

    @property
    def minimum_segment_samples(self) -> int:
        return max(21, self.pad_samples + 1)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return dict(value)


def _positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite number.") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a positive finite number.")
    return result


def validate_sample_axis(
    sample_axis: np.ndarray,
    contract: CanonicalIncrementContract,
) -> np.ndarray:
    """Validate and return a float64 regular axis."""
    axis = np.asarray(sample_axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("sample_axis must be one-dimensional with at least two samples.")
    expected = axis[0] + np.arange(axis.size, dtype=np.float64) * contract.sample_interval
    if not np.allclose(
        axis,
        expected,
        rtol=0.0,
        atol=contract.axis_absolute_tolerance,
    ):
        raise ValueError(
            "sample_axis does not match increment_contract.sample_interval "
            "within the declared absolute tolerance."
        )
    return axis


def _finite_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def canonical_lowpass(
    values: np.ndarray,
    sample_axis: np.ndarray,
    contract: CanonicalIncrementContract,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply canonical SOS forward/backward filtering per finite segment."""
    axis = validate_sample_axis(sample_axis, contract)
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
        contract.design_order,
        contract.cutoff_cycles_per_unit,
        btype="lowpass",
        fs=1.0 / contract.sample_interval,
        output="sos",
    )
    pad = contract.pad_samples
    for row_index, row_mask in enumerate(flat_mask):
        for start, stop in _finite_segments(row_mask):
            segment = flat[row_index, start:stop]
            if segment.size < contract.minimum_segment_samples:
                raise ValueError(
                    "finite segment is shorter than canonical minimum "
                    f"({segment.size} < {contract.minimum_segment_samples})."
                )
            padded = np.pad(segment, pad, mode="reflect")
            filtered = sosfiltfilt(sos, padded, padtype=None)
            result[row_index, start:stop] = filtered[pad : pad + segment.size]
    return result.reshape(array.shape)


def decompose_log_ai(
    target_log_ai: np.ndarray,
    sample_axis: np.ndarray,
    contract: CanonicalIncrementContract,
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
            "semantics": "canonical_complement_log_ai",
            "sample_domain": domain,
            "sample_unit": unit,
            "sample_interval": float(sample_interval),
            "sample_axis_uniform": True,
            "sample_axis_dtype": "float64",
            "sample_axis_absolute_tolerance": 1.0e-9,
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


__all__ = [
    "CanonicalIncrementContract",
    "canonical_lowpass",
    "decompose_log_ai",
    "generation_contract",
    "validate_sample_axis",
]
