"""Canonical log(AI) decomposition rules for synthetic benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.signal import butter, sosfiltfilt

from cup.utils.masks import true_runs


CANONICAL_CONTRACT_VERSION = "canonical_increment_v1"
CANONICAL_SEMANTICS = "canonical_complement_log_ai"
VALUE_DOMAIN = "log(AI)"
LOG_BASE = "natural"
AI_UNIT_CONVENTION = "m/s*g/cm3"
LOWPASS_IMPLEMENTATION = "scipy_butter_sosfiltfilt"
LOWPASS_CUTOFF_DEFINITION = "single_pass_minus_3db_final_minus_6db"
SAMPLE_INTERVAL_RELATIVE_TOLERANCE = 1.0e-6
SAMPLE_INTERVAL_ABSOLUTE_TOLERANCE = 1.0e-9


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


def _nonnegative_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite nonnegative number.") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number.")
    return result


def _exact_int(value: Any, label: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not np.isfinite(number) or number != int(number):
        raise ValueError(f"{label} must be an integer.")
    return int(number)


@dataclass(frozen=True)
class CanonicalIncrementContract:
    """Resolved numerical and semantic contract for one regular axis."""

    sample_domain: str
    sample_unit: str
    sample_interval: float
    depth_basis: str | None
    cutoff: float
    cutoff_kind: str
    buffer_axis_units: float
    contract_version: str = CANONICAL_CONTRACT_VERSION
    semantics: str = CANONICAL_SEMANTICS
    value_domain: str = VALUE_DOMAIN
    log_base: str = LOG_BASE
    ai_unit_convention: str = AI_UNIT_CONVENTION
    sample_interval_relative_tolerance: float = SAMPLE_INTERVAL_RELATIVE_TOLERANCE
    sample_interval_absolute_tolerance: float = SAMPLE_INTERVAL_ABSOLUTE_TOLERANCE
    design_order: int = 6
    effective_zero_phase_order: int = 12
    implementation: str = LOWPASS_IMPLEMENTATION
    cutoff_definition: str = LOWPASS_CUTOFF_DEFINITION
    buffer_mode: str = "reflect"
    sample_axis_dtype: str = "float64"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CanonicalIncrementContract":
        raw = _mapping(value, "increment_contract")
        if str(raw.get("contract_version") or "") != CANONICAL_CONTRACT_VERSION:
            raise ValueError(
                "increment_contract.contract_version must be "
                f"{CANONICAL_CONTRACT_VERSION!r}."
            )
        for key, expected in (
            ("semantics", CANONICAL_SEMANTICS),
            ("value_domain", VALUE_DOMAIN),
            ("log_base", LOG_BASE),
            ("ai_unit_convention", AI_UNIT_CONVENTION),
        ):
            if str(raw.get(key) or "") != expected:
                raise ValueError(f"increment_contract.{key} must be {expected!r}.")
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
        relative_tolerance = _nonnegative_float(
            raw.get("sample_interval_relative_tolerance"),
            "increment_contract.sample_interval_relative_tolerance",
        )
        absolute_tolerance = _nonnegative_float(
            raw.get("sample_interval_absolute_tolerance"),
            "increment_contract.sample_interval_absolute_tolerance",
        )
        lowpass = _mapping(raw.get("lowpass"), "increment_contract.lowpass")
        if str(lowpass.get("implementation") or "") != LOWPASS_IMPLEMENTATION:
            raise ValueError(
                "increment_contract.lowpass.implementation must be "
                f"{LOWPASS_IMPLEMENTATION}."
            )
        if _exact_int(lowpass.get("design_order"), "increment_contract.lowpass.design_order") != 6:
            raise ValueError("Canonical lowpass design_order must be 6.")
        if _exact_int(
            lowpass.get("effective_zero_phase_order"),
            "increment_contract.lowpass.effective_zero_phase_order",
        ) != 12:
            raise ValueError("Canonical lowpass effective_zero_phase_order must be 12.")
        if str(lowpass.get("cutoff_definition") or "") != LOWPASS_CUTOFF_DEFINITION:
            raise ValueError("Unsupported canonical lowpass cutoff_definition.")
        if str(lowpass.get("buffer_mode") or "") != "reflect":
            raise ValueError("Canonical lowpass buffer_mode must be reflect.")
        cutoff_kind = "cutoff_hz" if domain == "time" else "cutoff_wavelength_m"
        cutoff = _positive_float(
            lowpass.get(cutoff_kind), f"increment_contract.lowpass.{cutoff_kind}"
        )
        unexpected_cutoff = "cutoff_wavelength_m" if domain == "time" else "cutoff_hz"
        if unexpected_cutoff in lowpass:
            raise ValueError(
                f"{domain} increment contracts must not declare {unexpected_cutoff}."
            )
        expected_cutoff = 15.0 if domain == "time" else 400.0
        if not math.isclose(cutoff, expected_cutoff, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"{domain} canonical cutoff must be {expected_cutoff}.")
        buffer_axis_units = _positive_float(
            lowpass.get("buffer_axis_units"),
            "increment_contract.lowpass.buffer_axis_units",
        )
        expected_buffer = 0.4 if domain == "time" else 400.0
        if not math.isclose(
            buffer_axis_units, expected_buffer, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"{domain} canonical buffer_axis_units must be {expected_buffer}."
            )
        return cls(
            contract_version=CANONICAL_CONTRACT_VERSION,
            semantics=CANONICAL_SEMANTICS,
            sample_domain=domain,
            sample_unit=expected_unit,
            sample_interval=sample_interval,
            depth_basis="tvdss" if domain == "depth" else None,
            value_domain=VALUE_DOMAIN,
            log_base=LOG_BASE,
            ai_unit_convention=AI_UNIT_CONVENTION,
            sample_interval_relative_tolerance=relative_tolerance,
            sample_interval_absolute_tolerance=absolute_tolerance,
            cutoff=cutoff,
            cutoff_kind=cutoff_kind,
            buffer_axis_units=buffer_axis_units,
            design_order=6,
            effective_zero_phase_order=12,
            implementation=LOWPASS_IMPLEMENTATION,
            cutoff_definition=LOWPASS_CUTOFF_DEFINITION,
            buffer_mode="reflect",
            sample_axis_dtype="float64",
        )

    @property
    def cutoff_cycles_per_unit(self) -> float:
        return self.cutoff if self.sample_domain == "time" else 1.0 / self.cutoff

    @property
    def pad_samples(self) -> int:
        return int(math.ceil(self.buffer_axis_units / self.sample_interval))

    @property
    def minimum_segment_samples(self) -> int:
        return max(21, self.pad_samples + 1)

    def as_dict(self) -> dict[str, Any]:
        lowpass: dict[str, Any] = {
            "implementation": self.implementation,
            "design_order": self.design_order,
            "effective_zero_phase_order": self.effective_zero_phase_order,
            "cutoff_definition": self.cutoff_definition,
            "buffer_mode": self.buffer_mode,
            "buffer_axis_units": self.buffer_axis_units,
            self.cutoff_kind: self.cutoff,
        }
        result: dict[str, Any] = {
            "contract_version": self.contract_version,
            "semantics": self.semantics,
            "sample_domain": self.sample_domain,
            "sample_unit": self.sample_unit,
            "sample_interval": self.sample_interval,
            "sample_axis_uniform": True,
            "sample_axis_dtype": self.sample_axis_dtype,
            "sample_interval_relative_tolerance": self.sample_interval_relative_tolerance,
            "sample_interval_absolute_tolerance": self.sample_interval_absolute_tolerance,
            "value_domain": self.value_domain,
            "log_base": self.log_base,
            "ai_unit_convention": self.ai_unit_convention,
            "lowpass": lowpass,
        }
        if self.depth_basis is not None:
            result["depth_basis"] = self.depth_basis
        return result


def validate_increment_contract(
    value: CanonicalIncrementContract | Mapping[str, Any],
) -> CanonicalIncrementContract:
    """Parse and validate a materialized canonical increment contract."""
    if isinstance(value, CanonicalIncrementContract):
        value = value.as_dict()
    return CanonicalIncrementContract.from_mapping(value)


def validate_sample_axis(
    sample_axis: np.ndarray,
    contract: CanonicalIncrementContract | Mapping[str, Any],
) -> np.ndarray:
    """Cast a numeric axis to float64 and validate its regular spacing."""
    resolved = validate_increment_contract(contract)
    try:
        axis = np.asarray(sample_axis, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_axis must be numeric.") from exc
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("sample_axis must be one-dimensional with at least two samples.")
    if not np.all(np.isfinite(axis)):
        raise ValueError("sample_axis must not contain NaN or Inf.")
    differences = np.diff(axis)
    if np.any(differences <= 0.0):
        raise ValueError("sample_axis must be strictly increasing without duplicates.")
    expected = axis[0] + np.arange(axis.size, dtype=np.float64) * resolved.sample_interval
    if not np.allclose(
        axis,
        expected,
        rtol=resolved.sample_interval_relative_tolerance,
        atol=resolved.sample_interval_absolute_tolerance,
    ):
        raise ValueError(
            "sample_axis does not match increment_contract.sample_interval "
            "within the declared relative and absolute tolerances."
        )
    return axis


def canonical_lowpass(
    values: np.ndarray,
    sample_axis: np.ndarray,
    contract: CanonicalIncrementContract | Mapping[str, Any],
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
        for start, stop in true_runs(row_mask):
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


__all__ = [
    "CanonicalIncrementContract",
    "canonical_lowpass",
    "generation_contract",
    "validate_increment_contract",
    "validate_sample_axis",
]
