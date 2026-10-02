"""Small, explicit contracts for canonical impedance decomposition.

The contract is intentionally a numerical and semantic contract, not a
provenance fingerprint.  Producers write it into manifests; consumers parse
and compare it before using materialized arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


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
            raise ValueError(
                f"{domain} canonical cutoff must be {expected_cutoff}."
            )
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


def validate_contract_compatibility(
    increment_contract: CanonicalIncrementContract | Mapping[str, Any],
    lfm_contract: Mapping[str, Any],
) -> None:
    """Ensure an LFM producer contract matches a canonical increment contract."""
    increment = validate_increment_contract(increment_contract)
    lfm = validate_lfm_producer_contract(lfm_contract)
    pairs = (
        ("sample_domain", increment.sample_domain, lfm["sample_domain"]),
        ("sample_unit", increment.sample_unit, lfm["sample_unit"]),
        ("sample_interval", increment.sample_interval, lfm["sample_interval"]),
        (
            "sample_interval_relative_tolerance",
            increment.sample_interval_relative_tolerance,
            lfm["sample_interval_relative_tolerance"],
        ),
        (
            "sample_interval_absolute_tolerance",
            increment.sample_interval_absolute_tolerance,
            lfm["sample_interval_absolute_tolerance"],
        ),
        ("value_domain", increment.value_domain, lfm["value_domain"]),
        ("log_base", increment.log_base, lfm["log_base"]),
        ("ai_unit_convention", increment.ai_unit_convention, lfm["ai_unit_convention"]),
        ("implementation", increment.implementation, lfm["implementation"]),
        ("design_order", increment.design_order, lfm["design_order"]),
        (
            "effective_zero_phase_order",
            increment.effective_zero_phase_order,
            lfm["effective_zero_phase_order"],
        ),
        ("cutoff_definition", increment.cutoff_definition, lfm["cutoff_definition"]),
        ("buffer_mode", increment.buffer_mode, lfm["buffer_mode"]),
        ("buffer_axis_units", increment.buffer_axis_units, lfm["buffer_axis_units"]),
    )
    for name, expected, actual in pairs:
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"lfm_contract.{name} is incompatible with increment_contract.")
        elif actual != expected:
            raise ValueError(f"lfm_contract.{name} is incompatible with increment_contract.")
    if increment.depth_basis != lfm.get("depth_basis"):
        raise ValueError("lfm_contract.depth_basis is incompatible with increment_contract.")
    cutoff_key = increment.cutoff_kind
    if not math.isclose(float(lfm[cutoff_key]), increment.cutoff, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"lfm_contract.{cutoff_key} is incompatible with increment_contract.")


def validate_lfm_producer_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate metadata written by an external or synthetic LFM producer."""
    raw = _mapping(value, "lfm_contract")
    required = {
        "producer_kind",
        "producer_schema",
        "sample_domain",
        "sample_unit",
        "sample_interval",
        "sample_axis_uniform",
        "sample_axis_dtype",
        "sample_interval_relative_tolerance",
        "sample_interval_absolute_tolerance",
        "value_domain",
        "log_base",
        "ai_unit_convention",
        "canonical_lowpass_applied_to",
        "canonical_lowpass_application_count",
        "well_control_lowpass_application_count",
        "final_volume_lowpass_application_count",
        "post_lowpass_vertical_warp_applied",
        "implementation",
        "design_order",
        "effective_zero_phase_order",
        "cutoff_definition",
        "buffer_mode",
        "buffer_axis_units",
        "variant_selection",
        "final_background_complement_response_status",
        "final_background_complement_response_rms",
        "final_background_complement_response_ratio",
        "final_background_max_trace_response_ratio",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"lfm_contract lacks required keys: {missing}")
    domain = str(raw["sample_domain"]).strip().lower()
    if domain not in {"time", "depth"}:
        raise ValueError("lfm_contract.sample_domain must be time or depth.")
    expected_unit = "s" if domain == "time" else "m"
    if raw["sample_unit"] != expected_unit:
        raise ValueError(f"lfm_contract.sample_unit must be {expected_unit!r}.")
    if raw["sample_axis_uniform"] is not True or raw["sample_axis_dtype"] != "float64":
        raise ValueError("lfm_contract must declare a float64 uniform sample axis.")
    if domain == "depth":
        if str(raw.get("depth_basis") or "").lower() != "tvdss":
            raise ValueError("Depth lfm_contract requires depth_basis=tvdss.")
    elif raw.get("depth_basis") not in (None, ""):
        raise ValueError("Time lfm_contract must not declare depth_basis.")
    for key, expected in (
        ("value_domain", VALUE_DOMAIN),
        ("log_base", LOG_BASE),
        ("ai_unit_convention", AI_UNIT_CONVENTION),
        ("implementation", LOWPASS_IMPLEMENTATION),
        ("cutoff_definition", LOWPASS_CUTOFF_DEFINITION),
        ("buffer_mode", "reflect"),
    ):
        if raw[key] != expected:
            raise ValueError(f"lfm_contract.{key} must be {expected!r}.")
    if _exact_int(raw["design_order"], "lfm_contract.design_order") != 6 or _exact_int(
        raw["effective_zero_phase_order"], "lfm_contract.effective_zero_phase_order"
    ) != 12:
        raise ValueError("lfm_contract Butterworth orders must be 6 and 12.")
    _positive_float(raw["sample_interval"], "lfm_contract.sample_interval")
    _nonnegative_float(
        raw["sample_interval_relative_tolerance"],
        "lfm_contract.sample_interval_relative_tolerance",
    )
    _nonnegative_float(
        raw["sample_interval_absolute_tolerance"],
        "lfm_contract.sample_interval_absolute_tolerance",
    )
    _positive_float(raw["buffer_axis_units"], "lfm_contract.buffer_axis_units")
    for key in (
        "canonical_lowpass_application_count",
        "well_control_lowpass_application_count",
        "final_volume_lowpass_application_count",
    ):
        if _exact_int(raw[key], f"lfm_contract.{key}") < 0:
            raise ValueError(f"lfm_contract.{key} must be nonnegative.")
    if raw["canonical_lowpass_applied_to"] not in {
        "target_log_ai",
        "well_controls_before_spatial_modeling",
        "none",
    }:
        raise ValueError("lfm_contract.canonical_lowpass_applied_to is unsupported.")
    if not isinstance(raw["post_lowpass_vertical_warp_applied"], bool):
        raise ValueError("lfm_contract.post_lowpass_vertical_warp_applied must be boolean.")
    if not isinstance(raw["variant_selection"], Mapping):
        raise ValueError("lfm_contract.variant_selection must be a mapping.")
    qc_status = str(raw["final_background_complement_response_status"] or "").strip().lower()
    if qc_status not in {"not_computed", "measured"}:
        raise ValueError(
            "lfm_contract.final_background_complement_response_status must be "
            "'not_computed' or 'measured'."
        )
    for key in (
        "final_background_complement_response_rms",
        "final_background_complement_response_ratio",
        "final_background_max_trace_response_ratio",
    ):
        value = raw[key]
        if qc_status == "not_computed":
            if value is not None:
                raise ValueError(
                    f"lfm_contract.{key} must be null when complement-response QC is not computed."
                )
        else:
            _nonnegative_float(value, f"lfm_contract.{key}")
    cutoff_key = "cutoff_hz" if domain == "time" else "cutoff_wavelength_m"
    unexpected_cutoff = "cutoff_wavelength_m" if domain == "time" else "cutoff_hz"
    if unexpected_cutoff in raw:
        raise ValueError(
            f"{domain} lfm_contract must not declare {unexpected_cutoff}."
        )
    expected_cutoff = 15.0 if domain == "time" else 400.0
    if cutoff_key not in raw or not math.isclose(
        float(raw[cutoff_key]), expected_cutoff, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(f"lfm_contract.{cutoff_key} must be {expected_cutoff}.")
    expected_buffer = 0.4 if domain == "time" else 400.0
    if not math.isclose(
        float(raw["buffer_axis_units"]), expected_buffer, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            f"{domain} lfm_contract.buffer_axis_units must be {expected_buffer}."
        )
    return raw


def validate_synthoseis_lfm_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed LFM production profile used by Synthoseis-lite v4."""
    raw = validate_lfm_producer_contract(value)
    if raw["producer_kind"] != "synthoseis_lite":
        raise ValueError("Synthoseis LFM contracts require producer_kind='synthoseis_lite'.")
    if raw["producer_schema"] != "synthoseis_lite_v4":
        raise ValueError("Synthoseis LFM contracts require producer_schema='synthoseis_lite_v4'.")
    if raw["canonical_lowpass_applied_to"] != "target_log_ai":
        raise ValueError(
            "Synthoseis LFM contracts require canonical_lowpass_applied_to='target_log_ai'."
        )
    expected_counts = {
        "canonical_lowpass_application_count": 1,
        "well_control_lowpass_application_count": 0,
        "final_volume_lowpass_application_count": 0,
    }
    for key, expected in expected_counts.items():
        if int(raw[key]) != expected:
            raise ValueError(
                f"Synthoseis LFM contracts require {key}={expected}."
            )
    if raw["post_lowpass_vertical_warp_applied"] is not False:
        raise ValueError(
            "Synthoseis LFM contracts require post_lowpass_vertical_warp_applied=false."
        )
    return raw


def build_lfm_producer_contract(
    increment_contract: CanonicalIncrementContract | Mapping[str, Any],
    *,
    producer_schema: str,
    variant_selection: Mapping[str, Any] | str,
    producer_kind: str = "synthoseis_lite",
    canonical_lowpass_applied_to: str = "target_log_ai",
    canonical_lowpass_application_count: int = 1,
    well_control_lowpass_application_count: int = 0,
    final_volume_lowpass_application_count: int = 0,
    post_lowpass_vertical_warp_applied: bool = False,
    final_background_complement_response_status: str = "not_computed",
    final_background_complement_response_rms: float | None = None,
    final_background_complement_response_ratio: float | None = None,
    final_background_max_trace_response_ratio: float | None = None,
) -> dict[str, Any]:
    """Create the small manifest contract shared by synthetic and field LFM producers."""
    contract = validate_increment_contract(increment_contract)
    if isinstance(variant_selection, str):
        variant_selection = {"selected": variant_selection}
    result: dict[str, Any] = {
        "producer_kind": str(producer_kind),
        "producer_schema": str(producer_schema),
        "sample_domain": contract.sample_domain,
        "sample_unit": contract.sample_unit,
        "sample_interval": contract.sample_interval,
        "sample_axis_uniform": True,
        "sample_axis_dtype": "float64",
        "sample_interval_relative_tolerance": contract.sample_interval_relative_tolerance,
        "sample_interval_absolute_tolerance": contract.sample_interval_absolute_tolerance,
        "value_domain": contract.value_domain,
        "log_base": contract.log_base,
        "ai_unit_convention": contract.ai_unit_convention,
        "canonical_lowpass_applied_to": canonical_lowpass_applied_to,
        "canonical_lowpass_application_count": int(canonical_lowpass_application_count),
        "well_control_lowpass_application_count": int(well_control_lowpass_application_count),
        "final_volume_lowpass_application_count": int(final_volume_lowpass_application_count),
        "post_lowpass_vertical_warp_applied": bool(post_lowpass_vertical_warp_applied),
        "implementation": contract.implementation,
        "design_order": contract.design_order,
        "effective_zero_phase_order": contract.effective_zero_phase_order,
        "cutoff_definition": contract.cutoff_definition,
        "buffer_mode": contract.buffer_mode,
        "buffer_axis_units": contract.buffer_axis_units,
        "variant_selection": dict(variant_selection),
        "final_background_complement_response_status": str(
            final_background_complement_response_status
        ),
        "final_background_complement_response_rms": (
            None
            if final_background_complement_response_rms is None
            else float(final_background_complement_response_rms)
        ),
        "final_background_complement_response_ratio": (
            None
            if final_background_complement_response_ratio is None
            else float(final_background_complement_response_ratio)
        ),
        "final_background_max_trace_response_ratio": (
            None
            if final_background_max_trace_response_ratio is None
            else float(final_background_max_trace_response_ratio)
        ),
    }
    if contract.depth_basis is not None:
        result["depth_basis"] = contract.depth_basis
    result[contract.cutoff_kind] = contract.cutoff
    validate_lfm_producer_contract(result)
    return result


__all__ = [
    "AI_UNIT_CONVENTION",
    "CANONICAL_CONTRACT_VERSION",
    "CANONICAL_SEMANTICS",
    "CanonicalIncrementContract",
    "LOG_BASE",
    "LOWPASS_CUTOFF_DEFINITION",
    "LOWPASS_IMPLEMENTATION",
    "SAMPLE_INTERVAL_ABSOLUTE_TOLERANCE",
    "SAMPLE_INTERVAL_RELATIVE_TOLERANCE",
    "validate_contract_compatibility",
    "build_lfm_producer_contract",
    "validate_increment_contract",
    "validate_lfm_producer_contract",
    "validate_synthoseis_lfm_contract",
    "validate_sample_axis",
]
