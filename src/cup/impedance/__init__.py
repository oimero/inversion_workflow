"""Shared impedance contracts and canonical increment operators."""

from cup.impedance.canonical import (
    canonical_lowpass,
    decompose_log_ai,
    generation_contract,
)
from cup.impedance.contracts import (
    CanonicalIncrementContract,
    build_lfm_producer_contract,
    validate_contract_compatibility,
    validate_increment_contract,
    validate_lfm_producer_contract,
    validate_sample_axis,
)

__all__ = [
    "CanonicalIncrementContract",
    "build_lfm_producer_contract",
    "canonical_lowpass",
    "decompose_log_ai",
    "generation_contract",
    "validate_contract_compatibility",
    "validate_increment_contract",
    "validate_lfm_producer_contract",
    "validate_sample_axis",
]
