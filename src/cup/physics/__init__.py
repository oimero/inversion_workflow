"""Shared physical models, canonical AI contracts, and acoustic forward modeling.

Top-level functions are the NumPy backend.  Import
``cup.physics.torch_backend`` explicitly for differentiable PyTorch kernels;
this keeps PyTorch optional for non-learning ``cup`` users.
"""

from cup.physics.numpy_backend import (
    DEFAULT_OUTPUT_CHUNK_SIZE,
    ai_from_velocity,
    build_depth_operator,
    forward_depth,
    forward_time,
    reflectivity_from_log_ai,
    velocity_from_ai,
)
from cup.physics.calibration import AIVelocityRelation
from cup.physics.canonical import (
    canonical_lowpass,
    decompose_log_ai,
    generation_contract,
)
from cup.physics.contracts import (
    CanonicalIncrementContract,
    build_lfm_producer_contract,
    validate_contract_compatibility,
    validate_increment_contract,
    validate_lfm_producer_contract,
    validate_sample_axis,
    validate_synthoseis_lfm_contract,
)
from cup.physics.rock_physics import (
    EqualWellHuberFit,
    WellAiVpSamples,
    fit_equal_well_huber,
    well_fit_metrics,
)


__all__ = [
    "AIVelocityRelation",
    "CanonicalIncrementContract",
    "EqualWellHuberFit",
    "DEFAULT_OUTPUT_CHUNK_SIZE",
    "ai_from_velocity",
    "build_lfm_producer_contract",
    "build_depth_operator",
    "canonical_lowpass",
    "decompose_log_ai",
    "forward_depth",
    "forward_time",
    "generation_contract",
    "reflectivity_from_log_ai",
    "validate_contract_compatibility",
    "validate_increment_contract",
    "validate_lfm_producer_contract",
    "validate_sample_axis",
    "validate_synthoseis_lfm_contract",
    "velocity_from_ai",
    "WellAiVpSamples",
    "fit_equal_well_huber",
    "well_fit_metrics",
]
