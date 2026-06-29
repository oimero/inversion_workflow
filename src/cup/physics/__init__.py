"""Shared physical models for rock physics and acoustic forward modeling.

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
from cup.physics.rock_physics import (
    EqualWellHuberFit,
    WellAiVpSamples,
    fit_equal_well_huber,
    well_fit_metrics,
)


__all__ = [
    "AIVelocityRelation",
    "EqualWellHuberFit",
    "DEFAULT_OUTPUT_CHUNK_SIZE",
    "ai_from_velocity",
    "build_depth_operator",
    "forward_depth",
    "forward_time",
    "reflectivity_from_log_ai",
    "velocity_from_ai",
    "WellAiVpSamples",
    "fit_equal_well_huber",
    "well_fit_metrics",
]
