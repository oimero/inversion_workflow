"""Unified acoustic forward models.

Top-level functions are the NumPy backend.  Import
``cup.forward.torch_backend`` explicitly for differentiable PyTorch kernels;
this keeps PyTorch optional for non-learning ``cup`` users.
"""

from cup.forward.numpy_backend import (
    DEFAULT_OUTPUT_CHUNK_SIZE,
    ai_from_velocity,
    build_depth_operator,
    forward_depth,
    forward_time,
    reflectivity_from_log_ai,
    velocity_from_ai,
)
from cup.forward.calibration import AIVelocityRelation


__all__ = [
    "AIVelocityRelation",
    "DEFAULT_OUTPUT_CHUNK_SIZE",
    "ai_from_velocity",
    "build_depth_operator",
    "forward_depth",
    "forward_time",
    "reflectivity_from_log_ai",
    "velocity_from_ai",
]
