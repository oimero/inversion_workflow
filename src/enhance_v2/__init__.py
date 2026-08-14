"""Deterministic conditional residual texture transfer."""

from .library import build_residual_library
from .transfer import transfer_residual_texture

__all__ = ["build_residual_library", "transfer_residual_texture"]
