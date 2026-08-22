"""Deterministic conditional residual texture transfer."""

from .library import build_residual_library
from .transfer import transfer_residual_field, transfer_residual_texture
from .volume import ResidualTextureVolumeTransfer, VolumeTransferConfig, VolumeTransferResult, ZoneSurface

__all__ = [
    "build_residual_library",
    "ResidualTextureVolumeTransfer",
    "transfer_residual_field",
    "transfer_residual_texture",
    "VolumeTransferConfig",
    "VolumeTransferResult",
    "ZoneSurface",
]
