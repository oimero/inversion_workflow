"""Schema versions shared by well-workflow artifact producers and consumers."""

from __future__ import annotations


WELL_AUTO_TIE_SCHEMA_VERSION = "well_auto_tie_v3"
DEPTH_WAVELET_BATCH_SCHEMA_VERSION = "wavelet_batch_synthetic_depth_v4"


__all__ = [
    "DEPTH_WAVELET_BATCH_SCHEMA_VERSION",
    "WELL_AUTO_TIE_SCHEMA_VERSION",
]
