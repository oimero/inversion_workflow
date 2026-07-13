"""Small public protocol shared by time and depth benchmark readers."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class SyntheticSampleProtocol(Protocol):
    """Fields consumed by domain-independent training/evaluation code."""

    sample_id: str
    sample_kind: str
    sample_domain: str
    target_log_ai: np.ndarray
    canonical_background_log_ai: np.ndarray
    target_increment_log_ai: np.ndarray
    input_lfm_log_ai: np.ndarray
    seismic_input: np.ndarray
    seismic_model_consistent: np.ndarray
    valid_mask: np.ndarray
    lateral_m: np.ndarray

    @property
    def sample_axis(self) -> np.ndarray:
        """Return the regular model sampling axis for this sample."""
        ...

    @property
    def row(self) -> dict[str, Any]:
        ...


__all__ = ["SyntheticSampleProtocol"]
