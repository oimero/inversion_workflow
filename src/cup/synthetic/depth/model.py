"""Materialized depth-domain generation records.

The records in this module contain no generation policy.  They are the seam
between depth object construction, LFM/seismic adapters, and the artifact
writer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cup.synthetic.core.geometry import SectionGeometry as DepthSectionGeometry
from cup.synthetic.core.records import StructuredSampleRecord
from cup.synthetic.core.scenarios import GenerationScenario


@dataclass(frozen=True)
class DepthGeneratedSection:
    realization_id: str
    scenario: GenerationScenario
    geometry: DepthSectionGeometry
    tvdss_highres_m: np.ndarray
    tvdss_model_m: np.ndarray
    log_ai_highres: np.ndarray
    vp_highres_mps: np.ndarray
    model_target_log_ai: np.ndarray
    vp_model_mps: np.ndarray
    seismic_observed: np.ndarray
    seismic_model_consistent: np.ndarray
    subgrid_forward_residual: np.ndarray
    lfm_log_ai: np.ndarray
    valid_mask_model: np.ndarray
    categorical: dict[str, np.ndarray]
    object_catalog: list[dict[str, Any]]
    object_lateral_coefficients: list[dict[str, Any]]
    qc: dict[str, Any]
    structured_sample: StructuredSampleRecord | None = None


__all__ = ["DepthGeneratedSection", "DepthSectionGeometry"]
