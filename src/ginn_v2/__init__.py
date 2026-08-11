"""Shared domain-neutral contracts for the GINN V2 workflow."""

from ginn_v2.adapters import DepthDomainAdapter, TimeDomainAdapter
from ginn_v2.contracts import CommonObservationBatch, ForwardClosureResult
from ginn_v2.scales import (
    BODY_SMOOTHING_FWHM_M,
    gaussian_smooth_numpy,
    gaussian_smooth_torch,
)

__all__ = [
    "BODY_SMOOTHING_FWHM_M",
    "CommonObservationBatch",
    "DepthDomainAdapter",
    "ForwardClosureResult",
    "TimeDomainAdapter",
    "gaussian_smooth_numpy",
    "gaussian_smooth_torch",
]
