"""Shared domain-neutral contracts for the GINN V2 workflow."""

from ginn_v2.adapters import DepthDomainAdapter, TimeDomainAdapter
from ginn_v2.contracts import CommonObservationBatch, ForwardClosureResult
from ginn_v2.data import (
    ArrayTraceSource,
    InputNormalization,
    PatchBatch,
    PatchKey,
    PatchReader,
    SurveyTraceSource,
    candidate_patch_keys,
    fit_lfm_normalization,
)
from ginn_v2.evaluation import EvaluationMetrics, GateReport, GateThresholds, evaluate_gates
from ginn_v2.inverter import BodyInverter, BodyResult
from ginn_v2.model import BodyNetworkConfig, CenterTraceBodyNet
from ginn_v2.scales import (
    BODY_SMOOTHING_FWHM_M,
    gaussian_smooth_numpy,
    gaussian_smooth_torch,
)
from ginn_v2.trainer import (
    BodyInversionConfig,
    BodyInversionData,
    BodyInversionTrainer,
    build_body_inversion_data,
)

__all__ = [
    "BODY_SMOOTHING_FWHM_M",
    "ArrayTraceSource",
    "BodyNetworkConfig",
    "BodyInverter",
    "BodyResult",
    "CommonObservationBatch",
    "CenterTraceBodyNet",
    "DepthDomainAdapter",
    "EvaluationMetrics",
    "ForwardClosureResult",
    "GateReport",
    "GateThresholds",
    "InputNormalization",
    "PatchBatch",
    "PatchKey",
    "PatchReader",
    "BodyInversionConfig",
    "BodyInversionData",
    "BodyInversionTrainer",
    "SurveyTraceSource",
    "TimeDomainAdapter",
    "build_body_inversion_data",
    "candidate_patch_keys",
    "evaluate_gates",
    "fit_lfm_normalization",
    "gaussian_smooth_numpy",
    "gaussian_smooth_torch",
]
