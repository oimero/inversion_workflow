"""Structured GINN V2: a two-scale conditional geological generator."""

from ginn_v2.contracts import (
    BandlimitedEvidence,
    GenerationPolicy,
    ObservationTile,
    StructuredPrediction,
    VolumeInferenceResult,
)
from ginn_v2.evidence import BandlimitedEvidenceNetwork, EvidenceNetworkConfig
from ginn_v2.generator import ConditionalGenerator
from ginn_v2.inference import (
    forward_diagnostic,
    infer_fused_section,
    infer_section,
    infer_volume,
)
from ginn_v2.learning import evaluate_generator, train_generator

__all__ = [
    "BandlimitedEvidence",
    "BandlimitedEvidenceNetwork",
    "ConditionalGenerator",
    "EvidenceNetworkConfig",
    "GenerationPolicy",
    "ObservationTile",
    "StructuredPrediction",
    "VolumeInferenceResult",
    "evaluate_generator",
    "forward_diagnostic",
    "infer_fused_section",
    "infer_section",
    "infer_volume",
    "train_generator",
]
