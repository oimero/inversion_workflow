"""Structured GINN V2: a two-scale conditional geological generator."""

from ginn_v2.contracts import (
    ObservableEvidence,
    ObservableTargetContract,
    ObservationTile,
    StructuredPrediction,
)
from ginn_v2.generator import ConditionalGenerator
from ginn_v2.inference import (
    forward_diagnostic,
    infer_fused_section,
    infer_section,
    infer_volume,
)
from ginn_v2.learning import (
    evaluate_generator,
    train_generator,
)
from ginn_v2.real_field import (
    RealSectionObservations,
    load_real_section_observations,
)

__all__ = [
    "ConditionalGenerator",
    "ObservationTile",
    "ObservableEvidence",
    "ObservableTargetContract",
    "RealSectionObservations",
    "StructuredPrediction",
    "evaluate_generator",
    "forward_diagnostic",
    "infer_fused_section",
    "infer_section",
    "infer_volume",
    "load_real_section_observations",
    "train_generator",
]
