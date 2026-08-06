"""Structured GINN V2: a section-level conditional event-track generator."""

from ginn_v2.contracts import (
    DomainMismatchError,
    EventTrack,
    GenerationPolicy,
    InputContractError,
    NumericalFailure,
    ObservableEvidence,
    ObservationTile,
    StructuredEnsemble,
    VolumeInferenceResult,
)
from ginn_v2.generator import ConditionalGenerator
from ginn_v2.inference import forward_diagnostic, infer_section
from ginn_v2.learning import (
    LearningConfig,
    evaluate_generator,
    train_generator,
    validate_observation_contract,
)

__all__ = [
    "ConditionalGenerator",
    "DomainMismatchError",
    "EventTrack",
    "GenerationPolicy",
    "InputContractError",
    "LearningConfig",
    "NumericalFailure",
    "ObservableEvidence",
    "ObservationTile",
    "StructuredEnsemble",
    "VolumeInferenceResult",
    "evaluate_generator",
    "forward_diagnostic",
    "infer_section",
    "train_generator",
    "validate_observation_contract",
]
