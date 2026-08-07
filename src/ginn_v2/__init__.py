"""Structured GINN V2: a section-level conditional event-track generator."""

from ginn_v2.contracts import (
    DomainMismatchError,
    BandlimitedEvidence,
    EventTrack,
    EventTrackRealization,
    GenerationPolicy,
    InputContractError,
    NumericalFailure,
    ObservationTile,
    StructuredEnsemble,
    VolumeInferenceResult,
)
from ginn_v2.generator import ConditionalGenerator
from ginn_v2.events import (
    EventEvaluationConfig,
    EventGeneratorConfig,
    EventLearningConfig,
    EventPolicyCalibrationConfig,
    EventTrackNetwork,
    calibrate_event_generation_policy,
    evaluate_event_generator,
    load_event_generation_policy,
    train_event_generator,
)
from ginn_v2.inference import forward_diagnostic, infer_section
from ginn_v2.learning import (
    EvaluationConfig,
    LearningConfig,
    TargetPreflightConfig,
    evaluate_generator,
    preflight_evidence_targets,
    train_generator,
    validate_observation_contract,
)

__all__ = [
    "ConditionalGenerator",
    "BandlimitedEvidence",
    "DomainMismatchError",
    "EventTrack",
    "EventTrackRealization",
    "EventTrackNetwork",
    "EventEvaluationConfig",
    "EventGeneratorConfig",
    "EventLearningConfig",
    "EventPolicyCalibrationConfig",
    "EvaluationConfig",
    "GenerationPolicy",
    "InputContractError",
    "LearningConfig",
    "TargetPreflightConfig",
    "NumericalFailure",
    "ObservationTile",
    "StructuredEnsemble",
    "VolumeInferenceResult",
    "calibrate_event_generation_policy",
    "evaluate_generator",
    "evaluate_event_generator",
    "forward_diagnostic",
    "infer_section",
    "load_event_generation_policy",
    "preflight_evidence_targets",
    "train_generator",
    "train_event_generator",
    "validate_observation_contract",
]
