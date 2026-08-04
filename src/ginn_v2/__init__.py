"""Structured GINN V2: a two-scale conditional geological generator."""

from ginn_v2.contracts import (
    CoefficientVarianceCalibration,
    GenerationPolicy,
    ObservableEvidence,
    ObservableTargetContract,
    ObservationTile,
    SegmentExtent,
    SegmentParameterDistribution,
    StructuredPrediction,
)
from ginn_v2.evidence import EvidenceNetworkConfig, ObservableEvidenceNetwork
from ginn_v2.generator import (
    ConditionalGenerator,
    SegmentProfileHeadConfig,
)
from ginn_v2.learning import (
    CoefficientVarianceCalibrationConfig,
    MapProfileProbeConfig,
    SegmentProfileLearningConfig,
    TargetAuditConfig,
    audit_observable_targets,
    calibrate_coefficient_variance,
    calibrate_semi_markov_fusion,
    evaluate_generator,
    evaluate_map_reconstruction,
    evaluate_segment_profile_head,
    evaluate_semi_markov_oracle,
    evaluate_structured_ensemble,
    train_generator,
    train_map_profile_probe,
    train_segment_profile_head,
)

__all__ = [
    "CoefficientVarianceCalibration",
    "CoefficientVarianceCalibrationConfig",
    "ConditionalGenerator",
    "EvidenceNetworkConfig",
    "GenerationPolicy",
    "ObservationTile",
    "ObservableEvidence",
    "ObservableEvidenceNetwork",
    "ObservableTargetContract",
    "MapProfileProbeConfig",
    "SegmentExtent",
    "SegmentParameterDistribution",
    "SegmentProfileHeadConfig",
    "SegmentProfileLearningConfig",
    "StructuredPrediction",
    "TargetAuditConfig",
    "audit_observable_targets",
    "calibrate_coefficient_variance",
    "calibrate_semi_markov_fusion",
    "evaluate_generator",
    "evaluate_map_reconstruction",
    "evaluate_segment_profile_head",
    "evaluate_semi_markov_oracle",
    "evaluate_structured_ensemble",
    "train_generator",
    "train_map_profile_probe",
    "train_segment_profile_head",
]
