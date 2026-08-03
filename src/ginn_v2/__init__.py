"""Structured GINN V2: a two-scale conditional geological generator."""

from ginn_v2.contracts import (
    ObservableEvidence,
    ObservableTargetContract,
    ObservationTile,
)
from ginn_v2.evidence import EvidenceNetworkConfig, ObservableEvidenceNetwork
from ginn_v2.generator import ConditionalGenerator
from ginn_v2.learning import (
    TargetAuditConfig,
    audit_observable_targets,
    calibrate_semi_markov_fusion,
    evaluate_generator,
    evaluate_semi_markov_oracle,
    train_generator,
)

__all__ = [
    "ConditionalGenerator",
    "EvidenceNetworkConfig",
    "ObservationTile",
    "ObservableEvidence",
    "ObservableEvidenceNetwork",
    "ObservableTargetContract",
    "TargetAuditConfig",
    "audit_observable_targets",
    "calibrate_semi_markov_fusion",
    "evaluate_generator",
    "evaluate_semi_markov_oracle",
    "train_generator",
]
