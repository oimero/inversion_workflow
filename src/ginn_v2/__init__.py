"""Structured GINN V2: a small interface over structured truth and physics."""

from ginn_v2.decoder import (
    DecoderResult,
    RawSegmentParameters,
    decode_numpy,
    decode_torch,
)
from ginn_v2.forward import (
    ForwardContext,
    forward_numpy,
    forward_torch,
)
from ginn_v2.oracle import (
    OracleContractError,
    OracleReport,
    ProjectionResult,
    project_log_ai_to_model_grid,
    run_oracle,
)
from ginn_v2.runtime import configure_training_logger, resolve_device
from ginn_v2.truth import (
    ARTIFACT_TYPE,
    LatentTrace,
    ObservedTrace,
    SegmentTruth,
    StructuredSample,
    StructuredTruthArtifactReader,
    StructuredTruthArtifactWriter,
    StructuredTruthAdapter,
    ZoneTruth,
    assert_structured_sample_equal,
)

__all__ = [
    "ARTIFACT_TYPE",
    "DecoderResult",
    "ForwardContext",
    "LatentTrace",
    "ObservedTrace",
    "OracleContractError",
    "OracleReport",
    "ProjectionResult",
    "RawSegmentParameters",
    "SegmentTruth",
    "StructuredSample",
    "StructuredTruthAdapter",
    "StructuredTruthArtifactReader",
    "StructuredTruthArtifactWriter",
    "ZoneTruth",
    "assert_structured_sample_equal",
    "configure_training_logger",
    "decode_numpy",
    "decode_torch",
    "forward_numpy",
    "forward_torch",
    "project_log_ai_to_model_grid",
    "resolve_device",
    "run_oracle",
]
