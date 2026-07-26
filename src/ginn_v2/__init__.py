"""Structured GINN V2: a small interface over structured truth and physics."""

from ginn_v2.anchor import (
    AnchoredSegment,
    LfmAnchoredStructuredSample,
    anchor_to_lfm,
    decode_lfm_anchored_numpy,
    decode_lfm_anchored_torch,
    load_zone_ai_bounds,
)
from ginn_v2.data import (
    ParentSplitManifest,
    TeacherForcingBatch,
    TeacherForcingDataModule,
    build_parent_split_manifest,
    collate_teacher_forcing_samples,
    freeze_parent_split_manifest,
)
from ginn_v2.controls import (
    StateDurationBaseline,
    fit_state_duration_baseline,
    run_stage1_step1_controls,
)
from ginn_v2.decoder import (
    DecoderResult,
    RawSegmentParameters,
    decode_numpy,
    decode_torch,
)
from ginn_v2.model import (
    TeacherForcedParameterModel,
    TeacherForcingLossConfig,
    TeacherForcingModelConfig,
    batch_to_torch,
    teacher_forcing_loss,
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
    run_artifact_oracle,
    run_oracle,
)
from ginn_v2.runtime import configure_training_logger, resolve_device
from ginn_v2.truth import (
    LatentTrace,
    ObservedTrace,
    SegmentTruth,
    StructuredSample,
    StructuredTruthAdapter,
    ZoneTruth,
    assert_structured_sample_equal,
)

__all__ = [
    "AnchoredSegment",
    "DecoderResult",
    "ForwardContext",
    "LfmAnchoredStructuredSample",
    "LatentTrace",
    "ObservedTrace",
    "OracleContractError",
    "OracleReport",
    "ParentSplitManifest",
    "ProjectionResult",
    "RawSegmentParameters",
    "SegmentTruth",
    "StateDurationBaseline",
    "StructuredSample",
    "StructuredTruthAdapter",
    "TeacherForcedParameterModel",
    "TeacherForcingBatch",
    "TeacherForcingDataModule",
    "TeacherForcingLossConfig",
    "TeacherForcingModelConfig",
    "ZoneTruth",
    "anchor_to_lfm",
    "assert_structured_sample_equal",
    "batch_to_torch",
    "build_parent_split_manifest",
    "collate_teacher_forcing_samples",
    "configure_training_logger",
    "decode_lfm_anchored_numpy",
    "decode_lfm_anchored_torch",
    "decode_numpy",
    "decode_torch",
    "forward_numpy",
    "forward_torch",
    "fit_state_duration_baseline",
    "freeze_parent_split_manifest",
    "load_zone_ai_bounds",
    "project_log_ai_to_model_grid",
    "resolve_device",
    "run_artifact_oracle",
    "run_oracle",
    "run_stage1_step1_controls",
    "teacher_forcing_loss",
]
