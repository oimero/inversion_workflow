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
    DirectionalEvidence,
    SingleTraceStructuredModel,
    TeacherForcedParameterModel,
    TeacherForcingLossConfig,
    TeacherForcingModelConfig,
    batch_to_torch,
    teacher_forcing_loss,
)
from ginn_v2.hsmm import (
    HsmmPrior,
    HsmmResult,
    HsmmSegment,
    ZoneHsmmPrior,
    exact_hsmm,
    fit_hsmm_prior,
    freeze_hsmm_prior,
)
from ginn_v2.structure import (
    CenterTracePosterior,
    StructuredLossConfig,
    infer_center_trace,
    structured_training_loss,
)
from ginn_v2.structure_training import (
    Stage1Step2Config,
    run_stage1_step2,
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
    forward_context_from_sample,
    project_log_ai_to_model_grid,
    run_artifact_oracle,
    run_oracle,
)
from ginn_v2.observability import (
    TruthBoundaryOracleConfig,
    run_truth_boundary_oracle,
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
    "DirectionalEvidence",
    "ForwardContext",
    "HsmmPrior",
    "HsmmResult",
    "HsmmSegment",
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
    "Stage1Step2Config",
    "StructuredSample",
    "StructuredLossConfig",
    "StructuredTruthAdapter",
    "TeacherForcedParameterModel",
    "TeacherForcingBatch",
    "TeacherForcingDataModule",
    "TeacherForcingLossConfig",
    "TeacherForcingModelConfig",
    "TruthBoundaryOracleConfig",
    "SingleTraceStructuredModel",
    "CenterTracePosterior",
    "ZoneHsmmPrior",
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
    "exact_hsmm",
    "fit_hsmm_prior",
    "forward_numpy",
    "forward_torch",
    "fit_state_duration_baseline",
    "forward_context_from_sample",
    "freeze_parent_split_manifest",
    "freeze_hsmm_prior",
    "load_zone_ai_bounds",
    "project_log_ai_to_model_grid",
    "resolve_device",
    "run_artifact_oracle",
    "run_oracle",
    "run_stage1_step1_controls",
    "run_stage1_step2",
    "run_truth_boundary_oracle",
    "teacher_forcing_loss",
    "infer_center_trace",
    "structured_training_loss",
]
