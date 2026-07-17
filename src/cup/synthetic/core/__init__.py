"""Explicit stable exports for shared Synthoseis benchmark contracts."""

from cup.synthetic.core.artifacts import (
    build_attempt_plan,
    geometry_feasibility_rows,
    limit_attempt_plan,
    rejection_reason_summary,
    validate_dataset_metadata,
    validate_debug_attempt_limit,
    validate_training_manifest,
    write_dataset,
)
from cup.synthetic.core.contracts import (
    build_mask_contract,
    build_seismic_input_contract,
    validate_mask_contract,
    validate_seismic_input_contract,
)
from cup.synthetic.core.views import (
    build_seismic_view_metadata,
    validate_seismic_view_metadata,
)
from cup.synthetic.core.pipeline import (
    SeismicViewContext,
    SeismicViewPipeline,
    SyntheticBenchmarkPipeline,
    SyntheticDomainAdapter,
)

__all__ = [
    "build_attempt_plan",
    "build_mask_contract",
    "build_seismic_input_contract",
    "geometry_feasibility_rows",
    "limit_attempt_plan",
    "rejection_reason_summary",
    "validate_dataset_metadata",
    "validate_debug_attempt_limit",
    "validate_mask_contract",
    "validate_seismic_input_contract",
    "build_seismic_view_metadata",
    "validate_seismic_view_metadata",
    "SeismicViewContext",
    "SeismicViewPipeline",
    "SyntheticBenchmarkPipeline",
    "SyntheticDomainAdapter",
    "validate_training_manifest",
    "write_dataset",
]
