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
from cup.synthetic.core.lfm import (
    build_lfm_degradation_metadata,
    validate_lfm_degradation_metadata,
)
from cup.synthetic.core.seismic_variants import (
    build_seismic_variant_metadata,
    validate_seismic_variant_metadata,
)

__all__ = [
    "build_attempt_plan",
    "build_lfm_degradation_metadata",
    "build_mask_contract",
    "build_seismic_input_contract",
    "build_seismic_variant_metadata",
    "geometry_feasibility_rows",
    "limit_attempt_plan",
    "rejection_reason_summary",
    "validate_dataset_metadata",
    "validate_debug_attempt_limit",
    "validate_lfm_degradation_metadata",
    "validate_mask_contract",
    "validate_seismic_input_contract",
    "validate_seismic_variant_metadata",
    "validate_training_manifest",
    "write_dataset",
]
