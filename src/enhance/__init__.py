"""Resolution-enhancement utilities and stage-2 training components."""

from enhance.config import EnhancementConfig
from enhance.prior import (
    WellResolutionPriorBundle,
    ai_to_reflectivity,
    fit_delta_to_base_ai_bounds,
    fit_residual_to_lfm_bounds,
    highpass_log_ai_residual,
    load_well_resolution_prior_npz,
    save_well_resolution_prior_npz,
    summarize_well_resolution_prior,
    validate_ai_bounds,
    validate_well_resolution_prior,
)
from enhance.loss import EnhancementLoss, compose_enhanced_ai
from enhance.model import DilatedResNet1D
from enhance.trainer import EnhancementTrainer

__all__ = [
    "EnhancementConfig",
    "EnhancementLoss",
    "EnhancementTrainer",
    "DilatedResNet1D",
    "WellResolutionPriorBundle",
    "ai_to_reflectivity",
    "fit_delta_to_base_ai_bounds",
    "fit_residual_to_lfm_bounds",
    "highpass_log_ai_residual",
    "load_well_resolution_prior_npz",
    "save_well_resolution_prior_npz",
    "summarize_well_resolution_prior",
    "validate_ai_bounds",
    "validate_well_resolution_prior",
    "compose_enhanced_ai",
]
