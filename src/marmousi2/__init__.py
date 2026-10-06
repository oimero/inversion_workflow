"""Small, explicit adapters for the Marmousi2 benchmark.

The benchmark package owns only data preparation and numerical evaluation.
Model training and volume inference remain in :mod:`ginn_v2`; the prepared
configuration emitted here points at those existing workflow entry points.
"""

from .lfm_models import LfmArray, build_truth_huber_trend
from .evaluation import EVALUATION_SCOPES, evaluate_marmousi2_prediction
from .benchmark import comparison_plan, run_comparison_plan, write_comparison_plan
from .raw import MarmousiModels, TimeImpedanceModel, read_marmousi_models, resample_models_to_time
from .seismic import (
    estimate_training_wavelet,
    read_processed_time_segy,
    resample_processed_seismic,
    training_well_correlations,
)

__all__ = [
    "LfmArray",
    "MarmousiModels",
    "TimeImpedanceModel",
    "build_truth_huber_trend",
    "EVALUATION_SCOPES",
    "evaluate_marmousi2_prediction",
    "comparison_plan",
    "run_comparison_plan",
    "write_comparison_plan",
    "estimate_training_wavelet",
    "read_processed_time_segy",
    "read_marmousi_models",
    "resample_models_to_time",
    "resample_processed_seismic",
    "training_well_correlations",
]
