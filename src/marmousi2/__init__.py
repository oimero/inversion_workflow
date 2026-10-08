"""Small, explicit adapters for the Marmousi2 benchmark.

The benchmark package owns data preparation and numerical evaluation. Model
training and volume inference remain in the main workflow; this package only
adapts saved predictions for the leakage-aware evaluation protocol.
"""

from .lfm_models import LfmArray, build_truth_huber_trend
from .evaluation import EVALUATION_SCOPES, evaluate_marmousi2_prediction
from .benchmark import evaluate_prediction, load_prediction_array, save_prediction_npz
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
    "evaluate_prediction",
    "load_prediction_array",
    "save_prediction_npz",
    "estimate_training_wavelet",
    "read_processed_time_segy",
    "read_marmousi_models",
    "resample_models_to_time",
    "resample_processed_seismic",
    "training_well_correlations",
]
