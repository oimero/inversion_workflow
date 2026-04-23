"""ginn_depth — depth-domain GINN components."""

from .config import DepthGINNConfig
from .data import DepthLfmVolume, load_lfm_depth_npz, load_wavelet_csv
from .loss import GINNLoss
from .model import DilatedResBlock, DilatedResNet1D
from .physics import DepthForwardModel, DepthWaveletMatrixBuilder, reflectivity
from .trainer import Trainer

__all__ = [
    "DepthGINNConfig",
    "DepthForwardModel",
    "DepthLfmVolume",
    "DepthWaveletMatrixBuilder",
    "DilatedResBlock",
    "DilatedResNet1D",
    "GINNLoss",
    "Trainer",
    "load_lfm_depth_npz",
    "load_wavelet_csv",
    "reflectivity",
]
