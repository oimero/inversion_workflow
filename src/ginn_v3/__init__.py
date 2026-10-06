"""PIAI inversion with independent network, data, training, and artifact contracts."""

from ginn_v3.config import InferenceConfig, NetworkConfig, TrainingConfig
from ginn_v3.workflow import load_body, train_body

__all__ = ["InferenceConfig", "NetworkConfig", "TrainingConfig", "load_body", "train_body"]
