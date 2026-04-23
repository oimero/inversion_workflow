"""Configuration schema for depth-domain GINN training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import yaml

ValidationSplitMode = Literal["none", "spatial_block"]
ValidationBlockAnchor = Literal["maxmax", "maxmin", "minmax", "minmin", "center"]

_PATH_FIELDS = {
    "seismic_file",
    "top_horizon_file",
    "bot_horizon_file",
    "ai_lfm_file",
    "vp_lfm_file",
    "wavelet_file",
    "checkpoint_dir",
}


@dataclass
class DepthGINNConfig:
    """Depth-domain GINN training configuration."""

    seismic_file: Path = Path("your_depth_seismic.segy")
    top_horizon_file: Path = Path("your_top_horizon")
    bot_horizon_file: Path = Path("your_bottom_horizon")
    ai_lfm_file: Path = Path("your_ai_lfm_depth.npz")
    vp_lfm_file: Path = Path("your_vp_lfm_depth.npz")
    wavelet_file: Path = Path("your_wavelet.csv")

    segy_iline: int = 189
    segy_xline: int = 193
    segy_istep: int = 1
    segy_xstep: int = 1

    target_layer_min_thickness: float | None = None
    target_layer_nearest_distance_limit: float | None = None
    target_layer_outlier_threshold: float | None = None
    target_layer_outlier_min_neighbor_count: int = 2

    wavelet_gain: float | None = None
    wavelet_gain_num_traces: int = 256
    wavelet_amplitude_threshold: float = 1e-7

    in_channels: int = 3
    hidden_channels: int = 64
    out_channels: int = 1
    num_res_blocks: int = 8
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    kernel_size: int = 3

    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    lambda_l2: float = 0.03
    lambda_tv: float = 0.0
    ai_min: float = 3000.0
    ai_max: float = 30000.0
    zero_residual_outside_mask: bool = True
    boundary_effect_samples: int = 30

    validation_split_mode: ValidationSplitMode = "spatial_block"
    validation_fraction: float = 0.10
    validation_gap_traces: int = 8
    validation_block_anchor: ValidationBlockAnchor = "maxmin"
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    early_stopping_warmup: int = 5

    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    checkpoint_dir: Path = Path("checkpoints")
    log_interval: int = 50
    save_every: int = 5

    def __post_init__(self) -> None:
        for field_name in _PATH_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                setattr(self, field_name, Path(value))

        if not isinstance(self.dilations, tuple):
            self.dilations = tuple(self.dilations)

        if len(self.dilations) != self.num_res_blocks:
            raise ValueError(f"len(dilations)={len(self.dilations)} != num_res_blocks={self.num_res_blocks}")
        if self.in_channels != 3:
            raise ValueError("Depth GINN expects in_channels=3: seismic + AI LFM + target-layer mask.")
        if self.wavelet_gain is not None and self.wavelet_gain <= 0.0:
            raise ValueError(f"wavelet_gain must be positive when provided, got {self.wavelet_gain}.")
        if self.wavelet_gain_num_traces <= 0:
            raise ValueError(f"wavelet_gain_num_traces must be positive, got {self.wavelet_gain_num_traces}.")
        if self.wavelet_amplitude_threshold < 0.0:
            raise ValueError(
                f"wavelet_amplitude_threshold must be non-negative, got {self.wavelet_amplitude_threshold}."
            )
        if self.lambda_tv < 0.0:
            raise ValueError(f"lambda_tv must be non-negative, got {self.lambda_tv}.")
        if self.target_layer_min_thickness is not None and self.target_layer_min_thickness <= 0.0:
            raise ValueError(
                f"target_layer_min_thickness must be positive when provided, got {self.target_layer_min_thickness}."
            )
        if self.target_layer_nearest_distance_limit is not None and self.target_layer_nearest_distance_limit <= 0.0:
            raise ValueError(
                "target_layer_nearest_distance_limit must be positive when provided, "
                f"got {self.target_layer_nearest_distance_limit}."
            )
        if self.target_layer_outlier_threshold is not None and self.target_layer_outlier_threshold <= 0.0:
            raise ValueError(
                "target_layer_outlier_threshold must be positive when provided, "
                f"got {self.target_layer_outlier_threshold}."
            )
        if self.target_layer_outlier_min_neighbor_count < 1:
            raise ValueError(
                "target_layer_outlier_min_neighbor_count must be >= 1, "
                f"got {self.target_layer_outlier_min_neighbor_count}."
            )
        if self.ai_min <= 0.0:
            raise ValueError(f"ai_min must be positive, got {self.ai_min}.")
        if self.ai_max <= self.ai_min:
            raise ValueError(f"ai_max must be greater than ai_min, got ai_min={self.ai_min}, ai_max={self.ai_max}.")
        if self.boundary_effect_samples < 0:
            raise ValueError(f"boundary_effect_samples must be non-negative, got {self.boundary_effect_samples}.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(f"validation_fraction must be within [0, 1), got {self.validation_fraction}.")
        if self.validation_gap_traces < 0:
            raise ValueError(f"validation_gap_traces must be non-negative, got {self.validation_gap_traces}.")
        if self.early_stopping_patience < 0:
            raise ValueError(f"early_stopping_patience must be non-negative, got {self.early_stopping_patience}.")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError(f"early_stopping_min_delta must be non-negative, got {self.early_stopping_min_delta}.")
        if self.early_stopping_warmup < 0:
            raise ValueError(f"early_stopping_warmup must be non-negative, got {self.early_stopping_warmup}.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, base_dir: Path | None = None) -> "DepthGINNConfig":
        normalized = dict(data)
        for field_name in _PATH_FIELDS:
            if field_name not in normalized or normalized[field_name] is None:
                continue
            path = Path(normalized[field_name])
            if base_dir is not None and not path.is_absolute():
                path = (base_dir / path).resolve()
            normalized[field_name] = path

        if "dilations" in normalized and normalized["dilations"] is not None:
            normalized["dilations"] = tuple(normalized["dilations"])

        return cls(**normalized)

    @classmethod
    def from_yaml(cls, config_file: str | Path, *, base_dir: Path | None = None) -> "DepthGINNConfig":
        config_path = Path(config_file).resolve()
        with config_path.open("r", encoding="utf-8") as fp:
            raw_data = yaml.safe_load(fp) or {}
        if not isinstance(raw_data, dict):
            raise ValueError(f"Expected a mapping in config file {config_path}, got {type(raw_data).__name__}.")

        resolved_base_dir = config_path.parent if base_dir is None else Path(base_dir).resolve()
        return cls.from_dict(raw_data, base_dir=resolved_base_dir)

    def to_json_dict(self) -> Dict[str, Any]:
        return _to_json_compatible(asdict(self))


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_json_compatible(item) for key, item in value.items()}
    return value
