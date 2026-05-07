"""Configuration schema for depth-domain GINN training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import yaml

WaveletSource = Literal["precomputed_wavelet", "ricker_wavelet"]
GainSource = Literal["fixed_gain", "dynamic_gain_model"]
ValidationSplitMode = Literal["none", "spatial_block"]
ValidationBlockAnchor = Literal["maxmax", "maxmin", "minmax", "minmin", "center"]

_PATH_FIELDS = {
    "seismic_file",
    "top_horizon_file",
    "bot_horizon_file",
    "ai_lfm_file",
    "vp_lfm_file",
    "wavelet_file",
    "dynamic_gain_model",
    "resolution_prior_file",
    "checkpoint_dir",
}


@dataclass
class DepthGINNConfig:
    """Depth-domain GINN training configuration."""

    # ── 地震信息 ──────────────────────────────────────────────
    seismic_file: Path = Path("your_depth_seismic.segy")
    segy_iline: int = 189
    segy_xline: int = 193
    segy_istep: int = 1
    segy_xstep: int = 1

    # ── 目的层 ────────────────────────────────────────────────
    top_horizon_file: Path = Path("your_top_horizon")
    bot_horizon_file: Path = Path("your_bottom_horizon")
    target_layer_min_thickness: float | None = None
    target_layer_nearest_distance_limit: float | None = None
    target_layer_outlier_threshold: float | None = None
    target_layer_outlier_min_neighbor_count: int = 2

    # ── 低频模型 ──────────────────────────────────────────────
    ai_lfm_file: Path = Path("your_ai_lfm_depth.npz")
    resolution_prior_file: Path | None = None

    # ── 深度域子波 ────────────────────────────────────────────
    vp_lfm_file: Path = Path("your_vp_lfm_depth.npz")
    wavelet_source: WaveletSource = "precomputed_wavelet"
    wavelet_file: Path | None = Path("your_wavelet.csv")
    wavelet_type: str = "ricker"
    wavelet_freq: float = 25.0
    wavelet_dt: float = 0.001
    wavelet_length: int = 301
    wavelet_amplitude_threshold: float = 1e-7

    # ── 振幅补偿 ──────────────────────────────────────────────
    gain_source: GainSource = "fixed_gain"
    fixed_gain: float | None = None
    fixed_gain_num_traces: int = 256
    dynamic_gain_model: Path | None = None

    # ── 网络结构 ──────────────────────────────────────────────
    include_lfm_input: bool = True
    include_mask_input: bool = True
    include_dynamic_gain_input: bool = False
    in_channels: int = 3
    hidden_channels: int = 64
    out_channels: int = 1
    num_res_blocks: int = 8
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    kernel_size: int = 3

    # ── 优化与训练循环 ────────────────────────────────────────
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # ── 损失与物理约束 ────────────────────────────────────────
    lambda_l2: float = 0.03
    lambda_tv: float = 0.0
    ai_min: float = 3000.0
    ai_max: float = 30000.0
    zero_residual_outside_mask: bool = True
    boundary_effect_samples: int | None = None

    # ── 井先验 synthetic 预训练 ────────────────────────────────
    synthetic_pretrain_enabled: bool = False
    synthetic_pretrain_epochs: int = 0
    synthetic_traces_per_epoch: int = 0
    synthetic_batch_size: int | None = None
    synthetic_patch_fraction: float = 0.7
    synthetic_unresolved_fraction: float = 0.3
    synthetic_cluster_min_events: int = 2
    synthetic_cluster_max_events: int = 5
    synthetic_cluster_main_lobe_samples: int | None = None
    synthetic_residual_highpass_samples: int = 31
    synthetic_seismic_rms_match: bool = True
    synthetic_seismic_rms_target: float = 1.0
    synthetic_lambda_waveform: float = 1.0
    synthetic_lambda_residual_lowpass: float = 0.2
    synthetic_lambda_spectrum: float = 0.05
    synthetic_lambda_rms: float = 0.05
    synthetic_residual_lowpass_samples: int = 17

    # ── 验证与早停 ────────────────────────────────────────────
    validation_split_mode: ValidationSplitMode = "spatial_block"
    validation_fraction: float = 0.10
    validation_gap_traces: int = 8
    validation_block_anchor: ValidationBlockAnchor = "maxmin"
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 1e-4
    early_stopping_warmup: int = 5

    # ── 运行时与输出 ──────────────────────────────────────────
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    checkpoint_dir: Path = Path("checkpoints")
    log_interval: int = 50
    save_every: int = 5

    def __post_init__(self) -> None:
        for field_name in _PATH_FIELDS:
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, Path):
                setattr(self, field_name, Path(value))

        if not isinstance(self.dilations, tuple):
            self.dilations = tuple(self.dilations)

        if len(self.dilations) != self.num_res_blocks:
            raise ValueError(f"len(dilations)={len(self.dilations)} != num_res_blocks={self.num_res_blocks}")
        expected_in_channels = 1 + int(self.include_lfm_input) + int(self.include_mask_input) + int(
            self.include_dynamic_gain_input
        )
        if self.in_channels != expected_in_channels:
            raise ValueError(
                f"in_channels={self.in_channels} does not match enabled input channels "
                f"(expected {expected_in_channels}: seismic + enabled AI LFM/mask/dynamic gain)."
            )
        valid_wavelet_sources = {"precomputed_wavelet", "ricker_wavelet"}
        if self.wavelet_source not in valid_wavelet_sources:
            raise ValueError(
                f"Unsupported wavelet_source={self.wavelet_source!r}, expected one of {sorted(valid_wavelet_sources)}"
            )
        if self.wavelet_source == "precomputed_wavelet" and self.wavelet_file is None:
            raise ValueError("wavelet_file is required when wavelet_source='precomputed_wavelet'.")
        valid_gain_sources = {"fixed_gain", "dynamic_gain_model"}
        if self.gain_source not in valid_gain_sources:
            raise ValueError(
                f"Unsupported gain_source={self.gain_source!r}, expected one of {sorted(valid_gain_sources)}"
            )
        if self.gain_source == "dynamic_gain_model" and self.dynamic_gain_model is None:
            raise ValueError("dynamic_gain_model is required when gain_source='dynamic_gain_model'.")
        if self.fixed_gain is not None and self.fixed_gain <= 0.0:
            raise ValueError(f"fixed_gain must be positive when provided, got {self.fixed_gain}.")
        if self.fixed_gain_num_traces <= 0:
            raise ValueError(f"fixed_gain_num_traces must be positive, got {self.fixed_gain_num_traces}.")
        if self.wavelet_freq <= 0.0:
            raise ValueError(f"wavelet_freq must be positive, got {self.wavelet_freq}.")
        if self.wavelet_dt <= 0.0:
            raise ValueError(f"wavelet_dt must be positive, got {self.wavelet_dt}.")
        if self.wavelet_length < 2:
            raise ValueError(f"wavelet_length must be at least 2, got {self.wavelet_length}.")
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
        if self.boundary_effect_samples is not None and self.boundary_effect_samples < 0:
            raise ValueError(f"boundary_effect_samples must be non-negative, got {self.boundary_effect_samples}.")
        if self.synthetic_pretrain_enabled:
            if self.resolution_prior_file is None:
                raise ValueError("resolution_prior_file is required when synthetic_pretrain_enabled=True.")
            if self.synthetic_pretrain_epochs <= 0:
                raise ValueError("synthetic_pretrain_epochs must be positive when synthetic pretrain is enabled.")
            if self.synthetic_traces_per_epoch <= 0:
                raise ValueError("synthetic_traces_per_epoch must be positive when synthetic pretrain is enabled.")
        elif self.synthetic_pretrain_epochs < 0:
            raise ValueError(f"synthetic_pretrain_epochs must be non-negative, got {self.synthetic_pretrain_epochs}.")
        elif self.synthetic_traces_per_epoch < 0:
            raise ValueError(f"synthetic_traces_per_epoch must be non-negative, got {self.synthetic_traces_per_epoch}.")
        if self.synthetic_batch_size is not None and self.synthetic_batch_size <= 0:
            raise ValueError(f"synthetic_batch_size must be positive when provided, got {self.synthetic_batch_size}.")
        if self.synthetic_patch_fraction < 0.0 or self.synthetic_unresolved_fraction < 0.0:
            raise ValueError("synthetic_patch_fraction and synthetic_unresolved_fraction must be non-negative.")
        if self.synthetic_patch_fraction + self.synthetic_unresolved_fraction <= 0.0:
            raise ValueError("synthetic patch/unresolved fractions must have a positive sum.")
        if self.synthetic_cluster_min_events < 1:
            raise ValueError("synthetic_cluster_min_events must be >= 1.")
        if self.synthetic_cluster_max_events < self.synthetic_cluster_min_events:
            raise ValueError("synthetic_cluster_max_events must be >= synthetic_cluster_min_events.")
        if self.synthetic_cluster_main_lobe_samples is not None and self.synthetic_cluster_main_lobe_samples < 1:
            raise ValueError("synthetic_cluster_main_lobe_samples must be positive when provided.")
        if self.synthetic_residual_highpass_samples < 3:
            raise ValueError("synthetic_residual_highpass_samples must be >= 3.")
        if self.synthetic_seismic_rms_target <= 0.0:
            raise ValueError("synthetic_seismic_rms_target must be positive.")
        for field_name in (
            "synthetic_lambda_waveform",
            "synthetic_lambda_residual_lowpass",
            "synthetic_lambda_spectrum",
            "synthetic_lambda_rms",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative, got {getattr(self, field_name)}.")
        if self.synthetic_residual_lowpass_samples < 1:
            raise ValueError("synthetic_residual_lowpass_samples must be >= 1.")
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
        optional_path_fields = {"wavelet_file", "dynamic_gain_model", "resolution_prior_file"}
        for field_name in _PATH_FIELDS:
            if field_name not in normalized or normalized[field_name] is None:
                continue
            if (
                field_name in optional_path_fields
                and isinstance(normalized[field_name], str)
                and not normalized[field_name].strip()
            ):
                normalized[field_name] = None
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
