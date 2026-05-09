"""Configuration schema for stage-2 resolution enhancement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import yaml

_PATH_FIELDS = {
    "depth_config_file",
    "base_ai_file",
    "resolution_prior_file",
    "checkpoint_dir",
}

DeltaSupervisionMask = Literal["core", "loss"]


@dataclass
class EnhancementConfig:
    """Synthetic-only stage-2 enhancement configuration."""

    # ── Stage-1 base and well prior ───────────────────────────
    depth_config_file: Path = Path("experiments/ginn_depth/train.yaml")
    base_ai_file: Path = Path("your_stage1_base_ai_depth.npz")
    resolution_prior_file: Path = Path("your_well_resolution_prior.npz")

    # ── Network inputs ────────────────────────────────────────
    include_base_ai_input: bool = True
    include_mask_input: bool = False
    include_dynamic_gain_input: bool = False
    in_channels: int = 2
    hidden_channels: int = 64
    out_channels: int = 1
    num_res_blocks: int = 5
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16)
    kernel_size: int = 3

    # ── Optimization ──────────────────────────────────────────
    batch_size: int = 16
    epochs: int = 10
    lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # ── Delta loss ────────────────────────────────────────────
    lambda_delta_lowpass: float = 0.2
    lambda_delta_highpass: float = 1.0
    lambda_delta_rms: float = 0.05
    lambda_delta_rms_underfit: float = 0.0
    delta_rms_floor: float = 0.7
    delta_lowpass_samples: int = 17
    delta_highpass_samples: int = 7

    # ── Synthetic data generation ─────────────────────────────
    synthetic_traces_per_epoch: int = 1024
    synthetic_batch_size: int | None = None
    synthetic_patch_fraction: float = 0.7
    synthetic_unresolved_fraction: float = 0.3
    synthetic_well_patch_scale_min: float = 0.35
    synthetic_well_patch_scale_max: float = 0.80
    synthetic_cluster_min_events: int = 2
    synthetic_cluster_max_events: int = 5
    synthetic_cluster_amp_abs_p95_min: float = 0.45
    synthetic_cluster_amp_abs_p99_max: float = 1.00
    synthetic_cluster_main_lobe_samples: int | None = None
    synthetic_unresolved_oversample_factor: int = 6
    synthetic_residual_highpass_samples: int = 31
    synthetic_residual_highpass_samples_loss: int = 7
    synthetic_seismic_rms_match: bool = True
    synthetic_seismic_rms_target: float = 1.0
    synthetic_quality_gate_enabled: bool = True
    synthetic_max_residual_near_clip_fraction: float | None = 0.02
    synthetic_max_seismic_rms_ratio: float | None = 2.0
    synthetic_max_seismic_abs_p99_ratio: float | None = 2.5
    synthetic_max_resample_attempts: int = 8
    delta_supervision_mask: DeltaSupervisionMask = "core"

    # ── AI bounds and runtime ─────────────────────────────────
    ai_min: float = 3000.0
    ai_max: float = 30000.0
    zero_delta_outside_mask: bool = True
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    checkpoint_dir: Path = Path("checkpoints_enhance")
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
        expected_in_channels = 1 + int(self.include_base_ai_input) + int(self.include_mask_input) + int(
            self.include_dynamic_gain_input
        )
        if self.in_channels != expected_in_channels:
            raise ValueError(
                f"in_channels={self.in_channels} does not match enabled input channels "
                f"(expected {expected_in_channels}: seismic + enabled base_ai/mask/dynamic gain)."
            )
        if self.synthetic_traces_per_epoch <= 0:
            raise ValueError("synthetic_traces_per_epoch must be positive.")
        if self.synthetic_batch_size is not None and self.synthetic_batch_size <= 0:
            raise ValueError("synthetic_batch_size must be positive when provided.")
        if self.ai_min <= 0.0 or self.ai_max <= self.ai_min:
            raise ValueError(f"Invalid AI bounds: ai_min={self.ai_min}, ai_max={self.ai_max}.")
        for name in (
            "lambda_delta_lowpass",
            "lambda_delta_highpass",
            "lambda_delta_rms",
            "lambda_delta_rms_underfit",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.delta_lowpass_samples < 1 or self.delta_highpass_samples < 1:
            raise ValueError("delta low/high-pass windows must be positive.")
        if self.delta_supervision_mask not in ("core", "loss"):
            raise ValueError(
                f"delta_supervision_mask={self.delta_supervision_mask!r} must be one of ['core', 'loss']."
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, base_dir: Path | None = None) -> "EnhancementConfig":
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
    def from_yaml(cls, config_file: str | Path, *, base_dir: Path | None = None) -> "EnhancementConfig":
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
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    return value
