"""ginn.config — GINN 配置 schema、校验与加载。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import yaml

LfmSource = Literal["precomputed_lfm", "filtered_inversion_lfm"]
ValidationSplitMode = Literal["none", "spatial_block"]
ValidationBlockAnchor = Literal["maxmax", "maxmin", "minmax", "minmin", "center"]

_PATH_FIELDS = {
    "seismic_file",
    "top_horizon_file",
    "bot_horizon_file",
    "lfm_precomputed_file",
    "lfm_initial_inversion_file",
    "checkpoint_dir",
}


@dataclass
class GINNConfig:
    """GINN 训练与推理的完整配置。

    所有路径在实际使用前由入口脚本填充；此处默认值仅供参考。
    """

    # ── 数据路径 ──────────────────────────────────────────────
    seismic_file: Path = Path("your_seismic_file.sgy")  # 输入地震体路径。
    top_horizon_file: Path = Path("your_top_horizon_file")  # 目标层顶界解释面路径。
    bot_horizon_file: Path = Path("your_bot_horizon_file")  # 目标层底界解释面路径。

    # ── 地震几何（SEG-Y 头字节位置与步长，与 notebook 一致） ──
    segy_iline: int = 189  # inline 头字节位置。
    segy_xline: int = 193  # xline 头字节位置。
    segy_istep: int = 1  # inline 抽样步长。
    segy_xstep: int = 1  # xline 抽样步长。

    # ── 目的层 ────────────────────────────────────────────────
    target_layer_min_thickness: float | None = None  # 相邻层位最小厚度；为空时使用 sample_step。
    target_layer_nearest_distance_limit: float | None = None  # nearest 兜底最大距离；为空时不限制。
    target_layer_outlier_threshold: float | None = 0.02  # 孤立层位点剔除阈值；为空时禁用。
    target_layer_outlier_min_neighbor_count: int = 2  # 孤立点判断所需最小十字邻域有效点数。

    # ── 低频模型 ──────────────────────────────────────────────
    lfm_source: LfmSource = "precomputed_lfm"  # 低频模型来源：预计算结果或对阻抗体低通。
    lfm_precomputed_file: Path | None = Path("your_precomputed_lfm")  # precomputed_lfm
    lfm_initial_inversion_file: Path | None = Path("your_filtered_inversion_lfm")  # filtered_inversion_lfm
    lfm_filter_dt: float = 0.001  # 从初始反演体低通生成 LFM 时的采样间隔（秒）。
    lfm_cutoff_hz: float = 10.0  # 生成 LFM 时的 Butterworth 低通截止频率（Hz）。
    lfm_filter_order: int = 6  # 生成 LFM 时的零相位滤波器阶数。

    # ── 子波 ──────────────────────────────────────────────────
    wavelet_type: str = "ricker"  # 正演使用的子波类型。
    wavelet_freq: float = 25.0  # 子波主频（Hz）。
    wavelet_dt: float = 0.001  # 子波采样间隔（秒）。
    wavelet_length: int = 301  # 子波长度（采样点数，建议奇数）。
    wavelet_gain: float | None = None  # 子波增益；为空时根据样本道自动估计。
    wavelet_gain_num_traces: int = 256  # 自动估计子波增益时采样的有效道数。

    # ── 网络结构 ──────────────────────────────────────────────
    in_channels: int = 3  # 网络输入通道数：地震 + LFM + 目的层 mask。
    hidden_channels: int = 64  # 残差块内部的隐藏通道数。
    out_channels: int = 1  # 网络输出通道数，对应高频扰动。
    num_res_blocks: int = 8  # 残差块数量。
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)  # 各残差块的 dilation 序列。
    kernel_size: int = 3  # 一维卷积核大小。

    # ── 优化与训练循环 ────────────────────────────────────────
    batch_size: int = 16  # 每个 batch 的道数。
    epochs: int = 50  # 最大训练轮数。
    lr: float = 1e-3  # Adam 初始学习率。
    weight_decay: float = 1e-4  # Adam 权重衰减系数。
    grad_clip: float = 1.0  # 梯度裁剪阈值。

    # ── 损失与物理约束 ────────────────────────────────────────
    lambda_l2: float = 0.03  # 高频扰动 L2 正则化权重，约束阻抗尺度不要漂移。
    lambda_tv: float = 0.0  # 高频扰动 TV 正则化权重，抑制高频 ringing。
    ai_min: float = 5000.0  # 目标层内允许的波阻抗下界。
    ai_max: float = 20000.0  # 目标层内允许的波阻抗上界。
    zero_residual_outside_mask: bool = True  # 是否将层外高频扰动通过 taper 平滑压回 0。
    boundary_effect_samples: int = 30  # 同时用于 waveform loss 内缩和高频扰动 halo 宽度。

    # ── 验证与早停 ────────────────────────────────────────────
    validation_split_mode: ValidationSplitMode = "spatial_block"  # 验证集切分方式。
    validation_fraction: float = 0.10  # 验证集目标占比。
    validation_gap_traces: int = 8  # 训练区与验证区之间保留的空间缓冲带宽度（道数）。
    validation_block_anchor: ValidationBlockAnchor = "maxmin"  # 验证块落在工区哪个角/位置。
    early_stopping_patience: int = 8  # 连续多少个 epoch 无显著改善后停止训练。
    early_stopping_min_delta: float = 1e-4  # 视为“显著改善”的最小 val_loss 降幅。
    early_stopping_warmup: int = 5  # 早停开始生效前至少先训练的 epoch 数。

    # ── 运行时与输出 ──────────────────────────────────────────
    device: str = "cuda"  # 首选训练设备；若 CUDA 不可用会自动回退到 CPU。
    num_workers: int = 0  # DataLoader worker 数；Windows 下默认 0 更稳妥。
    pin_memory: bool = True  # CUDA 训练时是否启用 pinned memory。
    checkpoint_dir: Path = Path("checkpoints")  # checkpoint 输出目录。
    log_interval: int = 50  # 每隔多少个训练 batch 打一次日志。
    save_every: int = 5  # 每隔多少个 epoch 额外保存一次常规 checkpoint。

    def __post_init__(self) -> None:
        for field_name in _PATH_FIELDS:
            value = getattr(self, field_name)
            if value is None:
                continue
            if not isinstance(value, Path):
                setattr(self, field_name, Path(value))

        if not isinstance(self.dilations, tuple):
            self.dilations = tuple(self.dilations)

        # 确保 dilations 长度与 num_res_blocks 一致
        if len(self.dilations) != self.num_res_blocks:
            raise ValueError(f"len(dilations)={len(self.dilations)} != num_res_blocks={self.num_res_blocks}")
        if self.in_channels != 3:
            raise ValueError(
                "GINN now expects in_channels=3 because the target-layer mask is part of the network input."
            )

        valid_lfm_sources = {"precomputed_lfm", "filtered_inversion_lfm"}
        if self.lfm_source not in valid_lfm_sources:
            raise ValueError(f"Unsupported lfm_source={self.lfm_source!r}, expected one of {sorted(valid_lfm_sources)}")
        if self.lfm_source == "precomputed_lfm" and self.lfm_precomputed_file is None:
            raise ValueError("lfm_precomputed_file is required when lfm_source='precomputed_lfm'.")
        if self.lfm_source == "filtered_inversion_lfm" and self.lfm_initial_inversion_file is None:
            raise ValueError("lfm_initial_inversion_file is required when lfm_source='filtered_inversion_lfm'.")
        valid_validation_modes = {"none", "spatial_block"}
        if self.validation_split_mode not in valid_validation_modes:
            raise ValueError(
                "Unsupported validation_split_mode="
                f"{self.validation_split_mode!r}, expected one of {sorted(valid_validation_modes)}"
            )
        valid_validation_anchors = {"maxmax", "maxmin", "minmax", "minmin", "center"}
        if self.validation_block_anchor not in valid_validation_anchors:
            raise ValueError(
                "Unsupported validation_block_anchor="
                f"{self.validation_block_anchor!r}, expected one of {sorted(valid_validation_anchors)}"
            )

        if self.wavelet_gain is not None and self.wavelet_gain <= 0.0:
            raise ValueError(f"wavelet_gain must be positive when provided, got {self.wavelet_gain}.")
        if self.wavelet_gain_num_traces <= 0:
            raise ValueError(f"wavelet_gain_num_traces must be positive, got {self.wavelet_gain_num_traces}.")
        if self.lfm_filter_dt <= 0.0:
            raise ValueError(f"lfm_filter_dt must be positive, got {self.lfm_filter_dt}.")
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
    def from_dict(cls, data: Dict[str, Any], *, base_dir: Path | None = None) -> "GINNConfig":
        normalized = dict(data)
        optional_path_fields = {"lfm_precomputed_file", "lfm_initial_inversion_file"}

        for field_name in _PATH_FIELDS:
            if field_name not in normalized:
                continue

            raw_value = normalized[field_name]
            if raw_value is None:
                continue

            if field_name in optional_path_fields and isinstance(raw_value, str) and not raw_value.strip():
                normalized[field_name] = None
                continue

            path = Path(raw_value)
            if base_dir is not None and not path.is_absolute():
                path = (base_dir / path).resolve()
            normalized[field_name] = path

        if "dilations" in normalized and normalized["dilations"] is not None:
            normalized["dilations"] = tuple(normalized["dilations"])

        return cls(**normalized)

    @classmethod
    def from_yaml(cls, config_file: str | Path, *, base_dir: Path | None = None) -> "GINNConfig":
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
