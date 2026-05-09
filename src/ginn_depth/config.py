"""ginn_depth.config — 深度域 GINN 配置 schema、校验与加载。"""

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
    "checkpoint_dir",
}


@dataclass
class DepthGINNConfig:
    """深度域 GINN 训练与推理的完整配置。"""

    # ── 地震信息 ──────────────────────────────────────────────
    # 输入深度域地震体及 SEG-Y 几何读取方式。这里的头字节位置和步长要十分小心。
    seismic_file: Path = Path("your_depth_seismic.segy")  # 输入深度域地震体路径。
    segy_iline: int = 189  # inline 头字节位置。
    segy_xline: int = 193  # xline 头字节位置。
    segy_istep: int = 1  # inline 抽样步长。
    segy_xstep: int = 1  # xline 抽样步长。

    # ── 目的层 ────────────────────────────────────────────────
    # 目的层顶/底界决定 waveform loss 和 residual 自由度的有效区域。下面几个
    # 可选 QC 参数用于防御层位异常；除非层位诊断显示薄层、跳点或孤立异常，
    # 否则建议保持默认。
    top_horizon_file: Path = Path("your_top_horizon")  # 目的层顶界解释面路径。
    bot_horizon_file: Path = Path("your_bottom_horizon")  # 目的层底界解释面路径。
    target_layer_min_thickness: float | None = None  # 相邻层位最小厚度；为空时使用地震样本间距。
    target_layer_nearest_distance_limit: float | None = None  # 层位解释 nearest 插值的最远距离；为空时不限制。
    target_layer_outlier_threshold: float | None = 30  # 孤立层位点剔除阈值；为空时禁用。
    target_layer_outlier_min_neighbor_count: int = 2  # 孤立点判断所需最小十字邻域有效点数。

    # ── 低频模型 ──────────────────────────────────────────────
    # LFM 是 AI 的低频锚点，网络只预测相对 LFM 的高频扰动。
    ai_lfm_file: Path = Path("your_ai_lfm_depth.npz")  # 深度域 AI 低频模型 NPZ。

    # ── 深度域子波 ────────────────────────────────────────────
    # 深度域需要创建深度域的等效子波核矩阵。真实实验优先使用井震标定得到
    # 的预计算子波；Ricker 子波主要用于快速实验或缺少标定子波时的兜底。
    vp_lfm_file: Path = Path("your_vp_lfm_depth.npz")  # 深度域 Vp 低频模型 NPZ。
    wavelet_source: WaveletSource = "precomputed_wavelet"  # 子波来源：预计算子波或 Ricker 子波。
    wavelet_file: Path | None = Path("your_wavelet.csv")  # 预计算子波 CSV。
    wavelet_type: str = "ricker"  # 生成 Ricker 子波时使用的子波类型名。
    wavelet_freq: float = 25.0  # Ricker 子波主频（Hz）。
    wavelet_dt: float = 0.001  # 子波采样间隔（秒）。
    wavelet_length: int = 301  # 子波长度（采样点数，建议奇数）。
    wavelet_amplitude_threshold: float = 1e-7  # 深度域卷积矩阵中裁剪微小子波尾巴的阈值。

    # ── 振幅补偿 ──────────────────────────────────────────────
    # 振幅补偿让正演地震和归一化观测地震处于同一量级。fixed_gain 是全局
    # 标量；dynamic_gain_model 是随样点变化的增益体，通常应配合
    # include_dynamic_gain_input=True，让网络看到增益上下文。
    gain_source: GainSource = "fixed_gain"  # 振幅补偿来源：固定标量增益或动态增益体。
    fixed_gain: float | None = None  # 固定标量增益；gain_source=fixed_gain 且为空时自动估计。
    fixed_gain_num_traces: int = 256  # 自动估计固定增益时采样的有效道数。
    dynamic_gain_model: Path | None = None  # gain_source=dynamic_gain_model 时使用的预计算动态增益体。

    # ── 网络结构 ──────────────────────────────────────────────
    # 网络输入通道顺序为：地震、可选 LFM、可选 mask、可选 dynamic gain log-ratio。
    # in_channels 必须等于启用通道数。dilation 序列决定纵向感受野，改动会影响
    # 网络能利用的地震上下文尺度。
    include_lfm_input: bool = True  # 是否将 LFM 作为网络输入通道；地震通道始终启用。
    include_mask_input: bool = True  # 是否将目的层 mask 作为网络输入通道。
    include_dynamic_gain_input: bool = False  # 是否将 log-normalized dynamic gain 作为网络输入通道。
    in_channels: int = 3  # 网络输入通道数：地震 + 可选 LFM/mask/dynamic gain。
    hidden_channels: int = 64  # 残差块内部的隐藏通道数。
    out_channels: int = 1  # 网络输出通道数，对应高频扰动。
    num_res_blocks: int = 8  # 残差块数量。
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)  # 各残差块的 dilation 序列。
    kernel_size: int = 3  # 一维卷积核大小。

    # ── 优化与训练循环 ────────────────────────────────────────
    # 常规 Adam 训练参数。未来如需加入新的训练阶段，应优先在专用 config 中
    # 显式配置，避免把关键训练常数藏在 trainer 里。
    batch_size: int = 16  # 每个 batch 的道数。
    epochs: int = 30  # 最大训练轮数。
    lr: float = 1e-3  # Adam 初始学习率。
    weight_decay: float = 1e-4  # Adam 权重衰减系数。
    grad_clip: float = 1.0  # 梯度裁剪阈值。

    # ── 损失与物理约束 ────────────────────────────────────────
    # 真实数据训练损失由 waveform MAE、residual L2 和 residual TV 组成。
    # L2/TV 越强，越能抑制不稳定高频，但也越容易洗掉分辨率；做高分辨率实验
    # 时应和 baseline 对照，不要只看 waveform loss。
    lambda_l2: float = 0.03  # 高频扰动 L2 正则化权重，约束阻抗尺度不要漂移。
    lambda_tv: float = 0.0  # 高频扰动 TV 正则化权重，抑制层内高频 ringing。
    ai_min: float = 3000.0  # 目的层内允许的波阻抗下界。
    ai_max: float = 30000.0  # 目的层内允许的波阻抗上界。
    zero_residual_outside_mask: bool = True  # 是否将层外高频扰动通过 taper 平滑压回 0。
    boundary_effect_samples: int | None = None  # 为空时按子波 5% 有效半支撑自动计算。

    # ── 验证与早停 ────────────────────────────────────────────
    # 地震道空间相关性很强，因此默认使用空间块验证，而不是随机道验证。
    # gap_traces 是训练块和验证块之间的缓冲带，用来降低空间泄漏。
    validation_split_mode: ValidationSplitMode = "spatial_block"  # 验证集切分方式。
    validation_fraction: float = 0.10  # 验证集目标占比。
    validation_gap_traces: int = 8  # 训练区与验证区之间保留的空间缓冲带宽度（道数）。
    validation_block_anchor: ValidationBlockAnchor = "maxmin"  # 验证块落在工区哪个角/位置。
    early_stopping_patience: int = 8  # 连续多少个 epoch 无显著改善后停止训练。
    early_stopping_min_delta: float = 1e-4  # 视为“显著改善”的最小 val_loss 降幅。
    early_stopping_warmup: int = 5  # 早停开始生效前至少先训练的 epoch 数。

    # ── 运行时与输出 ──────────────────────────────────────────
    # 运行设备、DataLoader 和 checkpoint 输出设置。Windows 下 num_workers=0
    # 最稳；checkpoint_dir 会保存 metrics.csv、run_summary.json 和模型权重。
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
        optional_path_fields = {"wavelet_file", "dynamic_gain_model"}
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
