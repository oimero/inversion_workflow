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

    # ── 合成数据来源 ─────────────────────────────────────────
    # depth_config_file 只用于复用 depth GINN 的地震、层位、Vp、子波、gain、
    # 几何和 mask/taper 数据工程；base_ai_file 会被接到原 GINN 的 AI LFM
    # 槽位，作为 stage-2 的低频/base 阻抗。resolution_prior_file 提供井上
    # high-resolution log-AI residual 先验，用来构造 synthetic delta 目标。
    depth_config_file: Path = Path("experiments/ginn_depth/train.yaml")  # depth GINN 数据配置。
    base_ai_file: Path = Path("your_stage1_base_ai_depth.npz")  # stage-1 GINN 输出的 base AI。
    resolution_prior_file: Path = Path("your_well_resolution_prior.npz")  # 井分辨率残差先验。

    # ── 合成样本分布 ─────────────────────────────────────────
    # 每个 epoch 在线随机生成 synthetic traces。well_patch 直接抽井残差片段；
    # unresolved_cluster 在高采样轴上加入薄互层 packet，再降采样回训练采样。
    synthetic_traces_per_epoch: int = 1024  # 每个 epoch 生成的 synthetic 道数。
    synthetic_batch_size: int | None = None  # 为空时使用 batch_size。
    synthetic_patch_fraction: float = 0.7  # 井残差 patch 样本占比权重。
    synthetic_unresolved_fraction: float = 0.3  # 不可分辨薄互层样本占比权重。
    synthetic_well_patch_scale_min: float = 0.35  # 井残差 patch 随机缩放下界。
    synthetic_well_patch_scale_max: float = 0.80  # 井残差 patch 随机缩放上界。
    synthetic_cluster_min_events: int = 2  # 单个薄互层 packet 的最少跳变数。
    synthetic_cluster_max_events: int = 5  # 单个薄互层 packet 的最多跳变数。
    synthetic_cluster_amp_abs_p95_min: float = 0.45  # 薄互层振幅下界，按井残差 abs-p95 缩放。
    synthetic_cluster_amp_abs_p99_max: float = 1.00  # 薄互层振幅上界，按井残差 abs-p99 缩放。
    synthetic_cluster_main_lobe_samples: int | None = None  # 为空时使用合成器默认主瓣宽度。
    synthetic_unresolved_oversample_factor: int = 6  # 薄互层内部高采样倍数。
    synthetic_residual_highpass_samples: int = 31  # well_patch 残差高通窗口。

    # ── 合成样本 QC 与监督区域 ───────────────────────────────
    # delta_supervision_mask 控制 synthetic delta 放置和 delta loss 的区域。
    # core 使用完整目的层 mask；loss 使用 GINN eroded loss_mask，主要用于旧
    # 行为对照。seismic RMS/QC 仍可使用原 loss_mask 评估合成地震是否过强。
    delta_supervision_mask: DeltaSupervisionMask = "core"  # delta 监督区域：core 或 loss。
    synthetic_residual_highpass_samples_loss: int = 7  # QC 中统计 residual 高频能量的窗口。
    synthetic_seismic_rms_match: bool = True  # 是否把 synthetic seismic RMS 匹配到目标尺度。
    synthetic_seismic_rms_target: float = 1.0  # synthetic seismic RMS 目标值。
    synthetic_quality_gate_enabled: bool = True  # 是否启用 synthetic 样本质量门控。
    synthetic_max_residual_near_clip_fraction: float | None = 0.02  # near-clipping 最大比例。
    synthetic_max_seismic_rms_ratio: float | None = 2.0  # synthetic/real RMS 最大比值。
    synthetic_max_seismic_abs_p99_ratio: float | None = 2.5  # synthetic/real abs-p99 最大比值。
    synthetic_min_base_target_waveform_corr: float | None = None  # base/target 地震互相关硬门控；默认只做 QC。
    synthetic_max_resample_attempts: int = 8  # 单样本最多重采样次数。

    # ── AI 合成边界 ──────────────────────────────────────────
    # enhance 会预测 delta_log_ai，并组合 enhanced_ai = base_ai * exp(delta)。
    # 下面的边界只属于 synthetic/enhance，不写回 GINN 配置。
    ai_min: float = 3000.0  # synthetic/enhanced AI 下界。
    ai_max: float = 30000.0  # synthetic/enhanced AI 上界。
    zero_delta_outside_mask: bool = True  # 是否用 taper 将目的层外 delta 平滑压回 0。

    # ── 网络输入与结构 ───────────────────────────────────────
    # 默认只使用 seismic + base_ai，两通道训练。mask/gain 仍保留为可选实验，
    # 但不作为奥卡姆版默认输入。in_channels 必须等于启用通道数。
    include_base_ai_input: bool = True  # 是否输入 stage-1 base AI。
    include_mask_input: bool = False  # 是否把目的层 mask 作为网络输入。
    include_dynamic_gain_input: bool = False  # 是否输入 dynamic gain log-ratio。
    in_channels: int = 2  # 网络输入通道数：地震 + 可选 base/mask/dynamic gain。
    hidden_channels: int = 64  # 残差块隐藏通道数。
    out_channels: int = 1  # 输出 delta_log_ai 通道数。
    num_res_blocks: int = 5  # 残差块数量。
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16)  # 各残差块 dilation 序列。
    kernel_size: int = 3  # 一维卷积核大小。

    # ── Delta 监督损失 ───────────────────────────────────────
    # 训练 loop 不做物理正演反传；loss 直接比较预测和合成目标 delta_log_ai。
    # highpass 项负责细节形状，lowpass/RMS 项负责趋势和能量尺度。
    lambda_delta_lowpass: float = 0.2  # delta 低频分量 SmoothL1 权重。
    lambda_delta_highpass: float = 1.0  # delta 高频分量 SmoothL1 权重。
    lambda_delta_rms: float = 0.05  # delta RMS 尺度匹配权重。
    lambda_delta_rms_underfit: float = 0.0  # 预测能量不足惩罚权重。
    delta_rms_floor: float = 0.7  # underfit 惩罚触发比例。
    delta_lowpass_samples: int = 17  # lowpass 对比窗口。
    delta_highpass_samples: int = 7  # highpass 对比窗口。

    # ── 优化与运行时 ────────────────────────────────────────
    # 常规 Adam 训练参数和 DataLoader/checkpoint 输出设置。Windows 下
    # num_workers=0 最稳；checkpoint_dir 保存 metrics.csv、run_summary.json
    # 和模型权重。
    batch_size: int = 16  # 推理 batch size；训练 batch 默认也用它。
    epochs: int = 10  # 训练 epoch 数。
    lr: float = 5e-4  # Adam 初始学习率。
    weight_decay: float = 1e-4  # Adam 权重衰减。
    grad_clip: float = 1.0  # 梯度裁剪阈值。
    monitor_samples: int = 64  # 固定 monitor synthetic 样本数；设为 0 可关闭。
    monitor_seed: int = 20260507  # 固定 monitor synthetic 抽样随机种子。
    device: str = "cuda"  # 首选训练设备；CUDA 不可用时回退 CPU。
    num_workers: int = 0  # DataLoader worker 数。
    pin_memory: bool = True  # CUDA 训练时是否启用 pinned memory。
    checkpoint_dir: Path = Path("checkpoints_enhance")  # checkpoint 输出目录。
    log_interval: int = 50  # 每隔多少个 batch 打日志。
    save_every: int = 5  # 每隔多少个 epoch 保存常规 checkpoint。

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
        if self.synthetic_min_base_target_waveform_corr is not None and not (
            -1.0 <= self.synthetic_min_base_target_waveform_corr <= 1.0
        ):
            raise ValueError("synthetic_min_base_target_waveform_corr must be within [-1, 1] when provided.")
        if self.monitor_samples < 0:
            raise ValueError("monitor_samples must be non-negative.")
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
