"""ginn.config — GINN 配置 schema、校验与加载。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import yaml

LmfSource = Literal["wtie_time_lfm", "filtered_inversion_lmf"]

_PATH_FIELDS = {
    "seismic_file",
    "inversion_file",
    "top_horizon_file",
    "bot_horizon_file",
    "precomputed_lmf_file",
    "checkpoint_dir",
}


@dataclass
class GINNConfig:
    """GINN 训练与推理的完整配置。

    所有路径在实际使用前由入口脚本填充；此处默认值仅供参考。
    """

    # ── 数据路径 ──────────────────────────────────────────────
    seismic_file: Path = Path("data/raw/mero se 0116_1ms_new_84_coord.Sgy")
    inversion_file: Path = Path("data/raw/inverted_Zp.sgy")
    top_horizon_file: Path = Path("data/interpre_time/bve_top_t")
    bot_horizon_file: Path = Path("data/interpre_time/itp_bot_t")

    # ── 地震几何（SEG-Y 头字节位置与步长，与 notebook 一致） ──
    segy_iline: int = 5
    segy_xline: int = 21
    segy_istep: int = 1
    segy_xstep: int = 4

    # ── 采样参数 ──────────────────────────────────────────────
    dt: float = 0.001  # 采样间隔 (s)

    # ── 子波 ──────────────────────────────────────────────────
    wavelet_type: str = "ricker"
    wavelet_freq: float = 25.0  # Hz
    wavelet_dt: float = 0.001  # s
    wavelet_length: int = 101  # 采样点数（奇数，中心对称）
    wavelet_gain: float | None = None  # 若为空，则按样本道自动估计
    wavelet_gain_num_traces: int = 256  # 自动估计 wavelet_gain 时采样的有效道数

    # ── 低频模型 ──────────────────────────────────────────────
    lmf_source: LmfSource = "filtered_inversion_lmf"
    precomputed_lmf_file: Path = Path("data/output_lfm_time_from_wtie/lfm_time_from_wtie.npz")
    lmf_cutoff_hz: float = 10.0  # Butterworth 低通截止频率 (Hz)
    lmf_filter_order: int = 6  # 零相位滤波器阶数

    # ── 网络 ──────────────────────────────────────────────────
    in_channels: int = 2  # 地震 + LMF
    hidden_channels: int = 64
    out_channels: int = 1
    num_res_blocks: int = 8
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    kernel_size: int = 3

    # ── 训练 ──────────────────────────────────────────────────
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lambda_reg: float = 0.1  # 残差 L2 正则化权重（防止尺度发散）
    ai_min: float = 4000.0  # 目标层内允许的波阻抗下界
    ai_max: float = 20000.0  # 目标层内允许的波阻抗上界
    zero_residual_outside_mask: bool = True  # 将层外残差平滑压回 0
    device: str = "cuda"
    num_workers: int = 0  # Windows 下大数组无法 pickle，设 0
    pin_memory: bool = True

    # ── 掩码 ──────────────────────────────────────────────────
    mask_erosion_samples: int = 30  # 同时用于 loss mask 收缩与 residual halo/taper 宽度

    # ── 输出 ──────────────────────────────────────────────────
    checkpoint_dir: Path = Path("checkpoints")
    log_interval: int = 50  # 每 N 个 batch 打印一次
    save_every: int = 5  # 每 N 个 epoch 保存 checkpoint

    def __post_init__(self) -> None:
        for field_name in _PATH_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, Path):
                setattr(self, field_name, Path(value))

        if not isinstance(self.dilations, tuple):
            self.dilations = tuple(self.dilations)

        # 确保 dilations 长度与 num_res_blocks 一致
        if len(self.dilations) != self.num_res_blocks:
            raise ValueError(f"len(dilations)={len(self.dilations)} != num_res_blocks={self.num_res_blocks}")

        valid_lmf_sources = {"wtie_time_lfm", "filtered_inversion_lmf"}
        if self.lmf_source not in valid_lmf_sources:
            raise ValueError(f"Unsupported lmf_source={self.lmf_source!r}, expected one of {sorted(valid_lmf_sources)}")

        if self.wavelet_gain is not None and self.wavelet_gain <= 0.0:
            raise ValueError(f"wavelet_gain must be positive when provided, got {self.wavelet_gain}.")
        if self.wavelet_gain_num_traces <= 0:
            raise ValueError(f"wavelet_gain_num_traces must be positive, got {self.wavelet_gain_num_traces}.")
        if self.ai_min <= 0.0:
            raise ValueError(f"ai_min must be positive, got {self.ai_min}.")
        if self.ai_max <= self.ai_min:
            raise ValueError(f"ai_max must be greater than ai_min, got ai_min={self.ai_min}, ai_max={self.ai_max}.")
        if self.mask_erosion_samples < 0:
            raise ValueError(f"mask_erosion_samples must be non-negative, got {self.mask_erosion_samples}.")

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, base_dir: Path | None = None) -> "GINNConfig":
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
