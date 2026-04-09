"""ginn.config — 全局超参数配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


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
    n_samples: int = 1201  # 每道采样点数

    # ── 子波 ──────────────────────────────────────────────────
    wavelet_type: str = "ricker"
    wavelet_freq: float = 25.0  # Hz
    wavelet_dt: float = 0.001  # s
    wavelet_length: int = 301  # 采样点数（奇数，中心对称）
    wavelet_gain: float = 10.0  # 子波振幅增益

    # ── 低频模型 ──────────────────────────────────────────────
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
    device: str = "cuda"
    num_workers: int = 0  # Windows 下大数组无法 pickle，设 0
    pin_memory: bool = True

    # ── 掩码 ──────────────────────────────────────────────────
    mask_erosion_samples: int = 30  # 掩码边界收缩采样点数（实际间距最小 69ms，安全上限 ~34）

    # ── 输出 ──────────────────────────────────────────────────
    checkpoint_dir: Path = Path("checkpoints")
    log_interval: int = 50  # 每 N 个 batch 打印一次
    save_every: int = 5  # 每 N 个 epoch 保存 checkpoint

    def __post_init__(self) -> None:
        # 确保 dilations 长度与 num_res_blocks 一致
        if len(self.dilations) != self.num_res_blocks:
            raise ValueError(f"len(dilations)={len(self.dilations)} != num_res_blocks={self.num_res_blocks}")
