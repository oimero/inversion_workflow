"""GINN 训练入口脚本。

使用方法
--------
在仓库根目录下运行：

    python experiments/ginn/run_training.py

需要将 ``src`` 加入 PYTHONPATH，或通过 ``train_network.ps1`` 类似脚本启动。
"""

import logging
import sys
from pathlib import Path

# ── 确保 src/ 在搜索路径中 ──
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent.parent
_src_dir = _repo_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from ginn.config import GINNConfig
from ginn.trainer import Trainer


def main() -> None:
    # ── 日志 ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    data_root = _repo_root / "data"

    cfg = GINNConfig(
        # ── 数据路径 ──────────────────────────────────────────────
        seismic_file=data_root / "raw" / "mero se 0116_1ms_new_84_coord.Sgy",
        inversion_file=data_root / "raw" / "inverted_Zp.sgy",
        top_horizon_file=data_root / "interpre_time" / "bve_top_t",
        bot_horizon_file=data_root / "interpre_time" / "itp_bot_t",
        # ── 地震几何（SEG-Y 头字节位置与步长，与 notebook 一致） ──
        segy_iline=5,
        segy_xline=21,
        segy_istep=1,
        segy_xstep=4,
        # ── 采样参数 ──────────────────────────────────────────────
        dt=0.001,
        n_samples=1201,
        # ── 子波 ──────────────────────────────────────────────────
        wavelet_type="ricker",
        wavelet_freq=25.0,
        wavelet_dt=0.001,
        wavelet_length=301,
        wavelet_gain=10.0,
        # ── 低频模型 ──────────────────────────────────────────────
        lmf_source="wtie_time_lfm",
        precomputed_lmf_file=data_root / "lfm_time_from_wtie_output" / "lfm_time_from_wtie.npz",
        lmf_cutoff_hz=10.0,
        lmf_filter_order=6,
        # ── 网络 ──────────────────────────────────────────────────
        in_channels=2,
        hidden_channels=64,
        out_channels=1,
        num_res_blocks=8,
        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
        kernel_size=3,
        # ── 训练 ──────────────────────────────────────────────────
        batch_size=16,
        epochs=50,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        lambda_reg=0.1,  # 残差 L2 正则化权重
        device="cuda",
        num_workers=0,
        pin_memory=True,
        # ── 掩码 ──────────────────────────────────────────────────
        # mask_erosion_samples=30,  # 实际间距最小 69ms，安全收缩 30 点/侧
        # ── 输出 ──────────────────────────────────────────────────
        checkpoint_dir=_script_dir / "checkpoints",
        log_interval=50,
        save_every=5,
    )

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
