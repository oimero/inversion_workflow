"""ginn.data — 数据加载、预处理与 Dataset 定义。

职责
----
1. 读取 SEG-Y 地震体和反演体
2. 从反演体生成低频模型（Butterworth 低通滤波）
3. 生成理论子波（Ricker）
4. 从层位文件生成 3D 布尔掩码
5. 封装为 PyTorch Dataset
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from cup.petrel.load import import_interpretation_petrel, import_seismic
from cup.seismic.process import TargetLayer, interpolate_interpretation_surface
from cup.seismic.survey import open_survey
from ginn.config import GINNConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  SEG-Y I/O
# ═══════════════════════════════════════════════════════════════

# TODO：待提取为单独函数

# ═══════════════════════════════════════════════════════════════
#  低频模型
# ═══════════════════════════════════════════════════════════════


def make_lowfreq_model(
    impedance_volume: np.ndarray,
    dt_s: float = 0.001,
    cutoff_hz: float = 10.0,
    order: int = 6,
) -> np.ndarray:
    """对反演阻抗体逐道做 Butterworth 零相位低通滤波，生成低频模型。

    Parameters
    ----------
    impedance_volume : np.ndarray
        反演阻抗体，shape ``(n_il, n_xl, n_sample)``。
    dt_s : float
        采样间隔，秒。
    cutoff_hz : float
        低通截止频率，Hz。
    order : int
        滤波器阶数。

    Returns
    -------
    np.ndarray
        低频模型，shape 与输入相同。
    """
    from wtie.processing.spectral import apply_butter_lowpass_filter

    fs = 1.0 / dt_s
    n_il, n_xl, n_sample = impedance_volume.shape
    lmf = np.zeros_like(impedance_volume)

    for i in range(n_il):
        for j in range(n_xl):
            trace = impedance_volume[i, j, :].astype(np.float64)
            if np.all(trace == 0):
                continue
            lmf[i, j, :] = apply_butter_lowpass_filter(
                trace,
                highcut=cutoff_hz,
                fs=fs,
                order=order,
                zero_phase=True,
            )

    logger.info(
        "Generated LMF: cutoff=%.1f Hz, order=%d, shape=%s",
        cutoff_hz,
        order,
        lmf.shape,
    )
    return lmf.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  子波
# ═══════════════════════════════════════════════════════════════


def make_wavelet(
    wavelet_type: str = "ricker",
    freq: float = 25.0,
    dt: float = 0.001,
    length: int = 101,
    gain: float = 1.0,
) -> np.ndarray:
    """生成理论子波。

    Parameters
    ----------
    wavelet_type : str
        子波类型，当前支持 ``"ricker"``。
    freq : float
        主频 (Hz)。
    dt : float
        采样间隔 (s)。
    length : int
        采样点数。
    gain : float
        子波振幅增益参数。

    Returns
    -------
    np.ndarray
        子波振幅，shape ``(length,)``。
    """
    if wavelet_type == "ricker":
        from wtie.modeling.wavelet import ricker

        _, y = ricker(freq, dt, length)
    else:
        raise ValueError(f"Unsupported wavelet_type: {wavelet_type}")
    y = y * float(gain)

    logger.info(
        "Generated %s wavelet: freq=%.1f Hz, dt=%.4f s, length=%d, gain=%.2f", wavelet_type, freq, dt, length, gain
    )
    return y.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  层位掩码
# ═══════════════════════════════════════════════════════════════

# TODO：待提取为单独函数

# ═══════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ═══════════════════════════════════════════════════════════════


class SeismicTraceDataset(Dataset):
    """逐道 1D 地震数据 Dataset。

    每个 item 包含：
    - ``input``：(2, n_t) — 归一化地震道 + 归一化 LMF
    - ``obs``：(1, n_t) — 归一化观测地震道（用于损失计算）
    - ``mask``：(1, n_t) — 布尔掩码
    - ``lmf_raw``：(1, n_t) — 原始 LMF（用于正演时恢复阻抗）

    Parameters
    ----------
    seismic : np.ndarray
        地震体，shape ``(n_il, n_xl, n_t)``。
    lmf : np.ndarray
        低频模型，shape ``(n_il, n_xl, n_t)``。
    mask : np.ndarray
        布尔掩码，shape ``(n_il, n_xl, n_t)``。
    """

    def __init__(
        self,
        seismic: np.ndarray,
        lmf: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        n_il, n_xl, n_t = seismic.shape
        assert lmf.shape == seismic.shape
        assert mask.shape == seismic.shape

        # 找出掩码中至少有 1 个 True 的道
        has_valid = mask.reshape(n_il * n_xl, n_t).any(axis=1)
        valid_indices = np.flatnonzero(has_valid)

        self._seismic_flat = seismic.reshape(n_il * n_xl, n_t)
        self._lmf_flat = lmf.reshape(n_il * n_xl, n_t)
        self._mask_flat = mask.reshape(n_il * n_xl, n_t)
        self._valid_indices = valid_indices

        # 全局归一化统计量（仅在有效掩码区域内计算）
        valid_seis = seismic[mask]
        self._seis_rms = float(np.sqrt(np.mean(valid_seis**2))) + 1e-10

        # LMF 归一化：除以全局绝对最大值（保持正值，避免负数导致反射率除零）
        valid_lmf = lmf[mask]
        self._lmf_scale = float(np.abs(valid_lmf).max()) + 1e-10

        logger.info(
            "Dataset: %d valid traces / %d total, seis_rms=%.4f, lmf_scale=%.2f",
            len(valid_indices),
            n_il * n_xl,
            self._seis_rms,
            self._lmf_scale,
        )

    def __len__(self) -> int:
        return len(self._valid_indices)

    @property
    def seis_rms(self) -> float:
        """地震道全局 RMS（用于反归一化）。"""
        return self._seis_rms

    @property
    def lmf_scale(self) -> float:
        """LMF 全局缩放因子（绝对最大值）。"""
        return self._lmf_scale

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        flat_idx = self._valid_indices[idx]

        seis = self._seismic_flat[flat_idx].copy()  # (n_t,)
        lmf = self._lmf_flat[flat_idx].copy()  # (n_t,)
        m = self._mask_flat[flat_idx].copy()  # (n_t,)

        # 保留 LMF 原始量纲用于物理正演
        lmf_raw = lmf.copy()

        # 归一化：地震道除以 RMS，LMF 除以全局最大值（保持正值）
        seis_norm = seis / self._seis_rms
        lmf_norm = lmf / self._lmf_scale

        # 构造 2 通道输入
        x = np.stack([seis_norm, lmf_norm], axis=0)  # (2, n_t)

        return {
            "input": torch.from_numpy(x).float(),  # (2, n_t)
            "obs": torch.from_numpy(seis_norm[np.newaxis]).float(),  # (1, n_t)
            "mask": torch.from_numpy(m[np.newaxis]).bool(),  # (1, n_t)
            "lmf_raw": torch.from_numpy(lmf_raw[np.newaxis]).float(),  # (1, n_t)
        }


def build_dataset(cfg: GINNConfig) -> Tuple[SeismicTraceDataset, np.ndarray, Dict[str, Any]]:
    """根据配置构建完整 Dataset。

    Returns
    -------
    dataset : SeismicTraceDataset
    wavelet : np.ndarray
    geometry : dict
        地震体几何元信息。
    """
    logger.info("Loading seismic volume...")
    seismic = import_seismic(
        cfg.seismic_file,
        seismic_type="segy",
        iline=cfg.segy_iline,
        xline=cfg.segy_xline,
        istep=cfg.segy_istep,
        xstep=cfg.segy_xstep,
    )

    seismic_ctx = open_survey(
        cfg.seismic_file,
        seismic_type="segy",
        segy_options={
            "iline": cfg.segy_iline,
            "xline": cfg.segy_xline,
            "istep": cfg.segy_istep,
            "xstep": cfg.segy_xstep,
        },
    )
    geometry = seismic_ctx.query_geometry(domain="time")

    n_il, n_xl, n_sample = seismic.shape
    geometry_n_il = int(geometry["n_il"])
    geometry_n_xl = int(geometry["n_xl"])
    geometry_n_sample = int(geometry["n_sample"])
    if (n_il, n_xl, n_sample) != (geometry_n_il, geometry_n_xl, geometry_n_sample):
        raise ValueError(
            "Seismic volume shape does not match queried geometry: "
            f"volume={(n_il, n_xl, n_sample)}, geometry={(geometry_n_il, geometry_n_xl, geometry_n_sample)}"
        )

    # 在这里实例化 TargetLayer 对象，顺便输出一个层位解释的三维布尔体
    # TODO：理论上，低频模型应该由 TargetLayer 构建
    # 然后将 TargetLayer 跑出来的低频模型跑一遍确定性反演
    # 再将这个反演体作为网络的输入

    logger.info("Loading inversion volume...")
    inversion = import_seismic(
        cfg.inversion_file,
        seismic_type="segy",
        # iline=cfg.segy_iline,
        # xline=cfg.segy_xline,
        # istep=cfg.segy_istep,
        # xstep=cfg.segy_xstep,
    )
    if inversion.shape != seismic.shape:
        raise ValueError(f"Inversion shape {inversion.shape} does not match seismic shape {seismic.shape}.")

    logger.info("Loading and interpolating top/bottom interpretation horizons...")
    top_df_raw = import_interpretation_petrel(cfg.top_horizon_file)
    bot_df_raw = import_interpretation_petrel(cfg.bot_horizon_file)

    top_df_interp = interpolate_interpretation_surface(
        interpretation_df=top_df_raw,
        geometry=geometry,
        outlier_threshold=20.0,
        min_neighbor_count=2,
        keep_nan=True,
    )
    bot_df_interp = interpolate_interpretation_surface(
        interpretation_df=bot_df_raw,
        geometry=geometry,
        outlier_threshold=20.0,
        min_neighbor_count=2,
        keep_nan=True,
    )

    logger.info("Building horizon mask from TargetLayer...")
    target_layer = TargetLayer(
        interpolated_horizon_dfs={"top": top_df_interp, "bottom": bot_df_interp},
        geometry=geometry,
        top_name="top",
        bottom_name="bottom",
    )
    mask = target_layer.to_mask()

    logger.info("Generating low-frequency model...")
    lmf = make_lowfreq_model(
        inversion,
        dt_s=cfg.dt,
        cutoff_hz=cfg.lmf_cutoff_hz,
        order=cfg.lmf_filter_order,
    )

    logger.info("Generating wavelet...")
    wavelet = make_wavelet(
        wavelet_type=cfg.wavelet_type,
        freq=cfg.wavelet_freq,
        dt=cfg.wavelet_dt,
        length=cfg.wavelet_length,
        gain=cfg.wavelet_gain,
    )

    logger.info("Building horizon mask from TargetLayer (erosion disabled for now)...")
    if mask.shape != seismic.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match seismic shape {seismic.shape}.")

    dataset = SeismicTraceDataset(seismic, lmf, mask)
    return dataset, wavelet, geometry
