"""ginn.data — 数据加载、预处理与 Dataset 定义。

职责
----
1. 读取 SEG-Y 地震体和反演体
2. 读取预计算低频模型或从反演体生成低频模型
3. 生成理论子波（Ricker）
4. 从层位文件生成 3D 布尔掩码
5. 封装为 PyTorch Dataset
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
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
    dt_s: float,
    cutoff_hz: float,
    order: int,
) -> np.ndarray:
    """对反演阻抗体沿时间轴做 Butterworth 零相位低通滤波，生成低频模型。

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
    fs = 1.0 / dt_s
    n_il, n_xl, n_sample = impedance_volume.shape
    nyquist = 0.5 * fs
    if cutoff_hz <= 0.0 or cutoff_hz >= nyquist:
        raise ValueError(f"cutoff_hz must be within (0, {nyquist}), got {cutoff_hz}.")

    # 与旧实现保持一致：零相位前后各做一次滤波，因此单程滤波器阶数取 order // 2。
    single_pass_order = max(1, order // 2)
    sos = butter(single_pass_order, cutoff_hz, btype="low", fs=fs, output="sos")

    traces = impedance_volume.reshape(n_il * n_xl, n_sample).astype(np.float64, copy=False)
    active_mask = np.any(traces != 0.0, axis=1)
    filtered = traces.copy()
    if np.any(active_mask):
        filtered[active_mask] = sosfiltfilt(sos, traces[active_mask], axis=-1)
    lmf = filtered.reshape(n_il, n_xl, n_sample)

    logger.info(
        "Generated LMF: cutoff=%.1f Hz, order=%d, shape=%s",
        cutoff_hz,
        order,
        lmf.shape,
    )
    return lmf.astype(np.float32)


def load_lowfreq_model(lowfreq_file: Path) -> np.ndarray:
    """从 ``.npz`` 或 ``.npy`` 文件读取预计算低频模型体。"""
    lowfreq_path = Path(lowfreq_file)
    if not lowfreq_path.exists():
        raise FileNotFoundError(lowfreq_path)

    suffix = lowfreq_path.suffix.lower()
    if suffix == ".npy":
        volume = np.load(lowfreq_path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(lowfreq_path, allow_pickle=False) as archive:
            if "volume" in archive.files:
                volume = archive["volume"]
            elif len(archive.files) == 1:
                volume = archive[archive.files[0]]
            else:
                raise ValueError(
                    f"Expected key 'volume' in precomputed LMF archive or exactly one array, got {archive.files}."
                )

            if "metadata_json" in archive.files:
                metadata = json.loads(np.asarray(archive["metadata_json"]).item())
                logger.info("Loaded precomputed LMF metadata: %s", metadata)
    else:
        raise ValueError(f"Unsupported low-frequency model file type: {lowfreq_path.suffix}")

    return np.asarray(volume, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  子波
# ═══════════════════════════════════════════════════════════════


def make_wavelet(
    wavelet_type: str,
    freq: float,
    dt: float,
    length: int,
    gain: float,
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
    - ``input``：(2, n_sample) — 归一化地震道 + 归一化 LMF
    - ``obs``：(1, n_sample) — 归一化观测地震道（用于损失计算）
    - ``mask``：(1, n_sample) — 布尔掩码
    - ``lmf_raw``：(1, n_sample) — 原始 LMF（用于正演时恢复阻抗）

    Parameters
    ----------
    seismic : np.ndarray
        地震体，shape ``(n_il, n_xl, n_sample)``。
    lmf : np.ndarray
        低频模型，shape ``(n_il, n_xl, n_sample)``。
    mask : np.ndarray
        布尔掩码，shape ``(n_il, n_xl, n_sample)``。
    """

    def __init__(
        self,
        seismic: np.ndarray,
        lmf: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        n_il, n_xl, n_sample = seismic.shape
        assert lmf.shape == seismic.shape
        assert mask.shape == seismic.shape

        # 找出掩码中至少有 1 个 True 的道
        has_valid = mask.reshape(n_il * n_xl, n_sample).any(axis=1)
        valid_indices = np.flatnonzero(has_valid)

        self._seismic_flat = seismic.reshape(n_il * n_xl, n_sample)
        self._lmf_flat = lmf.reshape(n_il * n_xl, n_sample)
        self._mask_flat = mask.reshape(n_il * n_xl, n_sample)
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

        seis = self._seismic_flat[flat_idx].copy()  # (n_sample,)
        lmf = self._lmf_flat[flat_idx].copy()  # (n_sample,)
        m = self._mask_flat[flat_idx].copy()  # (n_sample,)

        # 保留 LMF 原始量纲用于物理正演
        lmf_raw = lmf.copy()

        # 归一化：地震道除以 RMS，LMF 除以全局最大值（保持正值）
        seis_norm = seis / self._seis_rms
        lmf_norm = lmf / self._lmf_scale

        # 构造 2 通道输入
        x = np.stack([seis_norm, lmf_norm], axis=0)  # (2, n_sample)

        return {
            "input": torch.from_numpy(x).float(),  # (2, n_sample)
            "obs": torch.from_numpy(seis_norm[np.newaxis]).float(),  # (1, n_sample)
            "mask": torch.from_numpy(m[np.newaxis]).bool(),  # (1, n_sample)
            "lmf_raw": torch.from_numpy(lmf_raw[np.newaxis]).float(),  # (1, n_sample)
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

    logger.info("Loading and interpolating top/bottom interpretation horizons...")
    top_df_raw = import_interpretation_petrel(cfg.top_horizon_file)
    bot_df_raw = import_interpretation_petrel(cfg.bot_horizon_file)

    top_df_interp = interpolate_interpretation_surface(
        interpretation_df=top_df_raw,
        geometry=geometry,
        outlier_threshold=0.02,
        min_neighbor_count=2,
        keep_nan=True,
    )
    bot_df_interp = interpolate_interpretation_surface(
        interpretation_df=bot_df_raw,
        geometry=geometry,
        outlier_threshold=0.02,
        min_neighbor_count=2,
        keep_nan=True,
    )

    logger.info("Building horizon mask from TargetLayer...")
    target_layer = TargetLayer(
        interpolated_horizon_dfs={"top": top_df_interp, "bottom": bot_df_interp},
        geometry=geometry,
        horizon_names=["top", "bottom"],
    )
    mask = target_layer.to_mask()

    if cfg.lmf_source == "wtie_time_lfm":
        logger.info("Loading precomputed low-frequency model from %s...", cfg.precomputed_lmf_file)
        lmf = load_lowfreq_model(cfg.precomputed_lmf_file)
    elif cfg.lmf_source == "filtered_inversion_lmf":
        logger.info("Loading inversion volume...")
        inversion = import_seismic(
            cfg.inversion_file,
            seismic_type="segy",
        )
        if inversion.shape != seismic.shape:
            raise ValueError(f"Inversion shape {inversion.shape} does not match seismic shape {seismic.shape}.")

        logger.info("Generating low-frequency model from inversion volume...")
        lmf = make_lowfreq_model(
            inversion,
            dt_s=cfg.dt,
            cutoff_hz=cfg.lmf_cutoff_hz,
            order=cfg.lmf_filter_order,
        )
    else:
        raise ValueError(f"Unsupported lmf_source: {cfg.lmf_source}")

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
    if lmf.shape != seismic.shape:
        raise ValueError(f"LMF shape {lmf.shape} does not match seismic shape {seismic.shape}.")

    dataset = SeismicTraceDataset(seismic, lmf, mask)
    return dataset, wavelet, geometry
