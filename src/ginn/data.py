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

from ginn.config import GINNConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  SEG-Y I/O
# ═══════════════════════════════════════════════════════════════

def load_segy_volume(
    segy_path: Path,
    iline: int = 5,
    xline: int = 21,
    istep: int = 1,
    xstep: int = 4,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """读取三维叠后 SEG-Y 数据体。

    Parameters
    ----------
    segy_path : Path
        SEG-Y 文件路径。
    iline, xline : int
        Inline/Crossline 头字节位置。
    istep, xstep : int
        Inline/Crossline 步长。

    Returns
    -------
    volume : np.ndarray
        三维数据体，shape ``(n_il, n_xl, n_t)``，float32。
    meta : dict
        几何元信息，包含 ``n_il``, ``n_xl``, ``n_t``,
        ``il_min``, ``xl_min``, ``il_step``, ``xl_step``, ``dt_ms``。
    """
    import cigsegy

    segy = cigsegy.Pysegy(str(segy_path))
    try:
        meta_info = cigsegy.tools.get_metaInfo(segy, apply_scalar=True)

        offset_keyloc = int(meta_info.get("offset", 37))
        ostep = int(meta_info.get("ostep", 1))

        segy.setLocations(iline, xline, offset_keyloc)
        segy.setSteps(istep, xstep, ostep)
        segy.set_segy_type(3)
        segy.scan()

        geominfo = cigsegy.tools.full_scan(
            segy, iline=iline, xline=xline, offset=offset_keyloc, is4d=False,
        )
        geom = np.asarray(geominfo["geom"])
        n_il, n_xl = geom.shape
        n_t = int(meta_info["nt"])
        dt_ms = float(meta_info["dt"]) / 1000.0  # μs → ms

        # 读取全部道
        volume = np.zeros((n_il, n_xl, n_t), dtype=np.float32)
        for i in range(n_il):
            for j in range(n_xl):
                trace_idx = int(geom[i, j])
                if trace_idx < 0:
                    continue  # 缺失道保持零
                trace = segy.collect(trace_idx, trace_idx + 1, 0, n_t).squeeze()
                volume[i, j, :] = trace

        meta = {
            "n_il": n_il,
            "n_xl": n_xl,
            "n_t": n_t,
            "il_min": float(geominfo["iline"]["min_iline"]),
            "xl_min": float(geominfo["xline"]["min_xline"]),
            "il_step": float(geominfo["iline"]["istep"]),
            "xl_step": float(geominfo["xline"]["xstep"]),
            "dt_ms": dt_ms,
            "start_time_ms": float(meta_info.get("start_time", 0)),
        }
    finally:
        segy.close()

    logger.info("Loaded SEG-Y: %s  shape=%s  dt=%.1f ms", segy_path.name, volume.shape, dt_ms)
    return volume, meta


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
        反演阻抗体，shape ``(n_il, n_xl, n_t)``。
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
    n_il, n_xl, n_t = impedance_volume.shape
    lmf = np.zeros_like(impedance_volume)

    for i in range(n_il):
        for j in range(n_xl):
            trace = impedance_volume[i, j, :].astype(np.float64)
            if np.all(trace == 0):
                continue
            lmf[i, j, :] = apply_butter_lowpass_filter(
                trace, highcut=cutoff_hz, fs=fs, order=order, zero_phase=True,
            )

    logger.info(
        "Generated LMF: cutoff=%.1f Hz, order=%d, shape=%s",
        cutoff_hz, order, lmf.shape,
    )
    return lmf.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  子波
# ═══════════════════════════════════════════════════════════════

def make_wavelet(
    wavelet_type: str = "ricker",
    freq: float = 25.0,
    dt: float = 0.001,
    length: int = 301,
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

    logger.info("Generated %s wavelet: freq=%.1f Hz, dt=%.4f s, length=%d", wavelet_type, freq, dt, length)
    return y.astype(np.float32)


# ═══════════════════════════════════════════════════════════════
#  层位掩码
# ═══════════════════════════════════════════════════════════════

def load_horizon_mask(
    top_horizon_file: Path,
    bot_horizon_file: Path,
    seismic_file: Path,
    n_il: int,
    n_xl: int,
    n_t: int,
    dt_s: float = 0.001,
    start_time_ms: float = 0.0,
    erosion: int = 0,
    iline: int = 5,
    xline: int = 21,
    istep: int = 1,
    xstep: int = 4,
) -> np.ndarray:
    """从 Petrel 层位文件生成三维布尔掩码。

    Parameters
    ----------
    top_horizon_file : Path
        顶层位文件路径。
    bot_horizon_file : Path
        底层位文件路径。
    seismic_file : Path
        地震体文件路径（用于获取几何信息）。
    n_il, n_xl, n_t : int
        数据体 inline/crossline/采样维度大小。
    dt_s : float
        采样间隔 (s)。
    start_time_ms : float
        地震道起始时间 (ms)。层位绝对时间会减去此值后转为采样索引。
    erosion : int
        掩码边界收缩采样点数。

    Returns
    -------
    np.ndarray
        布尔掩码，shape ``(n_il, n_xl, n_t)``。
        True 表示该点位于收缩后的有效区域内。
    """
    from cup.petrel.load import import_interpretation_petrel
    from cup.seismic.process import interpolate_interpretation_surface
    from cup.seismic.survey import query_seismic_geometry

    geometry = query_seismic_geometry(
        seismic_file, seismic_type="segy", domain="time",
        iline=iline, xline=xline, istep=istep, xstep=xstep,
    )

    def _load_surface(path: Path) -> np.ndarray:
        df = import_interpretation_petrel(path)
        interp_df = interpolate_interpretation_surface(
            interpretation_df=df,
            geometry=geometry,
            outlier_threshold=20.0,
            min_neighbor_count=2,
            keep_nan=True,
        )
        # 层位以 ms 为单位 → 转为采样索引
        il_min = int(geometry["inline_min"])
        xl_min = int(geometry["xline_min"])
        il_step = int(geometry["inline_step"])
        xl_step = int(geometry["xline_step"])

        surface_grid = np.full((n_il, n_xl), np.nan, dtype=np.float64)
        for _, row in interp_df.iterrows():
            il_idx = int((row["inline"] - il_min) / il_step)
            xl_idx = int((row["xline"] - xl_min) / xl_step)
            if 0 <= il_idx < n_il and 0 <= xl_idx < n_xl:
                val = row["interpretation"]
                if np.isfinite(val):
                    # 层位绝对时间 (ms) → 相对采样索引
                    surface_grid[il_idx, xl_idx] = (val - start_time_ms) / (dt_s * 1000.0)
        return surface_grid

    top_surface = _load_surface(top_horizon_file)
    bot_surface = _load_surface(bot_horizon_file)

    mask = np.zeros((n_il, n_xl, n_t), dtype=bool)
    for i in range(n_il):
        for j in range(n_xl):
            t_top = top_surface[i, j]
            t_bot = bot_surface[i, j]
            if np.isfinite(t_top) and np.isfinite(t_bot):
                # 边界收缩：顶向下 + 底向上
                idx_top = max(0, int(np.round(t_top)) + erosion)
                idx_bot = min(n_t, int(np.round(t_bot)) + 1 - erosion)
                if idx_top < idx_bot:
                    mask[i, j, idx_top:idx_bot] = True

    n_valid = mask.sum()
    coverage = n_valid / mask.size * 100
    logger.info(
        "Horizon mask: %d valid points (%.1f%%), erosion=%d samples",
        n_valid, coverage, erosion,
    )
    return mask


# ═══════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ═══════════════════════════════════════════════════════════════

class SeismicTraceDataset(Dataset):
    """逐道 1D 地震数据 Dataset。

    每个 item 包含：
    - ``input``：(2, n_t) — 归一化地震道 + 归一化 LMF
    - ``obs``：(1, n_t) — 原始地震道（用于损失计算）
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
        self._seis_rms = float(np.sqrt(np.mean(valid_seis ** 2))) + 1e-10

        # LMF 归一化：除以全局绝对最大值（保持正值，避免负数导致反射率除零）
        valid_lmf = lmf[mask]
        self._lmf_scale = float(np.abs(valid_lmf).max()) + 1e-10

        logger.info(
            "Dataset: %d valid traces / %d total, seis_rms=%.4f, lmf_scale=%.2f",
            len(valid_indices), n_il * n_xl, self._seis_rms, self._lmf_scale,
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

        seis = self._seismic_flat[flat_idx].copy()       # (n_t,)
        lmf = self._lmf_flat[flat_idx].copy()            # (n_t,)
        m = self._mask_flat[flat_idx].copy()              # (n_t,)

        # 记录原始值
        obs = seis.copy()
        lmf_raw = lmf.copy()

        # 归一化：地震道除以 RMS，LMF 除以全局最大值（保持正值）
        seis_norm = seis / self._seis_rms
        lmf_norm = lmf / self._lmf_scale

        # 构造 2 通道输入
        x = np.stack([seis_norm, lmf_norm], axis=0)  # (2, n_t)

        return {
            "input": torch.from_numpy(x).float(),                       # (2, n_t)
            "obs": torch.from_numpy(obs[np.newaxis]).float(),           # (1, n_t)
            "mask": torch.from_numpy(m[np.newaxis]).bool(),             # (1, n_t)
            "lmf_raw": torch.from_numpy(lmf_raw[np.newaxis]).float(),  # (1, n_t)
        }


def build_dataset(cfg: GINNConfig) -> Tuple[SeismicTraceDataset, np.ndarray, Dict[str, Any]]:
    """根据配置构建完整 Dataset。

    Returns
    -------
    dataset : SeismicTraceDataset
    wavelet : np.ndarray
    meta : dict
        地震体几何元信息。
    """
    logger.info("Loading seismic volume...")
    seismic, meta = load_segy_volume(
        cfg.seismic_file,
        iline=cfg.segy_iline, xline=cfg.segy_xline,
        istep=cfg.segy_istep, xstep=cfg.segy_xstep,
    )

    logger.info("Loading inversion volume...")
    inversion, _ = load_segy_volume(
        cfg.inversion_file,
        iline=cfg.segy_iline, xline=cfg.segy_xline,
        istep=cfg.segy_istep, xstep=cfg.segy_xstep,
    )

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
    )

    logger.info("Building horizon mask (erosion=%d, start_time=%.0f ms)...", cfg.mask_erosion_samples, meta["start_time_ms"])
    mask = load_horizon_mask(
        top_horizon_file=cfg.top_horizon_file,
        bot_horizon_file=cfg.bot_horizon_file,
        seismic_file=cfg.seismic_file,
        n_il=meta["n_il"], n_xl=meta["n_xl"], n_t=meta["n_t"],
        dt_s=cfg.dt,
        start_time_ms=meta["start_time_ms"],
        erosion=cfg.mask_erosion_samples,
        iline=cfg.segy_iline, xline=cfg.segy_xline,
        istep=cfg.segy_istep, xstep=cfg.segy_xstep,
    )

    dataset = SeismicTraceDataset(seismic, lmf, mask)
    return dataset, wavelet, meta
