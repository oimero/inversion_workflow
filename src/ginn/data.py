"""ginn.data — 数据加载、预处理与 Dataset 定义。

工作流：
1. 读取 SEG-Y 地震体和反演体
2. 读取预计算低频模型或从反演体生成低频模型
3. 生成理论子波（Ricker）
4. 从层位文件生成 3D 布尔掩码
5. 封装为 PyTorch Dataset
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
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


@dataclass
class DatasetBundle:
    """GINN 训练/验证/推理所需数据集集合。"""

    train_dataset: "SeismicTraceDataset"
    inference_dataset: "SeismicTraceDataset"
    val_dataset: "SeismicTraceDataset | None"
    wavelet: np.ndarray
    geometry: Dict[str, Any]
    split_metadata: Dict[str, Any]


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
    lfm = filtered.reshape(n_il, n_xl, n_sample)

    logger.info(
        "Generated LFM: cutoff=%.1f Hz, order=%d, shape=%s",
        cutoff_hz,
        order,
        lfm.shape,
    )
    return lfm.astype(np.float32)


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
                    f"Expected key 'volume' in precomputed LFM archive or exactly one array, got {archive.files}."
                )

            if "metadata_json" in archive.files:
                metadata = json.loads(np.asarray(archive["metadata_json"]).item())
                logger.info("Loaded precomputed LFM metadata: %s", metadata)
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


def estimate_wavelet_gain(
    seismic: np.ndarray,
    lfm: np.ndarray,
    mask: np.ndarray,
    unit_wavelet: np.ndarray,
    *,
    seis_rms: float,
    max_traces: int,
    candidate_trace_indices: np.ndarray | None = None,
    batch_size: int = 64,
) -> float:
    """基于样本道估计使合成地震与归一化观测同量级的子波增益。

    思路：
    1. 用单位增益子波对 LFM 做正演，得到 ``d_syn_unit``；
    2. 在同一批掩码样本上计算 ``d_syn_unit`` 的 RMS；
    3. 计算该批观测地震归一化后的 RMS；
    4. 令 ``gain = obs_norm_rms / syn_unit_rms``。
    """
    from ginn.physics import ForwardModel

    n_sample = seismic.shape[-1]
    seismic_flat = seismic.reshape(-1, n_sample)
    lfm_flat = lfm.reshape(-1, n_sample)
    mask_flat = mask.reshape(-1, n_sample).astype(bool, copy=False)

    valid_trace_indices = np.flatnonzero(mask_flat.any(axis=1))
    if candidate_trace_indices is not None:
        candidate_trace_indices = np.asarray(candidate_trace_indices, dtype=np.int64)
        valid_trace_indices = np.intersect1d(valid_trace_indices, candidate_trace_indices, assume_unique=False)
    if valid_trace_indices.size == 0:
        raise ValueError("Cannot auto-estimate wavelet gain because no valid traces were found in the mask.")

    n_selected = min(max_traces, valid_trace_indices.size)
    if n_selected < valid_trace_indices.size:
        rng = np.random.default_rng(0)
        selected = np.sort(rng.choice(valid_trace_indices, size=n_selected, replace=False))
    else:
        selected = valid_trace_indices

    forward_model = ForwardModel(unit_wavelet).cpu()
    syn_sq_sum = 0.0
    obs_norm_sq_sum = 0.0
    n_valid = 0

    with torch.no_grad():
        for start in range(0, selected.size, batch_size):
            batch_indices = selected[start : start + batch_size]

            lfm_batch = torch.from_numpy(lfm_flat[batch_indices][:, np.newaxis, :]).float()
            mask_batch = mask_flat[batch_indices][:, np.newaxis, :]
            d_syn_unit = forward_model(lfm_batch).cpu().numpy()

            seismic_batch = seismic_flat[batch_indices][:, np.newaxis, :] / float(seis_rms)
            valid_values = mask_batch

            syn_values = d_syn_unit[valid_values]
            obs_norm_values = seismic_batch[valid_values]

            syn_sq_sum += float(np.square(syn_values, dtype=np.float64).sum())
            obs_norm_sq_sum += float(np.square(obs_norm_values, dtype=np.float64).sum())
            n_valid += int(valid_values.sum())

    if n_valid <= 0:
        raise ValueError("Cannot auto-estimate wavelet gain because sampled valid point count is zero.")

    syn_rms = math.sqrt(syn_sq_sum / n_valid)
    obs_norm_rms = math.sqrt(obs_norm_sq_sum / n_valid)
    if syn_rms <= 0.0:
        raise ValueError(f"Cannot auto-estimate wavelet gain because synthetic RMS is non-positive: {syn_rms}.")

    gain = obs_norm_rms / syn_rms
    logger.info(
        "Auto wavelet gain from sampled traces: traces=%d, valid_points=%d, obs_norm_rms=%.4f, syn_unit_rms=%.4f, gain=%.4f",
        selected.size,
        n_valid,
        obs_norm_rms,
        syn_rms,
        gain,
    )
    return float(gain)


# ═══════════════════════════════════════════════════════════════
#  层位掩码
# ═══════════════════════════════════════════════════════════════


def _resolve_mask_bounds(mask_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从逐道布尔掩码解析 core 区间的起止采样点。

    Notes
    -----
    当前目的层 mask 由 top/bottom horizon 构造，按设计每条有效道只包含一个
    连续区间。这里利用该性质为所有道一次性解析边界，用于后续 erosion 与 halo。
    """
    if mask_flat.ndim != 2:
        raise ValueError(f"mask_flat must be a 2D array, got shape={mask_flat.shape}.")

    n_trace, n_sample = mask_flat.shape
    has_valid = mask_flat.any(axis=1)
    start = np.zeros(n_trace, dtype=np.int64)
    end = np.zeros(n_trace, dtype=np.int64)

    if np.any(has_valid):
        valid_mask = mask_flat[has_valid]
        start[has_valid] = np.argmax(valid_mask, axis=1)
        end[has_valid] = n_sample - np.argmax(valid_mask[:, ::-1], axis=1)

    return start, end, has_valid  # type: ignore


def _build_eroded_loss_mask(mask_flat: np.ndarray, erosion_samples: int) -> np.ndarray:
    """基于 core mask 构造用于 waveform loss 的内缩掩码。"""
    start, end, has_valid = _resolve_mask_bounds(mask_flat)
    n_sample = mask_flat.shape[1]
    sample_index = np.arange(n_sample, dtype=np.int64)[np.newaxis, :]
    lengths = end - start

    # 对过薄层段自动减小 erosion，至少保留 1 个采样点，避免整条道 loss 失效。
    erosion = np.minimum(int(erosion_samples), np.maximum((lengths - 1) // 2, 0))
    loss_start = start + erosion
    loss_end = end - erosion

    return (
        has_valid[:, np.newaxis]
        & (sample_index >= loss_start[:, np.newaxis])
        & (sample_index < loss_end[:, np.newaxis])
    )


def _build_residual_taper(mask_flat: np.ndarray, halo_samples: int) -> np.ndarray:
    """为 residual 构造 core+halo 支撑区的平滑 taper 权重。"""
    start, end, has_valid = _resolve_mask_bounds(mask_flat)
    n_sample = mask_flat.shape[1]
    sample_index = np.arange(n_sample, dtype=np.int64)[np.newaxis, :]
    halo = int(halo_samples)

    support_start = np.maximum(start - halo, 0)
    support_end = np.minimum(end + halo, n_sample)

    core_region = (
        has_valid[:, np.newaxis] & (sample_index >= start[:, np.newaxis]) & (sample_index < end[:, np.newaxis])
    )
    left_region = (
        has_valid[:, np.newaxis]
        & (sample_index >= support_start[:, np.newaxis])
        & (sample_index < start[:, np.newaxis])
    )
    right_region = (
        has_valid[:, np.newaxis] & (sample_index >= end[:, np.newaxis]) & (sample_index < support_end[:, np.newaxis])
    )

    taper = np.zeros(mask_flat.shape, dtype=np.float32)
    taper[core_region] = 1.0

    if halo > 0:
        left_denom = (start - support_start + 1).astype(np.float32)[:, np.newaxis]
        right_denom = (support_end - end + 1).astype(np.float32)[:, np.newaxis]

        left_weight = (sample_index - support_start[:, np.newaxis] + 1).astype(np.float32) / left_denom
        right_weight = (support_end[:, np.newaxis] - sample_index).astype(np.float32) / right_denom

        taper[left_region] = left_weight[left_region]
        taper[right_region] = right_weight[right_region]

    return taper


# ═══════════════════════════════════════════════════════════════
#  训练集/验证集切分
# ═══════════════════════════════════════════════════════════════


def _get_valid_trace_indices(mask_flat: np.ndarray) -> np.ndarray:
    """返回至少有一个有效采样点的道的展平索引。"""
    return np.flatnonzero(mask_flat.any(axis=1))


def _select_spatial_validation_split(
    valid_indices: np.ndarray,
    *,
    n_il: int,
    n_xl: int,
    validation_fraction: float,
    gap_traces: int,
    anchor: str,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """按 inline/xline 连续块切出验证集，并在周围留出缓冲带。"""
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    if valid_indices.size == 0:
        raise ValueError("Cannot build a validation split because no valid traces were found.")
    if validation_fraction <= 0.0:
        metadata = {
            "mode": "none",
            "train_trace_count": int(valid_indices.size),
            "val_trace_count": 0,
            "gap_trace_count": 0,
        }
        return valid_indices.copy(), np.empty(0, dtype=np.int64), metadata

    il_coords = valid_indices // n_xl
    xl_coords = valid_indices % n_xl
    il_min = int(il_coords.min())
    il_max = int(il_coords.max())
    xl_min = int(xl_coords.min())
    xl_max = int(xl_coords.max())
    il_extent = il_max - il_min + 1
    xl_extent = xl_max - xl_min + 1

    side_fraction = math.sqrt(validation_fraction)
    block_il = max(1, min(il_extent, int(math.ceil(il_extent * side_fraction))))
    block_xl = max(1, min(xl_extent, int(math.ceil(xl_extent * side_fraction))))

    if anchor == "maxmax":
        il_start = il_max - block_il + 1
        xl_start = xl_max - block_xl + 1
    elif anchor == "maxmin":
        il_start = il_max - block_il + 1
        xl_start = xl_min
    elif anchor == "minmax":
        il_start = il_min
        xl_start = xl_max - block_xl + 1
    elif anchor == "minmin":
        il_start = il_min
        xl_start = xl_min
    elif anchor == "center":
        il_start = il_min + max((il_extent - block_il) // 2, 0)
        xl_start = xl_min + max((xl_extent - block_xl) // 2, 0)
    else:
        raise ValueError(f"Unsupported validation block anchor: {anchor!r}")

    il_end = il_start + block_il
    xl_end = xl_start + block_xl

    val_mask = (il_coords >= il_start) & (il_coords < il_end) & (xl_coords >= xl_start) & (xl_coords < xl_end)
    val_indices = valid_indices[val_mask]
    if val_indices.size == 0:
        raise ValueError(
            "Validation block did not capture any valid traces. "
            f"Try a different validation_block_anchor or a larger validation_fraction (current={validation_fraction})."
        )

    gap = int(gap_traces)
    gap_il_start = max(il_start - gap, 0)
    gap_il_end = min(il_end + gap, n_il)
    gap_xl_start = max(xl_start - gap, 0)
    gap_xl_end = min(xl_end + gap, n_xl)

    exclusion_mask = (
        (il_coords >= gap_il_start) & (il_coords < gap_il_end) & (xl_coords >= gap_xl_start) & (xl_coords < gap_xl_end)
    )
    train_indices = valid_indices[~exclusion_mask]
    gap_only_mask = exclusion_mask & ~val_mask
    if train_indices.size == 0:
        raise ValueError(
            "Validation block plus gap removed all training traces. "
            f"Try a smaller validation_fraction or validation_gap_traces (current gap={gap})."
        )

    metadata = {
        "mode": "spatial_block",
        "anchor": anchor,
        "requested_validation_fraction": float(validation_fraction),
        "actual_validation_fraction": float(val_indices.size / valid_indices.size),
        "gap_traces": gap,
        "train_trace_count": int(train_indices.size),
        "val_trace_count": int(val_indices.size),
        "gap_trace_count": int(gap_only_mask.sum()),
        "block_il_start": int(il_start),
        "block_il_end": int(il_end),
        "block_xl_start": int(xl_start),
        "block_xl_end": int(xl_end),
    }
    return train_indices, val_indices, metadata


# ═══════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ═══════════════════════════════════════════════════════════════


class SeismicTraceDataset(Dataset):
    """逐道 1D 地震数据 Dataset。

    每个 item 包含：
    - ``input``：(2, n_sample) — 归一化地震道 + 归一化 LFM
    - ``obs``：(1, n_sample) — 归一化观测地震道（用于损失计算）
    - ``mask``：(1, n_sample) — core 布尔掩码（与旧接口兼容）
    - ``loss_mask``：(1, n_sample) — eroded core 掩码，仅用于 waveform loss
    - ``taper_weight``：(1, n_sample) — core+halo 平滑权重，用于 residual 收口
    - ``lfm_raw``：(1, n_sample) — 原始 LFM（用于正演时恢复阻抗）

    Parameters
    ----------
    seismic_flat : np.ndarray
        展平地震数据，shape ``(n_traces, n_sample)``。
    lfm_flat : np.ndarray
        展平低频模型，shape ``(n_traces, n_sample)``。
    mask_flat : np.ndarray
        展平布尔掩码，shape ``(n_traces, n_sample)``。
    loss_mask_flat : np.ndarray
        展平 eroded 损失掩码，shape ``(n_traces, n_sample)``。
    taper_flat : np.ndarray
        展平 taper 权重，shape ``(n_traces, n_sample)``。
    selected_indices : np.ndarray
        要使用的展平道索引。
    normalization_stats : tuple[float, float] | None
        如果提供，使用指定的 ``(seis_rms, lfm_scale)`` 而不是自行估计。
    """

    def __init__(
        self,
        seismic_flat: np.ndarray,
        lfm_flat: np.ndarray,
        mask_flat: np.ndarray,
        loss_mask_flat: np.ndarray,
        taper_flat: np.ndarray,
        selected_indices: np.ndarray,
        *,
        normalization_stats: tuple[float, float] | None = None,
    ) -> None:
        n_traces, n_sample = seismic_flat.shape
        assert lfm_flat.shape == seismic_flat.shape
        assert mask_flat.shape == seismic_flat.shape
        assert loss_mask_flat.shape == seismic_flat.shape
        assert taper_flat.shape == seismic_flat.shape

        selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
        if selected_indices.size == 0:
            raise ValueError("selected_indices must contain at least one trace.")

        self._seismic_flat = seismic_flat
        self._lfm_flat = lfm_flat
        self._mask_flat = mask_flat
        self._loss_mask_flat = loss_mask_flat
        self._taper_flat = taper_flat
        self._valid_indices = selected_indices

        if normalization_stats is None:
            selected_mask = self._mask_flat[self._valid_indices]
            selected_seismic = self._seismic_flat[self._valid_indices]
            selected_lfm = self._lfm_flat[self._valid_indices]

            valid_seis = selected_seismic[selected_mask]
            self._seis_rms = float(np.sqrt(np.mean(valid_seis**2))) + 1e-10

            valid_lfm = selected_lfm[selected_mask]
            self._lfm_scale = float(np.abs(valid_lfm).max()) + 1e-10
        else:
            self._seis_rms = float(normalization_stats[0])
            self._lfm_scale = float(normalization_stats[1])

        logger.info(
            "Dataset: %d selected traces / %d total, seis_rms=%.4f, lfm_scale=%.2f",
            len(self._valid_indices),
            n_traces,
            self._seis_rms,
            self._lfm_scale,
        )

    def __len__(self) -> int:
        return len(self._valid_indices)

    @property
    def seis_rms(self) -> float:
        """地震道全局 RMS（用于反归一化）。"""
        return self._seis_rms

    @property
    def lfm_scale(self) -> float:
        """LFM 全局缩放因子（绝对最大值）。"""
        return self._lfm_scale

    @property
    def valid_indices(self) -> np.ndarray:
        """当前数据集对应的展平 trace 索引。"""
        return self._valid_indices

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        flat_idx = self._valid_indices[idx]

        seis = self._seismic_flat[flat_idx].copy()  # (n_sample,)
        lfm = self._lfm_flat[flat_idx].copy()  # (n_sample,)
        core_mask = self._mask_flat[flat_idx].copy()  # (n_sample,)
        loss_mask = self._loss_mask_flat[flat_idx].copy()  # (n_sample,)
        taper_weight = self._taper_flat[flat_idx].copy()  # (n_sample,)

        # 保留 LFM 原始量纲用于物理正演
        lfm_raw = lfm.copy()

        # 归一化：地震道除以 RMS，LFM 除以全局最大值（保持正值）
        seis_norm = seis / self._seis_rms
        lfm_norm = lfm / self._lfm_scale

        # 构造 2 通道输入
        x = np.stack([seis_norm, lfm_norm], axis=0)  # (2, n_sample)

        return {
            "input": torch.from_numpy(x).float(),  # (2, n_sample)
            "obs": torch.from_numpy(seis_norm[np.newaxis]).float(),  # (1, n_sample)
            "mask": torch.from_numpy(core_mask[np.newaxis]).bool(),  # (1, n_sample)
            "loss_mask": torch.from_numpy(loss_mask[np.newaxis]).bool(),  # (1, n_sample)
            "taper_weight": torch.from_numpy(taper_weight[np.newaxis]).float(),  # (1, n_sample)
            "lfm_raw": torch.from_numpy(lfm_raw[np.newaxis]).float(),  # (1, n_sample)
        }


def build_dataset(cfg: GINNConfig) -> DatasetBundle:
    """根据配置构建训练/验证/推理数据集。

    Returns
    -------
    DatasetBundle
        包含训练、验证、推理数据集，以及子波与几何信息。
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

    if cfg.lfm_source == "wtie_time_lfm":
        logger.info("Loading precomputed low-frequency model from %s...", cfg.precomputed_lfm_file)
        lfm = load_lowfreq_model(cfg.precomputed_lfm_file)
    elif cfg.lfm_source == "filtered_inversion_lfm":
        logger.info("Loading inversion volume...")
        inversion = import_seismic(
            cfg.lfm_reference_impedance_file,
            seismic_type="segy",
        )
        if inversion.shape != seismic.shape:
            raise ValueError(f"Inversion shape {inversion.shape} does not match seismic shape {seismic.shape}.")

        logger.info("Generating low-frequency model from inversion volume...")
        lfm = make_lowfreq_model(
            inversion,
            dt_s=cfg.dt,
            cutoff_hz=cfg.lfm_cutoff_hz,
            order=cfg.lfm_filter_order,
        )
    else:
        raise ValueError(f"Unsupported lfm_source: {cfg.lfm_source}")

    logger.info("Building horizon mask from TargetLayer (erosion disabled for now)...")
    if mask.shape != seismic.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match seismic shape {seismic.shape}.")
    if lfm.shape != seismic.shape:
        raise ValueError(f"LFM shape {lfm.shape} does not match seismic shape {seismic.shape}.")

    # ── 展平并预计算掩码 / taper（只算一次） ──
    seismic_flat = seismic.reshape(n_il * n_xl, n_sample)
    lfm_flat = lfm.reshape(n_il * n_xl, n_sample)
    mask_flat = mask.reshape(n_il * n_xl, n_sample)
    loss_mask_flat = _build_eroded_loss_mask(mask_flat, erosion_samples=cfg.boundary_effect_samples)
    taper_flat = _build_residual_taper(mask_flat, halo_samples=cfg.boundary_effect_samples)
    all_valid_indices = _get_valid_trace_indices(mask_flat)
    logger.info(
        "Preprocessed masks: %d valid traces, boundary_effect_samples=%d",
        all_valid_indices.size,
        cfg.boundary_effect_samples,
    )

    # ── 验证集切分 ──
    train_indices = all_valid_indices
    val_indices: np.ndarray | None = None
    split_metadata: Dict[str, Any] = {
        "mode": "none",
        "train_trace_count": int(train_indices.size),
        "val_trace_count": 0,
        "gap_trace_count": 0,
    }
    if cfg.validation_split_mode == "spatial_block" and cfg.validation_fraction > 0.0:
        train_indices, val_indices, split_metadata = _select_spatial_validation_split(
            all_valid_indices,
            n_il=n_il,
            n_xl=n_xl,
            validation_fraction=cfg.validation_fraction,
            gap_traces=cfg.validation_gap_traces,
            anchor=cfg.validation_block_anchor,
        )
        logger.info("Validation split: %s", split_metadata)
    else:
        logger.info("Validation split disabled.")

    # ── 构建数据集（共享预计算的掩码 / taper，不重复计算） ──
    shared = (seismic_flat, lfm_flat, mask_flat, loss_mask_flat, taper_flat)
    train_dataset = SeismicTraceDataset(*shared, train_indices)
    train_norm_stats = (train_dataset.seis_rms, train_dataset.lfm_scale)
    val_dataset = None
    if val_indices is not None and val_indices.size > 0:
        val_dataset = SeismicTraceDataset(
            *shared,
            val_indices,
            normalization_stats=train_norm_stats,
        )
    inference_dataset = SeismicTraceDataset(
        *shared,
        all_valid_indices,
        normalization_stats=train_norm_stats,
    )

    logger.info("Generating wavelet...")
    resolved_wavelet_gain = cfg.wavelet_gain
    if resolved_wavelet_gain is None:
        unit_wavelet = make_wavelet(
            wavelet_type=cfg.wavelet_type,
            freq=cfg.wavelet_freq,
            dt=cfg.wavelet_dt,
            length=cfg.wavelet_length,
            gain=1.0,
        )
        resolved_wavelet_gain = estimate_wavelet_gain(
            seismic,
            lfm,
            mask,
            unit_wavelet,
            seis_rms=train_dataset.seis_rms,
            max_traces=cfg.wavelet_gain_num_traces,
            candidate_trace_indices=train_dataset.valid_indices,
        )
        cfg.wavelet_gain = resolved_wavelet_gain

    wavelet = make_wavelet(
        wavelet_type=cfg.wavelet_type,
        freq=cfg.wavelet_freq,
        dt=cfg.wavelet_dt,
        length=cfg.wavelet_length,
        gain=float(resolved_wavelet_gain),
    )
    return DatasetBundle(
        train_dataset=train_dataset,
        inference_dataset=inference_dataset,
        val_dataset=val_dataset,
        wavelet=wavelet,
        geometry=geometry,
        split_metadata=split_metadata,
    )
