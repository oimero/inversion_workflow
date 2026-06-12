"""ginn.data — 数据加载、预处理与 Dataset 定义。

工作流：
1. 读取 SEG-Y 地震体
2. 读取预计算低频模型，或从参考阻抗体生成低频模型
3. 解析子波并按需自动估计边界影响宽度
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
from cup.seismic.survey import open_survey
from cup.seismic.target_zone import TargetZone
from cup.utils.io import resolve_relative_path
from cup.seismic.wavelet import (
    DEFAULT_ACTIVE_SUPPORT_THRESHOLD,
    compute_wavelet_active_half_support_s,
    load_wavelet_csv,
    make_wavelet,
    validate_wavelet_dt,
)
from ginn.anchor import WellControlData, build_well_control_data
from ginn.config import GINNConfig
from ginn.masking import build_eroded_loss_mask as _build_eroded_loss_mask
from ginn.masking import build_residual_taper as _build_residual_taper
from ginn.masking import get_valid_trace_indices as _get_valid_trace_indices
from ginn.masking import select_spatial_validation_split as _select_spatial_validation_split

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

BOUNDARY_EFFECT_WAVELET_THRESHOLD = DEFAULT_ACTIVE_SUPPORT_THRESHOLD
DYNAMIC_GAIN_LOG_CLIP = 3.0


def compute_dynamic_gain_median(
    dynamic_gain_flat: np.ndarray,
    mask_flat: np.ndarray,
    selected_indices: np.ndarray,
) -> float:
    """Estimate a robust positive reference gain from selected masked samples."""
    selected_gain = dynamic_gain_flat[selected_indices]
    selected_mask = mask_flat[selected_indices]
    valid_gain = selected_gain[selected_mask]
    valid_gain = valid_gain[np.isfinite(valid_gain) & (valid_gain > 0.0)]
    if valid_gain.size == 0:
        raise ValueError("Cannot enable dynamic gain input: no positive finite gain values in selected mask.")
    return float(np.median(valid_gain))


def normalize_dynamic_gain_input(dynamic_gain: np.ndarray, gain_median: float) -> np.ndarray:
    """Convert multiplicative gain to a compact additive input channel."""
    if gain_median <= 0.0:
        raise ValueError(f"dynamic gain median must be positive, got {gain_median}.")
    safe_gain = np.maximum(dynamic_gain.astype(np.float32, copy=False), 1e-6)
    gain_norm = np.log(safe_gain / float(gain_median))
    return np.clip(gain_norm, -DYNAMIC_GAIN_LOG_CLIP, DYNAMIC_GAIN_LOG_CLIP).astype(np.float32)


@dataclass
class DatasetBundle:
    """GINN 训练/验证/推理所需数据集集合。"""

    train_dataset: "SeismicTraceDataset"
    inference_dataset: "SeismicTraceDataset"
    val_dataset: "SeismicTraceDataset | None"
    wavelet: np.ndarray
    geometry: Dict[str, Any]
    split_metadata: Dict[str, Any]
    lfm_metadata: Dict[str, Any]
    well_control_summary: Dict[str, Any]
    x_grid: np.ndarray
    y_grid: np.ndarray


def resolve_wavelet_from_config(cfg: GINNConfig, geometry: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate the unit-gain wavelet configured for time-domain GINN."""
    logger.info("Resolving wavelet from source=%s...", cfg.wavelet_source)
    if cfg.wavelet_source == "precomputed_wavelet":
        wavelet_time_s, base_wavelet = load_wavelet_csv(cfg.wavelet_file)  # type: ignore[arg-type]
        validate_wavelet_dt(wavelet_time_s, float(geometry["sample_step"]))
    elif cfg.wavelet_source == "ricker_wavelet":
        wavelet_time_s, base_wavelet = make_wavelet(
            wavelet_type=cfg.wavelet_type,
            freq=cfg.wavelet_freq,
            dt=cfg.wavelet_dt,
            length=cfg.wavelet_length,
            gain=1.0,
        )
    else:
        raise ValueError(f"Unsupported wavelet_source: {cfg.wavelet_source}")

    return np.asarray(wavelet_time_s, dtype=np.float64), np.asarray(base_wavelet, dtype=np.float32)


def compute_boundary_effect_samples_from_wavelet(
    wavelet_time_s: np.ndarray,
    wavelet: np.ndarray,
    seismic_sample_step_s: float,
    *,
    active_threshold: float = BOUNDARY_EFFECT_WAVELET_THRESHOLD,
) -> int:
    """Estimate boundary-effect width from the wavelet active half-support.

    The active support is where ``abs(wavelet)`` is at least ``active_threshold``
    times the wavelet peak amplitude. The returned value is the largest active
    time offset from the peak, converted to seismic sample count with ``ceil``.
    """
    if seismic_sample_step_s <= 0.0:
        raise ValueError(f"seismic_sample_step_s must be positive, got {seismic_sample_step_s}.")
    if not 0.0 < active_threshold <= 1.0:
        raise ValueError(f"active_threshold must be within (0, 1], got {active_threshold}.")

    half_support_s = compute_wavelet_active_half_support_s(
        wavelet_time_s,
        wavelet,
        active_threshold=active_threshold,
    )
    return int(math.ceil(half_support_s / float(seismic_sample_step_s)))


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

    # sosfiltfilt 会前后各滤波一次，单程阶数取一半以匹配配置中的总阶数。
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


def _json_scalar_to_dict(value: np.ndarray) -> dict[str, Any]:
    payload = np.asarray(value).item()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(str(payload))


def _load_lowfreq_npz_contract(lowfreq_file: Path) -> tuple[dict[str, Any], dict[str, Any], np.ndarray | None]:
    """Load metadata needed to verify a time-domain LFM NPZ against seismic geometry."""
    lowfreq_path = Path(lowfreq_file)
    if lowfreq_path.suffix.lower() != ".npz":
        return {}, {}, None

    with np.load(lowfreq_path, allow_pickle=False) as archive:
        metadata = _json_scalar_to_dict(archive["metadata_json"]) if "metadata_json" in archive.files else {}
        geometry = _json_scalar_to_dict(archive["geometry_json"]) if "geometry_json" in archive.files else {}
        samples = np.asarray(archive["samples"], dtype=np.float64) if "samples" in archive.files else None
    return metadata, geometry, samples


def _validate_time_lfm_contract(
    *,
    lfm_path: Path,
    lfm_geometry: dict[str, Any],
    lfm_samples: np.ndarray | None,
    seismic_geometry: dict[str, Any],
) -> None:
    """Ensure a precomputed LFM uses the same time axis as the training seismic."""
    if not lfm_geometry:
        raise ValueError("AI LFM NPZ must contain geometry_json for time-domain training.")

    sample_domain = str(lfm_geometry.get("sample_domain", "")).strip().lower()
    if sample_domain != "time":
        raise ValueError(f"AI LFM geometry_json.sample_domain must be 'time', got {sample_domain!r}.")

    sample_unit = str(lfm_geometry.get("sample_unit", "")).strip().lower()
    if sample_unit not in {"s", "sec", "second", "seconds"}:
        raise ValueError(f"AI LFM geometry_json.sample_unit must be seconds ('s'), got {sample_unit!r}.")

    n_sample = int(seismic_geometry["n_sample"])
    sample_min = float(seismic_geometry["sample_min"])
    sample_step = float(seismic_geometry["sample_step"])
    expected_samples = sample_min + np.arange(n_sample, dtype=np.float64) * sample_step

    if lfm_samples is None:
        raise ValueError("AI LFM NPZ must contain a 'samples' axis for time-domain training.")
    samples = np.asarray(lfm_samples, dtype=np.float64)
    if samples.shape != (n_sample,):
        raise ValueError(
            f"AI LFM samples axis length {samples.size} does not match seismic n_sample={n_sample}."
        )
    if np.any(~np.isfinite(samples)) or np.any(np.diff(samples) <= 0.0):
        raise ValueError("AI LFM samples axis must be finite and strictly increasing.")
    if not np.allclose(samples, expected_samples, rtol=0.0, atol=1e-6):
        max_diff = float(np.max(np.abs(samples - expected_samples)))
        raise ValueError(
            "AI LFM samples axis does not match the seismic time axis "
            f"(max_abs_diff={max_diff:.6g}s, lfm={lfm_path})."
        )


def _validate_dynamic_gain_npz_contract(
    *,
    gain_path: Path,
    archive: Any,
    volume: np.ndarray,
    seismic_geometry: dict[str, Any] | None,
) -> None:
    """Validate time-domain dynamic gain schema and axis metadata."""
    required_keys = {"metadata_json", "geometry_json", "samples", "inline", "xline"}
    missing = required_keys - set(archive.files)
    if missing:
        raise ValueError(f"Dynamic gain NPZ is missing required keys {sorted(missing)}: {gain_path}")

    metadata = _json_scalar_to_dict(archive["metadata_json"])
    expected = {
        "schema_version": "dynamic_gain_v2",
        "sample_domain": "time",
        "sample_unit": "s",
        "normalization": "seismic_raw_divided_by_train_mask_rms",
        "gain_model_is_relative_to_fixed_gain": False,
    }
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise ValueError(
                f"Dynamic gain metadata field {key!r} must be {expected_value!r}, got {actual!r}: {gain_path}"
            )
    if metadata.get("gain_reference") != "unit_wavelet_ginn_target_synthetic_to_normalized_observation":
        raise ValueError(
            "Dynamic gain metadata gain_reference must be "
            "'unit_wavelet_ginn_target_synthetic_to_normalized_observation'."
        )
    if metadata.get("anchor_target_band") != "lowpass_reference_to_ginn_cutoff":
        raise ValueError(
            "Dynamic gain metadata anchor_target_band must be "
            "'lowpass_reference_to_ginn_cutoff'."
        )
    train_mask_rms = float(metadata.get("train_mask_rms", np.nan))
    if not np.isfinite(train_mask_rms) or train_mask_rms <= 0.0:
        raise ValueError(f"Dynamic gain metadata train_mask_rms must be positive finite, got {train_mask_rms}.")

    if seismic_geometry is None:
        return

    expected_shape = (
        int(seismic_geometry["n_il"]),
        int(seismic_geometry["n_xl"]),
        int(seismic_geometry["n_sample"]),
    )
    if tuple(volume.shape) != expected_shape:
        raise ValueError(f"Dynamic gain volume shape {volume.shape} does not match seismic geometry {expected_shape}.")

    gain_geometry = _json_scalar_to_dict(archive["geometry_json"])
    for key in ("sample_domain", "sample_unit", "n_il", "n_xl", "n_sample"):
        if str(gain_geometry.get(key)) != str(seismic_geometry.get(key)):
            raise ValueError(
                f"Dynamic gain geometry_json.{key}={gain_geometry.get(key)!r} "
                f"does not match seismic geometry {seismic_geometry.get(key)!r}."
            )

    n_sample = int(seismic_geometry["n_sample"])
    samples = np.asarray(archive["samples"], dtype=np.float64)
    expected_samples = float(seismic_geometry["sample_min"]) + np.arange(n_sample, dtype=np.float64) * float(
        seismic_geometry["sample_step"]
    )
    if samples.shape != (n_sample,):
        raise ValueError(f"Dynamic gain samples shape {samples.shape} does not match n_sample={n_sample}.")
    if np.any(~np.isfinite(samples)) or np.any(np.diff(samples) <= 0.0):
        raise ValueError("Dynamic gain samples axis must be finite and strictly increasing.")
    if not np.allclose(samples, expected_samples, rtol=0.0, atol=1e-6):
        max_diff = float(np.max(np.abs(samples - expected_samples)))
        raise ValueError(f"Dynamic gain samples axis does not match seismic time axis (max_abs_diff={max_diff:.6g}s).")

    axis_specs = {
        "inline": (
            int(seismic_geometry["n_il"]),
            float(seismic_geometry["inline_min"]),
            float(seismic_geometry["inline_step"]),
        ),
        "xline": (
            int(seismic_geometry["n_xl"]),
            float(seismic_geometry["xline_min"]),
            float(seismic_geometry["xline_step"]),
        ),
    }
    for axis_name, (axis_size, axis_min, axis_step) in axis_specs.items():
        axis = np.asarray(archive[axis_name], dtype=np.float64)
        expected_axis = axis_min + np.arange(axis_size, dtype=np.float64) * axis_step
        if axis.shape != (axis_size,):
            raise ValueError(
                f"Dynamic gain {axis_name} axis shape {axis.shape} does not match expected {(axis_size,)}."
            )
        if np.any(~np.isfinite(axis)):
            raise ValueError(f"Dynamic gain {axis_name} axis must be finite.")
        if not np.allclose(axis, expected_axis, rtol=0.0, atol=1e-6):
            max_diff = float(np.max(np.abs(axis - expected_axis)))
            raise ValueError(
                f"Dynamic gain {axis_name} axis does not match seismic geometry (max_abs_diff={max_diff:.6g})."
            )


def load_dynamic_gain_model(gain_model_file: Path, seismic_geometry: dict[str, Any] | None = None) -> np.ndarray:
    """Load a time-domain ``dynamic_gain_v2`` NPZ volume."""
    gain_path = Path(gain_model_file)
    if not gain_path.exists():
        raise FileNotFoundError(gain_path)

    if gain_path.suffix.lower() != ".npz":
        raise ValueError(
            f"Dynamic gain model must be a dynamic_gain_v2 .npz file, got suffix {gain_path.suffix!r}: {gain_path}"
        )

    with np.load(gain_path, allow_pickle=False) as archive:
        if "volume" not in archive.files:
            raise ValueError(f"Expected key 'volume' in dynamic gain model archive, got {archive.files}.")
        volume = archive["volume"]
        _validate_dynamic_gain_npz_contract(
            gain_path=gain_path,
            archive=archive,
            volume=np.asarray(volume),
            seismic_geometry=seismic_geometry,
        )
        metadata = _json_scalar_to_dict(archive["metadata_json"])
        logger.info("Loaded dynamic gain metadata: %s", metadata)

    volume = np.asarray(volume, dtype=np.float32)
    if np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("Dynamic gain model must be finite and positive everywhere.")
    return volume


@dataclass(frozen=True)
class TraceQCData:
    """Read-only copies of one physical trace used by GINN QC."""

    seismic_raw: np.ndarray
    loss_mask: np.ndarray
    dynamic_gain: np.ndarray | None


class SeismicTraceDataset(Dataset):
    """逐道 1D 地震数据 Dataset。

    每个 item 包含：
    - ``input``：(3, n_sample) — 归一化地震道 + 归一化 LFM + 目的层 mask
    - ``obs``：(1, n_sample) — 归一化观测地震道（用于损失计算）
    - ``mask``：(1, n_sample) — core 布尔掩码
    - ``loss_mask``：(1, n_sample) — eroded core 掩码，仅用于 waveform loss
    - ``taper_weight``：(1, n_sample) — core+halo 平滑权重，用于残差收口
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
        dynamic_gain_flat: np.ndarray | None = None,
        include_lfm_input: bool = True,
        include_mask_input: bool = True,
        include_dynamic_gain_input: bool = False,
        normalization_stats: tuple[float, float] | None = None,
        dynamic_gain_median: float | None = None,
        well_control_data: WellControlData | None = None,
    ) -> None:
        n_traces, n_sample = seismic_flat.shape
        assert lfm_flat.shape == seismic_flat.shape
        assert mask_flat.shape == seismic_flat.shape
        assert loss_mask_flat.shape == seismic_flat.shape
        assert taper_flat.shape == seismic_flat.shape
        if dynamic_gain_flat is not None and dynamic_gain_flat.shape != seismic_flat.shape:
            raise ValueError(
                f"dynamic_gain_flat shape {dynamic_gain_flat.shape} does not match seismic shape {seismic_flat.shape}."
            )

        selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
        if selected_indices.size == 0:
            raise ValueError("selected_indices must contain at least one trace.")

        self._seismic_flat = seismic_flat
        self._lfm_flat = lfm_flat
        self._mask_flat = mask_flat
        self._loss_mask_flat = loss_mask_flat
        self._taper_flat = taper_flat
        self._dynamic_gain_flat = dynamic_gain_flat
        self._valid_indices = selected_indices
        self._include_lfm_input = bool(include_lfm_input)
        self._include_mask_input = bool(include_mask_input)
        self._include_dynamic_gain_input = bool(include_dynamic_gain_input)
        self._zero_anchor_target_log_ai = np.zeros((n_sample,), dtype=np.float32)
        self._zero_anchor_weight = np.zeros((n_sample,), dtype=np.float32)
        self._well_control = WellControlData.empty(n_sample) if well_control_data is None else well_control_data
        self._well_control_lookup = {
            int(flat_idx): row
            for row, flat_idx in enumerate(np.asarray(self._well_control.flat_indices, dtype=np.int64))
        }
        anchor_mask = np.isin(self._valid_indices, self._well_control.flat_indices)
        self._anchor_dataset_indices = np.flatnonzero(anchor_mask).astype(np.int64)
        self._ordinary_dataset_indices = np.flatnonzero(~anchor_mask).astype(np.int64)
        if self._include_dynamic_gain_input and self._dynamic_gain_flat is None:
            logger.warning("include_dynamic_gain_input=True but no dynamic gain model is loaded; using a zero channel.")

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

        if self._include_dynamic_gain_input and self._dynamic_gain_flat is not None:
            self._dynamic_gain_median = (
                float(dynamic_gain_median)
                if dynamic_gain_median is not None
                else compute_dynamic_gain_median(self._dynamic_gain_flat, self._mask_flat, self._valid_indices)
            )
        elif self._include_dynamic_gain_input:
            self._dynamic_gain_median = 1.0
        else:
            self._dynamic_gain_median = None

        logger.info(
            "Dataset: %d selected traces / %d total, seis_rms=%.4f, lfm_scale=%.2f, input_channels=%s",
            len(self._valid_indices),
            n_traces,
            self._seis_rms,
            self._lfm_scale,
            self.input_channel_names,
        )
        if self._anchor_dataset_indices.size:
            logger.info(
                "Dataset well-control samples: %d anchor-influenced / %d ordinary",
                int(self._anchor_dataset_indices.size),
                int(self._ordinary_dataset_indices.size),
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

    @property
    def lfm_flat(self) -> np.ndarray:
        """返回完整展平 LFM，用作未进入推理域 trace 的物理背景值。"""
        return self._lfm_flat

    @property
    def dynamic_gain_median(self) -> float | None:
        return self._dynamic_gain_median

    @property
    def anchor_dataset_indices(self) -> np.ndarray:
        return self._anchor_dataset_indices

    @property
    def ordinary_dataset_indices(self) -> np.ndarray:
        return self._ordinary_dataset_indices

    @property
    def well_control_summary(self) -> Dict[str, Any]:
        return dict(self._well_control.summary)

    @property
    def input_channel_names(self) -> tuple[str, ...]:
        channels = ["seismic"]
        if self._include_lfm_input:
            channels.append("lfm")
        if self._include_mask_input:
            channels.append("mask")
        if self._include_dynamic_gain_input:
            channels.append("dynamic_gain_log_ratio")
        return tuple(channels)

    def trace_qc_data(self, flat_idx: int) -> TraceQCData:
        """Return read-only QC arrays for an absolute flattened trace index."""
        index = int(flat_idx)
        if index < 0 or index >= self._seismic_flat.shape[0]:
            raise IndexError(f"flat_idx {index} is outside [0, {self._seismic_flat.shape[0]}).")

        seismic_raw = np.array(self._seismic_flat[index], dtype=np.float64, copy=True)
        loss_mask = np.array(self._loss_mask_flat[index], dtype=bool, copy=True)
        dynamic_gain = (
            None
            if self._dynamic_gain_flat is None
            else np.array(self._dynamic_gain_flat[index], dtype=np.float64, copy=True)
        )
        seismic_raw.setflags(write=False)
        loss_mask.setflags(write=False)
        if dynamic_gain is not None:
            dynamic_gain.setflags(write=False)
        return TraceQCData(
            seismic_raw=seismic_raw,
            loss_mask=loss_mask,
            dynamic_gain=dynamic_gain,
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        flat_idx = self._valid_indices[idx]

        seis = self._seismic_flat[flat_idx]  # (n_sample,)
        lfm = self._lfm_flat[flat_idx]  # (n_sample,)
        core_mask = self._mask_flat[flat_idx]  # (n_sample,)
        loss_mask = self._loss_mask_flat[flat_idx]  # (n_sample,)
        taper_weight = self._taper_flat[flat_idx]  # (n_sample,)
        dynamic_gain = self._dynamic_gain_flat[flat_idx] if self._dynamic_gain_flat is not None else None

        control_row = self._well_control_lookup.get(int(flat_idx))
        if control_row is None:
            anchor_target_log_ai = self._zero_anchor_target_log_ai
            anchor_weight = self._zero_anchor_weight
            well_influence = np.float32(0.0)
            waveform_weight_scale = np.float32(1.0)
            has_anchor = np.float32(0.0)
        else:
            anchor_target_log_ai = self._well_control.target_log_ai[control_row]
            anchor_weight = self._well_control.anchor_weight[control_row]
            well_influence = np.float32(self._well_control.well_influence[control_row])
            waveform_weight_scale = np.float32(self._well_control.waveform_weight_scale[control_row])
            has_anchor = np.float32(1.0)

        # 保留 LFM 原始量纲用于物理正演
        lfm_raw = lfm.copy()

        # 归一化：地震道除以 RMS，LFM 除以全局最大值（保持正值）
        seis_norm = seis / self._seis_rms
        lfm_norm = lfm / self._lfm_scale

        channels = [seis_norm.astype(np.float32)]
        if self._include_lfm_input:
            channels.append(lfm_norm.astype(np.float32))
        if self._include_mask_input:
            channels.append(core_mask.astype(np.float32))
        if self._include_dynamic_gain_input:
            if dynamic_gain is None:
                channels.append(np.zeros_like(seis_norm, dtype=np.float32))
            else:
                channels.append(normalize_dynamic_gain_input(dynamic_gain, self._dynamic_gain_median))  # type: ignore[arg-type]
        x = np.stack(channels, axis=0)

        item = {
            "input": torch.from_numpy(x).float(),  # (3, n_sample)
            "obs": torch.from_numpy(seis_norm[np.newaxis]).float(),  # (1, n_sample)
            "mask": torch.from_numpy(core_mask[np.newaxis]).bool(),  # (1, n_sample)
            "loss_mask": torch.from_numpy(loss_mask[np.newaxis]).bool(),  # (1, n_sample)
            "taper_weight": torch.from_numpy(taper_weight[np.newaxis]).float(),  # (1, n_sample)
            "lfm_raw": torch.from_numpy(lfm_raw[np.newaxis]).float(),  # (1, n_sample)
            "anchor_target_log_ai": torch.from_numpy(anchor_target_log_ai[np.newaxis]).float(),
            "anchor_weight": torch.from_numpy(anchor_weight[np.newaxis]).float(),
            "well_influence": torch.tensor([float(well_influence)], dtype=torch.float32),
            "waveform_weight_scale": torch.tensor([float(waveform_weight_scale)], dtype=torch.float32),
            "has_anchor": torch.tensor([float(has_anchor)], dtype=torch.float32),
        }
        if dynamic_gain is not None:
            item["dynamic_gain"] = torch.from_numpy(dynamic_gain[np.newaxis]).float()
        return item


def build_dataset(cfg: GINNConfig) -> DatasetBundle:
    """根据配置构建训练/验证/推理数据集。

    Returns
    -------
    DatasetBundle
        包含训练、验证、推理数据集，以及子波与几何信息。
    """
    logger.info("Loading seismic volume...")
    seismic_type = str(cfg.seismic_type).strip().lower()
    seismic = import_seismic(
        cfg.seismic_file,
        seismic_type=seismic_type,
        iline=cfg.segy_iline if seismic_type == "segy" else None,
        xline=cfg.segy_xline if seismic_type == "segy" else None,
        istep=cfg.segy_istep if seismic_type == "segy" else None,
        xstep=cfg.segy_xstep if seismic_type == "segy" else None,
    )

    seismic_ctx = open_survey(
        cfg.seismic_file,
        seismic_type=seismic_type,
        segy_options={
            "iline": cfg.segy_iline,
            "xline": cfg.segy_xline,
            "istep": cfg.segy_istep,
            "xstep": cfg.segy_xstep,
        }
        if seismic_type == "segy"
        else None,
    )
    geometry = seismic_ctx.describe_geometry(domain="time")
    ilines_xy = (
        float(geometry["inline_min"])
        + np.arange(int(geometry["n_il"]), dtype=np.float64) * float(geometry["inline_step"])
    )
    xlines_xy = (
        float(geometry["xline_min"])
        + np.arange(int(geometry["n_xl"]), dtype=np.float64) * float(geometry["xline_step"])
    )
    x_grid, y_grid = seismic_ctx.line_geometry.trace_xy_grids(ilines_xy, xlines_xy)

    n_il, n_xl, n_sample = seismic.shape
    geometry_n_il = int(geometry["n_il"])
    geometry_n_xl = int(geometry["n_xl"])
    geometry_n_sample = int(geometry["n_sample"])
    if (n_il, n_xl, n_sample) != (geometry_n_il, geometry_n_xl, geometry_n_sample):
        raise ValueError(
            "Seismic volume shape does not match queried geometry: "
            f"volume={(n_il, n_xl, n_sample)}, geometry={(geometry_n_il, geometry_n_xl, geometry_n_sample)}"
        )

    logger.info("Loading precomputed low-frequency model from %s...", cfg.ai_lfm_file)
    lfm = load_lowfreq_model(cfg.ai_lfm_file)  # type: ignore

    # ── read horizon files + target-layer QC from LFM NPZ metadata ──
    tl_min_thickness = None
    tl_nearest_limit = None
    tl_outlier_threshold = None
    tl_outlier_min_neighbor = 2
    top_horizon_file = None
    bot_horizon_file = None
    lfm_path = Path(str(cfg.ai_lfm_file))
    lfm_meta, lfm_geometry, lfm_samples = _load_lowfreq_npz_contract(lfm_path)
    _validate_time_lfm_contract(
        lfm_path=lfm_path,
        lfm_geometry=lfm_geometry,
        lfm_samples=lfm_samples,
        seismic_geometry=geometry,
    )
    tl_meta = lfm_meta.get("target_layer", {})
    tl_min_thickness = tl_meta.get("min_thickness")
    tl_nearest_limit = tl_meta.get("nearest_distance_limit")
    tl_outlier_threshold = tl_meta.get("outlier_threshold")
    tl_outlier_min_neighbor = tl_meta.get("outlier_min_neighbor_count", 2)
    hz_list = lfm_meta.get("horizons", [])
    if len(hz_list) >= 2:
        top_horizon_file = hz_list[0]["file"]
        bot_horizon_file = hz_list[-1]["file"]
    if top_horizon_file is None or bot_horizon_file is None:
        raise ValueError("AI LFM NPZ metadata must contain at least two sorted horizons.")

    logger.info("Loading raw top/bottom interpretation horizons...")
    top_horizon_file = str(resolve_relative_path(top_horizon_file, root=_REPO_ROOT))
    bot_horizon_file = str(resolve_relative_path(bot_horizon_file, root=_REPO_ROOT))
    top_df_raw = import_interpretation_petrel(top_horizon_file)
    bot_df_raw = import_interpretation_petrel(bot_horizon_file)

    logger.info("Building target layer from raw interpretations...")
    target_layer = TargetZone(
        raw_horizon_dfs={"top": top_df_raw, "bottom": bot_df_raw},
        geometry=geometry,
        horizon_names=["top", "bottom"],
        min_thickness=tl_min_thickness,
        nearest_distance_limit=tl_nearest_limit,
        outlier_threshold=tl_outlier_threshold,
        outlier_min_neighbor_count=tl_outlier_min_neighbor,
    )
    train_mask = target_layer.to_mask(use_valid_control_mask=True)
    inference_mask = target_layer.to_mask(use_valid_control_mask=False)

    if train_mask.shape != seismic.shape:
        raise ValueError(f"Training mask shape {train_mask.shape} does not match seismic shape {seismic.shape}.")
    if inference_mask.shape != seismic.shape:
        raise ValueError(f"Inference mask shape {inference_mask.shape} does not match seismic shape {seismic.shape}.")
    if lfm.shape != seismic.shape:
        raise ValueError(f"LFM shape {lfm.shape} does not match seismic shape {seismic.shape}.")

    dynamic_gain = None
    if cfg.gain_source == "dynamic_gain_model":
        logger.info("Loading dynamic gain model from %s...", cfg.dynamic_gain_model)
        dynamic_gain = load_dynamic_gain_model(cfg.dynamic_gain_model, seismic_geometry=geometry)  # type: ignore
        if dynamic_gain.shape != seismic.shape:
            raise ValueError(f"Dynamic gain shape {dynamic_gain.shape} does not match seismic shape {seismic.shape}.")

    wavelet_time_s, base_wavelet = resolve_wavelet_from_config(cfg, geometry)
    boundary_effect_samples = cfg.boundary_effect_samples
    if boundary_effect_samples is None:
        boundary_effect_samples = compute_boundary_effect_samples_from_wavelet(
            wavelet_time_s,
            base_wavelet,
            float(geometry["sample_step"]),
        )
        cfg.boundary_effect_samples = boundary_effect_samples
        logger.info(
            "Auto boundary_effect_samples=%d from wavelet active half-support (threshold=%.2f, sample_step=%.6f s)",
            boundary_effect_samples,
            BOUNDARY_EFFECT_WAVELET_THRESHOLD,
            float(geometry["sample_step"]),
        )
    else:
        boundary_effect_samples = int(boundary_effect_samples)
        cfg.boundary_effect_samples = boundary_effect_samples

    # ── 展平并预计算掩码 / taper（只算一次） ──
    seismic_flat = seismic.reshape(n_il * n_xl, n_sample)
    lfm_flat = lfm.reshape(n_il * n_xl, n_sample)
    dynamic_gain_flat = dynamic_gain.reshape(n_il * n_xl, n_sample) if dynamic_gain is not None else None
    train_mask_flat = train_mask.reshape(n_il * n_xl, n_sample)
    inference_mask_flat = inference_mask.reshape(n_il * n_xl, n_sample)
    train_loss_mask_flat = _build_eroded_loss_mask(train_mask_flat, erosion_samples=boundary_effect_samples)
    inference_loss_mask_flat = _build_eroded_loss_mask(
        inference_mask_flat,
        erosion_samples=boundary_effect_samples,
    )
    train_taper_flat = _build_residual_taper(train_mask_flat, halo_samples=boundary_effect_samples)
    inference_taper_flat = _build_residual_taper(inference_mask_flat, halo_samples=boundary_effect_samples)
    train_valid_indices = _get_valid_trace_indices(train_mask_flat)
    inference_valid_indices = _get_valid_trace_indices(inference_mask_flat)
    logger.info(
        "Preprocessed masks: train=%d traces, inference=%d traces, boundary_effect_samples=%d",
        train_valid_indices.size,
        inference_valid_indices.size,
        boundary_effect_samples,
    )

    # ── 验证集切分 ──
    train_indices = train_valid_indices
    val_indices: np.ndarray | None = None
    split_metadata: Dict[str, Any] = {
        "mode": "none",
        "train_trace_count": int(train_indices.size),
        "val_trace_count": 0,
        "gap_trace_count": 0,
        "inference_trace_count": int(inference_valid_indices.size),
    }
    if cfg.validation_split_mode == "spatial_block" and cfg.validation_fraction > 0.0:
        train_indices, val_indices, split_metadata = _select_spatial_validation_split(
            train_valid_indices,
            n_il=n_il,
            n_xl=n_xl,
            validation_fraction=cfg.validation_fraction,
            gap_traces=cfg.validation_gap_traces,
            anchor=cfg.validation_block_anchor,
        )
        split_metadata["inference_trace_count"] = int(inference_valid_indices.size)
        logger.info("Validation split: %s", split_metadata)
    else:
        logger.info("Validation split disabled.")

    # ── 构建 well-control map ──
    if cfg.well_control_enabled:
        train_well_control = build_well_control_data(
            anchor_file=cfg.log_ai_anchor_file,
            selected_indices=train_indices,
            sample_domain="time",
            n_sample=n_sample,
            n_traces=n_il * n_xl,
            geometry=geometry,
            lambda_log_ai_anchor=cfg.lambda_log_ai_anchor,
            radius_xy_m=cfg.log_ai_anchor_radius_xy_m,
            x_grid=x_grid,
            y_grid=y_grid,
            well_waveform_min_weight=cfg.well_waveform_min_weight,
            distance_decay=cfg.well_anchor_distance_decay,
            log_prefix="Time well-control",
        )
    else:
        train_well_control = WellControlData.empty(
            n_sample,
            summary={
                "enabled": False,
                "anchor_file": cfg.log_ai_anchor_file,
                "reason": "well_control_enabled_false",
            },
        )
    empty_well_control = WellControlData.empty(n_sample)
    split_metadata["well_control"] = train_well_control.summary

    # ── 构建数据集（共享预计算的掩码 / taper，不重复计算） ──
    train_shared = (seismic_flat, lfm_flat, train_mask_flat, train_loss_mask_flat, train_taper_flat)
    inference_shared = (
        seismic_flat,
        lfm_flat,
        inference_mask_flat,
        inference_loss_mask_flat,
        inference_taper_flat,
    )
    train_dataset = SeismicTraceDataset(
        *train_shared,
        train_indices,
        dynamic_gain_flat=dynamic_gain_flat,
        include_lfm_input=cfg.include_lfm_input,
        include_mask_input=cfg.include_mask_input,
        include_dynamic_gain_input=cfg.include_dynamic_gain_input,
        well_control_data=train_well_control,
    )
    train_norm_stats = (train_dataset.seis_rms, train_dataset.lfm_scale)
    val_dataset = None
    if val_indices is not None and val_indices.size > 0:
        val_dataset = SeismicTraceDataset(
            *train_shared,
            val_indices,
            dynamic_gain_flat=dynamic_gain_flat,
            include_lfm_input=cfg.include_lfm_input,
            include_mask_input=cfg.include_mask_input,
            include_dynamic_gain_input=cfg.include_dynamic_gain_input,
            normalization_stats=train_norm_stats,
            dynamic_gain_median=train_dataset.dynamic_gain_median,
            well_control_data=empty_well_control,
        )
    inference_dataset = SeismicTraceDataset(
        *inference_shared,
        inference_valid_indices,
        dynamic_gain_flat=dynamic_gain_flat,
        include_lfm_input=cfg.include_lfm_input,
        include_mask_input=cfg.include_mask_input,
        include_dynamic_gain_input=cfg.include_dynamic_gain_input,
        normalization_stats=train_norm_stats,
        dynamic_gain_median=train_dataset.dynamic_gain_median,
        well_control_data=empty_well_control,
    )

    if cfg.gain_source == "fixed_gain":
        resolved_fixed_gain = cfg.fixed_gain
        if resolved_fixed_gain is None:
            raise ValueError(
                "fixed_gain must be explicitly configured. LFM-based automatic gain estimation is invalid for "
                "the GINN target band; use recommended_fixed_gain.json from scripts/dynamic_gain.py."
            )
    elif cfg.gain_source == "dynamic_gain_model":
        resolved_fixed_gain = 1.0
        logger.info("Using dynamic gain model; fixed gain is disabled.")
    else:
        raise ValueError(f"Unsupported gain_source: {cfg.gain_source}")

    if cfg.wavelet_source == "ricker_wavelet":
        logger.info(
            "Generated %s wavelet: freq=%.1f Hz, dt=%.4f s, length=%d, fixed_gain=%.2f",
            cfg.wavelet_type,
            cfg.wavelet_freq,
            cfg.wavelet_dt,
            cfg.wavelet_length,
            float(resolved_fixed_gain),
        )
    wavelet = (base_wavelet * float(resolved_fixed_gain)).astype(np.float32)
    return DatasetBundle(
        train_dataset=train_dataset,
        inference_dataset=inference_dataset,
        val_dataset=val_dataset,
        wavelet=wavelet,
        geometry=geometry,
        split_metadata=split_metadata,
        lfm_metadata=lfm_meta,
        well_control_summary=train_well_control.summary,
        x_grid=x_grid,
        y_grid=y_grid,
    )
