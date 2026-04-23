"""Depth-domain data loading and dataset construction for GINN."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cup.petrel.load import import_interpretation_petrel, import_seismic
from cup.seismic.target_layer import TargetLayer
from ginn_depth.config import DepthGINNConfig
from ginn_depth.physics import DepthForwardModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DepthLfmVolume:
    """Precomputed depth-domain low-frequency model volume."""

    volume: np.ndarray
    ilines: np.ndarray
    xlines: np.ndarray
    samples: np.ndarray
    geometry: dict[str, Any]
    metadata: dict[str, Any]
    variance_volume: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.volume.shape)

    @property
    def dz(self) -> float:
        if self.samples.size < 2:
            raise ValueError("samples must contain at least two depth samples.")
        return float(np.median(np.diff(self.samples)))

    def nearest_indices(self, iline: float, xline: float) -> tuple[int, int]:
        il_idx = int(np.argmin(np.abs(self.ilines - float(iline))))
        xl_idx = int(np.argmin(np.abs(self.xlines - float(xline))))
        return il_idx, xl_idx

    def trace_by_index(self, il_idx: int, xl_idx: int) -> np.ndarray:
        return np.asarray(self.volume[int(il_idx), int(xl_idx), :])

    def trace_by_line(self, iline: float, xline: float) -> np.ndarray:
        il_idx, xl_idx = self.nearest_indices(iline, xline)
        return self.trace_by_index(il_idx, xl_idx)


def _json_scalar_to_dict(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            return {}
    if value is None:
        return {}
    text = str(value)
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def load_wavelet_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a time-domain wavelet CSV with columns ``time_s`` and ``amplitude``."""
    path = Path(path)
    df = pd.read_csv(path)
    required = {"time_s", "amplitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"wavelet CSV is missing columns: {sorted(missing)}")

    time_s = df["time_s"].to_numpy(dtype=np.float64)
    amplitude = df["amplitude"].to_numpy(dtype=np.float64)
    finite = np.isfinite(time_s) & np.isfinite(amplitude)
    if np.count_nonzero(finite) < 2:
        raise ValueError(f"wavelet CSV does not contain enough finite samples: {path}")

    time_s = time_s[finite]
    amplitude = amplitude[finite]
    order = np.argsort(time_s)
    time_s = time_s[order]
    amplitude = amplitude[order]
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("wavelet time_s samples must be strictly increasing after sorting.")
    return time_s, amplitude


def load_lfm_depth_npz(path: str | Path, *, volume_key: str = "volume") -> DepthLfmVolume:
    """Load an ``lfm_depth.py`` result saved as ``.npz``."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {volume_key, "ilines", "xlines", "samples"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"LFM npz is missing keys: {sorted(missing)}")

        volume = np.asarray(data[volume_key])
        ilines = np.asarray(data["ilines"], dtype=np.float64)
        xlines = np.asarray(data["xlines"], dtype=np.float64)
        samples = np.asarray(data["samples"], dtype=np.float64)
        variance_volume = np.asarray(data["variance_volume"]) if "variance_volume" in data.files else None
        geometry = _json_scalar_to_dict(data["geometry_json"]) if "geometry_json" in data.files else {}
        metadata = _json_scalar_to_dict(data["metadata_json"]) if "metadata_json" in data.files else {}

    if volume.ndim != 3:
        raise ValueError(f"Expected volume ndim=3, got {volume.ndim}.")
    if volume.shape != (ilines.size, xlines.size, samples.size):
        raise ValueError(
            "Volume shape does not match axes: "
            f"volume={volume.shape}, axes={(ilines.size, xlines.size, samples.size)}"
        )
    if samples.size < 2 or np.any(np.diff(samples) <= 0.0):
        raise ValueError("Depth samples must be strictly increasing.")

    return DepthLfmVolume(
        volume=volume,
        ilines=ilines,
        xlines=xlines,
        samples=samples,
        geometry=geometry,
        metadata=metadata,
        variance_volume=variance_volume,
    )


@dataclass
class DatasetBundle:
    """Depth-domain train/validation/inference datasets and physics inputs."""

    train_dataset: "DepthSeismicTraceDataset"
    inference_dataset: "DepthSeismicTraceDataset"
    val_dataset: "DepthSeismicTraceDataset | None"
    wavelet_time_s: np.ndarray
    wavelet_amp: np.ndarray
    depth_axis_m: np.ndarray
    geometry: Dict[str, Any]
    split_metadata: Dict[str, Any]


def _axis_step(axis: np.ndarray, axis_name: str) -> float:
    if axis.size < 2:
        raise ValueError(f"{axis_name} must contain at least two samples.")
    step = float(np.median(np.diff(axis)))
    if step <= 0.0:
        raise ValueError(f"{axis_name} must be strictly increasing.")
    return step


def geometry_from_axes(ilines: np.ndarray, xlines: np.ndarray, samples: np.ndarray) -> Dict[str, Any]:
    """Construct a TargetLayer-compatible geometry dict from explicit axes."""
    return {
        "n_il": int(ilines.size),
        "inline_min": float(ilines[0]),
        "inline_max": float(ilines[-1]),
        "inline_step": _axis_step(ilines, "ilines"),
        "n_xl": int(xlines.size),
        "xline_min": float(xlines[0]),
        "xline_max": float(xlines[-1]),
        "xline_step": _axis_step(xlines, "xlines"),
        "n_sample": int(samples.size),
        "sample_min": float(samples[0]),
        "sample_max": float(samples[-1]),
        "sample_step": _axis_step(samples, "samples"),
        "sample_domain": "depth",
        "sample_unit": "m",
    }


def estimate_wavelet_gain_depth(
    seismic: np.ndarray,
    ai_lfm: np.ndarray,
    vp_lfm: np.ndarray,
    mask: np.ndarray,
    wavelet_time_s: np.ndarray,
    wavelet_amp: np.ndarray,
    *,
    depth_axis_m: np.ndarray,
    seis_rms: float,
    max_traces: int,
    candidate_trace_indices: np.ndarray | None = None,
    batch_size: int = 16,
    amplitude_threshold: float = 0.0,
) -> float:
    """Estimate a scalar gain so synthetic depth seismic matches normalized observations."""
    n_sample = seismic.shape[-1]
    seismic_flat = seismic.reshape(-1, n_sample)
    ai_flat = ai_lfm.reshape(-1, n_sample)
    vp_flat = vp_lfm.reshape(-1, n_sample)
    mask_flat = mask.reshape(-1, n_sample).astype(bool, copy=False)

    valid_trace_indices = np.flatnonzero(mask_flat.any(axis=1))
    if candidate_trace_indices is not None:
        candidate_trace_indices = np.asarray(candidate_trace_indices, dtype=np.int64)
        valid_trace_indices = np.intersect1d(valid_trace_indices, candidate_trace_indices, assume_unique=False)
    if valid_trace_indices.size == 0:
        raise ValueError("Cannot auto-estimate wavelet gain because no valid depth traces were found in the mask.")

    n_selected = min(max_traces, valid_trace_indices.size)
    if n_selected < valid_trace_indices.size:
        rng = np.random.default_rng(0)
        selected = np.sort(rng.choice(valid_trace_indices, size=n_selected, replace=False))
    else:
        selected = valid_trace_indices

    forward_model = DepthForwardModel(
        wavelet_time_s,
        wavelet_amp,
        depth_axis_m=depth_axis_m,
        amplitude_threshold=amplitude_threshold,
    ).cpu()

    syn_sq_sum = 0.0
    obs_norm_sq_sum = 0.0
    n_valid = 0
    with torch.no_grad():
        for start in range(0, selected.size, batch_size):
            batch_indices = selected[start : start + batch_size]
            ai_batch = torch.from_numpy(ai_flat[batch_indices][:, np.newaxis, :]).float()
            vp_batch = torch.from_numpy(vp_flat[batch_indices]).float()
            mask_batch = mask_flat[batch_indices][:, np.newaxis, :]
            d_syn_unit = forward_model(ai_batch, vp_batch).cpu().numpy()

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
        "Auto depth-wavelet gain: traces=%d, valid_points=%d, obs_norm_rms=%.4f, syn_unit_rms=%.4f, gain=%.4f",
        selected.size,
        n_valid,
        obs_norm_rms,
        syn_rms,
        gain,
    )
    return float(gain)


def _resolve_mask_bounds(mask_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mask_flat.ndim != 2:
        raise ValueError(f"mask_flat must be 2D, got {mask_flat.shape}.")
    n_trace, n_sample = mask_flat.shape
    has_valid = mask_flat.any(axis=1)
    start = np.zeros(n_trace, dtype=np.int64)
    end = np.zeros(n_trace, dtype=np.int64)
    if np.any(has_valid):
        valid_mask = mask_flat[has_valid]
        start[has_valid] = np.argmax(valid_mask, axis=1)
        end[has_valid] = n_sample - np.argmax(valid_mask[:, ::-1], axis=1)
    return start, end, has_valid


def _build_eroded_loss_mask(mask_flat: np.ndarray, erosion_samples: int) -> np.ndarray:
    start, end, has_valid = _resolve_mask_bounds(mask_flat)
    n_sample = mask_flat.shape[1]
    sample_index = np.arange(n_sample, dtype=np.int64)[np.newaxis, :]
    lengths = end - start
    erosion = np.minimum(int(erosion_samples), np.maximum((lengths - 1) // 2, 0))
    loss_start = start + erosion
    loss_end = end - erosion
    return has_valid[:, np.newaxis] & (sample_index >= loss_start[:, np.newaxis]) & (sample_index < loss_end[:, np.newaxis])


def _build_residual_taper(mask_flat: np.ndarray, halo_samples: int) -> np.ndarray:
    start, end, has_valid = _resolve_mask_bounds(mask_flat)
    n_sample = mask_flat.shape[1]
    sample_index = np.arange(n_sample, dtype=np.int64)[np.newaxis, :]
    halo = int(halo_samples)
    support_start = np.maximum(start - halo, 0)
    support_end = np.minimum(end + halo, n_sample)

    core_region = has_valid[:, np.newaxis] & (sample_index >= start[:, np.newaxis]) & (sample_index < end[:, np.newaxis])
    left_region = (
        has_valid[:, np.newaxis]
        & (sample_index >= support_start[:, np.newaxis])
        & (sample_index < start[:, np.newaxis])
    )
    right_region = (
        has_valid[:, np.newaxis]
        & (sample_index >= end[:, np.newaxis])
        & (sample_index < support_end[:, np.newaxis])
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


def _get_valid_trace_indices(mask_flat: np.ndarray) -> np.ndarray:
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
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    if valid_indices.size == 0:
        raise ValueError("Cannot build a validation split because no valid traces were found.")
    if validation_fraction <= 0.0:
        metadata = {"mode": "none", "train_trace_count": int(valid_indices.size), "val_trace_count": 0, "gap_trace_count": 0}
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


class DepthSeismicTraceDataset(Dataset):
    """Per-trace depth-domain dataset for GINN."""

    def __init__(
        self,
        seismic_flat: np.ndarray,
        ai_lfm_flat: np.ndarray,
        vp_flat: np.ndarray,
        mask_flat: np.ndarray,
        loss_mask_flat: np.ndarray,
        taper_flat: np.ndarray,
        selected_indices: np.ndarray,
        *,
        normalization_stats: tuple[float, float] | None = None,
    ) -> None:
        n_traces, n_sample = seismic_flat.shape
        assert ai_lfm_flat.shape == seismic_flat.shape
        assert vp_flat.shape == seismic_flat.shape
        assert mask_flat.shape == seismic_flat.shape
        assert loss_mask_flat.shape == seismic_flat.shape
        assert taper_flat.shape == seismic_flat.shape

        selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
        if selected_indices.size == 0:
            raise ValueError("selected_indices must contain at least one trace.")

        self._seismic_flat = seismic_flat
        self._ai_lfm_flat = ai_lfm_flat
        self._vp_flat = vp_flat
        self._mask_flat = mask_flat
        self._loss_mask_flat = loss_mask_flat
        self._taper_flat = taper_flat
        self._valid_indices = selected_indices

        if normalization_stats is None:
            selected_mask = self._mask_flat[self._valid_indices]
            selected_seismic = self._seismic_flat[self._valid_indices]
            selected_lfm = self._ai_lfm_flat[self._valid_indices]
            valid_seis = selected_seismic[selected_mask]
            self._seis_rms = float(np.sqrt(np.mean(valid_seis**2))) + 1e-10
            valid_lfm = selected_lfm[selected_mask]
            self._lfm_scale = float(np.abs(valid_lfm).max()) + 1e-10
        else:
            self._seis_rms = float(normalization_stats[0])
            self._lfm_scale = float(normalization_stats[1])

        logger.info(
            "Depth dataset: %d selected traces / %d total, seis_rms=%.4f, ai_lfm_scale=%.2f",
            len(self._valid_indices),
            n_traces,
            self._seis_rms,
            self._lfm_scale,
        )

    def __len__(self) -> int:
        return len(self._valid_indices)

    @property
    def seis_rms(self) -> float:
        return self._seis_rms

    @property
    def lfm_scale(self) -> float:
        return self._lfm_scale

    @property
    def valid_indices(self) -> np.ndarray:
        return self._valid_indices

    @property
    def ai_lfm_flat(self) -> np.ndarray:
        return self._ai_lfm_flat

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        flat_idx = self._valid_indices[idx]
        seis = self._seismic_flat[flat_idx].copy()
        ai_lfm = self._ai_lfm_flat[flat_idx].copy()
        vp = self._vp_flat[flat_idx].copy()
        core_mask = self._mask_flat[flat_idx].copy()
        loss_mask = self._loss_mask_flat[flat_idx].copy()
        taper_weight = self._taper_flat[flat_idx].copy()

        ai_lfm_raw = ai_lfm.copy()
        velocity_raw = vp.copy()
        seis_norm = seis / self._seis_rms
        ai_lfm_norm = ai_lfm / self._lfm_scale
        x = np.stack([seis_norm, ai_lfm_norm, core_mask.astype(np.float32)], axis=0)

        return {
            "input": torch.from_numpy(x).float(),
            "obs": torch.from_numpy(seis_norm[np.newaxis]).float(),
            "mask": torch.from_numpy(core_mask[np.newaxis]).bool(),
            "loss_mask": torch.from_numpy(loss_mask[np.newaxis]).bool(),
            "taper_weight": torch.from_numpy(taper_weight[np.newaxis]).float(),
            "lfm_raw": torch.from_numpy(ai_lfm_raw[np.newaxis]).float(),
            "velocity_raw": torch.from_numpy(velocity_raw[np.newaxis]).float(),
        }


def build_dataset(cfg: DepthGINNConfig) -> DatasetBundle:
    """Construct depth-domain datasets from seismic, AI LFM, Vp LFM, and wavelet CSV."""
    logger.info("Loading depth-domain seismic volume...")
    seismic = import_seismic(
        cfg.seismic_file,
        seismic_type="segy",
        iline=cfg.segy_iline,
        xline=cfg.segy_xline,
        istep=cfg.segy_istep,
        xstep=cfg.segy_xstep,
    )
    logger.info("Loading AI/Vp depth LFMs...")
    ai_lfm = load_lfm_depth_npz(cfg.ai_lfm_file)
    vp_lfm = load_lfm_depth_npz(cfg.vp_lfm_file)

    if ai_lfm.shape != vp_lfm.shape:
        raise ValueError(f"AI/Vp LFM shape mismatch: ai={ai_lfm.shape}, vp={vp_lfm.shape}")
    if seismic.shape != ai_lfm.shape:
        raise ValueError(f"Seismic/LFM shape mismatch: seismic={seismic.shape}, ai_lfm={ai_lfm.shape}")
    if not np.allclose(ai_lfm.ilines, vp_lfm.ilines) or not np.allclose(ai_lfm.xlines, vp_lfm.xlines) or not np.allclose(ai_lfm.samples, vp_lfm.samples):
        raise ValueError("AI/Vp LFM axes do not match.")

    geometry = dict(ai_lfm.geometry) if ai_lfm.geometry else geometry_from_axes(ai_lfm.ilines, ai_lfm.xlines, ai_lfm.samples)
    geometry.setdefault("sample_domain", "depth")
    geometry.setdefault("sample_unit", "m")

    logger.info("Loading raw top/bottom depth horizons...")
    top_df_raw = import_interpretation_petrel(cfg.top_horizon_file)
    bot_df_raw = import_interpretation_petrel(cfg.bot_horizon_file)
    logger.info("Building target layer from raw depth interpretations...")
    target_layer = TargetLayer(
        raw_horizon_dfs={"top": top_df_raw, "bottom": bot_df_raw},
        geometry=geometry,
        horizon_names=["top", "bottom"],
        min_thickness=cfg.target_layer_min_thickness,
        nearest_distance_limit=cfg.target_layer_nearest_distance_limit,
        outlier_threshold=cfg.target_layer_outlier_threshold,
        outlier_min_neighbor_count=cfg.target_layer_outlier_min_neighbor_count,
    )
    train_mask = target_layer.to_mask(use_valid_control_mask=True)
    inference_mask = target_layer.to_mask(use_valid_control_mask=False)
    if train_mask.shape != seismic.shape:
        raise ValueError(f"Training mask shape {train_mask.shape} does not match seismic shape {seismic.shape}.")
    if inference_mask.shape != seismic.shape:
        raise ValueError(f"Inference mask shape {inference_mask.shape} does not match seismic shape {seismic.shape}.")

    n_il, n_xl, n_sample = seismic.shape
    seismic_flat = seismic.reshape(n_il * n_xl, n_sample)
    ai_lfm_flat = ai_lfm.volume.reshape(n_il * n_xl, n_sample)
    vp_flat = vp_lfm.volume.reshape(n_il * n_xl, n_sample)
    train_mask_flat = train_mask.reshape(n_il * n_xl, n_sample)
    inference_mask_flat = inference_mask.reshape(n_il * n_xl, n_sample)
    train_loss_mask_flat = _build_eroded_loss_mask(train_mask_flat, erosion_samples=cfg.boundary_effect_samples)
    inference_loss_mask_flat = _build_eroded_loss_mask(inference_mask_flat, erosion_samples=cfg.boundary_effect_samples)
    train_taper_flat = _build_residual_taper(train_mask_flat, halo_samples=cfg.boundary_effect_samples)
    inference_taper_flat = _build_residual_taper(inference_mask_flat, halo_samples=cfg.boundary_effect_samples)
    train_valid_indices = _get_valid_trace_indices(train_mask_flat)
    inference_valid_indices = _get_valid_trace_indices(inference_mask_flat)
    logger.info(
        "Preprocessed depth masks: train=%d traces, inference=%d traces, boundary_effect_samples=%d",
        train_valid_indices.size,
        inference_valid_indices.size,
        cfg.boundary_effect_samples,
    )

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

    train_shared = (seismic_flat, ai_lfm_flat, vp_flat, train_mask_flat, train_loss_mask_flat, train_taper_flat)
    inference_shared = (
        seismic_flat,
        ai_lfm_flat,
        vp_flat,
        inference_mask_flat,
        inference_loss_mask_flat,
        inference_taper_flat,
    )
    train_dataset = DepthSeismicTraceDataset(*train_shared, train_indices)
    train_norm_stats = (train_dataset.seis_rms, train_dataset.lfm_scale)
    val_dataset = None
    if val_indices is not None and val_indices.size > 0:
        val_dataset = DepthSeismicTraceDataset(*train_shared, val_indices, normalization_stats=train_norm_stats)
    inference_dataset = DepthSeismicTraceDataset(*inference_shared, inference_valid_indices, normalization_stats=train_norm_stats)

    logger.info("Loading depth-domain wavelet CSV...")
    wavelet_time_s, wavelet_amp = load_wavelet_csv(cfg.wavelet_file)
    resolved_wavelet_gain = cfg.wavelet_gain
    if resolved_wavelet_gain is None:
        resolved_wavelet_gain = estimate_wavelet_gain_depth(
            seismic,
            ai_lfm.volume,
            vp_lfm.volume,
            train_mask,
            wavelet_time_s,
            wavelet_amp,
            depth_axis_m=ai_lfm.samples,
            seis_rms=train_dataset.seis_rms,
            max_traces=cfg.wavelet_gain_num_traces,
            candidate_trace_indices=train_dataset.valid_indices,
            amplitude_threshold=cfg.wavelet_amplitude_threshold,
        )
        cfg.wavelet_gain = resolved_wavelet_gain
    wavelet_amp = (wavelet_amp * float(resolved_wavelet_gain)).astype(np.float32)
    wavelet_time_s = wavelet_time_s.astype(np.float32)

    return DatasetBundle(
        train_dataset=train_dataset,
        inference_dataset=inference_dataset,
        val_dataset=val_dataset,
        wavelet_time_s=wavelet_time_s,
        wavelet_amp=wavelet_amp,
        depth_axis_m=ai_lfm.samples.astype(np.float32),
        geometry=geometry,
        split_metadata=split_metadata,
    )
