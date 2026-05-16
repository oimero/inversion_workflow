"""Depth-domain data loading and dataset construction for GINN."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from cup.petrel.load import import_interpretation_petrel, import_seismic
from cup.seismic.spatial import build_trace_xy_grids, nominal_bin_spacing_m, xy_circle_mask
from cup.seismic.survey import open_survey
from cup.seismic.target_layer import TargetLayer
from cup.utils.io import resolve_repo_metadata_path
from cup.well.wavelet import (
    DEFAULT_ACTIVE_SUPPORT_THRESHOLD,
    compute_wavelet_active_half_support_s,
    load_wavelet_csv,
    make_wavelet,
)
from ginn.anchor import load_log_ai_anchor_npz, validate_log_ai_anchor
from ginn.data import compute_dynamic_gain_median, normalize_dynamic_gain_input
from ginn.masking import build_eroded_loss_mask as _build_eroded_loss_mask
from ginn.masking import build_residual_taper as _build_residual_taper
from ginn.masking import get_valid_trace_indices as _get_valid_trace_indices
from ginn.masking import select_spatial_validation_split as _select_spatial_validation_split
from ginn_depth.config import DepthGINNConfig
from ginn_depth.physics import DepthForwardModel

logger = logging.getLogger(__name__)

BOUNDARY_EFFECT_WAVELET_THRESHOLD = DEFAULT_ACTIVE_SUPPORT_THRESHOLD
BOUNDARY_EFFECT_VELOCITY_PERCENTILE = 25.0


def _find_repo_root(start: str | Path) -> Path:
    path = Path(start).resolve()
    for candidate in [path if path.is_dir() else path.parent, *(path if path.is_dir() else path.parent).parents]:
        if (candidate / "src").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find repository root from {path}.")


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
    lfm_metadata: Dict[str, Any]
    well_control_summary: Dict[str, Any]
    x_grid: np.ndarray
    y_grid: np.ndarray


@dataclass(frozen=True)
class WellControlData:
    """Sparse per-trace well-control targets for in-batch depth GINN supervision."""

    flat_indices: np.ndarray
    target_log_ai: np.ndarray
    anchor_weight: np.ndarray
    well_influence: np.ndarray
    waveform_weight_scale: np.ndarray
    summary: Dict[str, Any]

    @classmethod
    def empty(cls, n_sample: int, *, summary: Dict[str, Any] | None = None) -> "WellControlData":
        return cls(
            flat_indices=np.empty(0, dtype=np.int64),
            target_log_ai=np.zeros((0, int(n_sample)), dtype=np.float32),
            anchor_weight=np.zeros((0, int(n_sample)), dtype=np.float32),
            well_influence=np.zeros((0,), dtype=np.float32),
            waveform_weight_scale=np.ones((0,), dtype=np.float32),
            summary={} if summary is None else dict(summary),
        )


def resolve_wavelet_from_config(cfg: DepthGINNConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate the unit-gain wavelet configured for depth-domain GINN."""
    logger.info("Resolving depth-domain wavelet from source=%s...", cfg.wavelet_source)
    if cfg.wavelet_source == "precomputed_wavelet":
        wavelet_time_s, wavelet_amp = load_wavelet_csv(cfg.wavelet_file)  # type: ignore[arg-type]
    elif cfg.wavelet_source == "ricker_wavelet":
        wavelet_time_s, wavelet_amp = make_wavelet(
            wavelet_type=cfg.wavelet_type,
            freq=cfg.wavelet_freq,
            dt=cfg.wavelet_dt,
            length=cfg.wavelet_length,
            gain=1.0,
        )
    else:
        raise ValueError(f"Unsupported wavelet_source: {cfg.wavelet_source}")

    return np.asarray(wavelet_time_s, dtype=np.float64), np.asarray(wavelet_amp, dtype=np.float32)


def compute_boundary_effect_samples_from_depth_wavelet(
    wavelet_time_s: np.ndarray,
    wavelet_amp: np.ndarray,
    vp_lfm: np.ndarray,
    train_mask: np.ndarray,
    depth_axis_m: np.ndarray,
    *,
    active_threshold: float = BOUNDARY_EFFECT_WAVELET_THRESHOLD,
    velocity_percentile: float = BOUNDARY_EFFECT_VELOCITY_PERCENTILE,
) -> int:
    """Estimate boundary-effect width from wavelet support and target-layer P25 velocity."""
    if not 0.0 <= velocity_percentile <= 100.0:
        raise ValueError(f"velocity_percentile must be within [0, 100], got {velocity_percentile}.")
    if vp_lfm.shape != train_mask.shape:
        raise ValueError(f"vp_lfm shape {vp_lfm.shape} does not match train_mask shape {train_mask.shape}.")

    half_support_s = compute_wavelet_active_half_support_s(
        wavelet_time_s,
        wavelet_amp,
        active_threshold=active_threshold,
    )
    dz_m = _axis_step(np.asarray(depth_axis_m, dtype=np.float64), "depth_axis_m")
    masked_velocity = np.asarray(vp_lfm, dtype=np.float64)[np.asarray(train_mask, dtype=bool)]
    valid_velocity = masked_velocity[np.isfinite(masked_velocity) & (masked_velocity > 0.0)]
    if valid_velocity.size == 0:
        raise ValueError("Cannot auto-compute boundary_effect_samples because train_mask contains no valid Vp values.")

    velocity_mps = float(np.percentile(valid_velocity, velocity_percentile))
    return int(math.ceil(half_support_s * velocity_mps / (2.0 * dz_m)))


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
        return tuple(int(v) for v in self.volume.shape)  # type: ignore

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
    text = str(value)
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def load_lfm_depth_npz(path: str | Path, *, volume_key: str = "volume") -> DepthLfmVolume:
    """Load an ``lfm_precomputed_depth.py`` result saved as ``.npz``."""
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
            f"Volume shape does not match axes: volume={volume.shape}, axes={(ilines.size, xlines.size, samples.size)}"
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


def load_dynamic_gain_depth_model(path: str | Path) -> DepthLfmVolume:
    """Load a precomputed positive depth-domain dynamic gain model."""
    gain_model = load_lfm_depth_npz(path, volume_key="volume")
    volume = np.asarray(gain_model.volume, dtype=np.float32)
    if np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("Dynamic gain model must be finite and positive everywhere.")
    return DepthLfmVolume(
        volume=volume,
        ilines=gain_model.ilines,
        xlines=gain_model.xlines,
        samples=gain_model.samples,
        geometry=gain_model.geometry,
        metadata=gain_model.metadata,
        variance_volume=gain_model.variance_volume,
    )


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


def build_well_control_data(
    *,
    anchor_file: Path | None,
    selected_indices: np.ndarray,
    sample_domain: str,
    n_sample: int,
    n_traces: int,
    geometry: Dict[str, Any],
    lambda_log_ai_anchor: float,
    radius_xy_m: float,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    well_waveform_min_weight: float,
    distance_decay: str,
) -> WellControlData:
    """Build a sparse map of well-controlled traces for in-batch anchor loss."""
    selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    radius_xy_m = float(radius_xy_m)
    if lambda_log_ai_anchor <= 0.0 or anchor_file is None or selected_indices.size == 0 or radius_xy_m < 0.0:
        return WellControlData.empty(
            n_sample,
            summary={
                "enabled": False,
                "anchor_file": anchor_file,
                "reason": "disabled_or_empty",
            },
        )

    if distance_decay not in {"gaussian", "linear"}:
        raise ValueError(f"Unsupported well anchor distance decay: {distance_decay!r}.")

    anchor_bundle = load_log_ai_anchor_npz(anchor_file)
    validate_log_ai_anchor(
        anchor_bundle,
        sample_domain=sample_domain,
        n_sample=n_sample,
        n_traces=n_traces,
    )

    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)
    selected_set = {int(flat_idx) for flat_idx in selected_indices}
    n_il = int(geometry["n_il"])
    n_xl = int(geometry["n_xl"])
    if x_grid.shape != (n_il, n_xl) or y_grid.shape != (n_il, n_xl):
        raise ValueError(
            f"x_grid/y_grid shape must be {(n_il, n_xl)}, got {x_grid.shape} and {y_grid.shape}."
        )
    bin_spacing_m = nominal_bin_spacing_m(x_grid, y_grid)
    sigma = max(radius_xy_m / 2.0, 0.5 * bin_spacing_m)
    gaussian_edge = math.exp(-(radius_xy_m**2) / (2.0 * sigma**2)) if radius_xy_m > 0.0 else 0.0

    best_by_flat: dict[int, tuple[float, int]] = {}
    for row, flat_idx in enumerate(np.asarray(anchor_bundle.flat_indices, dtype=np.int64)):
        target = np.asarray(anchor_bundle.target_log_ai[row], dtype=np.float32)
        target_mask = np.asarray(anchor_bundle.anchor_mask[row], dtype=bool) & np.isfinite(target)
        if not np.any(target_mask):
            continue

        flat_ref = int(flat_idx)
        il_ref = flat_ref // n_xl
        xl_ref = flat_ref % n_xl
        if not (0 <= il_ref < n_il and 0 <= xl_ref < n_xl):
            continue
        circle_mask, distance_m = xy_circle_mask(
            x_grid,
            y_grid,
            center_x=float(x_grid[il_ref, xl_ref]),
            center_y=float(y_grid[il_ref, xl_ref]),
            radius_xy_m=radius_xy_m,
        )
        for il, xl in np.argwhere(circle_mask):
            il_int = int(il)
            xl_int = int(xl)
            flat = il_int * n_xl + xl_int
            if flat not in selected_set:
                continue
            dist = float(distance_m[il_int, xl_int])
            if dist < 1e-8:
                influence = 1.0
            elif distance_decay == "linear":
                influence = max(0.0, 1.0 - dist / max(radius_xy_m, 1e-6))
            else:
                raw_influence = math.exp(-(dist**2) / (2.0 * sigma**2))
                influence = (raw_influence - gaussian_edge) / max(1.0 - gaussian_edge, 1e-6)
                influence = max(0.0, min(1.0, influence))
            if influence <= 0.0:
                continue
            current = best_by_flat.get(flat)
            if current is None or influence > current[0]:
                best_by_flat[flat] = (float(influence), int(row))

    if not best_by_flat:
        return WellControlData.empty(
            n_sample,
            summary={
                "enabled": False,
                "anchor_file": anchor_file,
                "reason": "no_anchor_trace_in_selected_indices",
                "n_input_anchors": int(anchor_bundle.n_anchors),
            },
        )

    flat_indices = np.array(sorted(best_by_flat), dtype=np.int64)
    influences = np.array([best_by_flat[int(flat)][0] for flat in flat_indices], dtype=np.float32)
    rows = np.array([best_by_flat[int(flat)][1] for flat in flat_indices], dtype=np.int64)
    target_log_ai = np.asarray(anchor_bundle.target_log_ai[rows], dtype=np.float32)
    valid = np.asarray(anchor_bundle.anchor_mask[rows], dtype=bool) & np.isfinite(target_log_ai)
    anchor_weight = np.asarray(anchor_bundle.anchor_weight[rows], dtype=np.float32)
    anchor_weight = np.where(np.isfinite(anchor_weight) & (anchor_weight > 0.0), anchor_weight, 0.0)
    anchor_weight = (anchor_weight * valid.astype(np.float32)).astype(np.float32)
    waveform_weight_scale = (1.0 - (1.0 - float(well_waveform_min_weight)) * influences).astype(np.float32)

    anchor_names = np.asarray(anchor_bundle.anchor_names).astype(str)
    unique_rows, counts = np.unique(rows, return_counts=True)
    summary = {
        "enabled": True,
        "anchor_file": Path(anchor_file),
        "n_input_anchors": int(anchor_bundle.n_anchors),
        "n_controlled_traces": int(flat_indices.size),
        "radius_xy_m": radius_xy_m,
        "nominal_bin_spacing_m": float(bin_spacing_m),
        "distance_domain": "xy_m",
        "distance_decay": distance_decay,
        "well_waveform_min_weight": float(well_waveform_min_weight),
        "influence_min": float(np.min(influences)),
        "influence_mean": float(np.mean(influences)),
        "influence_max": float(np.max(influences)),
        "waveform_weight_min": float(np.min(waveform_weight_scale)),
        "waveform_weight_mean": float(np.mean(waveform_weight_scale)),
        "waveform_weight_max": float(np.max(waveform_weight_scale)),
        "controlled_traces_by_anchor": {
            str(anchor_names[int(row)]): int(count) for row, count in zip(unique_rows, counts)
        },
    }
    logger.info(
        "Depth well-control map: controlled_traces=%d, radius_xy_m=%.3f, influence=[%.3f, %.3f], waveform_weight=[%.3f, %.3f]",
        int(flat_indices.size),
        radius_xy_m,
        float(np.min(influences)),
        float(np.max(influences)),
        float(np.min(waveform_weight_scale)),
        float(np.max(waveform_weight_scale)),
    )
    return WellControlData(
        flat_indices=flat_indices,
        target_log_ai=target_log_ai,
        anchor_weight=anchor_weight,
        well_influence=influences,
        waveform_weight_scale=waveform_weight_scale,
        summary=summary,
    )


def estimate_fixed_gain_depth(
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
        raise ValueError("Cannot auto-estimate fixed gain because no valid depth traces were found in the mask.")

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
        raise ValueError("Cannot auto-estimate fixed gain because sampled valid point count is zero.")

    syn_rms = math.sqrt(syn_sq_sum / n_valid)
    obs_norm_rms = math.sqrt(obs_norm_sq_sum / n_valid)
    if syn_rms <= 0.0:
        raise ValueError(f"Cannot auto-estimate fixed gain because synthetic RMS is non-positive: {syn_rms}.")

    gain = obs_norm_rms / syn_rms
    logger.info(
        "Auto depth fixed gain: traces=%d, valid_points=%d, obs_norm_rms=%.4f, syn_unit_rms=%.4f, gain=%.4f",
        selected.size,
        n_valid,
        obs_norm_rms,
        syn_rms,
        gain,
    )
    return float(gain)


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
        dynamic_gain_flat: np.ndarray | None = None,
        include_lfm_input: bool = True,
        include_mask_input: bool = True,
        include_dynamic_gain_input: bool = False,
        normalization_stats: tuple[float, float] | None = None,
        dynamic_gain_median: float | None = None,
        well_control_data: WellControlData | None = None,
    ) -> None:
        n_traces, n_sample = seismic_flat.shape
        assert ai_lfm_flat.shape == seismic_flat.shape
        assert vp_flat.shape == seismic_flat.shape
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
        self._ai_lfm_flat = ai_lfm_flat
        self._vp_flat = vp_flat
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
            selected_lfm = self._ai_lfm_flat[self._valid_indices]
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
            "Depth dataset: %d selected traces / %d total, seis_rms=%.4f, ai_lfm_scale=%.2f, input_channels=%s",
            len(self._valid_indices),
            n_traces,
            self._seis_rms,
            self._lfm_scale,
            self.input_channel_names,
        )
        if self._anchor_dataset_indices.size:
            logger.info(
                "Depth dataset well-control samples: %d anchor-influenced / %d ordinary",
                int(self._anchor_dataset_indices.size),
                int(self._ordinary_dataset_indices.size),
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
            channels.append("ai_lfm")
        if self._include_mask_input:
            channels.append("mask")
        if self._include_dynamic_gain_input:
            channels.append("dynamic_gain_log_ratio")
        return tuple(channels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        flat_idx = self._valid_indices[idx]
        seis = self._seismic_flat[flat_idx]
        ai_lfm = self._ai_lfm_flat[flat_idx]
        vp = self._vp_flat[flat_idx]
        core_mask = self._mask_flat[flat_idx]
        loss_mask = self._loss_mask_flat[flat_idx]
        taper_weight = self._taper_flat[flat_idx]
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

        ai_lfm_raw = ai_lfm.copy()
        velocity_raw = vp.copy()
        seis_norm = seis / self._seis_rms
        ai_lfm_norm = ai_lfm / self._lfm_scale
        channels = [seis_norm.astype(np.float32)]
        if self._include_lfm_input:
            channels.append(ai_lfm_norm.astype(np.float32))
        if self._include_mask_input:
            channels.append(core_mask.astype(np.float32))
        if self._include_dynamic_gain_input:
            if dynamic_gain is None:
                channels.append(np.zeros_like(seis_norm, dtype=np.float32))
            else:
                channels.append(normalize_dynamic_gain_input(dynamic_gain, self._dynamic_gain_median))  # type: ignore[arg-type]
        x = np.stack(channels, axis=0)

        item = {
            "input": torch.from_numpy(x).float(),
            "obs": torch.from_numpy(seis_norm[np.newaxis]).float(),
            "mask": torch.from_numpy(core_mask[np.newaxis]).bool(),
            "loss_mask": torch.from_numpy(loss_mask[np.newaxis]).bool(),
            "taper_weight": torch.from_numpy(taper_weight[np.newaxis]).float(),
            "lfm_raw": torch.from_numpy(ai_lfm_raw[np.newaxis]).float(),
            "velocity_raw": torch.from_numpy(velocity_raw[np.newaxis]).float(),
            "flat_index": torch.tensor(int(flat_idx), dtype=torch.long),
            "anchor_target_log_ai": torch.from_numpy(anchor_target_log_ai[np.newaxis]).float(),
            "anchor_weight": torch.from_numpy(anchor_weight[np.newaxis]).float(),
            "well_influence": torch.tensor([float(well_influence)], dtype=torch.float32),
            "waveform_weight_scale": torch.tensor([float(waveform_weight_scale)], dtype=torch.float32),
            "has_anchor": torch.tensor([float(has_anchor)], dtype=torch.float32),
        }
        if dynamic_gain is not None:
            item["dynamic_gain"] = torch.from_numpy(dynamic_gain[np.newaxis]).float()
        return item

    def replace_base_ai(self, ai_lfm_flat: np.ndarray) -> None:
        """Replace the base AI volume while keeping mask/taper/well-control metadata.

        Parameters
        ----------
        ai_lfm_flat : np.ndarray
            New base AI flat array, shape ``(n_traces, n_sample)``.
        """
        base_flat = np.asarray(ai_lfm_flat, dtype=np.float32)
        if base_flat.shape != self._ai_lfm_flat.shape:
            raise ValueError(
                f"Base AI flat shape {base_flat.shape} does not match dataset AI shape {self._ai_lfm_flat.shape}."
            )
        self._ai_lfm_flat = base_flat
        selected_mask = self._mask_flat[self._valid_indices]
        selected_base_ai = base_flat[self._valid_indices]
        valid_base_ai = selected_base_ai[selected_mask]
        valid_base_ai = valid_base_ai[np.isfinite(valid_base_ai)]
        if valid_base_ai.size == 0:
            raise ValueError("No finite base AI samples inside the dataset mask.")
        max_abs = float(np.abs(valid_base_ai).max())
        if max_abs <= 0.0:
            raise ValueError("Base AI maximum absolute value must be positive.")
        self._lfm_scale = max_abs + 1e-10


class MixedWellBatchSampler(Sampler[list[int]]):
    """Batch sampler that replaces a fraction of each batch with well-controlled traces."""

    def __init__(
        self,
        dataset: DepthSeismicTraceDataset,
        *,
        batch_size: int,
        well_fraction: float,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if not 0.0 <= well_fraction <= 1.0:
            raise ValueError(f"well_fraction must be within [0, 1], got {well_fraction}.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.well_fraction = float(well_fraction)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0
        self.anchor_indices = np.asarray(dataset.anchor_dataset_indices, dtype=np.int64)
        self.ordinary_indices = np.asarray(dataset.ordinary_dataset_indices, dtype=np.int64)
        self.all_indices = np.arange(len(dataset), dtype=np.int64)

        if self.anchor_indices.size and self.well_fraction > 0.0:
            self.n_well_per_batch = int(round(self.batch_size * self.well_fraction))
            self.n_well_per_batch = min(max(self.n_well_per_batch, 1), self.batch_size)
        else:
            self.n_well_per_batch = 0
        self.n_ordinary_per_batch = self.batch_size - self.n_well_per_batch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return int(len(self.dataset) // self.batch_size)
        return int(math.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        n_batches = len(self)
        if self.n_well_per_batch <= 0 or self.anchor_indices.size == 0:
            shuffled = rng.permutation(self.all_indices)
            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                stop = min(start + self.batch_size, shuffled.size)
                batch = shuffled[start:stop]
                if batch.size == self.batch_size or not self.drop_last:
                    yield batch.astype(np.int64).tolist()
            return

        ordinary_pool = self.ordinary_indices if self.ordinary_indices.size else self.all_indices
        ordinary_needed = max(self.n_ordinary_per_batch, 0) * n_batches
        if ordinary_needed:
            repeats = int(math.ceil(ordinary_needed / max(ordinary_pool.size, 1)))
            ordinary_draws = np.concatenate([rng.permutation(ordinary_pool) for _ in range(repeats)])
        else:
            ordinary_draws = np.empty(0, dtype=np.int64)

        ordinary_cursor = 0
        for _ in range(n_batches):
            parts: list[np.ndarray] = []
            if self.n_ordinary_per_batch > 0:
                ordinary = ordinary_draws[ordinary_cursor : ordinary_cursor + self.n_ordinary_per_batch]
                ordinary_cursor += self.n_ordinary_per_batch
                parts.append(ordinary)
            well = rng.choice(self.anchor_indices, size=self.n_well_per_batch, replace=True)
            parts.append(np.asarray(well, dtype=np.int64))
            batch = np.concatenate(parts)
            rng.shuffle(batch)
            yield batch.astype(np.int64).tolist()


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
    logger.info("Loading AI/Vp depth LFMs...")
    ai_lfm = load_lfm_depth_npz(cfg.ai_lfm_file)
    vp_lfm = load_lfm_depth_npz(cfg.vp_lfm_file)

    if ai_lfm.shape != vp_lfm.shape:
        raise ValueError(f"AI/Vp LFM shape mismatch: ai={ai_lfm.shape}, vp={vp_lfm.shape}")
    if seismic.shape != ai_lfm.shape:
        raise ValueError(f"Seismic/LFM shape mismatch: seismic={seismic.shape}, ai_lfm={ai_lfm.shape}")
    if (
        not np.allclose(ai_lfm.ilines, vp_lfm.ilines)
        or not np.allclose(ai_lfm.xlines, vp_lfm.xlines)
        or not np.allclose(ai_lfm.samples, vp_lfm.samples)
    ):
        raise ValueError("AI/Vp LFM axes do not match.")

    dynamic_gain = None
    if cfg.gain_source == "dynamic_gain_model":
        logger.info("Loading depth dynamic gain model from %s...", cfg.dynamic_gain_model)
        dynamic_gain = load_dynamic_gain_depth_model(cfg.dynamic_gain_model)  # type: ignore
        if dynamic_gain.shape != ai_lfm.shape:
            raise ValueError(f"Dynamic gain shape {dynamic_gain.shape} does not match LFM shape {ai_lfm.shape}.")
        if (
            not np.allclose(ai_lfm.ilines, dynamic_gain.ilines)
            or not np.allclose(ai_lfm.xlines, dynamic_gain.xlines)
            or not np.allclose(ai_lfm.samples, dynamic_gain.samples)
        ):
            raise ValueError("Dynamic gain axes do not match AI/Vp LFM axes.")

    geometry = (
        dict(ai_lfm.geometry) if ai_lfm.geometry else geometry_from_axes(ai_lfm.ilines, ai_lfm.xlines, ai_lfm.samples)
    )
    geometry.setdefault("sample_domain", "depth")
    geometry.setdefault("sample_unit", "m")
    x_grid, y_grid = build_trace_xy_grids(seismic_ctx, ai_lfm.ilines, ai_lfm.xlines)

    # ── read horizon files + target-layer QC from AI LFM NPZ metadata ──
    hz_list = ai_lfm.metadata.get("horizons", [])
    if len(hz_list) < 2:
        raise ValueError("AI LFM NPZ metadata must contain at least two sorted horizons.")
    # Top = shallowest (index 0), bottom = deepest (index -1)
    repo_root = _find_repo_root(cfg.ai_lfm_file)
    top_horizon_file = resolve_repo_metadata_path(hz_list[0]["file"], root=repo_root)
    bot_horizon_file = resolve_repo_metadata_path(hz_list[-1]["file"], root=repo_root)
    logger.info("Top horizon: %s", top_horizon_file)
    logger.info("Bot horizon: %s", bot_horizon_file)

    tl_meta = ai_lfm.metadata.get("target_layer", {})
    tl_min_thickness = tl_meta.get("min_thickness")
    tl_nearest_limit = tl_meta.get("nearest_distance_limit")
    tl_outlier_threshold = tl_meta.get("outlier_threshold")
    tl_outlier_min_neighbor = tl_meta.get("outlier_min_neighbor_count", 2)
    logger.info(
        "Target-layer params from LFM NPZ metadata: min_thickness=%s, outlier_threshold=%s",
        tl_min_thickness,
        tl_outlier_threshold,
    )

    logger.info("Loading raw top/bottom depth horizons...")
    top_df_raw = import_interpretation_petrel(top_horizon_file)
    bot_df_raw = import_interpretation_petrel(bot_horizon_file)

    logger.info("Building target layer from raw depth interpretations...")
    target_layer = TargetLayer(
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

    wavelet_time_s, wavelet_amp = resolve_wavelet_from_config(cfg)
    boundary_effect_samples = cfg.boundary_effect_samples
    if boundary_effect_samples is None:
        boundary_effect_samples = compute_boundary_effect_samples_from_depth_wavelet(
            wavelet_time_s,
            wavelet_amp,
            vp_lfm.volume,
            train_mask,
            ai_lfm.samples,
        )
        cfg.boundary_effect_samples = boundary_effect_samples
        logger.info(
            "Auto depth boundary_effect_samples=%d from wavelet active half-support "
            "(threshold=%.2f) and train-mask Vp P%.0f",
            boundary_effect_samples,
            BOUNDARY_EFFECT_WAVELET_THRESHOLD,
            BOUNDARY_EFFECT_VELOCITY_PERCENTILE,
        )
    else:
        boundary_effect_samples = int(boundary_effect_samples)
        cfg.boundary_effect_samples = boundary_effect_samples

    n_il, n_xl, n_sample = seismic.shape
    seismic_flat = seismic.reshape(n_il * n_xl, n_sample)
    ai_lfm_flat = ai_lfm.volume.reshape(n_il * n_xl, n_sample)
    vp_flat = vp_lfm.volume.reshape(n_il * n_xl, n_sample)
    dynamic_gain_flat = dynamic_gain.volume.reshape(n_il * n_xl, n_sample) if dynamic_gain is not None else None
    train_mask_flat = train_mask.reshape(n_il * n_xl, n_sample)
    inference_mask_flat = inference_mask.reshape(n_il * n_xl, n_sample)
    train_loss_mask_flat = _build_eroded_loss_mask(train_mask_flat, erosion_samples=boundary_effect_samples)
    inference_loss_mask_flat = _build_eroded_loss_mask(inference_mask_flat, erosion_samples=boundary_effect_samples)
    train_taper_flat = _build_residual_taper(train_mask_flat, halo_samples=boundary_effect_samples)
    inference_taper_flat = _build_residual_taper(inference_mask_flat, halo_samples=boundary_effect_samples)
    train_valid_indices = _get_valid_trace_indices(train_mask_flat)
    inference_valid_indices = _get_valid_trace_indices(inference_mask_flat)
    logger.info(
        "Preprocessed depth masks: train=%d traces, inference=%d traces, boundary_effect_samples=%d",
        train_valid_indices.size,
        inference_valid_indices.size,
        boundary_effect_samples,
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

    if cfg.well_control_enabled:
        train_well_control = build_well_control_data(
            anchor_file=cfg.log_ai_anchor_file,
            selected_indices=train_indices,
            sample_domain="depth",
            n_sample=n_sample,
            n_traces=n_il * n_xl,
            geometry=geometry,
            lambda_log_ai_anchor=cfg.lambda_log_ai_anchor,
            radius_xy_m=cfg.log_ai_anchor_radius_xy_m,
            x_grid=x_grid,
            y_grid=y_grid,
            well_waveform_min_weight=cfg.well_waveform_min_weight,
            distance_decay=cfg.well_anchor_distance_decay,
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
    split_metadata["well_control"] = train_well_control.summary

    train_shared = (seismic_flat, ai_lfm_flat, vp_flat, train_mask_flat, train_loss_mask_flat, train_taper_flat)
    inference_shared = (
        seismic_flat,
        ai_lfm_flat,
        vp_flat,
        inference_mask_flat,
        inference_loss_mask_flat,
        inference_taper_flat,
    )
    train_dataset = DepthSeismicTraceDataset(
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
        val_dataset = DepthSeismicTraceDataset(
            *train_shared,
            val_indices,
            dynamic_gain_flat=dynamic_gain_flat,
            include_lfm_input=cfg.include_lfm_input,
            include_mask_input=cfg.include_mask_input,
            include_dynamic_gain_input=cfg.include_dynamic_gain_input,
            normalization_stats=train_norm_stats,
            dynamic_gain_median=train_dataset.dynamic_gain_median,
            well_control_data=WellControlData.empty(n_sample),
        )
    inference_dataset = DepthSeismicTraceDataset(
        *inference_shared,
        inference_valid_indices,
        dynamic_gain_flat=dynamic_gain_flat,
        include_lfm_input=cfg.include_lfm_input,
        include_mask_input=cfg.include_mask_input,
        include_dynamic_gain_input=cfg.include_dynamic_gain_input,
        normalization_stats=train_norm_stats,
        dynamic_gain_median=train_dataset.dynamic_gain_median,
        well_control_data=WellControlData.empty(n_sample),
    )

    if cfg.gain_source == "fixed_gain":
        resolved_fixed_gain = cfg.fixed_gain
        if resolved_fixed_gain is None:
            resolved_fixed_gain = estimate_fixed_gain_depth(
                seismic,
                ai_lfm.volume,
                vp_lfm.volume,
                train_mask,
                wavelet_time_s,
                wavelet_amp,
                depth_axis_m=ai_lfm.samples,
                seis_rms=train_dataset.seis_rms,
                max_traces=cfg.fixed_gain_num_traces,
                candidate_trace_indices=train_dataset.valid_indices,
                amplitude_threshold=cfg.wavelet_amplitude_threshold,
            )
            cfg.fixed_gain = resolved_fixed_gain
    elif cfg.gain_source == "dynamic_gain_model":
        resolved_fixed_gain = 1.0
        logger.info("Using depth dynamic gain model; fixed gain is disabled.")
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
    wavelet_amp = (wavelet_amp * float(resolved_fixed_gain)).astype(np.float32)
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
        lfm_metadata=dict(ai_lfm.metadata),
        well_control_summary=train_well_control.summary,
        x_grid=x_grid,
        y_grid=y_grid,
    )
