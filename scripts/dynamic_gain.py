"""Build a time-domain dynamic gain sidecar for GINN training.

The produced gain volume maps unit-wavelet LFM synthetics to the same
normalised seismic domain used by ``scripts/ginn_train.py``:

    seismic_norm = seismic_raw / train_mask_rms

Usage::

    python scripts/dynamic_gain.py
    python scripts/dynamic_gain.py --train-config experiments/ginn/train.yaml --config experiments/common.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.petrel.load import import_interpretation_petrel, import_seismic
from cup.seismic.target_zone import TargetZone
from cup.seismic.viz import plot_well_waveform_qc
from cup.utils.config import deep_merge_dict
from cup.utils.io import (
    latest_run,
    load_yaml_config,
    repo_relative_path,
    resolve_optional_path,
    sanitize_filename,
    write_json,
)
from cup.well.gain import (
    CANDIDATE_ATTRIBUTES,
    NORMALIZATION,
    SCHEMA_VERSION,
    assign_spatial_clusters,
    build_gain_volume as build_gain_volume_from_fit,
    fit_gain_relationship as fit_gain_relationship_from_samples,
    positive_ls_gain,
    recommended_fixed_gain,
    segment_attribute_values,
    write_gain_npz as write_dynamic_gain_npz,
)
from cup.well.wavelet import load_wavelet_csv, validate_wavelet_dt
from ginn.anchor import load_log_ai_anchor_npz, validate_log_ai_anchor
from ginn.config import GINNConfig
from ginn.data import (
    _load_lowfreq_npz_contract,
    _validate_time_lfm_contract,
    compute_boundary_effect_samples_from_wavelet,
    load_lowfreq_model,
)
from ginn.masking import build_eroded_loss_mask, get_valid_trace_indices, select_spatial_validation_split
from ginn.physics import ForwardModel
from wtie.optimize.similarity import dynamic_normalized_xcorr, normalized_xcorr
from wtie.processing import grid

matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 120

logger = logging.getLogger(__name__)


DEFAULT_CONFIG: dict[str, Any] = {
    "source_runs": {
        "mode": "latest",
        "well_constraints_dir": None,
        "lfm_precomputed_dir": None,
        "wavelet_generation_dir": None,
    },
    "segments": {
        "min_segment_valid_samples": 8,
        "max_segment_count_per_trace": 20,
        "min_segments_per_well": 1,
        "gain_eps": 1e-12,
    },
    "spatial_debias": {
        "enabled": True,
        "cluster_radius_m": 600.0,
    },
    "attributes": {
        "candidate_attributes": list(CANDIDATE_ATTRIBUTES),
        "attr_tie_threshold": 0.05,
        "attribute_floor_fraction": 0.10,
        "window_s": None,
    },
    "prediction": {
        "clip_percentiles": [5.0, 95.0],
        "gain_smoothing_samples": 1,
    },
    "runtime": {
        "forward_batch_traces": 256,
        "volume_batch_traces": 512,
    },
}


@dataclass(frozen=True)
class DynamicGainContext:
    cfg: GINNConfig
    script_cfg: dict[str, Any]
    train_config_file: Path
    output_dir: Path
    figure_dir: Path
    seismic_file: Path
    seismic: np.ndarray
    survey: Any
    geometry: dict[str, Any]
    ilines: np.ndarray
    xlines: np.ndarray
    samples: np.ndarray
    lfm: np.ndarray
    lfm_file: Path
    lfm_metadata: dict[str, Any]
    train_mask: np.ndarray
    train_loss_mask: np.ndarray
    train_indices: np.ndarray
    split_metadata: dict[str, Any]
    train_mask_rms: float
    wavelet_file: Path
    wavelet_time_s: np.ndarray
    unit_wavelet: np.ndarray
    anchor_file: Path
    source_dirs: dict[str, Path | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, default=Path("experiments/ginn/train.yaml"))
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _json_scalar_to_dict(value: np.ndarray) -> dict[str, Any]:
    payload = np.asarray(value).item()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(str(payload))


def _resolve_source_dirs(script_cfg: dict[str, Any], output_root: Path) -> dict[str, Path | None]:
    source_cfg = dict(script_cfg.get("source_runs") or {})
    mode = str(source_cfg.get("mode", "latest")).strip().lower()
    if mode != "latest":
        raise ValueError(f"dynamic_gain.source_runs.mode only supports 'latest', got {mode!r}.")

    dirs: dict[str, Path | None] = {}
    for key in ("well_constraints_dir", "lfm_precomputed_dir", "wavelet_generation_dir"):
        dirs[key] = resolve_optional_path(source_cfg.get(key), root=REPO_ROOT)
    return dirs


def _repo_path_or_none(path: Path | None) -> str | None:
    if path is None:
        return None
    return repo_relative_path(path, root=REPO_ROOT)


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
    except RuntimeError:
        pass
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", path)


def _samples_from_geometry(geometry: dict[str, Any]) -> np.ndarray:
    n_sample = int(geometry["n_sample"])
    return float(geometry["sample_min"]) + np.arange(n_sample, dtype=np.float64) * float(geometry["sample_step"])


def _line_axes_from_geometry(geometry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    ilines = float(geometry["inline_min"]) + np.arange(int(geometry["n_il"]), dtype=np.float64) * float(
        geometry["inline_step"]
    )
    xlines = float(geometry["xline_min"]) + np.arange(int(geometry["n_xl"]), dtype=np.float64) * float(
        geometry["xline_step"]
    )
    return ilines, xlines


def _validate_samples_axis(samples: np.ndarray, expected: np.ndarray, *, name: str) -> None:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.shape != expected.shape:
        raise ValueError(f"{name} samples shape {values.shape} does not match expected {expected.shape}.")
    if np.any(~np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} samples must be finite and strictly increasing.")
    if not np.allclose(values, expected, rtol=0.0, atol=1e-6):
        max_diff = float(np.max(np.abs(values - expected)))
        raise ValueError(f"{name} samples do not match seismic time axis (max_abs_diff={max_diff:.6g}s).")


def _resolve_lfm_file(cfg: GINNConfig, source_dirs: dict[str, Path | None], output_root: Path) -> Path:
    candidates = []
    if cfg.ai_lfm_file is not None:
        candidates.append(Path(cfg.ai_lfm_file))
    for path in candidates:
        if path.exists():
            return path.resolve()
    lfm_dir = source_dirs.get("lfm_precomputed_dir")
    if lfm_dir is None:
        lfm_dir = latest_run(output_root, "lfm_precomputed", "ai_lfm_time.npz")
    if lfm_dir is not None:
        candidates.append(lfm_dir / "ai_lfm_time.npz")
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Cannot resolve ai_lfm_time.npz from candidates: {candidates}")


def _resolve_wavelet_file(cfg: GINNConfig, source_dirs: dict[str, Path | None], output_root: Path) -> Path:
    candidates = []
    if cfg.wavelet_source == "precomputed_wavelet" and cfg.wavelet_file is not None:
        candidates.append(Path(cfg.wavelet_file))
    wavelet_dir = source_dirs.get("wavelet_generation_dir")
    if wavelet_dir is None and not any(path.exists() for path in candidates):
        wavelet_dir = latest_run(output_root, "wavelet_generation", "selected_wavelet.csv")
    if wavelet_dir is not None:
        candidates.append(wavelet_dir / "selected_wavelet.csv")
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        "Time dynamic gain requires the fifth-step selected_wavelet.csv. "
        f"Cannot resolve it from candidates: {candidates}"
    )


def _resolve_anchor_file(cfg: GINNConfig, source_dirs: dict[str, Path | None], output_root: Path) -> Path:
    candidates = []
    if cfg.log_ai_anchor_file is not None:
        candidates.append(Path(cfg.log_ai_anchor_file))
    constraints_dir = source_dirs.get("well_constraints_dir")
    if constraints_dir is None and not any(path.exists() for path in candidates):
        constraints_dir = latest_run(output_root, "well_constraints", "log_ai_anchor_time.npz")
    if constraints_dir is not None:
        candidates.append(constraints_dir / "log_ai_anchor_time.npz")
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Cannot resolve log_ai_anchor_time.npz from candidates: {candidates}")


def _resolve_horizon_path(value: Any) -> Path:
    text = str(value)
    path = Path(text)
    if path.is_absolute():
        return path
    candidates = [REPO_ROOT / path, REPO_ROOT / "data" / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _build_train_masks(
    *,
    cfg: GINNConfig,
    geometry: dict[str, Any],
    lfm_metadata: dict[str, Any],
    seismic_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    tl_meta = lfm_metadata.get("target_layer", {})
    horizons = list(lfm_metadata.get("horizons", []))
    if len(horizons) < 2:
        raise ValueError("AI LFM metadata must contain at least two sorted horizons.")

    top_file = _resolve_horizon_path(horizons[0]["file"])
    bottom_file = _resolve_horizon_path(horizons[-1]["file"])
    target_layer = TargetZone(
        raw_horizon_dfs={
            "top": import_interpretation_petrel(top_file),
            "bottom": import_interpretation_petrel(bottom_file),
        },
        geometry=geometry,
        horizon_names=["top", "bottom"],
        min_thickness=tl_meta.get("min_thickness"),
        nearest_distance_limit=tl_meta.get("nearest_distance_limit"),
        outlier_threshold=tl_meta.get("outlier_threshold"),
        outlier_min_neighbor_count=tl_meta.get("outlier_min_neighbor_count", 2),
    )
    train_mask = target_layer.to_mask(use_valid_control_mask=True)
    if train_mask.shape != seismic_shape:
        raise ValueError(f"Training mask shape {train_mask.shape} does not match seismic shape {seismic_shape}.")

    n_il, n_xl, n_sample = seismic_shape
    mask_flat = train_mask.reshape(n_il * n_xl, n_sample)
    boundary_effect_samples = cfg.boundary_effect_samples
    if boundary_effect_samples is None:
        raise ValueError("boundary_effect_samples must be resolved before building dynamic gain masks.")
    loss_mask_flat = build_eroded_loss_mask(mask_flat, erosion_samples=int(boundary_effect_samples))
    valid_indices = get_valid_trace_indices(mask_flat)
    train_indices = valid_indices
    split_metadata: dict[str, Any] = {
        "mode": "none",
        "train_trace_count": int(train_indices.size),
        "val_trace_count": 0,
        "gap_trace_count": 0,
        "inference_trace_count": int(valid_indices.size),
    }
    if cfg.validation_split_mode == "spatial_block" and cfg.validation_fraction > 0.0:
        train_indices, val_indices, split_metadata = select_spatial_validation_split(
            valid_indices,
            n_il=n_il,
            n_xl=n_xl,
            validation_fraction=cfg.validation_fraction,
            gap_traces=cfg.validation_gap_traces,
            anchor=cfg.validation_block_anchor,
        )
        split_metadata["inference_trace_count"] = int(valid_indices.size)
        split_metadata["val_trace_count"] = int(val_indices.size)
    return train_mask, loss_mask_flat.reshape(seismic_shape), train_indices, split_metadata


def _compute_train_mask_rms(seismic: np.ndarray, train_mask: np.ndarray, train_indices: np.ndarray) -> float:
    n_sample = seismic.shape[-1]
    seismic_flat = seismic.reshape(-1, n_sample)
    mask_flat = train_mask.reshape(-1, n_sample)
    values = seismic_flat[np.asarray(train_indices, dtype=np.int64)][mask_flat[np.asarray(train_indices, dtype=np.int64)]]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot compute train_mask_rms because the selected train mask has no finite seismic samples.")
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise ValueError(f"Invalid train_mask_rms={rms}.")
    return rms


def load_context(args: argparse.Namespace) -> DynamicGainContext:
    common_cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = deep_merge_dict(DEFAULT_CONFIG, dict(common_cfg.get("dynamic_gain") or {}))
    output_root = REPO_ROOT / str(common_cfg.get("output_root", "scripts/output"))
    source_dirs = _resolve_source_dirs(script_cfg, output_root)

    cfg = GINNConfig.from_yaml(args.train_config, base_dir=REPO_ROOT)
    if args.output_dir is None:
        output_dir = output_root / f"dynamic_gain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    lfm_file = _resolve_lfm_file(cfg, source_dirs, output_root)
    wavelet_file = _resolve_wavelet_file(cfg, source_dirs, output_root)
    anchor_file = _resolve_anchor_file(cfg, source_dirs, output_root)

    logger.info("Loading seismic volume: %s", cfg.seismic_file)
    seismic_type = str(cfg.seismic_type).strip().lower()
    seismic = import_seismic(
        cfg.seismic_file,
        seismic_type=seismic_type,
        iline=cfg.segy_iline if seismic_type == "segy" else None,
        xline=cfg.segy_xline if seismic_type == "segy" else None,
        istep=cfg.segy_istep if seismic_type == "segy" else None,
        xstep=cfg.segy_xstep if seismic_type == "segy" else None,
    )
    from cup.seismic.survey import open_survey

    survey = open_survey(
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
    geometry = survey.describe_geometry(domain="time")
    if str(geometry.get("sample_domain", "")).strip().lower() != "time":
        raise ValueError(f"Expected time-domain seismic geometry, got {geometry}.")
    if str(geometry.get("sample_unit", "")).strip().lower() not in {"s", "sec", "second", "seconds"}:
        raise ValueError(f"Expected seismic sample unit seconds, got {geometry}.")
    expected_shape = (int(geometry["n_il"]), int(geometry["n_xl"]), int(geometry["n_sample"]))
    if tuple(seismic.shape) != expected_shape:
        raise ValueError(f"Seismic shape {seismic.shape} does not match geometry {expected_shape}.")

    logger.info("Loading LFM: %s", lfm_file)
    lfm = load_lowfreq_model(lfm_file)
    lfm_metadata, lfm_geometry, lfm_samples = _load_lowfreq_npz_contract(lfm_file)
    _validate_time_lfm_contract(
        lfm_path=lfm_file,
        lfm_geometry=lfm_geometry,
        lfm_samples=lfm_samples,
        seismic_geometry=geometry,
    )
    if lfm.shape != seismic.shape:
        raise ValueError(f"LFM shape {lfm.shape} does not match seismic shape {seismic.shape}.")

    wavelet_time_s, unit_wavelet = load_wavelet_csv(wavelet_file)
    validate_wavelet_dt(wavelet_time_s, float(geometry["sample_step"]))
    unit_wavelet = np.asarray(unit_wavelet, dtype=np.float32)
    if cfg.boundary_effect_samples is None:
        cfg.boundary_effect_samples = compute_boundary_effect_samples_from_wavelet(
            np.asarray(wavelet_time_s, dtype=np.float64),
            unit_wavelet,
            float(geometry["sample_step"]),
        )
        logger.info("Auto boundary_effect_samples=%d from selected wavelet.", int(cfg.boundary_effect_samples))

    samples = _samples_from_geometry(geometry)
    ilines, xlines = _line_axes_from_geometry(geometry)
    train_mask, train_loss_mask, train_indices, split_metadata = _build_train_masks(
        cfg=cfg,
        geometry=geometry,
        lfm_metadata=lfm_metadata,
        seismic_shape=tuple(seismic.shape),
    )
    train_mask_rms = _compute_train_mask_rms(seismic, train_mask, train_indices)
    logger.info("train_mask_rms=%.6g from %d train traces", train_mask_rms, int(train_indices.size))

    return DynamicGainContext(
        cfg=cfg,
        script_cfg=script_cfg,
        train_config_file=(args.train_config if args.train_config.is_absolute() else REPO_ROOT / args.train_config).resolve(),
        output_dir=output_dir,
        figure_dir=figure_dir,
        seismic_file=Path(cfg.seismic_file).resolve(),
        seismic=np.asarray(seismic, dtype=np.float32),
        survey=survey,
        geometry=geometry,
        ilines=ilines,
        xlines=xlines,
        samples=samples,
        lfm=np.asarray(lfm, dtype=np.float32),
        lfm_file=lfm_file,
        lfm_metadata=lfm_metadata,
        train_mask=train_mask,
        train_loss_mask=train_loss_mask,
        train_indices=train_indices,
        split_metadata=split_metadata,
        train_mask_rms=train_mask_rms,
        wavelet_file=wavelet_file,
        wavelet_time_s=np.asarray(wavelet_time_s, dtype=np.float64),
        unit_wavelet=unit_wavelet,
        anchor_file=anchor_file,
        source_dirs=source_dirs,
    )


def build_unit_synthetic(lfm: np.ndarray, unit_wavelet: np.ndarray, *, batch_traces: int) -> np.ndarray:
    n_sample = int(lfm.shape[-1])
    flat = lfm.reshape(-1, n_sample)
    out = np.empty_like(flat, dtype=np.float32)
    model = ForwardModel(unit_wavelet).cpu()
    with torch.no_grad():
        for start in range(0, flat.shape[0], int(batch_traces)):
            end = min(start + int(batch_traces), flat.shape[0])
            batch = torch.from_numpy(flat[start:end, np.newaxis, :]).float()
            out[start:end] = model(batch).cpu().numpy()[:, 0, :]
            if start == 0 or end == flat.shape[0] or (start // max(int(batch_traces), 1)) % 20 == 0:
                logger.info("Forward modeled %d/%d traces", end, flat.shape[0])
    return out.reshape(lfm.shape)


def split_valid_indices(
    valid_indices: np.ndarray,
    *,
    min_valid_samples: int,
    max_segments: int,
    min_segments: int,
) -> list[np.ndarray]:
    valid_indices = np.asarray(valid_indices, dtype=np.int64).reshape(-1)
    if valid_indices.size < int(min_valid_samples):
        return []
    segment_count = int(min(max_segments, valid_indices.size // int(min_valid_samples)))
    segment_count = max(int(min_segments), segment_count)
    return [seg for seg in np.array_split(valid_indices, segment_count) if seg.size >= int(min_valid_samples)]


def contiguous_true_segments(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    return [segment for segment in np.split(indices, breaks) if segment.size > 0]


def _trace_xy_from_flat(ctx: DynamicGainContext, flat_idx: int) -> tuple[float, float, float, float]:
    n_xl = int(ctx.geometry["n_xl"])
    i_il = int(flat_idx) // n_xl
    i_xl = int(flat_idx) % n_xl
    inline = float(ctx.ilines[i_il])
    xline = float(ctx.xlines[i_xl])
    x_m, y_m = ctx.survey.line_geometry.line_to_coord(inline, xline)
    return inline, xline, float(x_m), float(y_m)


def estimate_well_gain_samples(ctx: DynamicGainContext, syn_unit: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchor = load_log_ai_anchor_npz(ctx.anchor_file)
    validate_log_ai_anchor(
        anchor,
        sample_domain="time",
        n_sample=int(ctx.geometry["n_sample"]),
        n_traces=int(ctx.geometry["n_il"]) * int(ctx.geometry["n_xl"]),
    )
    _validate_samples_axis(np.asarray(anchor.samples, dtype=np.float64), ctx.samples, name="log_ai_anchor_time.npz")

    n_sample = int(ctx.geometry["n_sample"])
    seismic_norm_flat = (ctx.seismic.reshape(-1, n_sample) / float(ctx.train_mask_rms)).astype(np.float32)
    syn_flat = syn_unit.reshape(-1, n_sample)
    loss_mask_flat = ctx.train_loss_mask.reshape(-1, n_sample)
    train_set = set(int(v) for v in np.asarray(ctx.train_indices, dtype=np.int64))

    seg_cfg = dict(ctx.script_cfg["segments"])
    min_samples = int(seg_cfg["min_segment_valid_samples"])
    max_segments = int(seg_cfg["max_segment_count_per_trace"])
    min_segments_per_well = int(seg_cfg["min_segments_per_well"])
    eps = float(seg_cfg["gain_eps"])

    rows: list[dict[str, Any]] = []
    skipped_not_train = 0
    skipped_no_segments = 0
    for anchor_row, flat_idx_value in enumerate(np.asarray(anchor.flat_indices, dtype=np.int64)):
        flat_idx = int(flat_idx_value)
        if flat_idx not in train_set:
            skipped_not_train += 1
            continue
        anchor_mask = np.asarray(anchor.anchor_mask[anchor_row], dtype=bool)
        valid = (
            anchor_mask
            & loss_mask_flat[flat_idx]
            & np.isfinite(seismic_norm_flat[flat_idx])
            & np.isfinite(syn_flat[flat_idx])
        )
        trace_segments: list[np.ndarray] = []
        for contiguous in contiguous_true_segments(valid):
            trace_segments.extend(
                split_valid_indices(
                    contiguous,
                    min_valid_samples=min_samples,
                    max_segments=max_segments,
                    min_segments=1,
                )
            )
        if not trace_segments:
            skipped_no_segments += 1
            continue

        inline, xline, x_m, y_m = _trace_xy_from_flat(ctx, flat_idx)
        well_name = str(np.asarray(anchor.anchor_names).astype(str)[anchor_row])
        for segment_index, indices in enumerate(trace_segments):
            gain = positive_ls_gain(
                seismic_norm_flat[flat_idx, indices],
                syn_flat[flat_idx, indices],
                eps=eps,
                min_valid_samples=min_samples,
            )
            if not np.isfinite(gain) or gain <= 0.0:
                continue
            attrs = segment_attribute_values(seismic_norm_flat[flat_idx, indices])
            row = {
                "well_name": well_name,
                "flat_idx": flat_idx,
                "inline": inline,
                "xline": xline,
                "x_m": x_m,
                "y_m": y_m,
                "segment_index": int(segment_index),
                "sample_start": int(indices[0]),
                "sample_end": int(indices[-1] + 1),
                "twt_start_s": float(ctx.samples[indices[0]]),
                "twt_end_s": float(ctx.samples[indices[-1]]),
                "n_valid_samples": int(indices.size),
                "gain": float(gain),
                "log_gain": float(np.log(gain)),
                "seismic_norm_rms": float(attrs["seismic_rms"]),
                "syn_unit_rms": float(np.sqrt(np.mean(syn_flat[flat_idx, indices].astype(np.float64) ** 2))),
                **attrs,
            }
            rows.append(row)

    sample_df = pd.DataFrame.from_records(rows)
    if sample_df.empty:
        raise ValueError("No positive finite dynamic gain samples were estimated from log_ai_anchor_time.npz.")

    counts = sample_df.groupby("well_name")["gain"].transform("count")
    sample_df = sample_df[counts >= min_segments_per_well].copy()
    if sample_df.empty:
        raise ValueError(
            "No wells satisfy dynamic_gain.segments.min_segments_per_well="
            f"{min_segments_per_well} after segment filtering."
        )

    summary = {
        "anchor_file": repo_relative_path(ctx.anchor_file, root=REPO_ROOT),
        "anchor_trace_count": int(anchor.flat_indices.size),
        "skipped_not_in_training_split": int(skipped_not_train),
        "skipped_without_valid_segments": int(skipped_no_segments),
        "segment_count": int(len(sample_df)),
        "well_count": int(sample_df["well_name"].nunique()),
    }
    return sample_df.reset_index(drop=True), summary


def fit_gain_relationship(ctx: DynamicGainContext, sample_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Thin wrapper — unpack config from *ctx* then delegate to ``cup.well.gain``."""
    attr_cfg = dict(ctx.script_cfg["attributes"])
    candidate_attributes = [str(v) for v in attr_cfg.get("candidate_attributes", list(CANDIDATE_ATTRIBUTES))]
    clip_percentiles_raw = ctx.script_cfg["prediction"].get("clip_percentiles", [5.0, 95.0])
    if len(clip_percentiles_raw) != 2:
        raise ValueError("dynamic_gain.prediction.clip_percentiles must contain two numbers.")
    clip_percentiles = (float(clip_percentiles_raw[0]), float(clip_percentiles_raw[1]))
    return fit_gain_relationship_from_samples(
        sample_df,
        candidate_attributes=candidate_attributes,
        attr_tie_threshold=float(attr_cfg.get("attr_tie_threshold", 0.05)),
        clip_percentiles=clip_percentiles,
        attribute_floor_fraction=float(attr_cfg.get("attribute_floor_fraction", 0.10)),
    )


def seconds_to_odd_samples(window_s: float, sample_step_s: float, *, min_samples: int = 3) -> int:
    n = int(round(float(window_s) / float(sample_step_s)))
    n = max(n, int(min_samples))
    if n % 2 == 0:
        n += 1
    return n


def _configured_attribute_window_samples(ctx: DynamicGainContext, sample_df: pd.DataFrame) -> int:
    value = ctx.script_cfg["attributes"].get("window_s")
    if value is None:
        durations = sample_df["twt_end_s"].to_numpy(dtype=np.float64) - sample_df["twt_start_s"].to_numpy(dtype=np.float64)
        duration = float(np.nanmedian(durations[np.isfinite(durations) & (durations > 0.0)]))
    else:
        duration = float(value)
    if not np.isfinite(duration) or duration <= 0.0:
        duration = float(ctx.geometry["sample_step"]) * 9.0
    return seconds_to_odd_samples(duration, float(ctx.geometry["sample_step"]), min_samples=3)


def build_gain_volume(ctx: DynamicGainContext, fit: dict[str, Any], sample_df: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    """Thin wrapper — unpack context then delegate to ``cup.well.gain``."""
    n_sample = int(ctx.geometry["n_sample"])
    flat = (ctx.seismic.reshape(-1, n_sample) / float(ctx.train_mask_rms)).astype(np.float32)
    window_samples = _configured_attribute_window_samples(ctx, sample_df)
    smoothing = int(ctx.script_cfg["prediction"].get("gain_smoothing_samples", 1))
    return build_gain_volume_from_fit(
        flat,
        fit=fit,
        sample_step_s=float(ctx.geometry["sample_step"]),
        seismic_shape=ctx.seismic.shape,
        window_samples=window_samples,
        gain_smoothing_samples=smoothing,
        batch_traces=int(ctx.script_cfg["runtime"].get("volume_batch_traces", 512)),
    )


def write_gain_npz(ctx: DynamicGainContext, gain_volume: np.ndarray, fit: dict[str, Any], volume_stats: dict[str, Any]) -> Path:
    """Thin wrapper — unpack context then delegate to ``cup.well.gain``."""
    path = ctx.output_dir / "dynamic_gain.npz"
    write_dynamic_gain_npz(
        path,
        gain_volume,
        samples=ctx.samples,
        ilines=ctx.ilines,
        xlines=ctx.xlines,
        geometry=ctx.geometry,
        wavelet_file=repo_relative_path(ctx.wavelet_file, root=REPO_ROOT),
        lfm_file=repo_relative_path(ctx.lfm_file, root=REPO_ROOT),
        train_mask_rms=ctx.train_mask_rms,
        fit=fit,
        volume_stats=volume_stats,
        lfm_metadata=ctx.lfm_metadata,
    )
    return path


def write_qc_figures(
    ctx: DynamicGainContext,
    sample_df: pd.DataFrame,
    attr_metrics: pd.DataFrame,
    well_gain: pd.DataFrame,
    cluster_gain: pd.DataFrame,
    fit: dict[str, Any],
    gain_volume: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.hist(sample_df["gain"].to_numpy(dtype=np.float64), bins=32, color="#4C78A8", alpha=0.85)
    ax.axvline(float(np.median(sample_df["gain"])), color="black", lw=1.2, label="segment median")
    ax.set_xlabel("Segment gain")
    ax.set_ylabel("Count")
    ax.set_title("Well segment gain distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_fig(ctx.figure_dir / "qc_01_gain_distribution.png")

    attr = str(fit["attribute_name"])
    fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    ax.scatter(sample_df[f"log_{attr}"], sample_df["log_gain"], s=22, alpha=0.72)
    finite = np.isfinite(sample_df[f"log_{attr}"]) & np.isfinite(sample_df["log_gain"])
    if np.any(finite):
        x0, x1 = np.percentile(sample_df.loc[finite, f"log_{attr}"], [1.0, 99.0])
        x_line = np.linspace(float(x0), float(x1), 100)
        ax.plot(x_line, float(fit["intercept"]) + float(fit["slope"]) * x_line, color="black", lw=1.2)
    ax.set_xlabel(f"ln({attr})")
    ax.set_ylabel("ln(gain)")
    ax.set_title("Selected attribute gain fit")
    ax.grid(True, alpha=0.25)
    _save_fig(ctx.figure_dir / "qc_02_attribute_fit.png")

    if not attr_metrics.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
        labels = attr_metrics["attribute"].astype(str).tolist()
        values = attr_metrics["pearson"].astype(float).tolist()
        ax.bar(labels, values, color="#59A14F")
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylabel("Pearson r")
        ax.set_title("Candidate attribute correlations")
        ax.grid(True, axis="y", alpha=0.25)
        _save_fig(ctx.figure_dir / "qc_03_attribute_metrics.png")

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.scatter(well_gain["spatial_cluster_id"], well_gain["gain"], s=28, alpha=0.75, label="well median")
    ax.scatter(cluster_gain["spatial_cluster_id"], cluster_gain["gain"], s=70, color="black", marker="_", label="cluster median")
    ax.set_xlabel("Spatial cluster")
    ax.set_ylabel("Gain")
    ax.set_title("Spatial debias fixed gain evidence")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save_fig(ctx.figure_dir / "qc_04_spatial_debias.png")

    i_il = gain_volume.shape[0] // 2
    i_xl = gain_volume.shape[1] // 2
    i_t = gain_volume.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.0), constrained_layout=True)
    im0 = axes[0].imshow(
        gain_volume[i_il].T,
        aspect="auto",
        origin="upper",
        extent=[ctx.xlines[0], ctx.xlines[-1], ctx.samples[-1], ctx.samples[0]],
        cmap="viridis",
    )
    axes[0].set_title(f"Dynamic gain inline @ {ctx.ilines[i_il]:.0f}")
    axes[0].set_xlabel("Xline")
    axes[0].set_ylabel("TWT (s)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = axes[1].imshow(
        gain_volume[:, i_xl, :].T,
        aspect="auto",
        origin="upper",
        extent=[ctx.ilines[0], ctx.ilines[-1], ctx.samples[-1], ctx.samples[0]],
        cmap="viridis",
    )
    axes[1].set_title(f"Dynamic gain xline @ {ctx.xlines[i_xl]:.0f}")
    axes[1].set_xlabel("Inline")
    axes[1].set_ylabel("TWT (s)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    im2 = axes[2].imshow(
        gain_volume[:, :, i_t].T,
        aspect="auto",
        origin="lower",
        extent=[ctx.ilines[0], ctx.ilines[-1], ctx.xlines[0], ctx.xlines[-1]],
        cmap="viridis",
    )
    axes[2].set_title(f"Dynamic gain slice @ {ctx.samples[i_t]:.3f} s")
    axes[2].set_xlabel("Inline")
    axes[2].set_ylabel("Xline")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)
    _save_fig(ctx.figure_dir / "qc_05_dynamic_gain_volume.png")


def _reflectivity_from_ai(ai: np.ndarray) -> np.ndarray:
    values = np.asarray(ai, dtype=np.float64).reshape(-1)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    upper = values[:-1]
    lower = values[1:]
    valid = np.isfinite(upper) & np.isfinite(lower)
    out[:-1][valid] = (lower[valid] - upper[valid]) / (lower[valid] + upper[valid] + 1e-10)
    return out


def _waveform_metrics(observed: np.ndarray, synthetic: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    obs = np.asarray(observed, dtype=np.float64).reshape(-1)
    syn = np.asarray(synthetic, dtype=np.float64).reshape(-1)
    valid = np.asarray(mask, dtype=bool).reshape(-1) & np.isfinite(obs) & np.isfinite(syn)
    obs_v = obs[valid]
    syn_v = syn[valid]
    diff = obs_v - syn_v
    obs_rms = float(np.sqrt(np.mean(obs_v**2))) if obs_v.size else float("nan")
    syn_rms = float(np.sqrt(np.mean(syn_v**2))) if syn_v.size else float("nan")
    if obs_v.size >= 2 and float(np.std(obs_v)) > 0.0 and float(np.std(syn_v)) > 0.0:
        corr = float(np.corrcoef(obs_v, syn_v)[0, 1])
    else:
        corr = float("nan")
    mae = float(np.mean(np.abs(diff))) if diff.size else float("nan")
    rmse = float(np.sqrt(np.mean(diff**2))) if diff.size else float("nan")
    bias = float(np.mean(syn_v - obs_v)) if diff.size else float("nan")
    return {
        "n_samples": int(valid.sum()),
        "corr": corr,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "observed_rms": obs_rms,
        "synthetic_rms": syn_rms,
        "rms_ratio": float(syn_rms / obs_rms) if np.isfinite(obs_rms) and obs_rms > 0.0 else float("nan"),
    }


def _qc_plot_slice(mask: np.ndarray, n_sample: int, *, pad_samples: int = 25) -> slice:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
    if indices.size == 0:
        return slice(0, int(n_sample))
    start = max(0, int(indices[0]) - int(pad_samples))
    end = min(int(n_sample), int(indices[-1]) + int(pad_samples) + 1)
    return slice(start, end)


def write_well_waveform_qc(
    ctx: DynamicGainContext,
    syn_unit: np.ndarray,
    gain_volume: np.ndarray,
    sample_df: pd.DataFrame,
) -> dict[str, Any]:
    well_qc_dir = ctx.output_dir / "well_qc"
    trace_dir = well_qc_dir / "traces"
    figure_dir = well_qc_dir / "figures"
    trace_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    anchor = load_log_ai_anchor_npz(ctx.anchor_file)
    validate_log_ai_anchor(
        anchor,
        sample_domain="time",
        n_sample=int(ctx.geometry["n_sample"]),
        n_traces=int(ctx.geometry["n_il"]) * int(ctx.geometry["n_xl"]),
    )
    _validate_samples_axis(np.asarray(anchor.samples, dtype=np.float64), ctx.samples, name="log_ai_anchor_time.npz")

    n_sample = int(ctx.geometry["n_sample"])
    seismic_norm_flat = (ctx.seismic.reshape(-1, n_sample) / float(ctx.train_mask_rms)).astype(np.float32)
    syn_unit_flat = syn_unit.reshape(-1, n_sample)
    gain_flat = gain_volume.reshape(-1, n_sample)
    synthetic_dynamic_flat = syn_unit_flat * gain_flat
    lfm_flat = ctx.lfm.reshape(-1, n_sample)
    loss_mask_flat = ctx.train_loss_mask.reshape(-1, n_sample)

    fit_groups = sample_df.groupby("flat_idx") if not sample_df.empty and "flat_idx" in sample_df.columns else {}
    metrics_rows: list[dict[str, Any]] = []
    anchor_names = np.asarray(anchor.anchor_names).astype(str)
    for anchor_row, flat_idx_value in enumerate(np.asarray(anchor.flat_indices, dtype=np.int64)):
        flat_idx = int(flat_idx_value)
        safe_name = sanitize_filename(str(anchor_names[anchor_row]))
        anchor_mask = np.asarray(anchor.anchor_mask[anchor_row], dtype=bool)
        loss_mask = np.asarray(loss_mask_flat[flat_idx], dtype=bool)
        finite = (
            anchor_mask
            & loss_mask
            & np.isfinite(seismic_norm_flat[flat_idx])
            & np.isfinite(synthetic_dynamic_flat[flat_idx])
        )
        used_sample_mask = np.zeros(n_sample, dtype=bool)
        fit_group = fit_groups.get_group(flat_idx) if hasattr(fit_groups, "groups") and flat_idx in fit_groups.groups else None
        if fit_group is not None:
            for _, row in fit_group.iterrows():
                start = max(0, int(row["sample_start"]))
                end = min(n_sample, int(row["sample_end"]))
                used_sample_mask[start:end] = True

        lfm_ai = np.asarray(lfm_flat[flat_idx], dtype=np.float64)
        reflectivity = _reflectivity_from_ai(lfm_ai)
        trace_df = pd.DataFrame(
            {
                "twt_s": ctx.samples,
                "seismic_norm": seismic_norm_flat[flat_idx],
                "synthetic_unit": syn_unit_flat[flat_idx],
                "dynamic_gain": gain_flat[flat_idx],
                "synthetic_dynamic": synthetic_dynamic_flat[flat_idx],
                "residual_dynamic": seismic_norm_flat[flat_idx] - synthetic_dynamic_flat[flat_idx],
                "lfm_ai": lfm_ai,
                "reflectivity": reflectivity,
                "anchor_mask": anchor_mask,
                "loss_mask": loss_mask,
                "valid_for_metrics": finite,
                "used_for_gain_fit_sample": used_sample_mask,
            }
        )
        trace_path = trace_dir / f"anchor_trace_{flat_idx}_{safe_name}.csv"
        figure_path = figure_dir / f"anchor_trace_{flat_idx}_{safe_name}.png"
        trace_df.to_csv(trace_path, index=False, encoding="utf-8-sig")

        plot_slice = _qc_plot_slice(finite | used_sample_mask, n_sample)
        plot_samples = np.asarray(ctx.samples[plot_slice], dtype=np.float64)
        synthetic_values = np.asarray(synthetic_dynamic_flat[flat_idx, plot_slice], dtype=np.float64)
        seismic_values = np.asarray(seismic_norm_flat[flat_idx, plot_slice], dtype=np.float64)
        lfm_plot = np.asarray(lfm_ai[plot_slice], dtype=np.float64)
        reflectivity_plot = np.asarray(reflectivity[plot_slice], dtype=np.float64)
        synthetic_trace = grid.Seismic(
            synthetic_values,
            plot_samples,
            "twt",
            name="Synthetic dynamic",
        )
        seismic_trace = grid.Seismic(
            seismic_values,
            plot_samples,
            "twt",
            name="Seismic normalized",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xcorr_values = normalized_xcorr(seismic_trace.values, synthetic_trace.values)
            xcorr_basis = synthetic_trace.sampling_rate * np.arange(-(synthetic_trace.size - 1), synthetic_trace.size)
            xcorr = grid.XCorr(xcorr_values, xcorr_basis, "tlag", name="XCorr")
            dxcorr = dynamic_normalized_xcorr(seismic_trace, synthetic_trace)
        fig, _ = plot_well_waveform_qc(
            grid.Log(lfm_plot, plot_samples, "twt", name="LFM AI"),
            grid.Reflectivity(reflectivity_plot, plot_samples, "twt", name="Reflectivity"),
            synthetic_trace,
            seismic_trace,
            xcorr,
            dxcorr,
            mode="amplitude",
            wiggle_scale_syn=1.0,
            wiggle_scale_real=1.0,
            figsize=(12.0, 7.5),
        )
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        metrics = _waveform_metrics(seismic_norm_flat[flat_idx], synthetic_dynamic_flat[flat_idx], finite)
        metrics_rows.append(
            {
                "anchor_name": str(anchor_names[anchor_row]),
                "flat_idx": flat_idx,
                "inline": float(np.asarray(anchor.inline, dtype=np.float64)[anchor_row]),
                "xline": float(np.asarray(anchor.xline, dtype=np.float64)[anchor_row]),
                "status": "ok" if int(metrics["n_samples"]) > 0 else "no_valid_samples",
                "used_for_gain_fit": bool(fit_group is not None and not fit_group.empty),
                "trace_csv": repo_relative_path(trace_path, root=REPO_ROOT),
                "figure": repo_relative_path(figure_path, root=REPO_ROOT),
                **metrics,
            }
        )

    metrics_df = pd.DataFrame.from_records(metrics_rows)
    metrics_path = well_qc_dir / "well_qc_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote dynamic gain well waveform QC to %s", well_qc_dir)
    return {
        "well_qc_dir": repo_relative_path(well_qc_dir, root=REPO_ROOT),
        "metrics": repo_relative_path(metrics_path, root=REPO_ROOT),
        "n_anchor_traces_qc": int(len(metrics_rows)),
    }


def write_outputs(
    ctx: DynamicGainContext,
    syn_unit: np.ndarray,
    sample_df: pd.DataFrame,
    fixed_payload: dict[str, Any],
    well_gain: pd.DataFrame,
    cluster_gain: pd.DataFrame,
    attr_metrics: pd.DataFrame,
    fit: dict[str, Any],
    gain_volume: np.ndarray,
    volume_stats: dict[str, Any],
    sample_summary: dict[str, Any],
) -> None:
    gain_npz = write_gain_npz(ctx, gain_volume, fit, volume_stats)
    fixed_payload = {
        **fixed_payload,
        "train_mask_rms": float(ctx.train_mask_rms),
    }
    write_json(ctx.output_dir / "recommended_fixed_gain.json", fixed_payload)
    sample_df.to_csv(ctx.output_dir / "dynamic_gain_samples.csv", index=False, encoding="utf-8-sig")
    well_gain.to_csv(ctx.output_dir / "dynamic_gain_well_medians.csv", index=False, encoding="utf-8-sig")
    cluster_gain.to_csv(ctx.output_dir / "dynamic_gain_cluster_medians.csv", index=False, encoding="utf-8-sig")
    attr_metrics.to_csv(ctx.output_dir / "dynamic_gain_attribute_metrics.csv", index=False, encoding="utf-8-sig")
    write_qc_figures(ctx, sample_df, attr_metrics, well_gain, cluster_gain, fit, gain_volume)
    well_qc_summary = write_well_waveform_qc(ctx, syn_unit, gain_volume, sample_df)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "train_config": repo_relative_path(ctx.train_config_file, root=REPO_ROOT),
            "seismic_file": repo_relative_path(ctx.seismic_file, root=REPO_ROOT),
            "ai_lfm_file": repo_relative_path(ctx.lfm_file, root=REPO_ROOT),
            "unit_wavelet_file": repo_relative_path(ctx.wavelet_file, root=REPO_ROOT),
            "anchor_file": repo_relative_path(ctx.anchor_file, root=REPO_ROOT),
            "source_dirs": {key: _repo_path_or_none(value) for key, value in ctx.source_dirs.items()},
        },
        "normalization": {
            "name": NORMALIZATION,
            "train_mask_rms": float(ctx.train_mask_rms),
            "split_metadata": ctx.split_metadata,
        },
        "fit": fit,
        "attribute_metrics": attr_metrics.to_dict(orient="records"),
        "samples": sample_summary,
        "recommended_fixed_gain": fixed_payload,
        "volume_stats": volume_stats,
        "well_qc": well_qc_summary,
        "outputs": {
            "dynamic_gain": repo_relative_path(gain_npz, root=REPO_ROOT),
            "recommended_fixed_gain": repo_relative_path(ctx.output_dir / "recommended_fixed_gain.json", root=REPO_ROOT),
            "dynamic_gain_samples": repo_relative_path(ctx.output_dir / "dynamic_gain_samples.csv", root=REPO_ROOT),
            "figures": repo_relative_path(ctx.figure_dir, root=REPO_ROOT),
            "well_qc": well_qc_summary["well_qc_dir"],
        },
        "config": ctx.script_cfg,
    }
    write_json(ctx.output_dir / "dynamic_gain_summary.json", summary)
    logger.info("Wrote dynamic gain outputs to %s", ctx.output_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    ctx = load_context(args)
    syn_unit = build_unit_synthetic(
        ctx.lfm,
        ctx.unit_wavelet,
        batch_traces=int(ctx.script_cfg["runtime"].get("forward_batch_traces", 256)),
    )
    sample_df, sample_summary = estimate_well_gain_samples(ctx, syn_unit)
    spatial_cfg = dict(ctx.script_cfg["spatial_debias"])
    sample_df = assign_spatial_clusters(
        sample_df,
        radius_m=float(spatial_cfg.get("cluster_radius_m", 600.0)),
        enabled=bool(spatial_cfg.get("enabled", True)),
    )
    fixed_payload, well_gain, cluster_gain = recommended_fixed_gain(sample_df)
    sample_df, fit, attr_metrics = fit_gain_relationship(ctx, sample_df)
    gain_volume, volume_stats = build_gain_volume(ctx, fit, sample_df)
    write_outputs(
        ctx,
        syn_unit,
        sample_df,
        fixed_payload,
        well_gain,
        cluster_gain,
        attr_metrics,
        fit,
        gain_volume,
        volume_stats,
        sample_summary,
    )


if __name__ == "__main__":
    main()
