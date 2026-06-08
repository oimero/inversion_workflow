"""Run time-domain GINN inference and export a stage-1 base-AI volume.

This is the ninth step of the time-domain workflow. It reads a trained GINN
checkpoint, replays the checkpoint config, and predicts the stage-1 impedance
volume over the full target interval.

Usage::

    python scripts/ginn_inversion.py
    python scripts/ginn_inversion.py --config experiments/common.yaml
    python scripts/ginn_inversion.py --checkpoint experiments/ginn/results/.../checkpoints/best.pt
    python scripts/ginn_inversion.py --output-dir scripts/output/ginn_inversion_test
    python scripts/ginn_inversion.py --skip-zgy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.seismic.survey import open_survey
from cup.seismic.viz import impedance_qc_metrics, plot_well_impedance_qc, sample_volume_at_points
from cup.utils.config import deep_merge_dict
from cup.utils.io import (
    load_yaml_config,
    repo_relative_path,
    resolve_relative_path,
    sanitize_filename,
    to_json_compatible,
    write_json,
)
from ginn.anchor import load_log_ai_anchor_npz, validate_log_ai_anchor
from ginn.config import GINNConfig
from ginn.trainer import Trainer

logger = logging.getLogger(__name__)


DEFAULT_CONFIG: dict[str, Any] = {
    "checkpoint_path": "experiments/ginn/results/2026060701/checkpoints/best.pt",
    "slice_mode": "inline",
    "slice_index": None,
    "clip_percentiles": [1.0, 99.0],
    "export_zgy": True,
    "zgy_inline_chunk_size": 16,
    "write_qc_context": False,
    "crossplot_max_samples": 200_000,
    "well_qc_enabled": True,
}


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override ginn_inversion.checkpoint_path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <output_root>/ginn_inversion_<timestamp>.",
    )
    parser.add_argument("--skip-zgy", action="store_true", help="Skip optional ZGY export.")
    parser.add_argument("--skip-well-qc", action="store_true", help="Skip anchor well impedance QC.")
    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
    except RuntimeError:
        pass
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _sampled_values(volume: np.ndarray, max_samples: int = 2_000_000) -> np.ndarray:
    flat = np.asarray(volume).ravel()
    stride = max(1, int(np.ceil(flat.size / max_samples)))
    values = flat[::stride]
    return values[np.isfinite(values)]


def _sampled_difference(left: np.ndarray, right: np.ndarray, max_samples: int = 2_000_000) -> np.ndarray:
    left_flat = np.asarray(left).ravel()
    right_flat = np.asarray(right).ravel()
    if left_flat.shape != right_flat.shape:
        raise ValueError(f"shape mismatch: {left.shape} vs {right.shape}")
    stride = max(1, int(np.ceil(left_flat.size / max_samples)))
    values = left_flat[::stride] - right_flat[::stride]
    return values[np.isfinite(values)]


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "min": None, "p01": None, "median": None, "p99": None, "max": None, "mean": None}
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _resolve_slice_index(mode: str, index: int | None, geometry: dict[str, Any]) -> int:
    if mode not in {"inline", "xline"}:
        raise ValueError(f"slice_mode must be 'inline' or 'xline', got {mode!r}")
    size = int(geometry["n_il"] if mode == "inline" else geometry["n_xl"])
    if index is None:
        return size // 2
    if not (0 <= index < size):
        raise IndexError(f"slice_index={index} out of range for {mode} size={size}")
    return int(index)


def _extract_section(volume: np.ndarray, mode: str, index: int) -> np.ndarray:
    if mode == "inline":
        return volume[index, :, :].T
    if mode == "xline":
        return volume[:, index, :].T
    raise ValueError(mode)


def _robust_limits(*arrays: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(arr, dtype=np.float32).ravel() for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot compute robust limits from empty finite values.")
    lo, hi = np.percentile(values, percentiles)
    if np.isclose(lo, hi):
        pad = max(abs(float(lo)) * 0.01, 1.0)
        lo = float(lo) - pad
        hi = float(hi) + pad
    return float(lo), float(hi)


def _axis_values(geometry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_il = int(geometry["n_il"])
    n_xl = int(geometry["n_xl"])
    n_sample = int(geometry["n_sample"])
    ilines = float(geometry["inline_min"]) + np.arange(n_il, dtype=np.float64) * float(geometry["inline_step"])
    xlines = float(geometry["xline_min"]) + np.arange(n_xl, dtype=np.float64) * float(geometry["xline_step"])
    samples = float(geometry["sample_min"]) + np.arange(n_sample, dtype=np.float64) * float(geometry["sample_step"])
    return ilines, xlines, samples


def _save_prediction_npz(
    path: Path,
    *,
    volume: np.ndarray,
    geometry: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    ilines, xlines, samples = _axis_values(geometry)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        volume=volume.astype(np.float32),
        ilines=ilines.astype(np.float32),
        xlines=xlines.astype(np.float32),
        samples=samples.astype(np.float32),
        geometry_json=json.dumps(to_json_compatible(geometry), ensure_ascii=False),
        metadata_json=json.dumps(to_json_compatible(metadata), ensure_ascii=False),
    )


def _save_qc_context(
    path: Path,
    *,
    lfm_volume: np.ndarray,
    mask_volume: np.ndarray,
    geometry: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    ilines, xlines, samples = _axis_values(geometry)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        lfm_volume=lfm_volume.astype(np.float32),
        mask=mask_volume.astype(bool),
        ilines=ilines.astype(np.float32),
        xlines=xlines.astype(np.float32),
        samples=samples.astype(np.float32),
        geometry_json=json.dumps(to_json_compatible(geometry), ensure_ascii=False),
        metadata_json=json.dumps(to_json_compatible(metadata), ensure_ascii=False),
    )


def _validate_checkpoint(checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    missing = {"config", "model_state_dict"} - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing required keys: {sorted(missing)}")
    if not isinstance(checkpoint["config"], dict):
        raise ValueError(f"Checkpoint {checkpoint_path} stores config as {type(checkpoint['config']).__name__}, expected dict.")


def _zgy_corners_from_survey(survey: Any, ilines: np.ndarray, xlines: np.ndarray) -> tuple[tuple[float, float], ...]:
    il0 = float(ilines[0])
    iln = float(ilines[-1])
    xl0 = float(xlines[0])
    xln = float(xlines[-1])
    geometry = survey.line_geometry
    return (
        tuple(float(v) for v in geometry.line_to_coord(il0, xl0)),
        tuple(float(v) for v in geometry.line_to_coord(iln, xl0)),
        tuple(float(v) for v in geometry.line_to_coord(il0, xln)),
        tuple(float(v) for v in geometry.line_to_coord(iln, xln)),
    )


def _try_write_zgy(
    zgy_file: Path,
    *,
    volume: np.ndarray,
    geometry: dict[str, Any],
    seismic_file: Path,
    seismic_type: str,
    inline_chunk_size: int,
) -> str:
    if seismic_type.lower() != "zgy":
        return "skipped_non_zgy_source"
    try:
        from pyzgy.write import SeismicWriter

        ilines, xlines, samples = _axis_values(geometry)
        if samples.size < 2:
            raise ValueError("ZGY export requires at least two samples.")
        sample_step_s = float(np.median(np.diff(samples)))
        if not np.allclose(np.diff(samples), sample_step_s, rtol=1e-6, atol=1e-9):
            raise ValueError("ZGY export requires a regular sample axis.")
        inline_inc = float(np.median(np.diff(ilines))) if ilines.size > 1 else 0.0
        xline_inc = float(np.median(np.diff(xlines))) if xlines.size > 1 else 0.0

        survey = open_survey(seismic_file, seismic_type=seismic_type)
        corners = _zgy_corners_from_survey(survey, ilines, xlines)

        zgy_file.parent.mkdir(parents=True, exist_ok=True)
        if zgy_file.exists():
            zgy_file.unlink()
        chunk = max(1, int(inline_chunk_size))
        export_volume = np.asarray(volume, dtype=np.float32)
        with SeismicWriter(
            zgy_file,
            tuple(int(v) for v in export_volume.shape),
            float(samples[0]) * 1000.0,
            sample_step_s * 1000.0,
            (float(ilines[0]), float(xlines[0])),
            (inline_inc, xline_inc),
            corners=corners,
        ) as writer:
            for il_start in range(0, export_volume.shape[0], chunk):
                il_end = min(export_volume.shape[0], il_start + chunk)
                writer.write_subvolume(export_volume[il_start:il_end], il_start, 0, 0)
        return "written"
    except Exception as exc:
        return f"failed:{exc}"


def _plot_prediction_slice(
    path: Path,
    *,
    pred_volume: np.ndarray,
    lfm_volume: np.ndarray,
    mask_volume: np.ndarray,
    geometry: dict[str, Any],
    slice_mode: str,
    slice_index: int,
    clip_percentiles: tuple[float, float],
) -> None:
    pred_section = _extract_section(pred_volume, slice_mode, slice_index)
    lfm_section = _extract_section(lfm_volume, slice_mode, slice_index)
    mask_section = _extract_section(mask_volume.astype(np.float32, copy=False), slice_mode, slice_index)
    diff_section = pred_section - lfm_section
    shared_vmin, shared_vmax = _robust_limits(pred_section, lfm_section, percentiles=clip_percentiles)
    finite_diff = diff_section[np.isfinite(diff_section)]
    diff_abs = float(np.percentile(np.abs(finite_diff), clip_percentiles[1])) if finite_diff.size else 1.0
    diff_abs = max(diff_abs, 1.0)

    _, _, samples = _axis_values(geometry)
    if slice_mode == "inline":
        _, xlines, _ = _axis_values(geometry)
        extent = [float(xlines[0]), float(xlines[-1]), float(samples[-1]), float(samples[0])]
        xlabel = "Xline"
    else:
        ilines, _, _ = _axis_values(geometry)
        extent = [float(ilines[0]), float(ilines[-1]), float(samples[-1]), float(samples[0])]
        xlabel = "Inline"

    fig, axes = plt.subplots(1, 4, figsize=(20, 7), constrained_layout=True)
    im0 = axes[0].imshow(pred_section, cmap="viridis", aspect="auto", origin="upper", extent=extent, vmin=shared_vmin, vmax=shared_vmax)
    axes[0].set_title(f"Predicted AI | {slice_mode}={slice_index}")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("TWT (s)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = axes[1].imshow(lfm_section, cmap="viridis", aspect="auto", origin="upper", extent=extent, vmin=shared_vmin, vmax=shared_vmax)
    axes[1].set_title("LFM")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("TWT (s)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    im2 = axes[2].imshow(diff_section, cmap="seismic", aspect="auto", origin="upper", extent=extent, vmin=-diff_abs, vmax=diff_abs)
    axes[2].set_title("Prediction - LFM")
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel("TWT (s)")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)
    im3 = axes[3].imshow(mask_section, cmap="gray_r", aspect="auto", origin="upper", extent=extent, vmin=0.0, vmax=1.0)
    axes[3].set_title("Target mask")
    axes[3].set_xlabel(xlabel)
    axes[3].set_ylabel("TWT (s)")
    fig.colorbar(im3, ax=axes[3], shrink=0.85)
    _save_fig(path)


def _plot_crossplot(path: Path, *, pred_volume: np.ndarray, lfm_volume: np.ndarray, max_samples: int) -> None:
    pred = np.asarray(pred_volume).ravel()
    lfm = np.asarray(lfm_volume).ravel()
    if pred.shape != lfm.shape:
        raise ValueError(f"shape mismatch: {pred_volume.shape} vs {lfm_volume.shape}")
    valid = np.isfinite(pred) & np.isfinite(lfm)
    pred = pred[valid]
    lfm = lfm[valid]
    if pred.size == 0:
        raise ValueError("Cannot plot crossplot from empty finite values.")
    stride = max(1, int(np.ceil(pred.size / max(1, int(max_samples)))))
    pred = pred[::stride]
    lfm = lfm[::stride]

    lo, hi = _robust_limits(pred, lfm, percentiles=(1.0, 99.0))
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.hexbin(lfm, pred, gridsize=120, bins="log", mincnt=1, cmap="viridis")
    ax.plot([lo, hi], [lo, hi], color="tab:red", lw=1.2, label="1:1")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("LFM AI")
    ax.set_ylabel("Predicted AI")
    ax.set_title("Prediction vs LFM")
    ax.legend(loc="upper left")
    _save_fig(path)


def _relative_outputs(paths: dict[str, Path | None]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key, path in paths.items():
        out[key] = None if path is None else repo_relative_path(path, root=REPO_ROOT)
    return out


def _resolve_anchor_point_table(anchor_metadata: dict[str, Any]) -> Path | None:
    value = anchor_metadata.get("point_table")
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value))
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = values.to_numpy(dtype=np.float64)
    w = np.maximum(weights.to_numpy(dtype=np.float64), 0.0)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        valid = np.isfinite(v)
        if not np.any(valid):
            return float("nan")
        return float(np.mean(v[valid]))
    return float(np.sum(v[valid] * w[valid]) / np.sum(w[valid]))


def _write_ginn_well_qc(
    *,
    pred_volume: np.ndarray,
    geometry: dict[str, Any],
    anchor_file: Path | None,
    output_dir: Path,
) -> tuple[Path | None, dict[str, Any]]:
    if anchor_file is None or not Path(anchor_file).exists():
        return None, {
            "status": "skipped",
            "reason": "missing_log_ai_anchor_file",
            "n_wells_qc": 0,
            "n_anchor_traces_qc": 0,
            "well_qc": None,
        }
    anchor = load_log_ai_anchor_npz(anchor_file)
    validate_log_ai_anchor(
        anchor,
        sample_domain="time",
        n_sample=int(geometry["n_sample"]),
        n_traces=int(geometry["n_il"]) * int(geometry["n_xl"]),
    )
    point_table = _resolve_anchor_point_table(anchor.metadata)
    if point_table is None or not point_table.exists():
        return None, {
            "status": "skipped",
            "reason": "missing_anchor_point_table",
            "n_wells_qc": 0,
            "n_anchor_traces_qc": 0,
            "well_qc": None,
        }

    point_df = pd.read_csv(point_table)
    required = {"well_name", "flat_idx", "sample_index", "twt_s", "inline_float", "xline_float", "ai_full", "well_low_ai", "weight"}
    missing = required - set(point_df.columns)
    if missing:
        return None, {
            "status": "skipped",
            "reason": f"point_table_missing_columns:{sorted(missing)}",
            "n_wells_qc": 0,
            "n_anchor_traces_qc": 0,
            "well_qc": None,
        }

    qc_dir = output_dir / "well_qc"
    figure_dir = qc_dir / "figures"
    trace_dir = qc_dir / "traces"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    ilines, xlines, samples = _axis_values(geometry)
    n_sample = samples.size
    point_by_flat = {int(flat_idx): group.copy() for flat_idx, group in point_df.groupby("flat_idx", sort=False)}
    metrics_rows: list[dict[str, Any]] = []

    for anchor_row, flat_idx_value in enumerate(np.asarray(anchor.flat_indices, dtype=np.int64)):
        flat_idx = int(flat_idx_value)
        anchor_name = str(np.asarray(anchor.anchor_names).astype(str)[anchor_row])
        anchor_mask = np.asarray(anchor.anchor_mask[anchor_row], dtype=bool)
        weight = np.asarray(anchor.anchor_weight[anchor_row], dtype=np.float64)
        low_ai = np.full(n_sample, np.nan, dtype=np.float64)
        low_ai[anchor_mask] = np.asarray(anchor.target_ai[anchor_row], dtype=np.float64)[anchor_mask]
        full_ai = np.full(n_sample, np.nan, dtype=np.float64)
        # Defaults are the aggregated anchor-trace coordinates. Point rows replace
        # them where per-sample floating coordinates exist, which matters for
        # deviated wells and dense-platform conflict audit.
        inline_values = np.full(n_sample, float(np.asarray(anchor.inline, dtype=np.float64)[anchor_row]), dtype=np.float64)
        xline_values = np.full(n_sample, float(np.asarray(anchor.xline, dtype=np.float64)[anchor_row]), dtype=np.float64)
        zones = np.full(n_sample, "", dtype=object)
        u_values = np.full(n_sample, np.nan, dtype=np.float64)

        group = point_by_flat.get(flat_idx)
        if group is not None and not group.empty:
            for sample_index, sample_group in group.groupby("sample_index", sort=False):
                idx = int(sample_index)
                if not 0 <= idx < n_sample:
                    continue
                w = sample_group["weight"]
                full_ai[idx] = _weighted_mean(sample_group["ai_full"], w)
                low_ai[idx] = _weighted_mean(sample_group["well_low_ai"], w)
                inline_values[idx] = _weighted_mean(sample_group["inline_float"], w)
                xline_values[idx] = _weighted_mean(sample_group["xline_float"], w)
                if "zone_name" in sample_group.columns:
                    zones[idx] = str(sample_group["zone_name"].astype(str).iloc[0])
                if "u_in_zone" in sample_group.columns:
                    u_values[idx] = _weighted_mean(sample_group["u_in_zone"], w)

        sample_indices = np.arange(n_sample, dtype=np.int64)
        pred_ai = sample_volume_at_points(
            pred_volume,
            ilines=ilines,
            xlines=xlines,
            inline_values=inline_values,
            xline_values=xline_values,
            sample_indices=sample_indices,
        )
        valid = anchor_mask & np.isfinite(low_ai) & np.isfinite(pred_ai)
        trace_df = pd.DataFrame(
            {
                "twt_s": samples,
                "sample_index": sample_indices,
                "inline_float": inline_values,
                "xline_float": xline_values,
                "zone_name": zones,
                "u_in_zone": u_values,
                "well_ai_full": full_ai,
                "well_ai_low": low_ai,
                "ginn_ai": pred_ai,
                "diff_vs_low": pred_ai - low_ai,
                "diff_vs_full": pred_ai - full_ai,
                "weight": weight,
                "anchor_mask": anchor_mask,
                "valid_for_metrics": valid,
            }
        )
        safe = sanitize_filename(anchor_name)
        trace_path = trace_dir / f"anchor_trace_{flat_idx}_{safe}.csv"
        figure_path = figure_dir / f"anchor_trace_{flat_idx}_{safe}.png"
        trace_df.to_csv(trace_path, index=False, encoding="utf-8-sig")
        low_metrics = impedance_qc_metrics(low_ai, pred_ai, mask=valid, weights=weight, prefix="low")
        full_metrics = impedance_qc_metrics(full_ai, pred_ai, mask=anchor_mask, weights=weight, prefix="full")
        metrics = {**low_metrics, **full_metrics}
        plot_well_impedance_qc(
            figure_path,
            samples_s=samples,
            full_ai=full_ai,
            low_ai=low_ai,
            model_ai=pred_ai,
            title=f"GINN anchor-trace QC | flat_idx={flat_idx} | {anchor_name}",
            full_label="Step-6 full-frequency anchor AI",
            low_label="Step-6 low-frequency anchor AI",
            model_label="Step-9 GINN predicted AI",
            metrics=metrics,
        )
        contributing_wells = sorted({str(v) for v in anchor_name.split(";") if str(v).strip()})
        metrics_rows.append(
            {
                "anchor_name": anchor_name,
                "contributing_wells": ";".join(contributing_wells),
                "n_contributing_wells": int(len(contributing_wells)),
                "flat_idx": flat_idx,
                "status": "ok" if int(low_metrics["low_n_samples"]) > 0 else "no_valid_samples",
                **metrics,
                "trace_qc_path": repo_relative_path(trace_path, root=REPO_ROOT),
                "figure_path": repo_relative_path(figure_path, root=REPO_ROOT),
            }
        )

    metrics_path = qc_dir / "well_qc_metrics.csv"
    pd.DataFrame.from_records(metrics_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    return metrics_path, {
        "status": "ok",
        "n_wells_qc": int(len(metrics_rows)),
        "n_anchor_traces_qc": int(len(metrics_rows)),
        "well_qc": repo_relative_path(qc_dir, root=REPO_ROOT),
        "well_qc_metrics": repo_relative_path(metrics_path, root=REPO_ROOT),
        "anchor_file": repo_relative_path(anchor_file, root=REPO_ROOT),
        "point_table": repo_relative_path(point_table, root=REPO_ROOT),
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    common_cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    if "ginn_inversion" not in common_cfg:
        raise ValueError("Missing 'ginn_inversion' section in config.")
    script_cfg = deep_merge_dict(DEFAULT_CONFIG, dict(common_cfg.get("ginn_inversion") or {}))

    output_root = resolve_relative_path(common_cfg.get("output_root", "scripts/output"), root=REPO_ROOT)
    if args.output_dir is None:
        output_dir = output_root / f"ginn_inversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = resolve_relative_path(args.output_dir, root=REPO_ROOT)
    figure_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    qc_dir = output_dir / "qc"
    trainer_context_dir = output_dir / "trainer_context"
    for path in (output_dir, figure_dir, metadata_dir, qc_dir, trainer_context_dir):
        path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint or Path(script_cfg["checkpoint_path"])
    checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else (REPO_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_checkpoint(checkpoint, checkpoint_path)
    checkpoint_train_config = GINNConfig.from_dict(checkpoint["config"], base_dir=REPO_ROOT)

    cfg = GINNConfig.from_dict(checkpoint["config"], base_dir=REPO_ROOT)
    cfg.device = "cuda" if torch.cuda.is_available() and str(cfg.device).lower().startswith("cuda") else "cpu"
    cfg.checkpoint_dir = trainer_context_dir

    logger.info("Building trainer context from checkpoint config...")
    trainer = Trainer(cfg)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.eval()

    logger.info("Predicting full stage-1 impedance volume...")
    pred_volume = trainer.predict_volume()
    n_il = int(trainer.geometry["n_il"])
    n_xl = int(trainer.geometry["n_xl"])
    n_sample = int(trainer.geometry["n_sample"])
    lfm_volume = trainer.dataset.lfm_flat.reshape(n_il, n_xl, n_sample)
    mask_volume = trainer.dataset._mask_flat.reshape(n_il, n_xl, n_sample)

    prediction_stats = {
        "prediction_ai": _stats(_sampled_values(pred_volume)),
        "lfm_ai": _stats(_sampled_values(lfm_volume)),
        "prediction_minus_lfm": _stats(_sampled_difference(pred_volume, lfm_volume)),
        "mask_coverage": float(mask_volume.mean()),
    }

    base_ai_npz = output_dir / "stage1_ginn_base_ai_time.npz"
    zgy_path = output_dir / "stage1_ginn_base_ai_time.zgy"
    qc_context_path = qc_dir / "prediction_context_time.npz"

    metadata: dict[str, Any] = {
        "artifact": base_ai_npz.name,
        "source_script": Path(__file__).name,
        "checkpoint_path": repo_relative_path(checkpoint_path, root=REPO_ROOT),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_best_loss": float(checkpoint.get("best_loss", np.nan)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "checkpoint_train_config": checkpoint_train_config.to_json_dict(),
        "replay_config": cfg.to_json_dict(),
        "ai_lfm_file": repo_relative_path(cfg.ai_lfm_file, root=REPO_ROOT),
        "wavelet_file": None if cfg.wavelet_file is None else repo_relative_path(cfg.wavelet_file, root=REPO_ROOT),
        "gain_source": cfg.gain_source,
        "dynamic_gain_model": None
        if cfg.dynamic_gain_model is None
        else repo_relative_path(cfg.dynamic_gain_model, root=REPO_ROOT),
        "prediction_stats": prediction_stats,
    }

    raw_slice_index = script_cfg.get("slice_index")
    slice_index = None if raw_slice_index is None else int(raw_slice_index)
    slice_mode = str(script_cfg.get("slice_mode", "inline"))
    resolved_slice_index = _resolve_slice_index(slice_mode, slice_index, trainer.geometry)
    clip_values = tuple(float(v) for v in script_cfg.get("clip_percentiles", [1.0, 99.0]))
    if len(clip_values) != 2:
        raise ValueError("clip_percentiles must contain exactly two values.")

    slice_figure_path = figure_dir / f"{slice_mode}_{resolved_slice_index:04d}_prediction_vs_lfm.png"
    crossplot_path = figure_dir / "prediction_vs_lfm_crossplot.png"
    _plot_prediction_slice(
        slice_figure_path,
        pred_volume=pred_volume,
        lfm_volume=lfm_volume,
        mask_volume=mask_volume,
        geometry=trainer.geometry,
        slice_mode=slice_mode,
        slice_index=resolved_slice_index,
        clip_percentiles=clip_values,  # type: ignore[arg-type]
    )
    _plot_crossplot(
        crossplot_path,
        pred_volume=pred_volume,
        lfm_volume=lfm_volume,
        max_samples=int(script_cfg.get("crossplot_max_samples", 200_000)),
    )

    qc_context_written = bool(script_cfg.get("write_qc_context", False))

    zgy_status = "disabled"
    if bool(script_cfg.get("export_zgy", True)) and not args.skip_zgy:
        zgy_status = _try_write_zgy(
            zgy_path,
            volume=pred_volume,
            geometry=trainer.geometry,
            seismic_file=cfg.seismic_file,
            seismic_type=cfg.seismic_type,
            inline_chunk_size=int(script_cfg.get("zgy_inline_chunk_size", 16)),
        )
    elif args.skip_zgy:
        zgy_status = "skipped_cli"

    well_qc_metrics_path: Path | None = None
    if args.skip_well_qc or not bool(script_cfg.get("well_qc_enabled", True)):
        well_qc_summary: dict[str, Any] = {
            "status": "skipped",
            "reason": "disabled",
            "n_wells_qc": 0,
            "n_anchor_traces_qc": 0,
            "well_qc": None,
        }
    else:
        well_qc_metrics_path, well_qc_summary = _write_ginn_well_qc(
            pred_volume=pred_volume,
            geometry=trainer.geometry,
            anchor_file=cfg.log_ai_anchor_file,
            output_dir=output_dir,
        )

    outputs = _relative_outputs(
        {
            "stage1_ginn_base_ai_time": base_ai_npz,
            "stage1_ginn_base_ai_time_zgy": zgy_path if zgy_status == "written" else None,
            "prediction_slice_figure": slice_figure_path,
            "prediction_crossplot": crossplot_path,
            "prediction_context_time": qc_context_path if qc_context_written else None,
            "well_qc_metrics": well_qc_metrics_path,
        }
    )
    outputs["well_qc"] = well_qc_summary.get("well_qc")
    summary = {
        "config": script_cfg,
        "checkpoint": {
            "path": repo_relative_path(checkpoint_path, root=REPO_ROOT),
            "epoch": metadata["checkpoint_epoch"],
            "best_epoch": metadata["checkpoint_best_epoch"],
            "best_loss": metadata["checkpoint_best_loss"],
        },
        "geometry": trainer.geometry,
        "prediction_stats": prediction_stats,
        "outputs": outputs,
        "zgy_export_status": zgy_status,
        "well_qc": well_qc_summary,
    }
    metadata["outputs"] = outputs
    metadata["zgy_export_status"] = zgy_status
    _save_prediction_npz(base_ai_npz, volume=pred_volume, geometry=trainer.geometry, metadata=metadata)
    if qc_context_written:
        _save_qc_context(
            qc_context_path,
            lfm_volume=lfm_volume,
            mask_volume=mask_volume,
            geometry=trainer.geometry,
            metadata=metadata,
        )
    write_json(metadata_dir / "run_summary.json", summary)

    print("=== GINN Inversion (Time) ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"NPZ: {base_ai_npz}")
    print(f"ZGY export: {zgy_status}")


if __name__ == "__main__":
    main()
