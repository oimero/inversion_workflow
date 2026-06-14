"""Run time-domain GINN inference and export a stage-1 base-AI volume.

This is the ninth step of the time-domain workflow. It reads a trained GINN
checkpoint, replays the checkpoint config, and predicts the stage-1 impedance
volume over the full target interval.

Usage::

    python scripts/ginn_inversion.py
    python scripts/ginn_inversion.py --config experiments/common.yaml
    python scripts/ginn_inversion.py --checkpoint experiments/ginn/results/.../checkpoints/best.pt
    python scripts/ginn_inversion.py --slice inline=400
    python scripts/ginn_inversion.py --output-dir scripts/output/ginn_inversion_test
    python scripts/ginn_inversion.py --skip-volume
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

from cup.seismic.geometry import validate_sample_indices
from cup.seismic.survey import open_survey
from cup.seismic.viz import (
    impedance_qc_metrics,
    plot_well_waveform_qc,
    waveform_qc_metrics,
)
from cup.time_config import TimeWorkflowConfig
from cup.utils.io import (
    build_segy_textual_header,
    load_yaml_config,
    repo_relative_path,
    resolve_relative_path,
    sanitize_filename,
    to_json_compatible,
    write_json,
)
from cup.well.assets import normalize_well_name
from cup.well.constraints import horizon_markers_from_zone_points
from ginn.anchor import load_log_ai_anchor_npz
from ginn.config import GINNConfig
from ginn.trainer import Trainer
from wtie.optimize.similarity import dynamic_normalized_xcorr, normalized_xcorr
from wtie.processing import grid

logger = logging.getLogger(__name__)


QC_CLIP_PERCENTILES = (1.0, 99.0)
CROSSPLOT_MAX_SAMPLES = 200_000


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to replay. Defaults to the latest experiments/ginn/results/*/checkpoints/best.pt.",
    )
    parser.add_argument(
        "--slice",
        default="inline",
        metavar="inline[=INDEX]|xline[=INDEX]",
        help="QC section direction and optional zero-based index. Defaults to the central inline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <output_root>/ginn_inversion_<timestamp>.",
    )
    parser.add_argument("--skip-volume", action="store_true", help="Skip optional survey-format volume export.")
    parser.add_argument(
        "--write-qc-context",
        action="store_true",
        help="Write the large LFM and mask QC context NPZ.",
    )
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


def _parse_slice_spec(value: str) -> tuple[str, int | None]:
    mode_text, separator, index_text = str(value).strip().casefold().partition("=")
    if mode_text not in {"inline", "xline"}:
        raise ValueError("--slice must be inline, xline, inline=INDEX, or xline=INDEX.")
    if not separator:
        return mode_text, None
    if not index_text.strip():
        raise ValueError("--slice index must not be empty.")
    try:
        index = int(index_text)
    except ValueError as exc:
        raise ValueError(f"--slice index must be an integer, got {index_text!r}.") from exc
    return mode_text, index


def _latest_ginn_checkpoint(results_root: Path) -> Path:
    candidates = [
        path
        for path in results_root.glob("*/checkpoints/best.pt")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No GINN checkpoint found under {results_root} matching */checkpoints/best.pt."
        )
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, path.parent.parent.name))[-1]


def _validate_checkpoint_seismic_contract(
    cfg: GINNConfig,
    workflow: TimeWorkflowConfig,
) -> None:
    shared_file = resolve_relative_path(
        workflow.seismic.file,
        root=resolve_relative_path(workflow.data_root, root=REPO_ROOT),
    )
    if str(cfg.seismic_type).casefold() != workflow.seismic.type:
        raise ValueError(
            "Checkpoint seismic_type does not match top-level seismic.type: "
            f"{cfg.seismic_type!r} != {workflow.seismic.type!r}."
        )
    if Path(cfg.seismic_file).resolve() != shared_file.resolve():
        raise ValueError(
            "Checkpoint seismic_file does not match top-level seismic.file: "
            f"{cfg.seismic_file} != {shared_file}."
        )


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


def _try_write_segy(
    segy_file: Path,
    *,
    volume: np.ndarray,
    seismic_file: Path,
    seismic_type: str,
    seismic_cfg: dict[str, Any],
) -> str:
    if seismic_type.lower() != "segy":
        return "skipped_non_segy_source"
    try:
        import cigsegy

        keylocs = [
            int(seismic_cfg["iline_byte"]),
            int(seismic_cfg["xline_byte"]),
            int(seismic_cfg["istep"]),
            int(seismic_cfg["xstep"]),
        ]
        textual = build_segy_textual_header(
            "Time-domain GINN stage-1 acoustic impedance",
            ["artifact=stage1_ginn_base_ai_time.npz", "source=ginn_inversion.py"],
        )
        cigsegy.create_by_sharing_header(
            str(segy_file),
            str(seismic_file),
            np.ascontiguousarray(np.asarray(volume, dtype=np.float32)),
            keylocs=keylocs,
            textual=textual,
        )
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


def _resolve_anchor_point_table(anchor_file: Path, metadata: dict[str, Any]) -> Path | None:
    raw_path = metadata.get("point_table")
    if raw_path in {None, ""}:
        return None
    path = Path(str(raw_path))
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _horizon_names_for_point_table(point_table_path: Path | None) -> dict[str, str]:
    if point_table_path is None:
        return {}
    summary_path = point_table_path.parent / "run_summary.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        f"horizon_{index}": Path(str(item.get("file", f"horizon_{index}"))).name
        for index, item in enumerate(summary.get("horizons") or [])
    }


def _point_table_for_anchor(anchor_name: str, flat_idx: int, point_df: pd.DataFrame) -> pd.DataFrame:
    if point_df.empty:
        return point_df
    out = point_df.loc[point_df["flat_idx"].astype("Int64") == int(flat_idx)].copy()
    if "well_name" in out.columns:
        key = normalize_well_name(anchor_name)
        named = out.loc[out["well_name"].map(lambda value: normalize_well_name(str(value))) == key].copy()
        if not named.empty:
            out = named
    return out


def _well_ai_on_axis(
    point_group: pd.DataFrame,
    samples: np.ndarray,
    *,
    value_column: str,
) -> np.ndarray:
    """Place one well-AI field by canonical TWT, validating the derived CSV index."""
    sample_axis = np.asarray(samples, dtype=np.float64).reshape(-1)
    values_on_axis = np.full(sample_axis.size, np.nan, dtype=np.float64)
    if point_group.empty:
        return values_on_axis

    required = {"twt_s", "seismic_sample_index", value_column}
    missing = required - set(point_group.columns)
    if missing:
        raise ValueError(f"Anchor point table is missing required columns: {sorted(missing)}")
    sample_indices = validate_sample_indices(
        sample_axis,
        point_group["twt_s"].to_numpy(dtype=np.float64),
        point_group["seismic_sample_index"].to_numpy(dtype=np.float64),
        field_name="seismic_sample_index",
    )
    ai_values = point_group[value_column].to_numpy(dtype=np.float64)
    weights = (
        np.maximum(point_group["weight"].to_numpy(dtype=np.float64), 0.0)
        if "weight" in point_group.columns
        else np.ones(ai_values.size, dtype=np.float64)
    )
    for sample_idx in np.unique(sample_indices):
        selected = (sample_indices == sample_idx) & np.isfinite(ai_values)
        if not np.any(selected):
            continue
        selected_weights = weights[selected]
        if not np.any(selected_weights > 0.0):
            selected_weights = np.ones(selected_weights.size, dtype=np.float64)
        values_on_axis[int(sample_idx)] = float(np.average(ai_values[selected], weights=selected_weights))
    return values_on_axis


def _reflectivity_from_ai(ai: np.ndarray) -> np.ndarray:
    values = np.asarray(ai, dtype=np.float64)
    out = np.zeros(values.shape, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0.0)
    pair_valid = valid[:-1] & valid[1:]
    upper = values[:-1][pair_valid]
    lower = values[1:][pair_valid]
    out[np.flatnonzero(pair_valid) + 1] = (lower - upper) / np.maximum(lower + upper, 1e-12)
    return out


def _expanded_mask_window(mask: np.ndarray, halo_samples: int) -> slice:
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    if indices.size == 0:
        raise ValueError("Cannot build a QC display window from an empty metric mask.")
    start = max(0, int(indices[0]) - int(halo_samples))
    stop = min(mask.size, int(indices[-1]) + int(halo_samples) + 1)
    return slice(start, stop)


def _forward_ginn_qc_trace(
    trainer: Trainer,
    predicted_ai: np.ndarray,
    dynamic_gain: np.ndarray | None,
) -> np.ndarray:
    ai_tensor = torch.from_numpy(
        np.asarray(predicted_ai, dtype=np.float32)
    ).view(1, 1, -1).to(trainer.device)
    gain_tensor = None
    if trainer.cfg.gain_source == "dynamic_gain_model":
        if dynamic_gain is None:
            raise ValueError("Checkpoint requires dynamic gain but the inference dataset has no gain trace.")
        gain_tensor = torch.from_numpy(
            np.asarray(dynamic_gain, dtype=np.float32)
        ).view(1, 1, -1).to(trainer.device)
    with torch.no_grad():
        return (
            trainer.forward_model(ai_tensor, gain=gain_tensor)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float64)
        )


def _write_ginn_well_qc(
    *,
    output_dir: Path,
    pred_volume: np.ndarray,
    anchor_file: Path | None,
    geometry: dict[str, Any],
    trainer: Trainer,
) -> dict[str, Any]:
    well_qc_dir = output_dir / "well_qc"
    trace_dir = well_qc_dir / "traces"
    figure_dir = well_qc_dir / "figures"
    trace_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = well_qc_dir / "well_qc_metrics.csv"

    if anchor_file is None:
        metrics_df = pd.DataFrame([{"status": "failed", "error": "checkpoint_config_has_no_log_ai_anchor_file"}])
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        return {"well_qc_dir": repo_relative_path(well_qc_dir, root=REPO_ROOT), "metrics": repo_relative_path(metrics_path, root=REPO_ROOT), "n_wells_qc": 0}

    anchor = load_log_ai_anchor_npz(anchor_file)
    point_table_path = _resolve_anchor_point_table(anchor_file, anchor.metadata)
    point_df = pd.read_csv(point_table_path) if point_table_path is not None and point_table_path.exists() else pd.DataFrame()
    horizon_names = _horizon_names_for_point_table(point_table_path)
    if not point_df.empty and "flat_idx" not in point_df.columns:
        raise ValueError(f"Anchor point table is missing flat_idx: {point_table_path}")

    n_sample = int(geometry["n_sample"])
    pred_flat = np.asarray(pred_volume, dtype=np.float32).reshape(-1, n_sample)
    samples = np.asarray(anchor.samples, dtype=np.float64)
    if samples.size != n_sample:
        raise ValueError(f"Anchor sample axis size {samples.size} does not match geometry n_sample={n_sample}.")
    metrics_rows: list[dict[str, Any]] = []

    for anchor_row, flat_idx_value in enumerate(np.asarray(anchor.flat_indices, dtype=np.int64)):
        flat_idx = int(flat_idx_value)
        anchor_name = str(np.asarray(anchor.anchor_names).astype(str)[anchor_row])
        safe = sanitize_filename(f"{flat_idx}_{anchor_name}")
        trace_path = trace_dir / f"well_qc_{safe}.csv"
        figure_path = figure_dir / f"well_qc_{safe}.png"

        base_row = {
            "well_name": anchor_name,
            "flat_idx": flat_idx,
            "inline": float(np.asarray(anchor.inline, dtype=np.float64)[anchor_row]),
            "xline": float(np.asarray(anchor.xline, dtype=np.float64)[anchor_row]),
            "gain_source": str(trainer.cfg.gain_source),
            "fixed_gain": (
                float(trainer.cfg.fixed_gain)
                if trainer.cfg.gain_source == "fixed_gain" and trainer.cfg.fixed_gain is not None
                else None
            ),
            "dynamic_gain_model": (
                None
                if trainer.cfg.dynamic_gain_model is None
                else repo_relative_path(trainer.cfg.dynamic_gain_model, root=REPO_ROOT)
            ),
        }
        try:
            if flat_idx < 0 or flat_idx >= pred_flat.shape[0]:
                raise IndexError("flat_idx_out_of_range")

            target_values = np.asarray(anchor.target_ai[anchor_row], dtype=np.float64)
            pred_values = np.asarray(pred_flat[flat_idx], dtype=np.float64)
            anchor_mask = np.asarray(anchor.anchor_mask[anchor_row], dtype=bool)
            qc_data = trainer.dataset.trace_qc_data(flat_idx)
            observed = np.asarray(qc_data.seismic_raw, dtype=np.float64) / float(trainer.dataset.seis_rms)
            loss_mask = np.asarray(qc_data.loss_mask, dtype=bool)

            point_group = _point_table_for_anchor(anchor_name, flat_idx, point_df)
            lfm_values = _well_ai_on_axis(point_group, samples, value_column="lfm_ai")
            reference_values = _well_ai_on_axis(point_group, samples, value_column="reference_ai")
            if not np.any(np.isfinite(lfm_values)):
                raise ValueError("No LFM well AI samples were found in the anchor point table.")

            synthetic = _forward_ginn_qc_trace(
                trainer,
                pred_values,
                qc_data.dynamic_gain,
            )
            reflectivity = _reflectivity_from_ai(pred_values)
            metric_mask = (
                anchor_mask
                & loss_mask
                & np.isfinite(target_values)
                & np.isfinite(lfm_values)
                & np.isfinite(pred_values)
                & np.isfinite(observed)
                & np.isfinite(synthetic)
            )
            if int(np.count_nonzero(metric_mask)) < 8:
                raise ValueError(f"Too few valid GINN waveform QC samples: {int(np.count_nonzero(metric_mask))}.")
            display_slice = _expanded_mask_window(
                metric_mask,
                halo_samples=int(trainer.cfg.boundary_effect_samples or 0),
            )
            display_twt = samples[display_slice]
            display_observed = observed[display_slice]
            display_synthetic = synthetic[display_slice]
            display_pred = pred_values[display_slice]
            display_lfm = lfm_values[display_slice]
            display_reference = reference_values[display_slice]
            display_target = target_values[display_slice]
            display_reflectivity = reflectivity[display_slice]

            observed_trace = grid.Seismic(
                display_observed,
                display_twt,
                "twt",
                name="Seismic normalized",
            )
            synthetic_trace = grid.Seismic(
                display_synthetic,
                display_twt,
                "twt",
                name="GINN synthetic",
            )
            xcorr_values = normalized_xcorr(observed_trace.values, synthetic_trace.values)
            xcorr_basis = synthetic_trace.sampling_rate * np.arange(
                -(synthetic_trace.size - 1),
                synthetic_trace.size,
            )
            xcorr = grid.XCorr(xcorr_values, xcorr_basis, "tlag", name="XCorr")
            dxcorr = dynamic_normalized_xcorr(observed_trace, synthetic_trace)
            lfm_trace = grid.Log(display_lfm, display_twt, "twt", name="LFM well AI")
            target_trace = grid.Log(display_target, display_twt, "twt", name="GINN target well AI")
            pred_trace = grid.Log(display_pred, display_twt, "twt", name="GINN predicted AI")
            reflectivity_trace = grid.Reflectivity(
                display_reflectivity,
                display_twt,
                "twt",
                name="GINN reflectivity",
            )

            trace_df = pd.DataFrame(
                {
                    "twt_s": samples,
                    "well_lfm_ai": lfm_values,
                    "well_reference_ai": reference_values,
                    "well_ginn_target_ai": target_values,
                    "ginn_predicted_ai": pred_values,
                    "reflectivity_ginn": reflectivity,
                    "seismic_raw": np.asarray(qc_data.seismic_raw, dtype=np.float64),
                    "seismic_normalized": observed,
                    "synthetic_ginn": synthetic,
                    "residual": observed - synthetic,
                    "dynamic_gain": (
                        np.full(n_sample, np.nan, dtype=np.float64)
                        if qc_data.dynamic_gain is None
                        else np.asarray(qc_data.dynamic_gain, dtype=np.float64)
                    ),
                    "anchor_mask": anchor_mask,
                    "loss_mask": loss_mask,
                    "valid_for_metrics": metric_mask,
                }
            )
            trace_df.to_csv(trace_path, index=False, encoding="utf-8-sig")

            waveform_metrics = waveform_qc_metrics(observed, synthetic, metric_mask)
            ai_rmse = float(
                np.sqrt(np.mean((pred_values[metric_mask] - target_values[metric_mask]) ** 2))
            )
            ai_traces = [target_trace, lfm_trace, pred_trace]
            if np.any(np.isfinite(display_reference)):
                ai_traces.insert(
                    0,
                    grid.Log(
                        display_reference,
                        display_twt,
                        "twt",
                        name="Reference well AI",
                    ),
                )
            fig, _axes = plot_well_waveform_qc(
                ai_traces,
                reflectivity_trace,
                synthetic_trace,
                observed_trace,
                xcorr,
                dxcorr,
                synthetic_ai=pred_trace,
                figsize=(12.0, 7.5),
                title=(
                    f"GINN synthetic | corr={waveform_metrics['corr']:.3f}, "
                    f"nmae={waveform_metrics['nmae']:.3f}, AI rmse={ai_rmse:.2e}"
                ),
                horizon_markers=horizon_markers_from_zone_points(
                    point_group,
                    horizon_names=horizon_names,
                ),
            )
            fig.savefig(figure_path, dpi=180, bbox_inches="tight")
            plt.close(fig)

            lfm_metric_trace = grid.Log(lfm_values, samples, "twt", name="LFM well AI")
            target_metric_trace = grid.Log(target_values, samples, "twt", name="GINN target well AI")
            pred_metric_trace = grid.Log(pred_values, samples, "twt", name="GINN predicted AI")
            impedance_metrics = impedance_qc_metrics(
                model_ai=pred_metric_trace,
                low_ai=lfm_metric_trace,
                full_ai=target_metric_trace,
                mask=metric_mask,
            )
            impedance_metrics = {
                key.replace("vs_low_", "vs_lfm_").replace("vs_full_", "vs_ginn_target_"): value
                for key, value in impedance_metrics.items()
            }
            metrics_rows.append(
                {
                    **base_row,
                    "status": "ok",
                    "trace_csv": repo_relative_path(trace_path, root=REPO_ROOT),
                    "figure": repo_relative_path(figure_path, root=REPO_ROOT),
                    "seismic_rms": float(trainer.dataset.seis_rms),
                    "xcorr_lag_s": float(xcorr.lag),
                    **impedance_metrics,
                    **{f"waveform_{key}": value for key, value in waveform_metrics.items()},
                }
            )
        except Exception as exc:
            plt.close("all")
            metrics_rows.append(
                {
                    **base_row,
                    "status": "failed",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )

    metrics_df = pd.DataFrame.from_records(metrics_rows)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    ok = metrics_df.loc[metrics_df["status"] == "ok"] if "status" in metrics_df else pd.DataFrame()
    summary: dict[str, Any] = {
        "well_qc_dir": repo_relative_path(well_qc_dir, root=REPO_ROOT),
        "metrics": repo_relative_path(metrics_path, root=REPO_ROOT),
        "n_wells_qc": int(len(ok)),
        "point_table": None if point_table_path is None else repo_relative_path(point_table_path, root=REPO_ROOT),
    }
    if not ok.empty and "vs_lfm_rmse" in ok:
        summary["mean_vs_lfm_rmse"] = float(ok["vs_lfm_rmse"].mean())
        summary["mean_vs_lfm_mae"] = float(ok["vs_lfm_mae"].mean())
        summary["mean_vs_lfm_corr"] = float(ok["vs_lfm_corr"].mean())
    return summary


def _relative_outputs(paths: dict[str, Path | None]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key, path in paths.items():
        out[key] = None if path is None else repo_relative_path(path, root=REPO_ROOT)
    return out


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
    workflow = TimeWorkflowConfig.from_mapping(common_cfg)
    script_cfg = dict(common_cfg.get("ginn_inversion") or {})
    if script_cfg:
        raise ValueError(
            "ginn_inversion no longer accepts common-YAML parameters; "
            "use --checkpoint, --slice, --skip-volume, or --write-qc-context."
        )

    output_root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
    if args.output_dir is None:
        output_dir = output_root / f"ginn_inversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = resolve_relative_path(args.output_dir, root=REPO_ROOT)
    figure_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    qc_dir = output_dir / "qc"
    well_qc_dir = output_dir / "well_qc"
    trainer_context_dir = output_dir / "trainer_context"
    for path in (output_dir, figure_dir, metadata_dir, qc_dir, well_qc_dir, trainer_context_dir):
        path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint or _latest_ginn_checkpoint(REPO_ROOT / "experiments" / "ginn" / "results")
    checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else (REPO_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_checkpoint(checkpoint, checkpoint_path)
    checkpoint_train_config = GINNConfig.from_dict(checkpoint["config"], base_dir=REPO_ROOT)

    cfg = GINNConfig.from_dict(checkpoint["config"], base_dir=REPO_ROOT)
    _validate_checkpoint_seismic_contract(cfg, workflow)
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
    segy_path = output_dir / "stage1_ginn_base_ai_time.segy"
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

    slice_mode, slice_index = _parse_slice_spec(args.slice)
    resolved_slice_index = _resolve_slice_index(slice_mode, slice_index, trainer.geometry)

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
        clip_percentiles=QC_CLIP_PERCENTILES,
    )
    _plot_crossplot(
        crossplot_path,
        pred_volume=pred_volume,
        lfm_volume=lfm_volume,
        max_samples=CROSSPLOT_MAX_SAMPLES,
    )

    qc_context_written = bool(args.write_qc_context)
    well_qc_summary = _write_ginn_well_qc(
        output_dir=output_dir,
        pred_volume=pred_volume,
        anchor_file=cfg.log_ai_anchor_file,
        geometry=trainer.geometry,
        trainer=trainer,
    )

    volume_status = "disabled"
    exported_volume_path: Path | None = None
    if not args.skip_volume:
        if cfg.seismic_type == "zgy":
            volume_status = _try_write_zgy(
                zgy_path,
                volume=pred_volume,
                geometry=trainer.geometry,
                seismic_file=cfg.seismic_file,
                seismic_type=cfg.seismic_type,
                inline_chunk_size=workflow.seismic.zgy_inline_chunk_size,
            )
            exported_volume_path = zgy_path if volume_status == "written" else None
        else:
            volume_status = _try_write_segy(
                segy_path,
                volume=pred_volume,
                seismic_file=cfg.seismic_file,
                seismic_type=cfg.seismic_type,
                seismic_cfg=workflow.seismic.as_dict(),
            )
            exported_volume_path = segy_path if volume_status == "written" else None
    else:
        volume_status = "skipped_cli"

    outputs = _relative_outputs(
        {
            "stage1_ginn_base_ai_time": base_ai_npz,
            "stage1_ginn_base_ai_time_volume": exported_volume_path,
            "prediction_slice_figure": slice_figure_path,
            "prediction_crossplot": crossplot_path,
            "prediction_context_time": qc_context_path if qc_context_written else None,
            "well_qc": well_qc_dir,
            "well_qc_metrics": well_qc_dir / "well_qc_metrics.csv",
        }
    )
    summary = {
        "runtime_options": {
            "slice": args.slice,
            "write_qc_context": qc_context_written,
            "export_volume": not args.skip_volume,
        },
        "checkpoint": {
            "path": repo_relative_path(checkpoint_path, root=REPO_ROOT),
            "epoch": metadata["checkpoint_epoch"],
            "best_epoch": metadata["checkpoint_best_epoch"],
            "best_loss": metadata["checkpoint_best_loss"],
        },
        "geometry": trainer.geometry,
        "prediction_stats": prediction_stats,
        "outputs": outputs,
        "well_qc": well_qc_summary,
        "volume_export_status": volume_status,
    }
    metadata["outputs"] = outputs
    metadata["volume_export_status"] = volume_status
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
    print(f"Volume export: {volume_status}")
    print(f"Well QC: {well_qc_dir}")


if __name__ == "__main__":
    main()
