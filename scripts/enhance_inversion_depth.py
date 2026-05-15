"""Run stage-2 enhancement inference and export an enhanced-AI volume.

Usage::

    python scripts/enhance_inversion_depth.py
    python scripts/enhance_inversion_depth.py --checkpoint experiments/enhance_depth/results/.../checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import lasio
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.petrel.load import import_well_heads_petrel  # noqa: E402
from cup.seismic.survey import open_survey  # noqa: E402
from cup.utils.io import (  # noqa: E402
    build_segy_textual_header,
    load_yaml_config,
    resolve_relative_path,
    sanitize_filename,
)
from enhance.config import EnhancementConfig  # noqa: E402
from enhance.loss import compose_enhanced_ai  # noqa: E402
from enhance.model import DilatedResNet1D  # noqa: E402
from ginn_depth.enhance import build_depth_enhancement_data_bundle  # noqa: E402
from wtie.optimize import tie as tie_utils  # noqa: E402
from wtie.optimize.logs import filter_log  # noqa: E402
from wtie.processing import grid  # noqa: E402

# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/common_depth.yaml"),
        help="Depth-domain common config YAML.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <output_root>/enhance_inversion_depth_<timestamp>.",
    )
    parser.add_argument("--skip-segy", action="store_true", help="Skip SEG-Y export.")
    parser.add_argument("--skip-well-qc", action="store_true", help="Skip LAS-vs-prediction well QC.")
    return parser.parse_args()


# =============================================================================
# Helpers
# =============================================================================


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
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1)),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
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


def _extract_mask_section(mask: np.ndarray, mode: str, index: int) -> np.ndarray:
    return _extract_section(mask.astype(np.float32, copy=False), mode, index).astype(bool)


def _robust_limits(*arrays: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(arr, dtype=np.float32).ravel() for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot compute robust limits from empty finite values.")
    lo, hi = np.percentile(values, percentiles)
    return float(lo), float(hi)


def _sample_axis_depth_m(geometry: dict[str, Any]) -> np.ndarray:
    n_sample = int(geometry["n_sample"])
    sample_min = float(geometry["sample_min"])
    sample_step = float(geometry["sample_step"])
    return sample_min + np.arange(n_sample, dtype=np.float64) * sample_step


def _bilinear_trace_from_volume(volume: np.ndarray, survey_ctx: Any, x: float, y: float) -> np.ndarray:
    i_float, j_float = survey_ctx.coord_to_index(x, y)
    i_float = float(np.clip(i_float, 0.0, volume.shape[0] - 1.0))
    j_float = float(np.clip(j_float, 0.0, volume.shape[1] - 1.0))
    i0 = int(np.floor(i_float))
    i1 = int(np.ceil(i_float))
    j0 = int(np.floor(j_float))
    j1 = int(np.ceil(j_float))
    wi = float(i_float - i0)
    wj = float(j_float - j0)
    t00 = volume[i0, j0, :]
    t01 = volume[i0, j1, :]
    t10 = volume[i1, j0, :]
    t11 = volume[i1, j1, :]
    return (1.0 - wi) * (1.0 - wj) * t00 + (1.0 - wi) * wj * t01 + wi * (1.0 - wj) * t10 + wi * wj * t11


def _find_curve_column(columns: pd.Index, candidates: tuple[str, ...]) -> Any | None:
    normalized = {str(col).strip().upper(): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _read_shifted_las_ai_curves(
    path: Path,
    *,
    log_filter_params: dict[str, float | int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    las = lasio.read(path)
    df = las.df()
    md = df.index.to_numpy(dtype=float)
    ai_col = _find_curve_column(df.columns, ("AI",))
    vp_col = _find_curve_column(df.columns, ("VP_MPS", "VP"))
    rho_col = _find_curve_column(df.columns, ("RHO_GCC", "RHO"))

    raw_ai: np.ndarray
    filtered_ai: np.ndarray
    if ai_col is not None:
        raw_ai = df[ai_col].to_numpy(dtype=float)
    elif vp_col is not None and rho_col is not None:
        raw_ai = df[vp_col].to_numpy(dtype=float) * df[rho_col].to_numpy(dtype=float)
    else:
        raise ValueError(f"Shifted LAS {path.name} must contain AI or VP+RHO curves.")

    median_size = int(log_filter_params["logs_median_size"])
    threshold = float(log_filter_params["logs_median_threshold"])
    std = float(log_filter_params["logs_std"])

    if vp_col is not None and rho_col is not None:
        vp = df[vp_col].to_numpy(dtype=float)
        rho = df[rho_col].to_numpy(dtype=float)
        logset = grid.LogSet(
            {
                "Vp": grid.Log(vp, md, "md", name="Vp", unit="m/s"),
                "Rho": grid.Log(rho, md, "md", name="Rho", unit="g/cm3"),
            }
        )
        filtered = tie_utils.filter_md_logs(
            logset,
            median_size=median_size,
            threshold=threshold,
            std=std,
            std2=0.8 * std,
        )
        filtered_ai = filtered.Vp.values * filtered.Rho.values
    elif ai_col is not None:
        ai_log = grid.Log(raw_ai, md, "md", name="AI", unit="m/s*g/cm3")
        filtered_log = filter_log(
            ai_log,
            median_size=median_size,
            threshold=threshold,
            std=std,
            std2=0.8 * std,
        )
        filtered_ai = filtered_log.values
    else:
        raise ValueError(f"Shifted LAS {path.name} must contain AI or VP+RHO curves.")

    return md, raw_ai, np.asarray(filtered_ai, dtype=float)


def _load_auto_tie_log_filter_params(cfg: dict[str, Any]) -> dict[str, float | int]:
    batch_cfg = cfg.get("wavelet_batch_synthetic_depth", {})
    if not batch_cfg:
        raise ValueError("Missing 'wavelet_batch_synthetic_depth' section for log filter parameters.")

    source_well_name = str(batch_cfg.get("source_well_name", ""))
    source_auto_tie_dir = batch_cfg.get("source_auto_tie_dir")
    source_run_summary: dict[str, Any] = {}
    if source_well_name and source_auto_tie_dir is not None:
        source_dir = resolve_relative_path(str(source_auto_tie_dir), root=REPO_ROOT)
        candidates = [
            source_dir / f"run_summary_{source_well_name}.json",
            source_dir / f"run_summary_auto_well_tie_{source_well_name}.json",
        ]
        summary_path = next((path for path in candidates if path.exists()), None)
        if summary_path is not None:
            source_run_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    params = {
        key: source_run_summary.get("auto_tie_best_parameters", {}).get(key)
        for key in ("logs_median_size", "logs_median_threshold", "logs_std")
    }
    if any(value is None for value in params.values()):
        fallback = batch_cfg.get("fallback_log_filter", {})
        params = {
            "logs_median_size": fallback.get("logs_median_size"),
            "logs_median_threshold": fallback.get("logs_median_threshold"),
            "logs_std": fallback.get("logs_std"),
        }
    if any(value is None for value in params.values()):
        raise ValueError("Missing auto-tie log filter parameters and fallback_log_filter.")
    return {
        "logs_median_size": int(params["logs_median_size"]),
        "logs_median_threshold": float(params["logs_median_threshold"]),
        "logs_std": float(params["logs_std"]),
    }


def _regularize_depth_curve(depth: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(depth) & np.isfinite(values)
    if int(valid.sum()) < 2:
        raise ValueError("depth curve has too few finite samples")
    depth = depth[valid]
    values = values[valid]
    order = np.argsort(depth)
    depth = depth[order]
    values = values[order]
    unique_depth, unique_idx = np.unique(depth, return_index=True)
    unique_values = values[unique_idx]
    if unique_depth.size < 2:
        raise ValueError("depth curve has too few unique samples")
    return unique_depth, unique_values


def _downsample_curve_to_axis(depth: np.ndarray, values: np.ndarray, sample_axis: np.ndarray) -> np.ndarray:
    """Average a high-resolution well curve into seismic depth sample bins."""
    depth, values = _regularize_depth_curve(depth, values)
    sample_axis = np.asarray(sample_axis, dtype=float)
    if sample_axis.size < 2:
        raise ValueError("sample_axis must contain at least two samples")
    dz = float(np.median(np.diff(sample_axis)))
    if not np.isfinite(dz) or dz <= 0.0:
        raise ValueError(f"invalid sample axis step: {dz}")

    edges = np.concatenate(
        [
            [sample_axis[0] - 0.5 * dz],
            0.5 * (sample_axis[:-1] + sample_axis[1:]),
            [sample_axis[-1] + 0.5 * dz],
        ]
    )
    out = np.full(sample_axis.shape, np.nan, dtype=np.float64)
    for idx in range(sample_axis.size):
        left = np.searchsorted(depth, edges[idx], side="left")
        right = np.searchsorted(depth, edges[idx + 1], side="left")
        if right > left:
            out[idx] = float(np.mean(values[left:right]))

    missing = ~np.isfinite(out)
    in_range = (sample_axis >= depth[0]) & (sample_axis <= depth[-1])
    fill = missing & in_range
    if np.any(fill):
        out[fill] = np.interp(sample_axis[fill], depth, values)
    return out.astype(np.float32)


def _save_prediction_npz(
    path: Path,
    *,
    volume: np.ndarray,
    geometry: dict[str, Any],
    samples: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    n_il = int(geometry["n_il"])
    n_xl = int(geometry["n_xl"])
    ilines = float(geometry["inline_min"]) + np.arange(n_il, dtype=np.float32) * float(geometry["inline_step"])
    xlines = float(geometry["xline_min"]) + np.arange(n_xl, dtype=np.float32) * float(geometry["xline_step"])
    np.savez_compressed(
        path,
        volume=volume.astype(np.float32),
        ilines=ilines.astype(np.float32),
        xlines=xlines.astype(np.float32),
        samples=np.asarray(samples, dtype=np.float32),
        geometry_json=json.dumps(geometry, ensure_ascii=False),
        metadata_json=json.dumps(metadata, ensure_ascii=False),
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = cfg.get("enhance_inversion_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'enhance_inversion_depth' section in config.")

    output_root = resolve_relative_path(str(cfg.get("output_root", "scripts/output")), root=REPO_ROOT)
    if args.checkpoint:
        checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else (REPO_ROOT / args.checkpoint).resolve()
    else:
        checkpoint_path = Path(script_cfg["checkpoint_path"])
        checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else (REPO_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"enhance_inversion_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dirs = {
        "root": output_dir,
        "metadata": output_dir / "metadata",
        "qc_figures": output_dir / "qc_figures",
        "well_qc": output_dir / "well_qc",
        "well_qc_figures": output_dir / "well_qc" / "figures",
        "well_qc_traces": output_dir / "well_qc" / "traces",
    }
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    print("=== Enhance Predict (Depth) ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")

    # ── Load model ──────────────────────────────────────────────
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_payload = checkpoint["config"]
    enhance_cfg = (
        EnhancementConfig.from_dict(cfg_payload, base_dir=REPO_ROOT)
        if isinstance(cfg_payload, dict)
        else cfg_payload
    )
    device = torch.device(enhance_cfg.device if torch.cuda.is_available() else "cpu")
    model = DilatedResNet1D(
        in_channels=enhance_cfg.in_channels,
        hidden_channels=enhance_cfg.hidden_channels,
        out_channels=enhance_cfg.out_channels,
        dilations=enhance_cfg.dilations,
        kernel_size=enhance_cfg.kernel_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # ── Build data bundle & run inference ───────────────────────
    bundle = build_depth_enhancement_data_bundle(enhance_cfg)
    depth_cfg = bundle.depth_cfg
    dataset = bundle.dataset_bundle.inference_dataset
    geometry = bundle.dataset_bundle.geometry
    sample_axis_m = _sample_axis_depth_m(geometry)
    n_il = int(geometry["n_il"])
    n_xl = int(geometry["n_xl"])
    n_sample = int(geometry["n_sample"])

    base_flat = dataset.ai_lfm_flat.astype(np.float32, copy=True)
    mask_flat = dataset._mask_flat.copy()  # type: ignore[attr-defined]
    enhanced_flat = base_flat.copy()
    delta_flat = np.zeros_like(base_flat, dtype=np.float32)

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=enhance_cfg.batch_size * 2,
        shuffle=False,
        num_workers=enhance_cfg.num_workers,
        pin_memory=enhance_cfg.pin_memory,
    )
    offset = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["input"].to(device)
            base_ai = batch["lfm_raw"].to(device)
            taper = batch["taper_weight"].to(device)
            delta = model(x)
            if enhance_cfg.zero_delta_outside_mask:
                delta = delta * taper.to(dtype=delta.dtype)
            enhanced = compose_enhanced_ai(base_ai, delta, ai_min=enhance_cfg.ai_min, ai_max=enhance_cfg.ai_max)
            n_batch = enhanced.shape[0]
            indices = dataset.valid_indices[offset : offset + n_batch]
            enhanced_flat[indices] = enhanced.squeeze(1).cpu().numpy()
            delta_flat[indices] = delta.squeeze(1).cpu().numpy()
            offset += n_batch

    enhanced_volume = enhanced_flat.reshape(n_il, n_xl, n_sample)
    delta_volume = delta_flat.reshape(n_il, n_xl, n_sample)
    base_volume = base_flat.reshape(n_il, n_xl, n_sample)
    mask_volume = mask_flat.reshape(n_il, n_xl, n_sample)

    # ── Export NPZ ──────────────────────────────────────────────
    enhanced_npz = output_dir / "enhanced_ai_depth.npz"

    prediction_stats = {
        "enhanced_ai": _stats(_sampled_values(enhanced_volume)),
        "base_ai": _stats(_sampled_values(base_volume)),
        "delta_log_ai": _stats(_sampled_values(delta_volume)),
        "enhanced_minus_base": _stats(_sampled_difference(enhanced_volume, base_volume)),
        "mask_coverage": float(mask_volume.mean()),
    }
    metadata = {
        "source_script": Path(__file__).name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "base_ai_file": str(enhance_cfg.base_ai_file),
        "prediction_stats": prediction_stats,
    }
    _save_prediction_npz(enhanced_npz, volume=enhanced_volume, geometry=geometry, samples=sample_axis_m, metadata=metadata)

    # ── QC figures ──────────────────────────────────────────────
    slice_mode = str(script_cfg.get("slice_mode", "inline"))
    raw_slice_index = script_cfg.get("slice_index")
    slice_index = None if raw_slice_index is None else int(raw_slice_index)
    resolved_slice_index = _resolve_slice_index(slice_mode, slice_index, geometry)
    clip_values = tuple(float(v) for v in script_cfg.get("clip_percentiles", [1.0, 99.0]))
    if len(clip_values) != 2:
        raise ValueError("clip_percentiles must contain exactly two values.")

    enhanced_section = _extract_section(enhanced_volume, slice_mode, resolved_slice_index)
    base_section = _extract_section(base_volume, slice_mode, resolved_slice_index)
    delta_section = _extract_section(delta_volume, slice_mode, resolved_slice_index)
    mask_section = _extract_mask_section(mask_volume, slice_mode, resolved_slice_index)

    shared_vmin, shared_vmax = _robust_limits(enhanced_section, base_section, percentiles=clip_values)
    delta_abs = float(np.percentile(np.abs(delta_section[np.isfinite(delta_section)]), clip_values[1]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    im0 = axes[0].imshow(
        enhanced_section, cmap="viridis", aspect="auto", origin="upper", vmin=shared_vmin, vmax=shared_vmax
    )
    axes[0].set_title(f"Enhanced AI | {slice_mode}={resolved_slice_index}")
    axes[0].set_xlabel("Trace index")
    axes[0].set_ylabel("Sample index")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = axes[1].imshow(base_section, cmap="viridis", aspect="auto", origin="upper", vmin=shared_vmin, vmax=shared_vmax)
    axes[1].set_title(f"Base AI (Stage-1) | {slice_mode}={resolved_slice_index}")
    axes[1].set_xlabel("Trace index")
    axes[1].set_ylabel("Sample index")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    im2 = axes[2].imshow(delta_section, cmap="seismic", aspect="auto", origin="upper", vmin=-delta_abs, vmax=delta_abs)
    axes[2].set_title(f"Delta log-AI | {slice_mode}={resolved_slice_index}")
    axes[2].set_xlabel("Trace index")
    axes[2].set_ylabel("Sample index")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)
    slice_figure_path = output_dirs["qc_figures"] / f"{slice_mode}_{resolved_slice_index:04d}_enhanced_vs_base.png"
    fig.savefig(slice_figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    im = ax.imshow(np.where(mask_section, 1.0, 0.0), cmap="gray_r", aspect="auto", origin="upper", vmin=0.0, vmax=1.0)
    ax.set_title(f"Mask | {slice_mode}={resolved_slice_index}")
    ax.set_xlabel("Trace index")
    ax.set_ylabel("Sample index")
    fig.colorbar(im, ax=ax, shrink=0.85)
    mask_figure_path = output_dirs["qc_figures"] / f"{slice_mode}_{resolved_slice_index:04d}_mask.png"
    fig.savefig(mask_figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ── SEG-Y export ────────────────────────────────────────────
    enhanced_segy_path = None
    if not args.skip_segy and bool(script_cfg.get("export_segy", True)):
        import cigsegy

        orig_segy_path = depth_cfg.seismic_file
        if not orig_segy_path.is_absolute():
            orig_segy_path = (REPO_ROOT / orig_segy_path).resolve()
        if not orig_segy_path.exists():
            raise FileNotFoundError(f"Original SEG-Y not found: {orig_segy_path}")

        enhanced_segy_path = output_dir / "enhanced_ai_depth.segy"
        textual = build_segy_textual_header(
            "Enhance stage-2 enhanced AI depth volume",
            [
                f"checkpoint={checkpoint_path.name}",
                f"epoch={metadata['checkpoint_epoch']}",
                f"base_ai_file={enhance_cfg.base_ai_file}",
            ],
        )
        cigsegy.create_by_sharing_header(
            str(enhanced_segy_path),
            str(orig_segy_path),
            np.ascontiguousarray(enhanced_volume.astype(np.float32)),
            keylocs=[depth_cfg.segy_iline, depth_cfg.segy_xline, depth_cfg.segy_istep, depth_cfg.segy_xstep],
            textual=textual,
        )


    # ── Well QC ─────────────────────────────────────────────────
    well_qc_metrics_path = None
    n_wells_qc = 0
    well_qc_summary: dict[str, float | None] = {"mean_rmse": None, "mean_mae": None, "mean_corr": None}
    if not args.skip_well_qc and bool(script_cfg.get("well_qc_enabled", True)):
        source_batch_dir = resolve_relative_path(str(script_cfg["source_batch_dir"]), root=REPO_ROOT)
        shifted_las_dir = resolve_relative_path(
            str(script_cfg.get("shifted_las_dir", "shifted_las")), root=source_batch_dir
        )
        data_root = resolve_relative_path(str(cfg.get("data_root", "data")), root=REPO_ROOT)
        well_heads_file = resolve_relative_path(str(cfg["well"]["well_heads_file"]), root=data_root)
        if not shifted_las_dir.exists():
            raise FileNotFoundError(f"Shifted LAS dir not found: {shifted_las_dir}")
        if not well_heads_file.exists():
            raise FileNotFoundError(f"Well heads file not found: {well_heads_file}")

        survey_ctx = open_survey(
            depth_cfg.seismic_file,
            seismic_type="segy",
            segy_options={
                "iline": depth_cfg.segy_iline,
                "xline": depth_cfg.segy_xline,
                "istep": depth_cfg.segy_istep,
                "xstep": depth_cfg.segy_xstep,
            },
        )
        well_heads = import_well_heads_petrel(well_heads_file)
        well_heads = well_heads.assign(Name_norm=well_heads["Name"].astype(str).str.upper())

        log_filter_params = _load_auto_tie_log_filter_params(cfg)

        metrics: list[dict[str, Any]] = []
        for las_path in sorted(shifted_las_dir.glob("*.las")):
            well_name_norm = las_path.stem.upper()
            head_match = well_heads.loc[well_heads["Name_norm"] == well_name_norm]
            if head_match.empty:
                metrics.append({"well_name": well_name_norm, "status": "failed", "error": "well head not matched"})
                continue

            try:
                head = head_match.iloc[0]
                kb_m = float(head["Well datum value"])
                md, ai_raw, ai_filtered = _read_shifted_las_ai_curves(
                    las_path,
                    log_filter_params=log_filter_params,
                )
                tvdss_raw = np.asarray(md, dtype=float) - kb_m
                valid_md = np.isfinite(tvdss_raw) & np.isfinite(ai_raw) & (ai_raw > 0.0)
                if int(valid_md.sum()) < 2:
                    raise ValueError("too few finite positive AI samples after MD→TVDSS")
                ref_depth, ref_ai_raw = _regularize_depth_curve(tvdss_raw[valid_md], ai_raw[valid_md])
                valid_filtered = np.isfinite(tvdss_raw) & np.isfinite(ai_filtered) & (ai_filtered > 0.0)
                ref_depth_filtered, ref_ai_filtered = _regularize_depth_curve(
                    tvdss_raw[valid_filtered],
                    ai_filtered[valid_filtered],
                )
                ref_ai_raw_sampled = _downsample_curve_to_axis(ref_depth, ref_ai_raw, sample_axis_m)
                ref_ai_filtered_sampled = _downsample_curve_to_axis(
                    ref_depth_filtered,
                    ref_ai_filtered,
                    sample_axis_m,
                )

                pred_trace = _bilinear_trace_from_volume(
                    enhanced_volume,
                    survey_ctx,
                    float(head["Surface X"]),
                    float(head["Surface Y"]),
                )
                pred_ai = np.asarray(pred_trace, dtype=np.float32)
                valid = np.isfinite(pred_ai) & np.isfinite(ref_ai_filtered_sampled)
                if not np.any(valid):
                    raise ValueError("no overlapping finite samples")

                diff = pred_ai[valid] - ref_ai_filtered_sampled[valid]
                corr = (
                    float(np.corrcoef(pred_ai[valid], ref_ai_filtered_sampled[valid])[0, 1])
                    if int(valid.sum()) > 1
                    else np.nan
                )
                valid_raw = np.isfinite(pred_ai) & np.isfinite(ref_ai_raw_sampled)
                diff_raw = pred_ai[valid_raw] - ref_ai_raw_sampled[valid_raw]
                corr_raw = (
                    float(np.corrcoef(pred_ai[valid_raw], ref_ai_raw_sampled[valid_raw])[0, 1])
                    if int(valid_raw.sum()) > 1
                    else np.nan
                )
                safe_name = sanitize_filename(str(head["Name"]))
                trace_qc_path = output_dirs["well_qc_traces"] / f"well_qc_{safe_name}.csv"
                pd.DataFrame(
                    {
                        "depth_m": sample_axis_m[valid],
                        "shifted_las_ai_downsampled": ref_ai_raw_sampled[valid],
                        "shifted_las_ai_filtered_downsampled": ref_ai_filtered_sampled[valid],
                        "enhanced_ai": pred_ai[valid],
                        "diff_ai": diff,
                    }
                ).to_csv(trace_qc_path, index=False)

                qc_plot_path = output_dirs["well_qc_figures"] / f"well_qc_{safe_name}.png"
                fig, ax = plt.subplots(figsize=(5, 10), constrained_layout=True)
                raw_plot_mask = (ref_depth >= sample_axis_m[valid][0]) & (ref_depth <= sample_axis_m[valid][-1])
                filtered_plot_mask = (ref_depth_filtered >= sample_axis_m[valid][0]) & (
                    ref_depth_filtered <= sample_axis_m[valid][-1]
                )
                ax.plot(
                    ref_ai_raw[raw_plot_mask],
                    ref_depth[raw_plot_mask],
                    label="Shifted LAS AI",
                    lw=0.8,
                    alpha=0.35,
                    color="gray",
                )
                ax.plot(
                    ref_ai_filtered[filtered_plot_mask],
                    ref_depth_filtered[filtered_plot_mask],
                    label="Shifted LAS AI (filtered)",
                    lw=1.6,
                    color="blue",
                )
                ax.plot(pred_ai[valid], sample_axis_m[valid], label="Enhanced predicted AI", lw=2, color="red")
                ax.invert_yaxis()
                ax.set_xlabel("AI")
                ax.set_ylabel("Depth (m)")
                ax.set_title(f"Well QC | {head['Name']}")
                ax.legend(loc="best")
                ax.grid(True, alpha=0.3, linestyle=":")
                fig.savefig(qc_plot_path, dpi=180, bbox_inches="tight")
                plt.close(fig)

                metrics.append(
                    {
                        "well_name": head["Name"],
                        "status": "ok",
                        "n_samples": int(np.sum(valid)),
                        "mae": float(np.mean(np.abs(diff))),
                        "rmse": float(np.sqrt(np.mean(diff**2))),
                        "bias": float(np.mean(diff)),
                        "corr": corr,
                        "mae_raw_downsampled": float(np.mean(np.abs(diff_raw))) if diff_raw.size else np.nan,
                        "rmse_raw_downsampled": float(np.sqrt(np.mean(diff_raw**2))) if diff_raw.size else np.nan,
                        "bias_raw_downsampled": float(np.mean(diff_raw)) if diff_raw.size else np.nan,
                        "corr_raw_downsampled": corr_raw,
                        "log_filter_params_json": json.dumps(log_filter_params, ensure_ascii=False),
                        "reference_file": las_path.name,
                        "trace_qc_path": str(trace_qc_path),
                        "figure_path": str(qc_plot_path),
                    }
                )
            except Exception as exc:
                metrics.append({"well_name": well_name_norm, "status": "failed", "error": str(exc)})

        metrics_df = pd.DataFrame(metrics)
        if not metrics_df.empty and "rmse" in metrics_df:
            ok = metrics_df["status"] == "ok"
            metrics_df = pd.concat(
                [metrics_df.loc[ok].sort_values("rmse"), metrics_df.loc[~ok]],
                ignore_index=True,
            )
        well_qc_metrics_path = output_dirs["well_qc"] / "well_qc_metrics.csv"
        metrics_df.to_csv(well_qc_metrics_path, index=False)
        ok_metrics = metrics_df.loc[metrics_df["status"] == "ok"] if "status" in metrics_df else pd.DataFrame()
        n_wells_qc = int(len(ok_metrics))
        if not ok_metrics.empty:
            well_qc_summary = {
                "mean_rmse": float(ok_metrics["rmse"].mean()),
                "mean_mae": float(ok_metrics["mae"].mean()),
                "mean_corr": float(ok_metrics["corr"].mean()),
            }

    # ── Run summary ─────────────────────────────────────────────
    run_summary = {
        "checkpoint_path": str(checkpoint_path),
        "epoch": metadata["checkpoint_epoch"],
        "device": str(device),
        "enhanced_npz": str(enhanced_npz),
        "enhanced_segy_path": None if enhanced_segy_path is None else str(enhanced_segy_path),
        "slice_mode": slice_mode,
        "slice_index": int(resolved_slice_index),
        "slice_figure_path": str(slice_figure_path),
        "mask_figure_path": str(mask_figure_path),
        "well_qc_metrics_path": None if well_qc_metrics_path is None else str(well_qc_metrics_path),
        "n_il": n_il,
        "n_xl": n_xl,
        "n_sample": n_sample,
        "sample_min": float(geometry["sample_min"]),
        "sample_max": float(geometry["sample_max"]),
        "sample_step": float(geometry["sample_step"]),
        "n_wells_qc": n_wells_qc,
        "well_qc": well_qc_summary,
        "prediction_stats": prediction_stats,
    }
    run_summary_path = output_dirs["metadata"] / "run_summary.json"
    with run_summary_path.open("w", encoding="utf-8") as fp:
        json.dump(run_summary, fp, ensure_ascii=False, indent=2)

    print(f"Saved enhanced AI NPZ: {enhanced_npz}")
    if enhanced_segy_path is not None:
        print(f"Saved enhanced AI SEG-Y: {enhanced_segy_path}")
    print(f"Saved QC figures: {output_dirs['qc_figures']}")
    print(f"Saved well QC: {output_dirs['well_qc']}")
    print(f"Saved run summary: {run_summary_path}")


if __name__ == "__main__":
    main()
