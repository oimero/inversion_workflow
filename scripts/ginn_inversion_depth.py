"""Run depth-domain GINN inference and export a stage-1 base-AI volume.

This is the scripted form of ``notebooks/ginn_inversion_depth@20260424.ipynb``.
The NPZ output is compatible with ``ginn_depth.data.load_lfm_depth_npz`` and can
be used as ``base_ai_file`` for stage-2 enhancement.

Usage::

    python scripts/ginn_inversion_depth.py
    python scripts/ginn_inversion_depth.py --checkpoint experiments/ginn_depth/results/.../checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Bootstrap


def _find_repo_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    if not (root / "src").exists():
        root = Path.cwd().resolve()
    if not (root / "src").exists():
        raise RuntimeError("Could not locate repo root containing 'src'.")
    return root


def _ensure_import_path(src_root: Path) -> None:
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


_ensure_import_path(_find_repo_root() / "src")

from cup.petrel.load import import_well_heads_petrel  # noqa: E402
from cup.seismic.survey import open_survey  # noqa: E402
from cup.utils.io import (  # noqa: E402
    build_segy_textual_header,
    load_yaml_config,
    resolve_relative_path,
    sanitize_filename,
)
from ginn_depth.config import DepthGINNConfig  # noqa: E402
from ginn_depth.trainer import Trainer  # noqa: E402

# CLI


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
        help="Output directory. Defaults to <output_root>/ginn_inversion_depth_<timestamp>.",
    )
    parser.add_argument("--skip-segy", action="store_true", help="Skip SEG-Y export.")
    parser.add_argument("--skip-well-qc", action="store_true", help="Skip LAS-vs-prediction well QC.")
    return parser.parse_args()


# Array/QC helpers


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


def _collect_las_references(las_dir: Path) -> dict[str, Path]:
    refs: dict[str, Path] = {}
    for las_path in sorted(las_dir.glob("*.las")):
        refs[las_path.stem.upper()] = las_path
    return refs


def _extract_ai_reference_from_las(
    las_path: Path,
    *,
    vp_unit: str,
    rho_unit: str,
    kb_m: float,
    depth_domain: str,
    log_filter_params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    from cup.petrel.load import load_vp_rho_logset_from_las
    from wtie.optimize import tie as tie_utils
    from wtie.processing.logs import interpolate_nans

    logset_md = load_vp_rho_logset_from_las(las_path, vp_unit=vp_unit, rho_unit=rho_unit)
    filtered_logset_md = tie_utils.filter_md_logs(
        logset_md,
        median_size=int(log_filter_params["logs_median_size"]),
        threshold=float(log_filter_params["logs_median_threshold"]),
        std=float(log_filter_params["logs_std"]),
        std2=0.8 * float(log_filter_params["logs_std"]),
    )
    depth = np.asarray(filtered_logset_md.basis, dtype=np.float64)
    vp = interpolate_nans(np.asarray(filtered_logset_md.Vp.values, dtype=np.float64), method="linear")
    rho = interpolate_nans(np.asarray(filtered_logset_md.Rho.values, dtype=np.float64), method="linear")
    if depth_domain == "tvdss":
        depth = depth - float(kb_m)
    elif depth_domain != "md":
        raise ValueError(f"well_qc_depth_domain must be 'tvdss' or 'md', got {depth_domain!r}")
    ai = vp * rho
    finite = np.isfinite(depth) & np.isfinite(ai) & (ai > 0.0)
    return depth[finite], ai[finite]


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


# Main


def main() -> None:
    args = parse_args()
    repo_root = _find_repo_root()
    cfg = load_yaml_config(args.config, base_dir=repo_root)
    script_cfg = cfg.get("ginn_inversion_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'ginn_inversion_depth' section in config.")

    output_root = resolve_relative_path(str(cfg.get("output_root", "scripts/output")), root=repo_root)
    checkpoint_path = args.checkpoint or Path(script_cfg["checkpoint_path"])
    checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else (repo_root / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"ginn_inversion_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
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

    print("=== GINN Inversion (Depth) ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_payload = checkpoint["config"]
    depth_cfg = (
        DepthGINNConfig.from_dict(cfg_payload, base_dir=repo_root) if isinstance(cfg_payload, dict) else cfg_payload
    )
    depth_cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(depth_cfg)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.eval()

    pred_volume = trainer.predict_volume()
    n_il = int(trainer.geometry["n_il"])
    n_xl = int(trainer.geometry["n_xl"])
    n_sample = int(trainer.geometry["n_sample"])
    lfm_volume = trainer.dataset.ai_lfm_flat.reshape(n_il, n_xl, n_sample)
    mask_volume = trainer.dataset._mask_flat.reshape(n_il, n_xl, n_sample)

    base_ai_npz = output_dir / "stage1_ginn_base_ai_depth.npz"

    prediction_stats = {
        "prediction_ai": _stats(_sampled_values(pred_volume)),
        "lfm_ai": _stats(_sampled_values(lfm_volume)),
        "prediction_minus_lfm": _stats(_sampled_difference(pred_volume, lfm_volume)),
        "mask_coverage": float(mask_volume.mean()),
    }
    metadata = {
        "artifact": base_ai_npz.name,
        "source_script": Path(__file__).name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_best_loss": float(checkpoint.get("best_loss", np.nan)),
        "intended_usage": "Set experiments/enhance_depth/train.yaml base_ai_file to this NPZ for stage-2 enhancement.",
        "prediction_stats": prediction_stats,
    }
    _save_prediction_npz(
        base_ai_npz,
        volume=pred_volume,
        geometry=trainer.geometry,
        samples=trainer.depth_axis_m,
        metadata=metadata,
    )

    slice_mode = str(script_cfg.get("slice_mode", "inline"))
    raw_slice_index = script_cfg.get("slice_index")
    slice_index = None if raw_slice_index is None else int(raw_slice_index)
    resolved_slice_index = _resolve_slice_index(slice_mode, slice_index, trainer.geometry)
    clip_values = tuple(float(v) for v in script_cfg.get("clip_percentiles", [1.0, 99.0]))
    if len(clip_values) != 2:
        raise ValueError("clip_percentiles must contain exactly two values.")

    pred_section = _extract_section(pred_volume, slice_mode, resolved_slice_index)
    lfm_section = _extract_section(lfm_volume, slice_mode, resolved_slice_index)
    mask_section = _extract_mask_section(mask_volume, slice_mode, resolved_slice_index)
    diff_section = pred_section - lfm_section
    auto_vmin, auto_vmax = _robust_limits(pred_section, lfm_section, percentiles=clip_values)  # type: ignore[arg-type]
    ai_display_min = script_cfg.get("ai_display_min", None)
    ai_display_max = script_cfg.get("ai_display_max", None)
    shared_vmin = auto_vmin if ai_display_min is None else float(ai_display_min)
    shared_vmax = auto_vmax if ai_display_max is None else float(ai_display_max)
    diff_abs = float(np.percentile(np.abs(diff_section[np.isfinite(diff_section)]), clip_values[1]))

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    im0 = axes[0].imshow(
        pred_section, cmap="viridis", aspect="auto", origin="upper", vmin=shared_vmin, vmax=shared_vmax
    )
    axes[0].set_title(f"Predicted AI | {slice_mode}={resolved_slice_index}")
    axes[0].set_xlabel("Trace index")
    axes[0].set_ylabel("Sample index")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = axes[1].imshow(lfm_section, cmap="viridis", aspect="auto", origin="upper", vmin=shared_vmin, vmax=shared_vmax)
    axes[1].set_title(f"LFM | {slice_mode}={resolved_slice_index}")
    axes[1].set_xlabel("Trace index")
    axes[1].set_ylabel("Sample index")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    im2 = axes[2].imshow(diff_section, cmap="seismic", aspect="auto", origin="upper", vmin=-diff_abs, vmax=diff_abs)
    axes[2].set_title(f"Prediction - LFM | {slice_mode}={resolved_slice_index}")
    axes[2].set_xlabel("Trace index")
    axes[2].set_ylabel("Sample index")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)
    slice_figure_path = output_dirs["qc_figures"] / f"{slice_mode}_{resolved_slice_index:04d}_prediction_vs_lfm.png"
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

    pred_segy_path = None
    if not args.skip_segy and bool(script_cfg.get("export_segy", True)):
        import cigsegy

        orig_segy_path = Path(depth_cfg.seismic_file)
        if not orig_segy_path.exists():
            raise FileNotFoundError(f"Original SEG-Y not found: {orig_segy_path}")
        pred_segy_path = output_dir / "stage1_ginn_base_ai_depth.segy"
        textual = build_segy_textual_header(
            "GINN-Depth predicted impedance volume",
            [
                f"artifact={base_ai_npz.name}",
                f"checkpoint={checkpoint_path.name}",
                f"epoch={metadata['checkpoint_epoch']}",
                f"best_loss={metadata['checkpoint_best_loss']:.6g}",
            ],
        )
        cigsegy.create_by_sharing_header(
            str(pred_segy_path),
            str(orig_segy_path),
            np.ascontiguousarray(pred_volume.astype(np.float32)),
            keylocs=[depth_cfg.segy_iline, depth_cfg.segy_xline, depth_cfg.segy_istep, depth_cfg.segy_xstep],
            textual=textual,
        )

    well_qc_metrics_path = None
    n_wells_qc = 0
    well_qc_summary: dict[str, float | None] = {"mean_rmse": None, "mean_mae": None, "mean_corr": None}
    if not args.skip_well_qc and bool(script_cfg.get("well_qc_enabled", True)):
        data_root = resolve_relative_path(str(cfg.get("data_root", "data")), root=repo_root)
        las_dir = resolve_relative_path(
            str(script_cfg.get("well_qc_las_dir", "vertical_well_las_target_qyz")), root=data_root
        )
        well_heads_file = resolve_relative_path(str(cfg["well"]["well_heads_file"]), root=data_root)
        if not las_dir.exists():
            raise FileNotFoundError(f"Well QC LAS dir not found: {las_dir}")
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
        sample_axis_m = _sample_axis_depth_m(trainer.geometry)
        well_heads = import_well_heads_petrel(well_heads_file)
        well_heads = well_heads.assign(Name_norm=well_heads["Name"].astype(str).str.upper())
        reference_files = _collect_las_references(las_dir)
        vp_unit = str(script_cfg.get("well_qc_las_vp_unit", "us/m"))
        rho_unit = str(script_cfg.get("well_qc_las_rho_unit", "g/cm3"))
        depth_domain = str(script_cfg.get("well_qc_depth_domain", "tvdss"))
        log_filter_params = script_cfg.get("well_qc_log_filter", {})
        required_filter_keys = ("logs_median_size", "logs_median_threshold", "logs_std")
        missing_filter_keys = [key for key in required_filter_keys if key not in log_filter_params]
        if missing_filter_keys:
            raise ValueError(f"Missing well_qc_log_filter keys: {missing_filter_keys}")

        metrics: list[dict[str, Any]] = []
        for well_name_norm, las_path in reference_files.items():
            head_match = well_heads.loc[well_heads["Name_norm"] == well_name_norm]
            if head_match.empty:
                metrics.append({"well_name": well_name_norm, "status": "failed", "error": "well head not matched"})
                continue

            try:
                head = head_match.iloc[0]
                kb_m = float(head["Well datum value"])
                ref_depth, ref_ai = _extract_ai_reference_from_las(
                    las_path,
                    vp_unit=vp_unit,
                    rho_unit=rho_unit,
                    kb_m=kb_m,
                    depth_domain=depth_domain,
                    log_filter_params=log_filter_params,
                )
                pred_trace = _bilinear_trace_from_volume(
                    pred_volume,
                    survey_ctx,
                    float(head["Surface X"]),
                    float(head["Surface Y"]),
                )
                pred_ai = np.interp(ref_depth, sample_axis_m, pred_trace, left=np.nan, right=np.nan)
                valid = np.isfinite(pred_ai) & np.isfinite(ref_ai)
                if not np.any(valid):
                    raise ValueError("no overlapping finite samples")

                diff = pred_ai[valid] - ref_ai[valid]
                corr = float(np.corrcoef(pred_ai[valid], ref_ai[valid])[0, 1]) if int(valid.sum()) > 1 else np.nan
                safe_name = sanitize_filename(str(head["Name"]))
                trace_qc_path = output_dirs["well_qc_traces"] / f"well_qc_{safe_name}.csv"
                pd.DataFrame(
                    {
                        "depth_m": ref_depth[valid],
                        "filtered_las_ai": ref_ai[valid],
                        "predicted_ai": pred_ai[valid],
                        "diff_ai": diff,
                    }
                ).to_csv(trace_qc_path, index=False)

                qc_plot_path = output_dirs["well_qc_figures"] / f"well_qc_{safe_name}.png"
                fig, ax = plt.subplots(figsize=(5, 10), constrained_layout=True)
                ax.plot(ref_ai[valid], ref_depth[valid], label="Filtered LAS AI", lw=2)
                ax.plot(pred_ai[valid], ref_depth[valid], label="GINN-Depth predicted AI", lw=2)
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
                        "reference_file": las_path.name,
                        "depth_domain": depth_domain,
                        "log_filter_median_size": int(log_filter_params["logs_median_size"]),
                        "log_filter_threshold": float(log_filter_params["logs_median_threshold"]),
                        "log_filter_std": float(log_filter_params["logs_std"]),
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

    run_summary = {
        "checkpoint_path": str(checkpoint_path),
        "epoch": metadata["checkpoint_epoch"],
        "best_epoch": metadata["checkpoint_best_epoch"],
        "best_loss": metadata["checkpoint_best_loss"],
        "device": str(trainer.device),
        "base_ai_npz": str(base_ai_npz),
        "pred_segy_path": None if pred_segy_path is None else str(pred_segy_path),
        "slice_mode": slice_mode,
        "slice_index": int(resolved_slice_index),
        "slice_figure_path": str(slice_figure_path),
        "mask_figure_path": str(mask_figure_path),
        "well_qc_metrics_path": None if well_qc_metrics_path is None else str(well_qc_metrics_path),
        "n_il": n_il,
        "n_xl": n_xl,
        "n_sample": n_sample,
        "sample_min": float(trainer.geometry["sample_min"]),
        "sample_max": float(trainer.geometry["sample_max"]),
        "sample_step": float(trainer.geometry["sample_step"]),
        "n_wells_qc": n_wells_qc,
        "well_qc": well_qc_summary,
        "prediction_stats": prediction_stats,
    }
    run_summary_path = output_dirs["metadata"] / "run_summary.json"
    with run_summary_path.open("w", encoding="utf-8") as fp:
        json.dump(run_summary, fp, ensure_ascii=False, indent=2)

    print(f"Saved base AI NPZ: {base_ai_npz}")
    if pred_segy_path is not None:
        print(f"Saved SEG-Y: {pred_segy_path}")
    print(f"Saved QC figures: {output_dirs['qc_figures']}")
    print(f"Saved well QC: {output_dirs['well_qc']}")
    print(f"Saved run summary: {run_summary_path}")


if __name__ == "__main__":
    main()
