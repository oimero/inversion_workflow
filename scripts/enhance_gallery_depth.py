"""Render a gallery of depth stage-2 enhancement synthetic samples.

The script uses the depth enhancement synthetic adapter, then writes figures for
visually checking whether the
generated AI, log-AI residual, reflectivity, and synthetic seismic waveforms are
reasonable.

Outputs:

- ``samples.csv``: one row per synthetic sample with compact metrics.
- ``index.html``: clickable browser gallery.
- ``overview_page_*.png``: compact multi-sample pages for fast scanning.
- ``samples/sample_*.png``: detailed five-panel plots for each sample.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGGER = logging.getLogger("enhance_gallery_depth")

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.statistics import normalized_cross_correlation, rms  # noqa: E402
from cup.utils.io import load_yaml_config, resolve_relative_path  # noqa: E402

# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/enhance_depth/train.yaml"),
        help="Stage-2 enhancement YAML config.",
    )
    parser.add_argument(
        "--common-config",
        type=Path,
        default=Path("experiments/common_depth.yaml"),
        help="Shared depth workflow YAML used for output_root.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of random synthetic samples to render.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260507,
        help="Random seed for NumPy and PyTorch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <output_root>/enhance_gallery_depth_<timestamp>.",
    )
    parser.add_argument("--dpi", type=int, default=130, help="Figure DPI.")
    parser.add_argument(
        "--overview-page-size",
        type=int,
        default=20,
        help="Number of samples in each overview page. Set <=0 to skip overview pages.",
    )
    parser.add_argument(
        "--skip-detail-plots",
        action="store_true",
        help="Only write CSV/index/overview pages; skip one detailed PNG per sample.",
    )
    parser.add_argument(
        "--show-clean-target",
        action="store_true",
        help="Show clean target_seismic in the seismic panel in addition to synthetic raw.",
    )
    parser.add_argument(
        "--show-dirty-input",
        action="store_true",
        help="Show augmented input_seismic in the seismic panel and overview pages.",
    )
    parser.add_argument(
        "--main-lobe-samples",
        type=int,
        default=None,
        help="Override synthetic_cluster_main_lobe_samples for this run.",
    )
    parser.add_argument(
        "--patch-fraction",
        type=float,
        default=None,
        help="Override synthetic_patch_fraction for this run.",
    )
    parser.add_argument(
        "--unresolved-fraction",
        type=float,
        default=None,
        help="Override synthetic_unresolved_fraction for this run.",
    )
    parser.add_argument(
        "--well-patch-scale-min",
        type=float,
        default=None,
        help="Override synthetic_well_patch_scale_min for this run.",
    )
    parser.add_argument(
        "--well-patch-scale-max",
        type=float,
        default=None,
        help="Override synthetic_well_patch_scale_max for this run.",
    )
    parser.add_argument(
        "--cluster-amp-abs-p95-min",
        type=float,
        default=None,
        help="Override synthetic_cluster_amp_abs_p95_min for this run.",
    )
    parser.add_argument(
        "--cluster-amp-abs-p99-max",
        type=float,
        default=None,
        help="Override synthetic_cluster_amp_abs_p99_max for this run.",
    )
    parser.add_argument(
        "--unresolved-oversample-factor",
        type=int,
        default=None,
        help="Override synthetic_unresolved_oversample_factor for this run.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    file_handler = logging.FileHandler(output_dir / "gallery.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def as_1d(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).squeeze()


def finite_core(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    valid = mask & np.isfinite(values)
    return values[valid]


def abs_percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(np.abs(values), percentile))


def safe_ratio(num: float, denom: float) -> float:
    if not np.isfinite(num) or not np.isfinite(denom) or abs(denom) < 1e-12:
        return float("nan")
    return float(num / denom)


def mode_name(sample: dict[str, Any]) -> str:
    return "well_patch" if int(sample["synthetic_mode"].item()) == 0 else "unresolved_cluster"


def sample_record(sample: dict[str, Any], index: int, image_path: Path | None) -> dict[str, Any]:
    core_mask = as_1d(sample["mask"]).astype(bool)
    waveform_mask = core_mask
    delta_mask = as_1d(sample["delta_loss_mask"]).astype(bool)
    real = finite_core(as_1d(sample["obs"]), waveform_mask)
    synthetic = finite_core(as_1d(sample["target_seismic"]), waveform_mask)
    synthetic_raw = finite_core(as_1d(sample["target_seismic_raw"]), waveform_mask)
    input_clean_full = as_1d(sample.get("input_seismic_clean", sample["target_seismic"]))
    input_augmented_full = as_1d(sample.get("input_seismic_augmented", sample["target_seismic"]))
    input_clean = finite_core(input_clean_full, waveform_mask)
    input_augmented = finite_core(input_augmented_full, waveform_mask)
    input_delta = finite_core(input_augmented_full - input_clean_full, waveform_mask)
    base_seismic_full = (
        as_1d(sample["base_seismic"]) if "base_seismic" in sample else np.zeros_like(as_1d(sample["target_seismic"]))
    )
    base_seismic = finite_core(base_seismic_full, waveform_mask)
    base_target_delta_rms = rms(finite_core(as_1d(sample["target_seismic"]) - base_seismic_full, waveform_mask))
    residual = finite_core(as_1d(sample["target_residual"]), delta_mask)
    ai = finite_core(as_1d(sample["target_ai"]), delta_mask)
    base_ai = finite_core(as_1d(sample["base_ai_raw"]), delta_mask)
    real_rms = rms(real)
    synthetic_rms = rms(synthetic)
    real_abs_p99 = abs_percentile(real, 99.0)
    synthetic_abs_p99 = abs_percentile(synthetic, 99.0)
    residual_rms = rms(residual)
    return {
        "sample_index": index,
        "mode": mode_name(sample),
        "real_rms": real_rms,
        "synthetic_rms": synthetic_rms,
        "synthetic_raw_rms": rms(synthetic_raw),
        "synthetic_to_real_rms": safe_ratio(synthetic_rms, real_rms),
        "real_abs_p99": real_abs_p99,
        "synthetic_abs_p99": synthetic_abs_p99,
        "synthetic_to_real_abs_p99": safe_ratio(synthetic_abs_p99, real_abs_p99),
        "target_obs_waveform_corr": normalized_cross_correlation(synthetic, real),
        "input_obs_waveform_corr": normalized_cross_correlation(input_augmented, real),
        "input_to_clean_rms_ratio": safe_ratio(rms(input_augmented), rms(input_clean)),
        "input_augmentation_delta_rms_fraction": safe_ratio(rms(input_delta), rms(input_clean)),
        "base_target_waveform_corr": normalized_cross_correlation(base_seismic, synthetic),
        "base_target_waveform_delta_rms_to_target_rms": safe_ratio(base_target_delta_rms, synthetic_rms),
        "residual_rms": residual_rms,
        "residual_abs_p95": abs_percentile(residual, 95.0),
        "residual_abs_p99": abs_percentile(residual, 99.0),
        "residual_abs_max": abs_percentile(residual, 100.0),
        "ai_min": float(np.nanmin(ai)) if ai.size else float("nan"),
        "ai_max": float(np.nanmax(ai)) if ai.size else float("nan"),
        "base_ai_min": float(np.nanmin(base_ai)) if base_ai.size else float("nan"),
        "base_ai_max": float(np.nanmax(base_ai)) if base_ai.size else float("nan"),
        "core_mask_fraction": float(np.mean(core_mask)),
        "waveform_mask_fraction": float(np.mean(waveform_mask)),
        "delta_mask_fraction": float(np.mean(delta_mask)),
        "rms_scale": float(sample["synthetic_rms_scale"].item()),
        "resample_attempts": int(sample.get("synthetic_resample_attempts").item())
        if "synthetic_resample_attempts" in sample
        else 1,
        "image": image_path.as_posix() if image_path is not None else "",
    }


def shade_mask(ax: plt.Axes, x: np.ndarray, mask: np.ndarray, *, label: str | None = None) -> None:
    if x.size != mask.size or mask.size == 0:
        return
    y0, y1 = ax.get_ylim()
    ax.fill_between(x, y0, y1, where=mask, color="0.90", alpha=0.55, linewidth=0, zorder=0, label=label)
    ax.set_ylim(y0, y1)


def plot_detail(
    sample: dict[str, Any],
    depth: np.ndarray,
    index: int,
    path: Path,
    dpi: int,
    *,
    show_clean_target: bool = False,
    show_dirty_input: bool = False,
) -> None:
    target_seismic = as_1d(sample["target_seismic"])
    input_augmented = as_1d(sample.get("input_seismic_augmented", sample["target_seismic"]))
    target_seismic_raw = as_1d(sample["target_seismic_raw"])
    base_seismic = as_1d(sample["base_seismic"]) if "base_seismic" in sample else None
    real_obs = as_1d(sample["obs"])
    base_ai = as_1d(sample["base_ai_raw"])
    ai = as_1d(sample["target_ai"])
    residual = as_1d(sample["target_residual"])
    reflectivity = as_1d(sample["raw_reflectivity"])
    taper = as_1d(sample["taper_weight"])
    core_mask = as_1d(sample["mask"]).astype(bool)
    waveform_mask = core_mask
    delta_mask = as_1d(sample["delta_loss_mask"]).astype(bool)
    sample_mode = mode_name(sample)
    highres_depth = as_1d(sample["depth_highres"]) if "depth_highres" in sample else None
    highres_ai = as_1d(sample["target_ai_highres"]) if "target_ai_highres" in sample else None
    highres_reflectivity = as_1d(sample["raw_reflectivity_highres"]) if "raw_reflectivity_highres" in sample else None

    fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(depth, real_obs, color="0.58", lw=1.0, label="real normalized seismic")
    if base_seismic is not None:
        axes[0].plot(depth, base_seismic, color="tab:blue", lw=0.95, alpha=0.90, label="base forward")
    axes[0].plot(depth, target_seismic_raw, color="tab:orange", lw=0.9, alpha=0.75, label="synthetic raw")
    if show_clean_target:
        axes[0].plot(depth, target_seismic, color="tab:green", lw=0.9, alpha=0.80, label="clean target")
    if show_dirty_input:
        axes[0].plot(depth, input_augmented, color="tab:red", lw=0.75, alpha=0.70, label="dirty input")
    axes[0].set_ylabel("seismic")

    axes[1].plot(depth, base_ai, color="0.40", lw=1.0, label="base AI")
    if highres_depth is not None and highres_ai is not None:
        axes[1].plot(highres_depth, highres_ai, color="tab:orange", lw=0.45, alpha=0.55, label="internal high-res AI")
    axes[1].plot(depth, ai, color="tab:red", lw=1.0, label="target AI")
    axes[1].set_ylabel("AI")

    axes[2].plot(depth, residual, color="tab:purple", lw=1.0, label="target logAI residual")
    axes[2].axhline(0.0, color="0.2", lw=0.7)
    axes[2].set_ylabel("logAI residual")

    refl_depth = depth[:-1] if depth.size == reflectivity.size + 1 else np.arange(reflectivity.size)
    if (
        highres_depth is not None
        and highres_reflectivity is not None
        and highres_depth.size == highres_reflectivity.size + 1
    ):
        axes[3].plot(
            highres_depth[:-1], highres_reflectivity, color="0.55", lw=0.35, alpha=0.65, label="internal high-res"
        )
    axes[3].plot(refl_depth, reflectivity, color="tab:green", lw=0.8, label="sampled reflectivity")
    axes[3].axhline(0.0, color="0.2", lw=0.7)
    axes[3].set_ylabel("reflectivity")

    axes[4].plot(depth, taper, color="tab:brown", lw=1.2, label="taper")
    axes[4].fill_between(depth, 0.0, core_mask.astype(float), color="0.75", alpha=0.35, label="core mask")
    axes[4].set_ylim(-0.05, 1.05)
    axes[4].set_ylabel("support")
    axes[4].set_xlabel("depth / m")
    axes[4].legend(loc="upper right", fontsize=8)

    for ax in axes[:4]:
        shade_mask(ax, depth, core_mask)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[4].grid(True, alpha=0.25)

    waveform_synthetic = finite_core(target_seismic, waveform_mask)
    delta_residual = finite_core(residual, delta_mask)
    title = (
        f"sample {index:03d} | mode={sample_mode} | "
        f"synthetic RMS={rms(waveform_synthetic):.3f} | residual abs p99={abs_percentile(delta_residual, 99.0):.3f}"
    )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_overview_page(
    samples: list[dict[str, Any]],
    records: list[dict[str, Any]],
    depth: np.ndarray,
    page_index: int,
    path: Path,
    dpi: int,
    *,
    show_dirty_input: bool = False,
) -> None:
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(15, max(2.0 * n_rows, 3.0)), sharex="col")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (sample, record) in enumerate(zip(samples, records)):
        seismic = as_1d(sample["target_seismic"])
        input_augmented = as_1d(sample.get("input_seismic_augmented", sample["target_seismic"]))
        real_obs = as_1d(sample["obs"])
        ai = as_1d(sample["target_ai"])
        base_ai = as_1d(sample["base_ai_raw"])
        residual = as_1d(sample["target_residual"])
        waveform_mask = as_1d(sample["mask"]).astype(bool)
        delta_mask = as_1d(sample["delta_loss_mask"]).astype(bool)

        axes[row, 0].plot(depth, real_obs, color="0.65", lw=0.8)
        axes[row, 0].plot(depth, seismic, color="tab:blue", lw=1.0)
        if show_dirty_input:
            axes[row, 0].plot(depth, input_augmented, color="tab:red", lw=0.7, alpha=0.75)
        axes[row, 1].plot(depth, base_ai, color="0.45", lw=0.8)
        axes[row, 1].plot(depth, ai, color="tab:red", lw=0.9)
        axes[row, 2].plot(depth, residual, color="tab:purple", lw=0.9)
        axes[row, 2].axhline(0.0, color="0.2", lw=0.5)

        synth = finite_core(seismic, waveform_mask)
        real = finite_core(real_obs, waveform_mask)
        axes[row, 3].scatter([rms(real)], [rms(synth)], s=18, color="tab:blue")
        lim = max(rms(real), rms(synth), 1e-6) * 1.25
        axes[row, 3].plot([0, lim], [0, lim], color="0.35", lw=0.7)
        axes[row, 3].set_xlim(0, lim)
        axes[row, 3].set_ylim(0, lim)

        for col in range(3):
            shade_mask(axes[row, col], depth, delta_mask)
        axes[row, 0].set_ylabel(f"#{record['sample_index']:03d}\n{record['mode']}", fontsize=7)
        for col in range(4):
            axes[row, col].grid(True, alpha=0.20)
            axes[row, col].tick_params(labelsize=7)

    seismic_title = "seismic: real gray, clean blue"
    if show_dirty_input:
        seismic_title += ", input red"
    titles = (seismic_title, "AI: base gray, target red", "logAI delta", "RMS")
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=10)
    for ax in axes[-1, :3]:
        ax.set_xlabel("depth / m")
    axes[-1, 3].set_xlabel("real RMS")
    axes[-1, 3].set_ylabel("synthetic RMS")
    fig.suptitle(f"Depth enhancement synthetic samples overview page {page_index}")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(key: str) -> dict[str, Any]:
        values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {"n": 0, "p50": None, "p95": None, "min": None, "max": None}
        return {
            "n": int(values.size),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    modes = sorted({record["mode"] for record in records})
    return {
        "n_samples": len(records),
        "mode_counts": {mode: sum(1 for record in records if record["mode"] == mode) for mode in modes},
        "synthetic_to_real_rms": stats("synthetic_to_real_rms"),
        "synthetic_to_real_abs_p99": stats("synthetic_to_real_abs_p99"),
        "target_obs_waveform_corr": stats("target_obs_waveform_corr"),
        "input_obs_waveform_corr": stats("input_obs_waveform_corr"),
        "input_to_clean_rms_ratio": stats("input_to_clean_rms_ratio"),
        "input_augmentation_delta_rms_fraction": stats("input_augmentation_delta_rms_fraction"),
        "base_target_waveform_corr": stats("base_target_waveform_corr"),
        "base_target_waveform_delta_rms_to_target_rms": stats("base_target_waveform_delta_rms_to_target_rms"),
        "residual_rms": stats("residual_rms"),
        "residual_abs_p99": stats("residual_abs_p99"),
        "core_mask_fraction": stats("core_mask_fraction"),
        "waveform_mask_fraction": stats("waveform_mask_fraction"),
        "delta_mask_fraction": stats("delta_mask_fraction"),
        "rms_scale": stats("rms_scale"),
        "resample_attempts": stats("resample_attempts"),
    }


def write_index_html(path: Path, records: list[dict[str, Any]], overview_paths: list[Path]) -> None:
    rows = []
    for record in records:
        image = html.escape(record["image"])
        link = (
            f'<a href="{image}">sample {int(record["sample_index"]):03d}</a>'
            if image
            else f"sample {int(record['sample_index']):03d}"
        )
        rows.append(
            "<tr>"
            f"<td>{link}</td>"
            f"<td>{html.escape(str(record['mode']))}</td>"
            f"<td>{float(record['synthetic_to_real_rms']):.3f}</td>"
            f"<td>{float(record['synthetic_to_real_abs_p99']):.3f}</td>"
            f"<td>{float(record['target_obs_waveform_corr']):.3f}</td>"
            f"<td>{float(record['input_obs_waveform_corr']):.3f}</td>"
            f"<td>{float(record['base_target_waveform_corr']):.3f}</td>"
            f"<td>{float(record['residual_abs_p99']):.4f}</td>"
            f"<td>{float(record['rms_scale']):.3f}</td>"
            f"<td>{int(record['resample_attempts'])}</td>"
            "</tr>"
        )

    overview_links = "\n".join(
        f'<li><a href="{html.escape(path_item.as_posix())}">{html.escape(path_item.name)}</a></li>'
        for path_item in overview_paths
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Depth Enhancement Synthetic Gallery</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #f2f2f2; }}
    a {{ color: #1f5fbf; }}
  </style>
</head>
<body>
  <h1>Depth Enhancement Synthetic Gallery</h1>
  <h2>Overview Pages</h2>
  <ul>
    {overview_links}
  </ul>
  <h2>Samples</h2>
  <table>
    <thead>
      <tr>
        <th>sample</th><th>mode</th><th>synth/real RMS</th><th>synth/real abs-p99</th>
        <th>target/obs corr</th><th>input/obs corr</th><th>base/target corr</th>
        <th>residual abs-p99</th><th>rms scale</th><th>attempts</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def apply_overrides(cfg: Any, args: argparse.Namespace) -> None:
    if args.main_lobe_samples is not None:
        cfg.synthetic_cluster_main_lobe_samples = int(args.main_lobe_samples)
    if args.patch_fraction is not None:
        cfg.synthetic_patch_fraction = float(args.patch_fraction)
    if args.unresolved_fraction is not None:
        cfg.synthetic_unresolved_fraction = float(args.unresolved_fraction)
    if args.well_patch_scale_min is not None:
        cfg.synthetic_well_patch_scale_min = float(args.well_patch_scale_min)
    if args.well_patch_scale_max is not None:
        cfg.synthetic_well_patch_scale_max = float(args.well_patch_scale_max)
    if args.cluster_amp_abs_p95_min is not None:
        cfg.synthetic_cluster_amp_abs_p95_min = float(args.cluster_amp_abs_p95_min)
    if args.cluster_amp_abs_p99_max is not None:
        cfg.synthetic_cluster_amp_abs_p99_max = float(args.cluster_amp_abs_p99_max)
    if args.unresolved_oversample_factor is not None:
        cfg.synthetic_unresolved_oversample_factor = int(args.unresolved_oversample_factor)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")

    import torch

    from enhance import load_well_resolution_prior_npz
    from enhance.config import EnhancementConfig
    from ginn_depth.enhance import build_depth_enhancement_bundle

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    common_cfg = load_yaml_config(args.common_config, base_dir=REPO_ROOT)
    output_root = resolve_relative_path(str(common_cfg.get("output_root", "scripts/output")), root=REPO_ROOT)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"enhance_gallery_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    samples_dir = output_dir / "samples"

    setup_logging(output_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Project root: %s", REPO_ROOT)
    LOGGER.info("Config: %s", config_path)
    LOGGER.info("Common config: %s", resolve_relative_path(args.common_config, root=REPO_ROOT))
    LOGGER.info("Output: %s", output_dir)
    LOGGER.info("Seed: %d", args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25

    cfg = EnhancementConfig.from_yaml(config_path, base_dir=REPO_ROOT)
    apply_overrides(cfg, args)
    cfg.device = "cpu"
    cfg.num_workers = 0
    cfg.pin_memory = False
    cfg.synthetic_traces_per_epoch = int(args.num_samples)

    LOGGER.info("Building depth enhancement dataset...")
    prior = load_well_resolution_prior_npz(cfg.resolution_prior_file)
    enhancement_bundle = build_depth_enhancement_bundle(cfg)
    dataset_bundle = enhancement_bundle.dataset_bundle
    synthetic_dataset = enhancement_bundle.synthetic_dataset

    LOGGER.info(
        "Synthetic config: samples=%d patch_fraction=%.3f unresolved_fraction=%.3f "
        "well_patch_scale=[%.3f, %.3f] cluster_amp=[%.3f*p95, %.3f*p99]",
        args.num_samples,
        cfg.synthetic_patch_fraction,
        cfg.synthetic_unresolved_fraction,
        cfg.synthetic_well_patch_scale_min,
        cfg.synthetic_well_patch_scale_max,
        cfg.synthetic_cluster_amp_abs_p95_min,
        cfg.synthetic_cluster_amp_abs_p99_max,
    )

    depth = dataset_bundle.depth_axis_m
    samples: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    LOGGER.info("Sampling and rendering %d traces...", args.num_samples)
    for idx in range(args.num_samples):
        sample = synthetic_dataset[idx]
        detail_rel = Path("samples") / f"sample_{idx:03d}_{mode_name(sample)}.png"
        detail_path = output_dir / detail_rel
        if not args.skip_detail_plots:
            plot_detail(
                sample,
                depth,
                idx,
                detail_path,
                args.dpi,
                show_clean_target=args.show_clean_target,
                show_dirty_input=args.show_dirty_input,
            )
        record = sample_record(sample, idx, None if args.skip_detail_plots else detail_rel)
        samples.append(sample)
        records.append(record)
        if (idx + 1) % max(1, args.num_samples // 10) == 0:
            LOGGER.info("Rendered %d/%d", idx + 1, args.num_samples)

    overview_paths: list[Path] = []
    if args.overview_page_size > 0:
        for page_start in range(0, len(samples), args.overview_page_size):
            page_samples = samples[page_start : page_start + args.overview_page_size]
            page_records = records[page_start : page_start + args.overview_page_size]
            page_idx = page_start // args.overview_page_size + 1
            rel_path = Path(f"overview_page_{page_idx:02d}.png")
            plot_overview_page(
                page_samples,
                page_records,
                depth,
                page_idx,
                output_dir / rel_path,
                args.dpi,
                show_dirty_input=args.show_dirty_input,
            )
            overview_paths.append(rel_path)

    summary = {
        "config": {
            "config_path": config_path,
            "resolution_prior_file": cfg.resolution_prior_file,
            "seed": args.seed,
            "num_samples": args.num_samples,
            "synthetic_patch_fraction": cfg.synthetic_patch_fraction,
            "synthetic_unresolved_fraction": cfg.synthetic_unresolved_fraction,
            "synthetic_well_patch_scale_min": cfg.synthetic_well_patch_scale_min,
            "synthetic_well_patch_scale_max": cfg.synthetic_well_patch_scale_max,
            "synthetic_cluster_min_events": cfg.synthetic_cluster_min_events,
            "synthetic_cluster_max_events": cfg.synthetic_cluster_max_events,
            "synthetic_cluster_amp_abs_p95_min": cfg.synthetic_cluster_amp_abs_p95_min,
            "synthetic_cluster_amp_abs_p99_max": cfg.synthetic_cluster_amp_abs_p99_max,
            "synthetic_cluster_main_lobe_samples": cfg.synthetic_cluster_main_lobe_samples,
            "synthetic_unresolved_oversample_factor": cfg.synthetic_unresolved_oversample_factor,
            "synthetic_residual_highpass_samples": cfg.synthetic_residual_highpass_samples,
            "delta_supervision_mask": cfg.delta_supervision_mask,
            "synthetic_quality_gate_enabled": cfg.synthetic_quality_gate_enabled,
            "synthetic_seismic_rms_match": cfg.synthetic_seismic_rms_match,
            "synthetic_min_target_obs_waveform_corr": cfg.synthetic_min_target_obs_waveform_corr,
            "synthetic_min_base_target_waveform_corr": cfg.synthetic_min_base_target_waveform_corr,
            "synthetic_input_augmentation_enabled": cfg.synthetic_input_augmentation_enabled,
            "synthetic_input_phase_deg_max": cfg.synthetic_input_phase_deg_max,
            "synthetic_input_amp_jitter": cfg.synthetic_input_amp_jitter,
            "synthetic_input_noise_rms_fraction": cfg.synthetic_input_noise_rms_fraction,
            "synthetic_input_spectral_tilt_max": cfg.synthetic_input_spectral_tilt_max,
            "gallery_show_clean_target": args.show_clean_target,
            "gallery_show_dirty_input": args.show_dirty_input,
        },
        "summary": summarize_records(records),
    }
    write_rows_csv(output_dir / "samples.csv", records)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2, default=json_default)
    write_index_html(output_dir / "index.html", records, overview_paths)

    LOGGER.info("Mode counts: %s", summary["summary"]["mode_counts"])
    LOGGER.info("synthetic_to_real_rms: %s", summary["summary"]["synthetic_to_real_rms"])
    LOGGER.info("residual_abs_p99: %s", summary["summary"]["residual_abs_p99"])
    LOGGER.info("Wrote: %s", output_dir / "index.html")
    LOGGER.info("Wrote: %s", output_dir / "samples.csv")
    LOGGER.info("Wrote detail plots under: %s", samples_dir)


if __name__ == "__main__":
    main()
