"""Batch QC for depth stage-2 enhancement synthetic samples.

The script samples the depth enhancement synthetic adapter and writes a compact QC bundle:

- ``qc.log``: human-readable verdicts and metric summaries.
- ``summary.json``: machine-readable aggregate statistics and flags.
- ``samples.csv``: one row per synthetic trace.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger("enhance_qc_depth")
torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/enhance_depth/train.yaml"),
        help="Stage-2 enhancement YAML config.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=512,
        help="Number of random synthetic samples to generate.",
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
        help="QC output directory. Defaults to data/output_enhance_qc_depth_<timestamp>.",
    )
    parser.add_argument(
        "--main-lobe-samples",
        type=int,
        default=None,
        help="Override synthetic_cluster_main_lobe_samples for this QC run.",
    )
    parser.add_argument(
        "--patch-fraction",
        type=float,
        default=None,
        help="Override synthetic_patch_fraction for this QC run.",
    )
    parser.add_argument(
        "--unresolved-fraction",
        type=float,
        default=None,
        help="Override synthetic_unresolved_fraction for this QC run.",
    )
    parser.add_argument(
        "--well-patch-scale-min",
        type=float,
        default=None,
        help="Override synthetic_well_patch_scale_min for this QC run.",
    )
    parser.add_argument(
        "--well-patch-scale-max",
        type=float,
        default=None,
        help="Override synthetic_well_patch_scale_max for this QC run.",
    )
    parser.add_argument(
        "--cluster-amp-abs-p95-min",
        type=float,
        default=None,
        help="Override synthetic_cluster_amp_abs_p95_min for this QC run.",
    )
    parser.add_argument(
        "--cluster-amp-abs-p99-max",
        type=float,
        default=None,
        help="Override synthetic_cluster_amp_abs_p99_max for this QC run.",
    )
    parser.add_argument(
        "--unresolved-oversample-factor",
        type=int,
        default=None,
        help="Override synthetic_unresolved_oversample_factor for this QC run.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "qc.log"
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def ensure_import_path(project_root: Path) -> None:
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))


def to_numpy_1d(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().squeeze()
    return np.asarray(value).squeeze()


def finite_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    valid = mask & np.isfinite(values)
    return values[valid]


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.full((window,), 1.0 / float(window), dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def highpass(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values - moving_average(values, window)


def masked_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    valid = mask & np.isfinite(pred) & np.isfinite(target)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - target[valid])))


def percentile_abs(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(np.abs(values), percentile))


def safe_ratio(num: float, denom: float) -> float:
    if not np.isfinite(num) or not np.isfinite(denom) or abs(denom) < 1e-12:
        return float("nan")
    return float(num / denom)


def count_abs_peaks(values: np.ndarray, threshold_fraction: float) -> int:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3 or not np.any(np.isfinite(values)):
        return 0
    amp = np.abs(np.nan_to_num(values, nan=0.0))
    peak = float(np.max(amp))
    if peak <= 0.0:
        return 0
    threshold = peak * float(threshold_fraction)
    center = amp[1:-1]
    peaks = (center >= threshold) & (center >= amp[:-2]) & (center >= amp[2:])
    return int(np.sum(peaks))


def width_above_fraction(values: np.ndarray, threshold_fraction: float) -> int:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return 0
    amp = np.abs(np.nan_to_num(values, nan=0.0))
    peak = float(np.max(amp))
    if peak <= 0.0:
        return 0
    indices = np.flatnonzero(amp >= peak * float(threshold_fraction))
    if indices.size == 0:
        return 0
    return int(indices[-1] - indices[0] + 1)


def local_window(center: int, n_sample: int, half_width: int) -> slice:
    start = max(0, int(center) - int(half_width))
    stop = min(int(n_sample), int(center) + int(half_width) + 1)
    return slice(start, stop)


def zero_residual_seismic(sample: dict[str, torch.Tensor], forward_model: Any) -> np.ndarray:
    base_ai = sample["base_ai_raw"].float().unsqueeze(0)
    vp = sample["velocity_raw"].float().unsqueeze(0)
    dynamic_gain = sample.get("dynamic_gain")
    gain = dynamic_gain.float().unsqueeze(0) if dynamic_gain is not None else None
    with torch.no_grad():
        seismic = forward_model(torch.clamp(base_ai, min=1e-6), vp, gain=gain)
    return to_numpy_1d(seismic)


def sample_metrics(
    sample: dict[str, torch.Tensor],
    *,
    main_lobe_samples: int,
    residual_clip_abs: float,
    highpass_samples: int,
    forward_model: Any,
) -> dict[str, Any]:
    mode = "well_patch" if int(sample["synthetic_mode"].item()) == 0 else "unresolved_cluster"
    loss_mask = to_numpy_1d(sample["loss_mask"]).astype(bool)
    taper = to_numpy_1d(sample["taper_weight"]).astype(np.float64)
    real = finite_values(to_numpy_1d(sample["obs"]), loss_mask)
    synthetic = finite_values(to_numpy_1d(sample["target_seismic"]), loss_mask)
    synthetic_raw = finite_values(to_numpy_1d(sample["target_seismic_raw"]), loss_mask)
    residual_full = to_numpy_1d(sample["target_residual"]).astype(np.float64)
    residual = finite_values(residual_full, loss_mask)
    residual_highpass_full = highpass(residual_full, highpass_samples)
    residual_highpass = finite_values(residual_highpass_full, loss_mask)
    ai = finite_values(to_numpy_1d(sample["target_ai"]), loss_mask)
    reflectivity_full = to_numpy_1d(sample["raw_reflectivity"]).astype(np.float64)
    zero_seismic = zero_residual_seismic(sample, forward_model)
    target_seismic_full = to_numpy_1d(sample["target_seismic"]).astype(np.float64)

    real_rms = rms(real)
    synthetic_rms = rms(synthetic)
    synthetic_raw_rms = rms(synthetic_raw)
    real_abs_p99 = percentile_abs(real, 99.0)
    synthetic_abs_p99 = percentile_abs(synthetic, 99.0)
    residual_abs_max = percentile_abs(residual, 100.0)
    residual_outside = residual_full[np.asarray(taper <= 0.0)]
    residual_inside = residual_full[np.asarray(taper > 0.0)]
    residual_energy = float(np.nansum(residual_full * residual_full))
    residual_loss_mask_energy = float(np.nansum(residual_full[loss_mask] * residual_full[loss_mask]))
    residual_loss_mask_energy_fraction = safe_ratio(residual_loss_mask_energy, residual_energy)

    active = np.flatnonzero(loss_mask & np.isfinite(residual_full))
    if active.size:
        center = int(active[np.argmax(np.abs(residual_full[active]))])
    else:
        center = int(np.argmax(np.abs(np.nan_to_num(residual_full, nan=0.0))))
    half = max(3, int(main_lobe_samples))
    win = local_window(center, residual_full.size, half)
    refl_stop = min(win.stop, reflectivity_full.size)
    refl_win = reflectivity_full[win.start : refl_stop]
    seismic_full = target_seismic_full

    residual_detail = residual_full[win]
    reflectivity_detail = refl_win
    detail_width_scale = 1.0
    if mode == "unresolved_cluster" and "target_residual_highres" in sample and "raw_reflectivity_highres" in sample:
        highres_residual = to_numpy_1d(sample["target_residual_highres"]).astype(np.float64)
        highres_reflectivity = to_numpy_1d(sample["raw_reflectivity_highres"]).astype(np.float64)
        factor = max(1, int(round((highres_residual.size - 1) / max(residual_full.size - 1, 1))))
        highres_center = min(highres_residual.size - 1, max(0, center * factor))
        highres_win = local_window(highres_center, highres_residual.size, half * factor)
        highres_refl_stop = min(highres_win.stop, highres_reflectivity.size)
        residual_detail = highres_residual[highres_win]
        reflectivity_detail = highres_reflectivity[highres_win.start : highres_refl_stop]
        detail_width_scale = float(factor)

    residual_peak_count = count_abs_peaks(residual_detail, 0.30)
    reflectivity_peak_count = count_abs_peaks(reflectivity_detail, 0.30)
    seismic_peak_count = count_abs_peaks(seismic_full[win], 0.50)
    residual_width = int(round(width_above_fraction(residual_detail, 0.50) / detail_width_scale))
    seismic_width = width_above_fraction(seismic_full[win], 0.50)

    ai_min = float(np.nanmin(ai)) if ai.size else float("nan")
    ai_max = float(np.nanmax(ai)) if ai.size else float("nan")

    return {
        "mode": mode,
        "real_rms": real_rms,
        "synthetic_rms": synthetic_rms,
        "synthetic_raw_rms": synthetic_raw_rms,
        "synthetic_to_real_rms": safe_ratio(synthetic_rms, real_rms),
        "real_abs_p99": real_abs_p99,
        "synthetic_abs_p99": synthetic_abs_p99,
        "synthetic_to_real_abs_p99": safe_ratio(synthetic_abs_p99, real_abs_p99),
        "residual_rms": rms(residual),
        "target_residual_highpass_rms": rms(residual_highpass),
        "residual_abs_p95": percentile_abs(residual, 95.0),
        "residual_abs_p99": percentile_abs(residual, 99.0),
        "residual_abs_max": residual_abs_max,
        "residual_loss_mask_energy_fraction": residual_loss_mask_energy_fraction,
        "residual_near_clip_fraction": float(np.mean(np.abs(residual) >= 0.98 * residual_clip_abs)) if residual.size else float("nan"),
        "residual_outside_taper_abs_mean": float(np.mean(np.abs(residual_outside))) if residual_outside.size else 0.0,
        "residual_inside_taper_abs_mean": float(np.mean(np.abs(residual_inside))) if residual_inside.size else 0.0,
        "residual_outside_taper_nonzero_fraction": float(np.mean(np.abs(residual_outside) > 1e-6)) if residual_outside.size else 0.0,
        "zero_waveform_mae": masked_mae(zero_seismic, target_seismic_full, loss_mask),
        "zero_waveform_mae_to_synthetic_rms": safe_ratio(masked_mae(zero_seismic, target_seismic_full, loss_mask), synthetic_rms),
        "ai_min": ai_min,
        "ai_max": ai_max,
        "rms_scale": float(sample["synthetic_rms_scale"].item()),
        "resample_attempts": float(sample.get("synthetic_resample_attempts", torch.tensor(1)).item()),
        "dominant_center": center,
        "dominant_window_residual_peak_count": residual_peak_count,
        "dominant_window_reflectivity_peak_count": reflectivity_peak_count,
        "dominant_window_seismic_peak_count": seismic_peak_count,
        "dominant_window_residual_width": residual_width,
        "dominant_window_seismic_width": seismic_width,
        "dominant_window_detail_to_seismic_peak_ratio": safe_ratio(float(max(residual_peak_count, reflectivity_peak_count)), float(max(seismic_peak_count, 1))),
        "dominant_window_seismic_to_residual_width_ratio": safe_ratio(float(seismic_width), float(max(residual_width, 1))),
    }


def robust_stats(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "p05": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modes = sorted({str(row["mode"]) for row in rows})
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    summary: dict[str, Any] = {
        "n_samples": len(rows),
        "mode_counts": {mode: sum(1 for row in rows if row["mode"] == mode) for mode in modes},
        "all": {key: robust_stats([float(row[key]) for row in rows]) for key in numeric_keys},
        "by_mode": {},
    }
    for mode in modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        summary["by_mode"][mode] = {key: robust_stats([float(row[key]) for row in mode_rows]) for key in numeric_keys}
    return summary


def add_quality_flags(summary: dict[str, Any], *, requested_patch_fraction: float, requested_unresolved_fraction: float) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []

    def flag(level: str, name: str, message: str) -> None:
        flags.append({"level": level, "name": name, "message": message})

    all_stats = summary["all"]
    n_samples = max(int(summary["n_samples"]), 1)
    mode_counts = summary["mode_counts"]
    total_fraction = requested_patch_fraction + requested_unresolved_fraction
    expected_cluster = requested_unresolved_fraction / total_fraction if total_fraction > 0 else 0.0
    observed_cluster = mode_counts.get("unresolved_cluster", 0) / n_samples
    if abs(observed_cluster - expected_cluster) > max(0.10, 3.0 / math.sqrt(n_samples)):
        flag("WARN", "mode_mix", f"unresolved_cluster fraction {observed_cluster:.3f} differs from configured {expected_cluster:.3f}.")
    else:
        flag("OK", "mode_mix", f"unresolved_cluster fraction {observed_cluster:.3f} is close to configured {expected_cluster:.3f}.")

    rms_ratio = all_stats["synthetic_to_real_rms"]["p50"]
    if rms_ratio is not None and 0.70 <= rms_ratio <= 1.30:
        flag("OK", "seismic_rms", f"median synthetic/real RMS ratio is {rms_ratio:.3f}.")
    else:
        flag("WARN", "seismic_rms", f"median synthetic/real RMS ratio is {rms_ratio}; expected roughly 0.70-1.30.")

    p99_ratio = all_stats["synthetic_to_real_abs_p99"]["p50"]
    if p99_ratio is not None and 0.60 <= p99_ratio <= 1.60:
        flag("OK", "seismic_abs_p99", f"median synthetic/real abs-p99 ratio is {p99_ratio:.3f}.")
    else:
        flag("WARN", "seismic_abs_p99", f"median synthetic/real abs-p99 ratio is {p99_ratio}; expected roughly 0.60-1.60.")

    outside_fraction = all_stats["residual_outside_taper_nonzero_fraction"]["p95"]
    if outside_fraction is not None and outside_fraction <= 0.005:
        flag("OK", "taper_support", f"p95 outside-taper nonzero fraction is {outside_fraction:.3e}.")
    else:
        flag("WARN", "taper_support", f"p95 outside-taper nonzero fraction is {outside_fraction}; residual may leak outside support.")

    near_clip = all_stats["residual_near_clip_fraction"]["p95"]
    if near_clip is not None and near_clip <= 0.02:
        flag("OK", "residual_clip", f"p95 residual near-clip fraction is {near_clip:.3e}.")
    else:
        flag("WARN", "residual_clip", f"p95 residual near-clip fraction is {near_clip}; synthetic residuals may be clip-dominated.")

    highpass_rms = all_stats["target_residual_highpass_rms"]["p50"]
    if highpass_rms is not None and highpass_rms >= 0.01:
        flag("OK", "target_highpass", f"median target residual highpass RMS is {highpass_rms:.3e}.")
    else:
        flag("WARN", "target_highpass", f"median target residual highpass RMS is {highpass_rms}; target may be too smooth.")

    support_fraction = all_stats["residual_loss_mask_energy_fraction"]["p50"]
    if support_fraction is not None and support_fraction >= 0.70:
        flag("OK", "loss_mask_support", f"median residual energy inside loss_mask is {support_fraction:.3f}.")
    else:
        flag("WARN", "loss_mask_support", f"median residual energy inside loss_mask is {support_fraction}; high-frequency target may sit outside supervised support.")

    zero_mae_ratio = all_stats["zero_waveform_mae_to_synthetic_rms"]["p50"]
    if zero_mae_ratio is not None and zero_mae_ratio >= 0.05:
        flag("OK", "zero_baseline", f"median zero-residual MAE/synthetic RMS is {zero_mae_ratio:.3f}.")
    else:
        flag("WARN", "zero_baseline", f"median zero-residual MAE/synthetic RMS is {zero_mae_ratio}; zero residual may be too competitive.")

    cluster = summary["by_mode"].get("unresolved_cluster")
    if cluster:
        detail_ratio = cluster["dominant_window_detail_to_seismic_peak_ratio"]["p50"]
        residual_peaks = cluster["dominant_window_residual_peak_count"]["p50"]
        reflectivity_peaks = cluster["dominant_window_reflectivity_peak_count"]["p50"]
        detail_peaks = max(float(residual_peaks or 0.0), float(reflectivity_peaks or 0.0))
        if detail_ratio is not None and detail_peaks >= 2.0 and detail_ratio >= 1.5:
            flag(
                "OK",
                "unresolved_cluster",
                f"cluster median detail peaks={detail_peaks:.1f}, detail/seismic peak ratio={detail_ratio:.2f}.",
            )
        else:
            flag(
                "WARN",
                "unresolved_cluster",
                "cluster samples may not clearly show many details per seismic peak "
                f"(median residual peaks={residual_peaks}, reflectivity peaks={reflectivity_peaks}, detail/seismic peak ratio={detail_ratio}).",
            )

    attempts_p95 = all_stats["resample_attempts"]["p95"]
    if attempts_p95 is not None and attempts_p95 <= 2.0:
        flag("OK", "resample_attempts", f"p95 resample attempts is {attempts_p95:.1f}.")
    else:
        flag("WARN", "resample_attempts", f"p95 resample attempts is {attempts_p95}; quality gates may be rejecting many samples.")

    return flags


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def log_summary(summary: dict[str, Any], flags: list[dict[str, str]]) -> None:
    LOGGER.info("Synthetic QC samples: %d", summary["n_samples"])
    LOGGER.info("Mode counts: %s", summary["mode_counts"])
    for key in (
        "synthetic_to_real_rms",
        "synthetic_to_real_abs_p99",
        "residual_rms",
        "target_residual_highpass_rms",
        "residual_loss_mask_energy_fraction",
        "residual_abs_p99",
        "residual_near_clip_fraction",
        "residual_outside_taper_nonzero_fraction",
        "zero_waveform_mae",
        "zero_waveform_mae_to_synthetic_rms",
        "rms_scale",
        "resample_attempts",
    ):
        stats = summary["all"][key]
        LOGGER.info(
            "%s: mean=%s p50=%s p95=%s min=%s max=%s",
            key,
            _fmt(stats["mean"]),
            _fmt(stats["p50"]),
            _fmt(stats["p95"]),
            _fmt(stats["min"]),
            _fmt(stats["max"]),
        )

    cluster = summary["by_mode"].get("unresolved_cluster")
    if cluster:
        for key in (
            "dominant_window_residual_peak_count",
            "dominant_window_reflectivity_peak_count",
            "dominant_window_seismic_peak_count",
            "dominant_window_detail_to_seismic_peak_ratio",
            "dominant_window_seismic_to_residual_width_ratio",
        ):
            stats = cluster[key]
            LOGGER.info("cluster %s: mean=%s p50=%s p95=%s", key, _fmt(stats["mean"]), _fmt(stats["p50"]), _fmt(stats["p95"]))

    LOGGER.info("Quality flags:")
    for item in flags:
        LOGGER.info("[%s] %s: %s", item["level"], item["name"], item["message"])


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_f):
        return "nan"
    return f"{value_f:.4g}"


def main() -> None:
    args = parse_args()
    project_root = Path.cwd().resolve()
    ensure_import_path(project_root)

    global torch
    import torch as torch_module

    torch = torch_module

    from enhance import load_well_resolution_prior_npz
    from enhance.config import EnhancementConfig
    from ginn_depth.enhance import build_depth_enhancement_bundle

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "data" / f"output_enhance_qc_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir

    setup_logging(output_dir)
    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Config: %s", config_path)
    LOGGER.info("Output: %s", output_dir)
    LOGGER.info("Seed: %d", args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = EnhancementConfig.from_yaml(config_path, base_dir=project_root)
    cfg.device = "cpu"
    cfg.num_workers = 0
    cfg.pin_memory = False
    cfg.synthetic_traces_per_epoch = int(args.num_samples)
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
    LOGGER.info("Resolution prior: %s", cfg.resolution_prior_file)
    LOGGER.info(
        "Synthetic mix: patch_fraction=%.3f unresolved_fraction=%.3f well_patch_scale=[%.3f, %.3f] "
        "cluster_amp=[%.3f*p95, %.3f*p99] main_lobe_samples=%s",
        cfg.synthetic_patch_fraction,
        cfg.synthetic_unresolved_fraction,
        cfg.synthetic_well_patch_scale_min,
        cfg.synthetic_well_patch_scale_max,
        cfg.synthetic_cluster_amp_abs_p95_min,
        cfg.synthetic_cluster_amp_abs_p99_max,
        cfg.synthetic_cluster_main_lobe_samples,
    )

    prior = load_well_resolution_prior_npz(cfg.resolution_prior_file)
    LOGGER.info(
        "Prior: wells=%d samples=%d valid_samples=%d residual_abs_p99=%s",
        prior.n_wells,
        prior.n_sample,
        int(prior.well_mask.sum()),
        _fmt(prior.summary.get("residual", {}).get("abs_p99")),
    )

    LOGGER.info("Building depth enhancement synthetic dataset...")
    enhancement_bundle = build_depth_enhancement_bundle(cfg)
    synthetic_dataset = enhancement_bundle.synthetic_dataset
    forward_model = synthetic_dataset.forward_model.cpu()
    main_lobe_samples = int(cfg.synthetic_cluster_main_lobe_samples or 12)

    LOGGER.info("Sampling %d synthetic traces...", args.num_samples)
    rows = []
    for idx in range(args.num_samples):
        sample = synthetic_dataset[idx]
        row = sample_metrics(
            sample,
            main_lobe_samples=main_lobe_samples,
            residual_clip_abs=synthetic_dataset.residual_max_abs,
            highpass_samples=cfg.synthetic_residual_highpass_samples_loss,
            forward_model=forward_model,
        )
        row["sample_index"] = idx
        rows.append(row)
        if (idx + 1) % max(1, args.num_samples // 10) == 0:
            LOGGER.info("Sampled %d/%d", idx + 1, args.num_samples)

    summary = summarize_rows(rows)
    summary["config"] = {
        "config_path": config_path,
        "resolution_prior_file": cfg.resolution_prior_file,
        "num_samples": args.num_samples,
        "seed": args.seed,
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
        "synthetic_residual_highpass_samples_loss": cfg.synthetic_residual_highpass_samples_loss,
        "synthetic_seismic_rms_match": cfg.synthetic_seismic_rms_match,
        "synthetic_seismic_rms_target": cfg.synthetic_seismic_rms_target,
        "synthetic_quality_gate_enabled": cfg.synthetic_quality_gate_enabled,
        "synthetic_max_residual_near_clip_fraction": cfg.synthetic_max_residual_near_clip_fraction,
        "synthetic_max_seismic_rms_ratio": cfg.synthetic_max_seismic_rms_ratio,
        "synthetic_max_seismic_abs_p99_ratio": cfg.synthetic_max_seismic_abs_p99_ratio,
        "synthetic_max_resample_attempts": cfg.synthetic_max_resample_attempts,
        "ai_min": cfg.ai_min,
        "ai_max": cfg.ai_max,
    }
    flags = add_quality_flags(
        summary,
        requested_patch_fraction=cfg.synthetic_patch_fraction,
        requested_unresolved_fraction=cfg.synthetic_unresolved_fraction,
    )
    summary["quality_flags"] = flags

    write_rows_csv(output_dir / "samples.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2, default=json_default)
    log_summary(summary, flags)
    LOGGER.info("Wrote: %s", output_dir / "samples.csv")
    LOGGER.info("Wrote: %s", output_dir / "summary.json")
    LOGGER.info("Wrote: %s", output_dir / "qc.log")


if __name__ == "__main__":
    main()
