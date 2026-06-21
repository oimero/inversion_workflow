"""Run R0 real-field zero-shot prediction for frozen GINN-v2 candidates."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sha256_file, write_json
from scripts.real_field_input_domain_diagnostic import _load_model_manifest, _synthetic_train_values
from ginn_v2.real_field import input_qc_frame, load_real_field_section, run_zero_shot_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None, help="Torch device. Use 'cuda' to fail if GPU is unavailable.")
    parser.add_argument("--stitch-strategy", choices=("uniform", "center_crop"), default=None)
    return parser.parse_args()


def _timestamped_output(prefix: str, explicit: Path | None, *, output_root: Path) -> Path:
    if explicit is not None:
        return resolve_relative_path(explicit, root=REPO_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{prefix}_{timestamp}"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _coerce_float(value: object) -> float:
    number = pd.to_numeric(value, errors="coerce")
    try:
        return float(number)
    except (TypeError, ValueError):
        return float("nan")


def _load_well_positions(cfg: dict) -> dict[str, tuple[float, float]]:
    inventory_text = str(cfg.get("well_inventory_file") or "").strip()
    if not inventory_text:
        return {}
    inventory_path = resolve_relative_path(inventory_text, root=REPO_ROOT)
    if not inventory_path.is_file():
        return {}
    inventory = pd.read_csv(inventory_path)
    if not {"well_name", "inline_float", "xline_float"}.issubset(inventory.columns):
        return {}
    out: dict[str, tuple[float, float]] = {}
    for _, row in inventory.iterrows():
        well = str(row.get("well_name", ""))
        inline = _coerce_float(row.get("inline_float"))
        xline = _coerce_float(row.get("xline_float"))
        if well and np.isfinite(inline) and np.isfinite(xline):
            out[well] = (inline, xline)
    return out


def _log_ai_at_twt(*, las_path: Path, tdt_path: Path, twt_s: np.ndarray) -> np.ndarray:
    import lasio

    las = lasio.read(str(las_path))
    frame = las.df()
    if "AI" not in frame.columns:
        raise ValueError(f"Filtered LAS lacks AI curve: {las_path}")
    md_axis = frame.index.to_numpy(dtype=np.float64)
    ai = frame["AI"].to_numpy(dtype=np.float64)
    finite_ai = np.isfinite(md_axis) & np.isfinite(ai) & (ai > 0.0)
    if int(np.count_nonzero(finite_ai)) < 2:
        return np.full_like(twt_s, np.nan, dtype=np.float64)
    tdt = pd.read_csv(tdt_path)
    if not {"twt_s", "md_m"}.issubset(tdt.columns):
        raise ValueError(f"Optimized TDT must contain twt_s/md_m: {tdt_path}")
    tdt_twt = pd.to_numeric(tdt["twt_s"], errors="coerce").to_numpy(dtype=np.float64)
    tdt_md = pd.to_numeric(tdt["md_m"], errors="coerce").to_numpy(dtype=np.float64)
    finite_tdt = np.isfinite(tdt_twt) & np.isfinite(tdt_md)
    if int(np.count_nonzero(finite_tdt)) < 2:
        return np.full_like(twt_s, np.nan, dtype=np.float64)
    order_tdt = np.argsort(tdt_twt[finite_tdt])
    md_at_twt = np.interp(
        twt_s,
        tdt_twt[finite_tdt][order_tdt],
        tdt_md[finite_tdt][order_tdt],
        left=np.nan,
        right=np.nan,
    )
    order_ai = np.argsort(md_axis[finite_ai])
    ai_at_twt = np.interp(
        md_at_twt,
        md_axis[finite_ai][order_ai],
        ai[finite_ai][order_ai],
        left=np.nan,
        right=np.nan,
    )
    return np.where(ai_at_twt > 0.0, np.log(ai_at_twt), np.nan)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) <= 0.0 or np.std(b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _trace_distance_to_section(
    *, well_inline: float, well_xline: float, section_ilines: np.ndarray, section_xlines: np.ndarray
) -> tuple[int, float]:
    distances = np.hypot(section_ilines - float(well_inline), section_xlines - float(well_xline))
    index = int(np.nanargmin(distances))
    return index, float(distances[index])


def _write_well_prediction_qc(output_dir: Path, *, run_cfg: dict) -> dict[str, object]:
    cfg = dict(run_cfg.get("well_qc") or {})
    if not bool(cfg.get("enabled", True)):
        return {"csv": "", "figures": {}, "status": "disabled"}
    well_auto_tie_text = str(cfg.get("well_auto_tie_dir") or "").strip()
    if not well_auto_tie_text:
        raise ValueError("real_field_zero_shot.well_qc.well_auto_tie_dir must be explicit when well_qc is enabled.")
    well_auto_tie_dir = resolve_relative_path(
        well_auto_tie_text,
        root=REPO_ROOT,
    )
    metrics_path = well_auto_tie_dir / "well_tie_metrics.csv"
    if not metrics_path.is_file():
        return {"csv": "", "figures": {}, "status": "missing_well_tie_metrics"}
    ties = pd.read_csv(metrics_path)
    ties = ties[ties["tie_status"].astype(str).eq("success")].copy()
    positions = _load_well_positions(cfg)
    model_arrays = {}
    for child in sorted(output_dir.iterdir()):
        if child.is_dir() and (child / "predictions.npz").is_file():
            model_arrays[child.name] = np.load(child / "predictions.npz", allow_pickle=True)
    if not model_arrays:
        return {"csv": "", "figures": {}, "status": "missing_predictions"}
    first = next(iter(model_arrays.values()))
    ilines = np.asarray(first["ilines"], dtype=np.float64)
    xlines = np.asarray(first["xlines"], dtype=np.float64)
    twt_s = np.asarray(first["twt_s"], dtype=np.float64)
    max_distance = float(cfg.get("max_line_distance", 25.0))
    rows: list[dict[str, object]] = []
    figure_outputs: dict[str, str] = {}
    figure_dir = output_dir / "figures" / "wells"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for _, tie in ties.iterrows():
        well = str(tie.get("well_name", ""))
        inline = _coerce_float(tie.get("inline_float"))
        xline = _coerce_float(tie.get("xline_float"))
        position_source = "well_tie_metrics"
        if (not np.isfinite(inline) or not np.isfinite(xline)) and well in positions:
            inline, xline = positions[well]
            position_source = "well_inventory"
        base = {"well_name": well, "well_inline": inline, "well_xline": xline, "well_position_source": position_source}
        if not np.isfinite(inline) or not np.isfinite(xline):
            rows.append({**base, "model_role": "", "status": "skipped_missing_well_position"})
            continue
        trace_idx, distance = _trace_distance_to_section(
            well_inline=inline,
            well_xline=xline,
            section_ilines=ilines,
            section_xlines=xlines,
        )
        base.update(
            {
                "nearest_section_trace": trace_idx,
                "nearest_section_inline": float(ilines[trace_idx]),
                "nearest_section_xline": float(xlines[trace_idx]),
                "section_line_distance": distance,
            }
        )
        if distance > max_distance:
            rows.append({**base, "model_role": "", "status": "skipped_outside_section_support", "max_line_distance": max_distance})
            continue
        try:
            well_log_ai = _log_ai_at_twt(
                las_path=resolve_relative_path(str(tie["filtered_las_file"]), root=REPO_ROOT),
                tdt_path=resolve_relative_path(str(tie["optimized_tdt_file"]), root=REPO_ROOT),
                twt_s=twt_s,
            )
        except Exception as exc:
            rows.append({**base, "model_role": "", "status": "skipped_well_log_projection_failed", "reason": str(exc)})
            continue
        fig, ax = plt.subplots(figsize=(4.5, 6.0))
        ax.plot(well_log_ai, twt_s, label="well filtered log(AI)", color="black", linewidth=1.5)
        for role, arrays in model_arrays.items():
            pred = np.asarray(arrays["stitched_pred_log_ai"], dtype=np.float64)[trace_idx]
            valid = (
                np.asarray(arrays["valid_mask_model"], dtype=bool)[trace_idx]
                & (np.asarray(arrays["stitching_weight"], dtype=np.float64)[trace_idx] > 0.0)
                & np.isfinite(well_log_ai)
                & np.isfinite(pred)
            )
            n_valid = int(np.count_nonzero(valid))
            residual = pred[valid] - well_log_ai[valid] if n_valid else np.asarray([])
            rows.append(
                {
                    **base,
                    "model_role": role,
                    "status": "ok" if n_valid >= 8 else "insufficient_valid_samples",
                    "n_valid": n_valid,
                    "rmse": float(np.sqrt(np.mean(residual * residual))) if n_valid else float("nan"),
                    "bias": float(np.mean(residual)) if n_valid else float("nan"),
                    "corr": _safe_corr(well_log_ai[valid], pred[valid]) if n_valid else float("nan"),
                    "pred_mean": float(np.mean(pred[valid])) if n_valid else float("nan"),
                    "well_log_ai_mean": float(np.mean(well_log_ai[valid])) if n_valid else float("nan"),
                }
            )
            ax.plot(pred, twt_s, label=role, linewidth=1.0)
        ax.invert_yaxis()
        ax.set_xlabel("log(AI)")
        ax.set_ylabel("TWT s")
        ax.set_title(f"R0 well prediction QC: {well}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = figure_dir / f"r0_well_prediction_qc_{well}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figure_outputs[well] = repo_relative_path(path, root=REPO_ROOT)
    csv_path = output_dir / "well_prediction_qc.csv"
    pd.DataFrame.from_records(rows).to_csv(csv_path, index=False)
    return {
        "csv": repo_relative_path(csv_path, root=REPO_ROOT),
        "figures": figure_outputs,
        "status": "ok",
    }


def _plot_zero_shot_qc(output_dir: Path) -> dict[str, str]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    model_arrays = {}
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir() or not (child / "predictions.npz").is_file():
            continue
        arrays = np.load(child / "predictions.npz", allow_pickle=True)
        model_arrays[child.name] = arrays
        lfm = arrays["lfm_input"]
        pred = arrays["stitched_pred_log_ai"]
        delta = arrays["pred_delta_vs_lfm"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, values, title in [
            (axes[0], lfm, "LFM log(AI)"),
            (axes[1], delta, "Pred - LFM"),
            (axes[2], pred, "Pred log(AI)"),
        ]:
            image = ax.imshow(values.T, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("lateral trace")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        axes[0].set_ylabel("TWT sample")
        fig.suptitle(f"R0 zero-shot QC: {child.name}")
        fig.tight_layout()
        path = figures / f"{child.name}_lfm_delta_pred.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs[f"{child.name}_triptych"] = repo_relative_path(path, root=REPO_ROOT)
    if "lateral" in model_arrays and "no_lateral" in model_arrays:
        diff = model_arrays["lateral"]["stitched_pred_log_ai"] - model_arrays["no_lateral"]["stitched_pred_log_ai"]
        fig, ax = plt.subplots(figsize=(6, 4))
        vmax = float(np.nanquantile(np.abs(diff), 0.99)) if np.isfinite(diff).any() else 1.0
        image = ax.imshow(diff.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title("lateral - no_lateral log(AI)")
        ax.set_xlabel("lateral trace")
        ax.set_ylabel("TWT sample")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = figures / "lateral_minus_no_lateral.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs["lateral_minus_no_lateral"] = repo_relative_path(path, root=REPO_ROOT)
    return outputs


def _configured_bands(run_cfg: dict) -> list[dict[str, float | str]]:
    spectral = dict(run_cfg.get("spectral_qc") or {})
    bands = spectral.get("bands")
    if not isinstance(bands, list) or not bands:
        raise ValueError("real_field_zero_shot.spectral_qc.bands must be a non-empty list.")
    out = []
    for item in bands:
        if not isinstance(item, dict):
            raise ValueError("Each spectral_qc band must be a mapping.")
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError("Each spectral_qc band requires name.")
        out.append(
            {
                "name": name,
                "low_hz": float(item.get("low_hz", 0.0)),
                "high_hz": float(item["high_hz"]) if item.get("high_hz") is not None else float("inf"),
            }
        )
    return out


def _band_rms(values: np.ndarray, valid: np.ndarray, *, dt_s: float, low_hz: float, high_hz: float) -> float:
    data = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(data)
    if data.ndim != 2 or not np.any(mask):
        return float("nan")
    prepared = np.zeros_like(data, dtype=np.float64)
    for idx in range(data.shape[0]):
        row_mask = mask[idx]
        if not np.any(row_mask):
            continue
        mean = float(np.mean(data[idx, row_mask]))
        prepared[idx] = np.where(row_mask, data[idx] - mean, 0.0)
    spectrum = np.fft.rfft(prepared, axis=1)
    freqs = np.fft.rfftfreq(data.shape[1], d=float(dt_s))
    band = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(band):
        return 0.0
    filtered = np.zeros_like(spectrum)
    filtered[:, band] = spectrum[:, band]
    component = np.fft.irfft(filtered, n=data.shape[1], axis=1)
    return float(np.sqrt(np.mean(component[mask] ** 2)))


def _band_component_2d(values: np.ndarray, *, dt_s: float, low_hz: float, high_hz: float) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    prepared = np.where(np.isfinite(data), data - np.nanmean(data, axis=1, keepdims=True), 0.0)
    spectrum = np.fft.rfft(prepared, axis=1)
    freqs = np.fft.rfftfreq(data.shape[1], d=float(dt_s))
    band = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    filtered = np.zeros_like(spectrum)
    filtered[:, band] = spectrum[:, band]
    return np.fft.irfft(filtered, n=data.shape[1], axis=1)


def _write_spectral_qc(output_dir: Path, *, run_cfg: dict, dt_s: float) -> dict[str, str]:
    bands = _configured_bands(run_cfg)
    rows = []
    arrays_by_role = {}
    for child in sorted(output_dir.iterdir()):
        if not child.is_dir() or not (child / "predictions.npz").is_file():
            continue
        arrays = np.load(child / "predictions.npz", allow_pickle=True)
        arrays_by_role[child.name] = arrays
        delta = arrays["pred_delta_vs_lfm"]
        valid = arrays["valid_mask_model"].astype(bool) & (arrays["stitching_weight"] > 0.0)
        full = _band_rms(delta, valid, dt_s=dt_s, low_hz=0.0, high_hz=float("inf"))
        summary_path = child / "real_field_zero_shot_model_summary.json"
        synthetic_delta_std = float("nan")
        if summary_path.is_file():
            import json

            summary = json.load(open(summary_path, encoding="utf-8"))
            synthetic_delta_std = float((summary.get("normalization") or {}).get("delta", {}).get("std", float("nan")))
        for band in bands:
            band_rms = _band_rms(
                delta,
                valid,
                dt_s=dt_s,
                low_hz=float(band["low_hz"]),
                high_hz=float(band["high_hz"]),
            )
            rows.append(
                {
                    "model_role": child.name,
                    "signal": "pred_delta_vs_lfm",
                    "band": band["name"],
                    "low_hz": band["low_hz"],
                    "high_hz": band["high_hz"],
                    "band_rms": band_rms,
                    "fullband_rms": full,
                    "energy_ratio": band_rms / full if np.isfinite(full) and full > 0 else float("nan"),
                    "synthetic_train_delta_std": synthetic_delta_std,
                    "pred_delta_spectrum_vs_synthetic_train": (
                        band_rms / synthetic_delta_std
                        if np.isfinite(synthetic_delta_std) and synthetic_delta_std > 0
                        else float("nan")
                    ),
                    "pred_delta_spectrum_vs_well": float("nan"),
                    "pred_delta_spectrum_vs_well_status": "reported_per_well_in_R1_well_forward_diagnostic_when_supported",
                }
            )
    spectral_path = output_dir / "real_field_spectral_qc.csv"
    pd.DataFrame.from_records(rows).to_csv(spectral_path, index=False)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    diff_rows = []
    if "lateral" in arrays_by_role and "no_lateral" in arrays_by_role:
        lateral = arrays_by_role["lateral"]
        no_lateral = arrays_by_role["no_lateral"]
        diff = lateral["stitched_pred_log_ai"] - no_lateral["stitched_pred_log_ai"]
        valid = (
            lateral["valid_mask_model"].astype(bool)
            & no_lateral["valid_mask_model"].astype(bool)
            & (lateral["stitching_weight"] > 0.0)
            & (no_lateral["stitching_weight"] > 0.0)
        )
        full = _band_rms(diff, valid, dt_s=dt_s, low_hz=0.0, high_hz=float("inf"))
        for band in bands:
            band_rms = _band_rms(
                diff,
                valid,
                dt_s=dt_s,
                low_hz=float(band["low_hz"]),
                high_hz=float(band["high_hz"]),
            )
            diff_rows.append(
                {
                    "comparison": "lateral_minus_no_lateral",
                    "band": band["name"],
                    "low_hz": band["low_hz"],
                    "high_hz": band["high_hz"],
                    "band_rms": band_rms,
                    "fullband_rms": full,
                    "energy_ratio": band_rms / full if np.isfinite(full) and full > 0 else float("nan"),
                    "lateral_minus_no_lateral_nullspace_energy": (
                        band_rms if str(band["name"]) == "highfreq_or_nullspace" else float("nan")
                    ),
                }
            )
        band_fig_path = figures / "lateral_minus_no_lateral_band_split.png"
        panel_specs = [{"name": "fullband", "values": diff}]
        panel_specs.extend(
            {
                "name": str(band["name"]),
                "values": _band_component_2d(
                    diff,
                    dt_s=dt_s,
                    low_hz=float(band["low_hz"]),
                    high_hz=float(band["high_hz"]),
                ),
            }
            for band in bands
        )
        fig, axes = plt.subplots(1, len(panel_specs), figsize=(4 * len(panel_specs), 4), sharey=True)
        if len(panel_specs) == 1:
            axes = [axes]
        for ax, panel in zip(axes, panel_specs):
            values = np.asarray(panel["values"], dtype=np.float64)
            finite = values[np.isfinite(values)]
            vmax = float(np.nanquantile(np.abs(finite), 0.99)) if finite.size else 1.0
            image = ax.imshow(values.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_title(str(panel["name"]))
            ax.set_xlabel("lateral trace")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        axes[0].set_ylabel("TWT sample")
        fig.suptitle("lateral - no_lateral band split")
        fig.tight_layout()
        fig.savefig(band_fig_path, dpi=160)
        plt.close(fig)
    diff_path = output_dir / "lateral_difference_band_qc.csv"
    pd.DataFrame.from_records(diff_rows).to_csv(diff_path, index=False)

    figure_path = figures / "spectral_band_energy_qc.png"
    if rows:
        frame = pd.DataFrame.from_records(rows)
        fig, ax = plt.subplots(figsize=(8, 4))
        for role, group in frame.groupby("model_role"):
            ax.plot(group["band"], group["band_rms"], marker="o", label=role)
        ax.set_ylabel("RMS log(AI)")
        ax.set_title("R0 pred_delta_vs_lfm band energy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_path, dpi=160)
        plt.close(fig)
    return {
        "real_field_spectral_qc": repo_relative_path(spectral_path, root=REPO_ROOT),
        "lateral_difference_band_qc": repo_relative_path(diff_path, root=REPO_ROOT),
        "lateral_minus_no_lateral_band_split_figure": (
            repo_relative_path(band_fig_path, root=REPO_ROOT) if "band_fig_path" in locals() and band_fig_path.is_file() else ""
        ),
        "spectral_band_energy_figure": (
            repo_relative_path(figure_path, root=REPO_ROOT) if figure_path.is_file() else ""
        ),
    }


def _input_distribution_warnings(qc_path: Path) -> list[dict[str, object]]:
    if not qc_path.is_file():
        return []
    frame = pd.read_csv(qc_path)
    warnings = []
    for _, row in frame.iterrows():
        name = str(row.get("input", ""))
        gt5 = pd.to_numeric(row.get("fraction_abs_normalized_gt_5"), errors="coerce")
        if np.isfinite(gt5) and float(gt5) > 0.05:
            warnings.append(
                {
                    "warning": "input_distribution_ood",
                    "input": name,
                    "fraction_abs_normalized_gt_5": float(gt5),
                    "threshold": 0.05,
                }
            )
    return warnings


def _source_file_hashes(section_metadata: dict, source_runs: dict) -> dict[str, dict[str, object]]:
    files: dict[str, str] = {
        "seismic_file": str(section_metadata.get("seismic_file") or ""),
        "lfm_file": str(section_metadata.get("lfm_file") or ""),
    }
    if section_metadata.get("target_mask_file"):
        files["target_mask_file"] = str(section_metadata["target_mask_file"])
    wavelet_dir = source_runs.get("wavelet_generation_dir")
    if wavelet_dir:
        files["selected_wavelet_csv"] = str(resolve_relative_path(wavelet_dir, root=REPO_ROOT) / "selected_wavelet.csv")
    out = {}
    for key, text in files.items():
        if not text:
            continue
        path = Path(text)
        if not path.is_absolute():
            path = resolve_relative_path(path, root=REPO_ROOT)
        if not path.is_file():
            out[key] = {"path": str(path), "sha256": "", "status": "missing"}
            continue
        out[key] = {
            "path": repo_relative_path(path, root=REPO_ROOT),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
            "status": "ok",
        }
    return out


def main() -> None:
    args = parse_args()
    cfg_path = resolve_relative_path(args.config, root=REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    run_cfg = dict(cfg.get("real_field_zero_shot") or {})
    if not run_cfg:
        raise ValueError("experiments/common.yaml lacks real_field_zero_shot section.")
    output_root = resolve_relative_path(cfg.get("output_root", "scripts/output"), root=REPO_ROOT)
    output_dir = _timestamped_output("real_field_zero_shot", args.output_dir, output_root=output_root)
    output_dir.mkdir(parents=True, exist_ok=False)

    device = str(args.device or run_cfg.get("device") or "auto")
    stitch_strategy = str(args.stitch_strategy or run_cfg.get("stitch_strategy") or "uniform")
    models = run_cfg.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("real_field_zero_shot.models must be a non-empty list.")
    expected = {
        "no_lateral": "trace1d_tcn_mismatch",
        "lateral": "trace1d_tcn_lateral_mixer_mismatch",
    }
    seen = {str(item.get("model_role")): str(item.get("model_id")) for item in models if isinstance(item, dict)}
    if seen != expected:
        raise ValueError(f"R0 first batch must contain exactly {expected}, got {seen}.")
    data_root = resolve_relative_path(cfg.get("data_root", "data"), root=REPO_ROOT)
    run_cfg_for_load = dict(run_cfg)
    inputs_for_load = dict(run_cfg_for_load.get("real_field_inputs") or {})
    transform_name = str(inputs_for_load.get("seismic_value_transform") or inputs_for_load.get("seismic_transform") or "identity")
    if transform_name not in {"identity", "raw", "none"} and "seismic_reference_stats" not in inputs_for_load:
        diag_cfg = dict(cfg.get("real_field_input_domain_diagnostic") or {})
        manifest = _load_model_manifest(dict(models[0]))
        values, metadata = _synthetic_train_values(
            manifest,
            input_name="seismic",
            max_patches=int(diag_cfg.get("max_synthetic_train_patches", 4096)),
            seed=int(diag_cfg.get("synthetic_train_sampling_seed", 20260620)),
        )
        from ginn_v2.real_field import finite_summary_stats

        inputs_for_load["seismic_reference_stats"] = finite_summary_stats(values)
        inputs_for_load["seismic_reference_sampling"] = metadata
        run_cfg_for_load["real_field_inputs"] = inputs_for_load
    section = load_real_field_section(config=run_cfg_for_load, root=REPO_ROOT, data_root=data_root)

    model_summaries = []
    input_qc_written = False
    for model_cfg in models:
        summary = run_zero_shot_model(
            section=section,
            model_cfg=model_cfg,
            output_dir=output_dir,
            root=REPO_ROOT,
            device_name=device,
            stitch_strategy=stitch_strategy,
        )
        model_summaries.append(summary)
        if not input_qc_written:
            qc = input_qc_frame(section, summary.get("normalization", {}) or {})
            qc.to_csv(output_dir / "model_input_qc.csv", index=False)
            input_qc_written = True
    figure_outputs = _plot_zero_shot_qc(output_dir)
    well_outputs = _write_well_prediction_qc(output_dir, run_cfg=run_cfg)
    dt_s = float(section.twt_s[1] - section.twt_s[0]) if section.twt_s.size > 1 else 0.002
    spectral_outputs = _write_spectral_qc(output_dir, run_cfg=run_cfg, dt_s=dt_s)

    source_runs = dict(run_cfg.get("source_runs") or {})
    boundary = dict(run_cfg.get("boundary") or {})
    dt_s = float(section.twt_s[1] - section.twt_s[0]) if section.twt_s.size > 1 else 0.002
    input_qc_path = output_dir / "model_input_qc.csv"
    source_hashes = _source_file_hashes(section.metadata, source_runs)
    summary = {
        "schema_version": "real_field_zero_shot_summary_v1",
        "status": "needs_forward_diagnostic",
        "config_file": repo_relative_path(cfg_path, root=REPO_ROOT),
        "device_requested": device,
        "stitch_strategy": stitch_strategy,
        "source_runs": source_runs,
        "section": section.metadata,
        "axis_contract": {
            "n_lateral": int(section.lfm.shape[0]),
            "n_twt": int(section.lfm.shape[1]),
            "twt_start_s": float(section.twt_s[0]),
            "twt_stop_s": float(section.twt_s[-1]),
            "dt_s": float(section.twt_s[1] - section.twt_s[0]) if section.twt_s.size > 1 else None,
        },
        "mask_contract": {
            "valid_fraction": float(section.valid_mask.mean()),
            "valid_samples": int(section.valid_mask.sum()),
        },
        "boundary_contract": {
            "loss_or_eval_erosion_s": float(boundary.get("loss_or_eval_erosion_s", 0.0) or 0.0),
            "prediction_taper_halo_s": float(boundary.get("prediction_taper_halo_s", 0.0) or 0.0),
            "forward_diagnostic_crop_s": float(boundary.get("forward_diagnostic_crop_s", 0.0) or 0.0),
            "loss_or_eval_erosion_samples": int(np.ceil(float(boundary.get("loss_or_eval_erosion_s", 0.0) or 0.0) / dt_s)),
            "prediction_taper_halo_samples": int(np.ceil(float(boundary.get("prediction_taper_halo_s", 0.0) or 0.0) / dt_s)),
            "forward_diagnostic_crop_samples": int(np.ceil(float(boundary.get("forward_diagnostic_crop_s", 0.0) or 0.0) / dt_s)),
            "dt_s": dt_s,
        },
        "source_file_sha256": source_hashes,
        "input_distribution_qc": {
            "path": repo_relative_path(input_qc_path, root=REPO_ROOT),
            "warnings": _input_distribution_warnings(input_qc_path),
        },
        "outputs": {
            "model_input_qc": repo_relative_path(input_qc_path, root=REPO_ROOT),
            "figures": figure_outputs,
            "well_prediction_qc": well_outputs,
            **spectral_outputs,
        },
        "models": model_summaries,
        "code_version_or_git_commit": str(run_cfg.get("code_version_or_git_commit") or _git_commit()),
    }
    wavelet_dir_text = (source_runs.get("wavelet_generation_dir") or "").strip()
    if wavelet_dir_text:
        wavelet_dir = resolve_relative_path(wavelet_dir_text, root=REPO_ROOT)
        selected = wavelet_dir / "selected_wavelet.csv"
        if selected.is_file():
            summary["wavelet_sha256"] = sha256_file(selected)
    write_json(output_dir / "real_field_zero_shot_summary.json", summary)
    print("=== Real Field Zero-Shot ===")
    print(f"Output: {output_dir}")
    print(f"Models: {len(model_summaries)}")
    print(f"Status: {summary['status']}")


if __name__ == "__main__":
    main()
