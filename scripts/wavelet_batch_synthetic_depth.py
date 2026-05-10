"""Batch depth-domain well tie using a fixed wavelet.

Takes a single auto-tie wavelet, generates synthetic seismograms for all wells,
scans bulk time shifts to find the best match, and exports depth-shifted LAS
files for downstream LFM building.

Usage::

    python scripts/wavelet_batch_synthetic_depth.py
    python scripts/wavelet_batch_synthetic_depth.py --well NW11
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.utils.io import (  # noqa: E402
    load_yaml_config,
    resolve_relative_path,
    sanitize_filename,
)

matplotlib.use("Agg")

plt.rcParams["figure.dpi"] = 120
pd.set_option("display.max_columns", 80)


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
    except RuntimeError:
        pass
    plt.savefig(str(path), dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


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
    parser.add_argument(
        "--well",
        type=str,
        default=None,
        help="Process a single well only (useful for testing).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to scripts/output/wavelet_batch_synthetic_depth_<timestamp>.",
    )
    return parser.parse_args()


# =============================================================================
# Depth / TWT conversion
# =============================================================================


def depth_curve_to_twt(
    depth_tvdss: np.ndarray,
    values: np.ndarray,
    tdt_df: pd.DataFrame,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    from wtie.processing.logs import interpolate_nans

    depth_tvdss = np.asarray(depth_tvdss, dtype=float)
    values = interpolate_nans(values, method="linear")
    z_tdt = tdt_df["tvdss_m"].to_numpy(dtype=float)
    t_tdt = tdt_df["twt_s"].to_numpy(dtype=float)
    z_min = max(float(np.nanmin(depth_tvdss)), float(z_tdt[0]))
    z_max = min(float(np.nanmax(depth_tvdss)), float(z_tdt[-1]))
    if z_max <= z_min:
        raise ValueError("Depth curve and local TDT do not overlap.")
    dz = float(np.nanmedian(np.diff(z_tdt)))
    z_regular = np.arange(z_min, z_max + 0.5 * dz, dz)
    values_z = np.interp(z_regular, depth_tvdss, values)
    twt_z = np.interp(z_regular, z_tdt, t_tdt)
    twt_regular = np.arange(twt_z[0], twt_z[-1] + 0.5 * dt_s, dt_s)
    values_t = np.interp(twt_regular, twt_z, values_z)
    return twt_regular, values_t


# =============================================================================
# Synthetic metrics
# =============================================================================


def least_squares_scale(seismic_norm: np.ndarray, synthetic_raw: np.ndarray, mask: np.ndarray) -> float:
    s = seismic_norm[mask]
    y = synthetic_raw[mask]
    denom = float(np.dot(y, y))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(s, y) / denom)


def metrics_for_synthetic(
    seismic_norm: np.ndarray,
    synthetic_raw: np.ndarray,
    eval_mask: np.ndarray,
) -> dict[str, Any]:
    mask = eval_mask & np.isfinite(seismic_norm) & np.isfinite(synthetic_raw)
    if int(mask.sum()) < 5:
        return {
            "scale": np.nan,
            "signed_ls_scale": np.nan,
            "corr": np.nan,
            "nmae": np.nan,
            "n_eval_samples": int(mask.sum()),
        }
    if np.std(synthetic_raw[mask]) <= 0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(seismic_norm[mask], synthetic_raw[mask])[0, 1])
    signed_scale = least_squares_scale(seismic_norm, synthetic_raw, mask)
    if not np.isfinite(signed_scale):
        return {
            "scale": np.nan,
            "signed_ls_scale": np.nan,
            "corr": corr,
            "nmae": np.nan,
            "n_eval_samples": int(mask.sum()),
        }
    scale = abs(float(signed_scale))
    synthetic_scaled = scale * synthetic_raw
    nmae = float(np.sum(np.abs(seismic_norm[mask] - synthetic_scaled[mask])) / np.sum(np.abs(seismic_norm[mask])))
    return {
        "scale": scale,
        "signed_ls_scale": float(signed_scale),
        "corr": corr,
        "nmae": nmae,
        "n_eval_samples": int(mask.sum()),
    }


def make_eval_mask(twt_s: np.ndarray, boundary_half_support_s: float) -> np.ndarray:
    if twt_s.size < 10:
        return np.ones_like(twt_s, dtype=bool)
    edge_mask = (twt_s >= twt_s[0] + boundary_half_support_s) & (twt_s <= twt_s[-1] - boundary_half_support_s)
    if int(edge_mask.sum()) >= max(10, int(0.2 * twt_s.size)):
        return edge_mask
    return np.ones_like(twt_s, dtype=bool)


def evaluate_shift(
    *,
    twt_s: np.ndarray,
    seismic_norm: np.ndarray,
    ref_twt_s: np.ndarray,
    ref_values: np.ndarray,
    wavelet_amp: np.ndarray,
    shift_s: float,
    modeler: Any,
    eval_mask_base: np.ndarray,
) -> dict[str, Any]:
    shifted_time = twt_s - shift_s
    shifted_ref = np.interp(shifted_time, ref_twt_s, ref_values, left=0.0, right=0.0)
    synthetic_raw = modeler(wavelet_amp, shifted_ref)
    valid_shift = (shifted_time >= ref_twt_s[0]) & (shifted_time <= ref_twt_s[-1])
    eval_mask = eval_mask_base & valid_shift
    metric = metrics_for_synthetic(seismic_norm, synthetic_raw, eval_mask)
    metric.update({"shift_s": float(shift_s)})
    return {
        "metrics": metric,
        "reflectivity_shifted": shifted_ref,
        "synthetic_raw": synthetic_raw,
        "eval_mask": eval_mask,
    }


# =============================================================================
# Depth shift
# =============================================================================


def compute_depth_shift_curve(tdt_df: pd.DataFrame, twt_s: np.ndarray, shift_s: float) -> pd.DataFrame:
    tdt_t = tdt_df["twt_s"].to_numpy(dtype=float)
    tdt_z = tdt_df["tvdss_m"].to_numpy(dtype=float)
    shifted_t = twt_s + shift_s
    valid = (twt_s >= tdt_t[0]) & (twt_s <= tdt_t[-1]) & (shifted_t >= tdt_t[0]) & (shifted_t <= tdt_t[-1])
    if int(valid.sum()) == 0:
        return pd.DataFrame(columns=["twt_s", "tvdss_m", "depth_shift_m"])
    z0 = np.interp(twt_s[valid], tdt_t, tdt_z)
    z1 = np.interp(shifted_t[valid], tdt_t, tdt_z)
    return pd.DataFrame({"twt_s": twt_s[valid], "tvdss_m": z0, "depth_shift_m": z1 - z0})


def build_shifted_md_logset_for_export(
    logset_md: Any,
    *,
    kb_m: float,
    depth_shift_df: pd.DataFrame,
) -> dict[str, Any]:
    from wtie.processing.grid import Log

    if depth_shift_df.empty:
        raise ValueError("Depth shift curve is empty; cannot export shifted LAS.")

    md_m = np.asarray(logset_md.basis, dtype=float)
    tvdss_m = md_m - float(kb_m)
    shift_z = depth_shift_df["tvdss_m"].to_numpy(dtype=float)
    shift_m = depth_shift_df["depth_shift_m"].to_numpy(dtype=float)
    finite_shift = np.isfinite(shift_z) & np.isfinite(shift_m)
    if int(finite_shift.sum()) < 2:
        raise ValueError("Depth shift curve has too few finite samples.")
    shift_z = shift_z[finite_shift]
    shift_m = shift_m[finite_shift]
    order = np.argsort(shift_z)
    shift_z = shift_z[order]
    shift_m = shift_m[order]
    unique_shift_z, unique_shift_idx = np.unique(shift_z, return_index=True)
    unique_shift_m = shift_m[unique_shift_idx]

    depth_shift_at_log = np.interp(
        tvdss_m,
        unique_shift_z,
        unique_shift_m,
        left=unique_shift_m[0],
        right=unique_shift_m[-1],
    )
    shifted_md_m = tvdss_m + depth_shift_at_log + float(kb_m)

    finite_md = np.isfinite(md_m)
    if int(finite_md.sum()) < 2:
        raise ValueError("Input logset has too few finite MD samples.")
    md_step_m = float(np.nanmedian(np.diff(md_m[finite_md])))
    if not np.isfinite(md_step_m) or md_step_m <= 0.0:
        raise ValueError(f"Invalid MD sample step: {md_step_m}")

    finite_shifted_md = np.isfinite(shifted_md_m)
    if int(finite_shifted_md.sum()) < 2:
        raise ValueError("Shifted MD has too few finite samples.")
    shifted_md_sorted = np.sort(shifted_md_m[finite_shifted_md])
    unique_shifted_md = np.unique(shifted_md_sorted)
    if unique_shifted_md.size < 2:
        raise ValueError("Shifted MD has too few unique samples.")
    regular_md = np.arange(
        float(unique_shifted_md[0]),
        float(unique_shifted_md[-1]) + 0.5 * md_step_m,
        md_step_m,
    )

    def interpolate_curve(values: np.ndarray, curve_name: str) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        valid = finite_shifted_md & np.isfinite(values)
        if int(valid.sum()) < 2:
            raise ValueError(f"Curve {curve_name} has too few finite shifted samples.")
        x = shifted_md_m[valid]
        y = values[valid]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        unique_x, unique_idx = np.unique(x, return_index=True)
        unique_y = y[unique_idx]
        if unique_x.size < 2:
            raise ValueError(f"Curve {curve_name} has too few unique shifted samples.")
        return np.interp(regular_md, unique_x, unique_y)

    vp_mps = interpolate_curve(logset_md.Vp.values, "VP_MPS")
    rho_gcc = interpolate_curve(logset_md.Rho.values, "RHO_GCC")
    ai = vp_mps * rho_gcc

    return {
        "VP_MPS": Log(vp_mps, regular_md, "md", name="VP_MPS", unit="m/s", allow_nan=False),
        "RHO_GCC": Log(rho_gcc, regular_md, "md", name="RHO_GCC", unit="g/cm3", allow_nan=False),
        "AI": Log(ai, regular_md, "md", name="AI", unit="m/s*g/cm3", allow_nan=False),
    }


# =============================================================================
# Single-well processor
# =============================================================================


def process_well(
    well_name: str,
    *,
    las_file: Path,
    las_vp_unit: str,
    las_rho_unit: str,
    kb_m: float,
    well_x: float,
    well_y: float,
    survey: Any,
    geometry_depth: dict[str, Any],
    well_heads_df: pd.DataFrame,
    wavelet_amp: np.ndarray,
    wavelet_dt_s: float,
    wavelet_active_half_support_s: float,
    shift_values_s: np.ndarray,
    auto_tie_log_filter_params: dict[str, Any],
    modeler: Any,
    output_dirs: dict[str, Path],
) -> dict[str, Any]:
    from cup.petrel.export import export_logsets_to_las
    from cup.petrel.load import load_vp_rho_logset_from_las
    from cup.utils.raw_trace import zscore_trace
    from wtie.optimize import tie as tie_utils
    from wtie.processing import grid
    from wtie.processing.logs import interpolate_nans

    name = sanitize_filename(well_name)
    row: dict[str, Any] = {"well_name": well_name, "status": "failed", "error": ""}

    il_float, xl_float = survey.coord_to_line(well_x, well_y)
    if not (geometry_depth["inline_min"] <= il_float <= geometry_depth["inline_max"]):
        raise ValueError(f"Inline outside survey range: {il_float}")
    if not (geometry_depth["xline_min"] <= xl_float <= geometry_depth["xline_max"]):
        raise ValueError(f"Crossline outside survey range: {xl_float}")

    logset_md = load_vp_rho_logset_from_las(las_file, vp_unit=las_vp_unit, rho_unit=las_rho_unit)
    md_m = logset_md.basis.astype(float)
    tvdss_m = md_m - kb_m
    vp_mps = interpolate_nans(logset_md.Vp.values, method="linear")

    seismic_depth_trace = survey.import_seismic_at_well(well_x=well_x, well_y=well_y, domain="depth")
    seis_depth = seismic_depth_trace.basis.astype(float)
    seis_amp = interpolate_nans(seismic_depth_trace.values, method="linear")

    overlap_z_min = max(float(np.nanmin(tvdss_m)), float(np.nanmin(seis_depth)))
    overlap_z_max = min(float(np.nanmax(tvdss_m)), float(np.nanmax(seis_depth)))
    if overlap_z_max <= overlap_z_min:
        raise ValueError(
            f"No TVDSS overlap. well=[{np.nanmin(tvdss_m)}, {np.nanmax(tvdss_m)}], "
            f"seismic=[{np.nanmin(seis_depth)}, {np.nanmax(seis_depth)}]"
        )

    win_mask = (tvdss_m >= overlap_z_min) & (tvdss_m <= overlap_z_max)
    if int(win_mask.sum()) < 10:
        raise ValueError(f"Too few well samples in overlap window: {int(win_mask.sum())}")

    tdt_df = grid.build_local_tdt_from_vp(
        tvdss_m=tvdss_m[win_mask],
        vp_mps=vp_mps[win_mask],
        md_m=md_m[win_mask],
    )
    tdt_md = grid.TimeDepthTable(twt=tdt_df["twt_s"].to_numpy(), md=tdt_df["md_m"].to_numpy())
    filtered_logset_md = tie_utils.filter_md_logs(
        logset_md,
        median_size=auto_tie_log_filter_params["logs_median_size"],
        threshold=auto_tie_log_filter_params["logs_median_threshold"],
        std=auto_tie_log_filter_params["logs_std"],
        std2=0.8 * auto_tie_log_filter_params["logs_std"],
    )
    logset_twt = tie_utils.convert_logs_from_md_to_twt(filtered_logset_md, None, tdt_md, wavelet_dt_s)  # type: ignore
    reflectivity_twt = tie_utils.compute_reflectivity(logset_twt)

    twt_seis, seis_twt = depth_curve_to_twt(seis_depth, seis_amp, tdt_df, wavelet_dt_s)
    twt_ref = reflectivity_twt.basis.astype(float)
    ref_twt = reflectivity_twt.values.astype(float)

    t_min = max(float(twt_seis[0]), float(twt_ref[0]))
    t_max = min(float(twt_seis[-1]), float(twt_ref[-1]))
    twt_s = np.arange(t_min, t_max + 0.5 * wavelet_dt_s, wavelet_dt_s)
    if twt_s.size < 10:
        raise ValueError(f"Too few common TWT samples: {twt_s.size}")

    seismic_raw = np.interp(twt_s, twt_seis, seis_twt)
    seismic_norm = zscore_trace(seismic_raw)
    eval_mask_base = make_eval_mask(twt_s, wavelet_active_half_support_s)

    scan_rows = []
    best = None
    for shift_s in shift_values_s:
        result = evaluate_shift(
            twt_s=twt_s,
            seismic_norm=seismic_norm,
            ref_twt_s=twt_ref,
            ref_values=ref_twt,
            wavelet_amp=wavelet_amp,
            shift_s=float(shift_s),
            modeler=modeler,
            eval_mask_base=eval_mask_base,
        )
        metric = result["metrics"]
        scan_rows.append(
            {
                "well_name": well_name,
                "shift_s": metric["shift_s"],
                "shift_ms": metric["shift_s"] * 1000.0,
                "corr": metric["corr"],
                "nmae": metric["nmae"],
                "scale": metric["scale"],
                "signed_ls_scale": metric["signed_ls_scale"],
                "n_eval_samples": metric["n_eval_samples"],
            }
        )
        if np.isfinite(metric["corr"]):
            if best is None or metric["corr"] > best["metrics"]["corr"]:
                best = result

    if best is None:
        raise ValueError("No finite shift-scan correlation values.")

    best_metrics = best["metrics"]
    best_shift_s = float(best_metrics["shift_s"])
    best_synthetic_scaled = best_metrics["scale"] * best["synthetic_raw"]
    depth_shift_df = compute_depth_shift_curve(tdt_df, twt_s, best_shift_s)

    # Save per-well CSVs
    scan_df = pd.DataFrame(scan_rows)
    qc_df = pd.DataFrame(
        {
            "twt_s": twt_s,
            "seismic_norm": seismic_norm,
            "reflectivity_shifted": best["reflectivity_shifted"],
            "synthetic_scaled": best_synthetic_scaled,
            "residual": seismic_norm - best_synthetic_scaled,
            "eval_mask": best["eval_mask"],
        }
    )

    scan_path = output_dirs["shift_scans"] / f"shift_scan_{name}.csv"
    qc_path = output_dirs["synthetic_qc"] / f"synthetic_qc_{name}.csv"
    depth_shift_path = output_dirs["depth_shift_curves"] / f"depth_shift_curve_{name}.csv"
    scan_df.to_csv(scan_path, index=False)
    qc_df.to_csv(qc_path, index=False)
    depth_shift_df.to_csv(depth_shift_path, index=False)

    # Export shifted LAS
    shifted_logset_for_export = build_shifted_md_logset_for_export(
        logset_md,
        kb_m=kb_m,
        depth_shift_df=depth_shift_df,
    )
    shifted_las_export = export_logsets_to_las(
        {well_name: shifted_logset_for_export},
        output_dirs["shifted_las"],
        curve_names=["VP_MPS", "RHO_GCC", "AI"],
        write_fmt="%.6f",
    )
    if not shifted_las_export["exported_files"]:
        raise ValueError(f"Failed to export shifted LAS: {shifted_las_export}")
    shifted_las_path = Path(shifted_las_export["exported_files"][0])

    # Per-well figures
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    t_ms = twt_s * 1000.0
    axes[0].plot(best["reflectivity_shifted"], t_ms, lw=0.8, color="tab:purple")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Reflectivity")
    axes[0].set_ylabel("Relative TWT (ms)")
    axes[0].set_title("Shifted reflectivity")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(seismic_norm, t_ms, lw=0.9, label="Seismic", color="black")
    axes[1].plot(best_synthetic_scaled, t_ms, lw=0.9, label="Synthetic", color="tab:red", alpha=0.85)
    axes[1].set_xlabel("Normalized amplitude")
    axes[1].set_title(f"{well_name}: corr={best_metrics['corr']:.3f}, shift={best_shift_s * 1000:.1f} ms")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(seismic_norm - best_synthetic_scaled, t_ms, lw=0.9, color="tab:gray")
    axes[2].axvline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[2].set_xlabel("Residual")
    axes[2].set_title("Residual")
    axes[2].grid(True, alpha=0.25)
    _save_fig(output_dirs["figures"] / f"qc_{name}_synthetic_vs_seismic.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(scan_df["shift_ms"], scan_df["corr"], lw=1.2, color="tab:blue")
    ax.axvline(best_shift_s * 1000.0, color="tab:red", lw=1.0, ls="--", label="best shift")
    ax.set_xlabel("Bulk time shift (ms)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"{well_name} shift scan")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    _save_fig(output_dirs["figures"] / f"qc_{name}_shift_scan.png")
    plt.close(fig)

    # Depth shift statistics
    if depth_shift_df.empty:
        median_depth_shift = np.nan
        mean_depth_shift = np.nan
        p10_depth_shift = np.nan
        p90_depth_shift = np.nan
    else:
        depth_values = depth_shift_df["depth_shift_m"].to_numpy(dtype=float)
        median_depth_shift = float(np.nanmedian(depth_values))
        mean_depth_shift = float(np.nanmean(depth_values))
        p10_depth_shift = float(np.nanpercentile(depth_values, 10))
        p90_depth_shift = float(np.nanpercentile(depth_values, 90))

    median_vp_mps = float(np.nanmedian(tdt_df["vp_mps"].to_numpy(dtype=float)))
    approx_depth_shift_m = float(median_vp_mps * best_shift_s / 2.0)

    row.update(
        {
            "status": "ok",
            "kb_m": kb_m,
            "well_x": well_x,
            "well_y": well_y,
            "inline_float": float(il_float),
            "xline_float": float(xl_float),
            "overlap_tvdss_min_m": overlap_z_min,
            "overlap_tvdss_max_m": overlap_z_max,
            "twt_min_s": float(twt_s[0]),
            "twt_max_s": float(twt_s[-1]),
            "n_samples": int(twt_s.size),
            "best_shift_s": best_shift_s,
            "best_shift_ms": best_shift_s * 1000.0,
            "corr": float(best_metrics["corr"]),
            "nmae": float(best_metrics["nmae"]),
            "scale": float(best_metrics["scale"]),
            "signed_ls_scale": float(best_metrics["signed_ls_scale"]),
            "n_eval_samples": int(best_metrics["n_eval_samples"]),
            "median_vp_mps": median_vp_mps,
            "median_depth_shift_m": median_depth_shift,
            "mean_depth_shift_m": mean_depth_shift,
            "p10_depth_shift_m": p10_depth_shift,
            "p90_depth_shift_m": p90_depth_shift,
            "approx_depth_shift_m": approx_depth_shift_m,
            "synthetic_qc_path": str(qc_path),
            "shift_scan_path": str(scan_path),
            "depth_shift_curve_path": str(depth_shift_path),
            "shifted_las_path": str(shifted_las_path),
            "synthetic_fig_path": str(output_dirs["figures"] / f"qc_{name}_synthetic_vs_seismic.png"),
            "shift_fig_path": str(output_dirs["figures"] / f"qc_{name}_shift_scan.png"),
        }
    )
    return row


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    data_root = resolve_relative_path(cfg.get("data_root", "data"), root=REPO_ROOT)

    script_cfg = cfg.get("wavelet_batch_synthetic_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'wavelet_batch_synthetic_depth' section in config.")

    las_vp_unit = str(script_cfg.get("las_vp_unit", "us/m"))
    las_rho_unit = str(script_cfg.get("las_rho_unit", "g/cm3"))

    # ── Resolve input paths ──

    las_dir_str = str(script_cfg["las_dir"])
    las_dir = resolve_relative_path(las_dir_str, root=data_root)
    well_heads_file = resolve_relative_path(str(cfg["well"]["well_heads_file"]), root=data_root)
    seismic_file = resolve_relative_path(str(cfg["segy"]["file"]), root=data_root)
    source_auto_tie_dir = REPO_ROOT / str(script_cfg["source_auto_tie_dir"])
    source_well_name = str(script_cfg["source_well_name"])
    wavelet_path = source_auto_tie_dir / f"wavelet_201ms_{source_well_name}.csv"
    # Try both naming conventions (run_summary_NW11.json from new script,
    # run_summary_auto_well_tie_NW11.json from old notebook).
    run_summary_candidates = [
        source_auto_tie_dir / f"run_summary_{source_well_name}.json",
        source_auto_tie_dir / f"run_summary_auto_well_tie_{source_well_name}.json",
    ]
    wavelet_run_summary_path = next((p for p in run_summary_candidates if p.exists()), None)

    for p in [las_dir, well_heads_file, seismic_file, wavelet_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    # ── Output dirs ──

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / cfg.get("output_root", "scripts/output")
        output_dir = output_root / f"wavelet_batch_synthetic_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir

    output_dirs = {
        "root": output_dir,
        "synthetic_qc": output_dir / "synthetic_qc",
        "shift_scans": output_dir / "shift_scans",
        "depth_shift_curves": output_dir / "depth_shift_curves",
        "shifted_las": output_dir / "shifted_las",
        "figures": output_dir / "figures",
    }
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ── Well list ──

    excluded_well_names = list(script_cfg.get("excluded_well_names", []))
    all_las_well_names = sorted(path.stem for path in las_dir.glob("*.las"))
    if not all_las_well_names:
        raise ValueError(f"No LAS files found in {las_dir}")
    excluded_set = set(excluded_well_names)
    unknown = sorted(excluded_set - set(all_las_well_names))
    if unknown:
        raise ValueError(f"Excluded wells do not have LAS files: {unknown}")
    well_names = [w for w in all_las_well_names if w not in excluded_set]
    if not well_names:
        raise ValueError("No wells left after applying excluded_well_names.")
    if args.well:
        if args.well not in well_names:
            raise ValueError(f"--well {args.well} not in well list (or excluded). Available: {well_names}")
        well_names = [args.well]

    print("=== Batch Synthetic Depth ===")
    print(f"Wavelet: {wavelet_path}")
    print(f"LAS dir: {las_dir}")
    print(f"Seismic: {seismic_file}")
    print(f"Output dir: {output_dir}")
    print(f"Wells to process ({len(well_names)}): {well_names}")

    # ── Load wavelet ──

    from cup.well.wavelet import (
        compute_wavelet_active_half_support_s,
        infer_wavelet_dt,
        load_wavelet_csv,
    )

    wavelet_time_s, wavelet_amp = load_wavelet_csv(wavelet_path)
    if wavelet_amp.size < 3:
        raise ValueError("Wavelet has too few samples.")
    if wavelet_amp.size % 2 == 0:
        raise ValueError("Expected an odd-sample centered wavelet.")
    wavelet_dt_s = infer_wavelet_dt(wavelet_time_s)
    wavelet_full_half_s = max(abs(float(wavelet_time_s[0])), abs(float(wavelet_time_s[-1])))
    # Uses cup.well.wavelet.DEFAULT_ACTIVE_SUPPORT_THRESHOLD (=0.05).
    wavelet_active_half_support_s = compute_wavelet_active_half_support_s(
        wavelet_time_s,
        wavelet_amp,
    )
    shift_min_ms = float(script_cfg["shift_min_ms"])
    shift_max_ms = float(script_cfg["shift_max_ms"])
    shift_values_s = np.arange(shift_min_ms / 1000.0, shift_max_ms / 1000.0 + 0.5 * wavelet_dt_s, wavelet_dt_s)

    # Load auto-tie run summary for log filter params (with fallback)
    if wavelet_run_summary_path is not None:
        source_run_summary = json.loads(wavelet_run_summary_path.read_text(encoding="utf-8"))
    else:
        source_run_summary = {}
    source_expected_shift_s = float(source_run_summary.get("auto_tie_best_parameters", {}).get("table_t_shift", np.nan))
    auto_tie_log_filter_params = {
        key: source_run_summary.get("auto_tie_best_parameters", {}).get(key)
        for key in ["logs_median_size", "logs_median_threshold", "logs_std"]
    }
    if any(value is None for value in auto_tie_log_filter_params.values()):
        fallback = script_cfg["fallback_log_filter"]
        auto_tie_log_filter_params = {
            "logs_median_size": fallback["logs_median_size"],
            "logs_median_threshold": fallback["logs_median_threshold"],
            "logs_std": fallback["logs_std"],
        }

    print(f"Wavelet samples={wavelet_amp.size}, dt={wavelet_dt_s * 1000:.3f} ms")
    print(
        f"Eval boundary half-support={wavelet_active_half_support_s * 1000:.1f} ms "
        f"(full half={wavelet_full_half_s * 1000:.1f} ms)"
    )
    print(f"Shift scan: {shift_values_s[0] * 1000:.1f} to {shift_values_s[-1] * 1000:.1f} ms, n={shift_values_s.size}")
    if np.isfinite(source_expected_shift_s):
        print(f"Source auto-tie table_t_shift: {source_expected_shift_s * 1000:.3f} ms")
    print(f"Auto-tie log filter params: {auto_tie_log_filter_params}")

    # ── Load shared resources ──

    from cup.petrel.load import import_well_heads_petrel
    from cup.seismic.survey import open_survey
    from wtie.modeling.modeling import ConvModeler

    well_heads_df = import_well_heads_petrel(well_heads_file)

    segy_cfg = cfg["segy"]
    segy_options = {
        "iline": segy_cfg["iline_byte"],
        "xline": segy_cfg["xline_byte"],
        "istep": segy_cfg["istep"],
        "xstep": segy_cfg["xstep"],
    }
    survey = open_survey(seismic_file, seismic_type="segy", segy_options=segy_options)
    geometry_depth = survey.query_geometry(domain="depth")
    modeler = ConvModeler()

    # ── Process wells ──

    metric_rows = []
    for well_name in well_names:
        print(f"\n=== {well_name} ===")
        las_file = las_dir / f"{well_name}.las"
        if not las_file.exists():
            row = {"well_name": well_name, "status": "failed", "error": "LAS file not found"}
            print("FAILED: LAS file not found")
            metric_rows.append(row)
            continue

        well_row = well_heads_df.loc[well_heads_df["Name"] == well_name]
        if well_row.empty:
            row = {"well_name": well_name, "status": "failed", "error": "Not found in well heads"}
            print("FAILED: Not found in well heads")
            metric_rows.append(row)
            continue
        well_row = well_row.iloc[0]
        kb_m = float(well_row["Well datum value"])
        well_x = float(well_row["Surface X"])
        well_y = float(well_row["Surface Y"])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                row = process_well(
                    well_name=well_name,
                    las_file=las_file,
                    las_vp_unit=las_vp_unit,
                    las_rho_unit=las_rho_unit,
                    kb_m=kb_m,
                    well_x=well_x,
                    well_y=well_y,
                    survey=survey,
                    geometry_depth=geometry_depth,
                    well_heads_df=well_heads_df,
                    wavelet_amp=wavelet_amp,
                    wavelet_dt_s=wavelet_dt_s,
                    wavelet_active_half_support_s=wavelet_active_half_support_s,
                    shift_values_s=shift_values_s,
                    auto_tie_log_filter_params=auto_tie_log_filter_params,
                    modeler=modeler,
                    output_dirs=output_dirs,
                )
            print(
                f"OK: shift={row['best_shift_ms']:.1f} ms, corr={row['corr']:.3f}, "
                f"nmae={row['nmae']:.3f}, depth_shift={row['median_depth_shift_m']:.1f} m"
            )
        except Exception as exc:
            row = {"well_name": well_name, "status": "failed", "error": str(exc)}
            print(f"FAILED: {exc}")
            traceback.print_exc(limit=2)
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "wavelet_batch_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved {metrics_path}")

    # ── Batch summary figures ──

    ok_df = metrics_df.loc[metrics_df["status"] == "ok"].copy()
    if ok_df.empty:
        print("No successful wells to plot batch summaries.")
    else:
        x = np.arange(len(ok_df))
        labels = ok_df["well_name"].tolist()

        # Fig 01: metric summary
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        axes[0].bar(x, ok_df["corr"], color="tab:blue", alpha=0.85)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha="right")
        axes[0].set_ylabel("Correlation")
        axes[0].set_title("Best-shift correlation")
        axes[0].set_ylim(-1, 1)
        axes[0].grid(True, axis="y", alpha=0.25)

        axes[1].bar(x, ok_df["nmae"], color="tab:orange", alpha=0.85)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")
        axes[1].set_ylabel("NMAE")
        axes[1].set_title("Best-shift NMAE")
        axes[1].grid(True, axis="y", alpha=0.25)

        axes[2].bar(x, ok_df["best_shift_ms"], color="tab:green", alpha=0.85)
        axes[2].axhline(0.0, color="black", lw=0.8)
        if np.isfinite(source_expected_shift_s):
            axes[2].axhline(
                source_expected_shift_s * 1000.0,
                color="tab:red",
                lw=1.0,
                ls="--",
                label=f"{source_well_name} auto-tie",
            )
            axes[2].legend(loc="best")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, rotation=45, ha="right")
        axes[2].set_ylabel("Best shift (ms)")
        axes[2].set_title("Bulk time shift")
        axes[2].grid(True, axis="y", alpha=0.25)
        _save_fig(output_dirs["figures"] / "qc_01_batch_metric_summary.png")

        # Fig 02: depth shift summary
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        axes[0].bar(x, ok_df["median_depth_shift_m"], color="tab:purple", alpha=0.85, label="median exact")
        axes[0].plot(
            x,
            ok_df["approx_depth_shift_m"],
            color="black",
            marker="o",
            lw=1.0,
            label="Vp*dt/2 approx",
        )
        axes[0].axhline(0.0, color="black", lw=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=45, ha="right")
        axes[0].set_ylabel("Depth shift (m)")
        axes[0].set_title("Time shift converted to depth")
        axes[0].legend(loc="best")
        axes[0].grid(True, axis="y", alpha=0.25)

        axes[1].errorbar(
            x,
            ok_df["median_depth_shift_m"],
            yerr=[
                ok_df["median_depth_shift_m"] - ok_df["p10_depth_shift_m"],
                ok_df["p90_depth_shift_m"] - ok_df["median_depth_shift_m"],
            ],
            fmt="o",
            color="tab:purple",
            ecolor="tab:gray",
            capsize=4,
        )
        axes[1].axhline(0.0, color="black", lw=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")
        axes[1].set_ylabel("Depth shift (m)")
        axes[1].set_title("P10-P90 exact depth shift")
        axes[1].grid(True, axis="y", alpha=0.25)
        _save_fig(output_dirs["figures"] / "qc_02_batch_depth_shift_summary.png")

    # ── Source well sanity check ──

    if source_well_name in metrics_df["well_name"].values and np.isfinite(source_expected_shift_s):
        source_row = metrics_df.loc[metrics_df["well_name"] == source_well_name].iloc[0]
        if source_row["status"] == "ok":
            delta_ms = float(source_row["best_shift_ms"] - source_expected_shift_s * 1000.0)
            print(f"\n{source_well_name} sanity check:")
            print(f"  batch best shift = {source_row['best_shift_ms']:.3f} ms")
            print(f"  auto-tie table_t_shift = {source_expected_shift_s * 1000.0:.3f} ms")
            print(f"  difference = {delta_ms:.3f} ms")

    # ── Manifest ──

    print(f"\n=== Outputs ===")
    print(f"Summary: {metrics_path}")
    print(f"Batch figures:")
    print(f"  {output_dirs['figures'] / 'qc_01_batch_metric_summary.png'}")
    print(f"  {output_dirs['figures'] / 'qc_02_batch_depth_shift_summary.png'}")
    print("Per-well artifacts:")
    for _, row in metrics_df.iterrows():
        print(f"  {row['well_name']} [{row['status']}]")
        if row["status"] == "ok":
            for key in [
                "synthetic_qc_path",
                "shift_scan_path",
                "depth_shift_curve_path",
                "shifted_las_path",
                "synthetic_fig_path",
                "shift_fig_path",
            ]:
                print(f"    {row[key]}")


if __name__ == "__main__":
    main()
