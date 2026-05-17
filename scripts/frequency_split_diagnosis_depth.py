"""Diagnose depth-domain well frequency split cutoffs from well-tie waveform fit.

The script sweeps candidate lowpass cutoff wavelengths on shifted LAS AI logs,
forward-models the lowpass AI with the fixed batch-tie wavelet, and reports
which cutoff is still supported by the real well-side seismic waveform.

Usage::

    python scripts/frequency_split_diagnosis_depth.py
    python scripts/frequency_split_diagnosis_depth.py --config experiments/common_depth.yaml
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cup.petrel.load import import_well_heads_petrel
from cup.seismic.survey import open_survey
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sanitize_filename
from cup.utils.raw_trace import zscore_trace
from cup.well.wavelet import compute_wavelet_active_half_support_s, infer_wavelet_dt, load_wavelet_csv
from wavelet_batch_synthetic_depth import depth_curve_to_twt, make_eval_mask, metrics_for_synthetic
from wtie.modeling.modeling import ConvModeler
from wtie.optimize import tie as tie_utils
from wtie.optimize.logs import filter_log
from wtie.processing import grid
from wtie.processing.logs import interpolate_nans
from wtie.processing.spectral import apply_butter_lowpass_filter


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
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <output_root>/frequency_split_diagnosis_depth_<timestamp>.",
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


def _find_curve(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    normalized = {str(col).strip().upper(): str(col) for col in df.columns}
    for name in names:
        hit = normalized.get(name.upper())
        if hit is not None:
            return hit
    return None


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


def _read_shifted_las_curves(
    path: Path,
    *,
    log_filter_params: dict[str, float | int],
    las_ai_source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    las = lasio.read(path)
    df = las.df()
    md = df.index.to_numpy(dtype=float)
    ai_col = _find_curve(df, ("AI",))
    vp_col = _find_curve(df, ("VP_MPS", "VP"))
    rho_col = _find_curve(df, ("RHO_GCC", "RHO"))
    if vp_col is None or rho_col is None:
        raise ValueError("Shifted LAS must contain VP_MPS/VP and RHO_GCC/RHO curves for TDT construction.")

    raw_vp = df[vp_col].to_numpy(dtype=float)
    raw_rho = df[rho_col].to_numpy(dtype=float)
    raw_ai = df[ai_col].to_numpy(dtype=float) if ai_col is not None else raw_vp * raw_rho

    if las_ai_source == "raw_shifted_las":
        return md, raw_ai, raw_vp
    if las_ai_source != "filtered_shifted_las":
        raise ValueError("las_ai_source must be filtered_shifted_las or raw_shifted_las.")

    logset = grid.LogSet(
        {
            "Vp": grid.Log(raw_vp, md, "md", name="Vp", unit="m/s"),
            "Rho": grid.Log(raw_rho, md, "md", name="Rho", unit="g/cm3"),
        }
    )
    if ai_col is not None:
        ai_log = grid.Log(raw_ai, md, "md", name="AI", unit="m/s*g/cm3")
        filtered_ai = filter_log(
            ai_log,
            median_size=int(log_filter_params["logs_median_size"]),
            threshold=float(log_filter_params["logs_median_threshold"]),
            std=float(log_filter_params["logs_std"]),
            std2=0.8 * float(log_filter_params["logs_std"]),
        ).values
        filtered_logs = tie_utils.filter_md_logs(
            logset,
            median_size=int(log_filter_params["logs_median_size"]),
            threshold=float(log_filter_params["logs_median_threshold"]),
            std=float(log_filter_params["logs_std"]),
            std2=0.8 * float(log_filter_params["logs_std"]),
        )
    else:
        filtered_logs = tie_utils.filter_md_logs(
            logset,
            median_size=int(log_filter_params["logs_median_size"]),
            threshold=float(log_filter_params["logs_median_threshold"]),
            std=float(log_filter_params["logs_std"]),
            std2=0.8 * float(log_filter_params["logs_std"]),
        )
        filtered_ai = filtered_logs.Vp.values * filtered_logs.Rho.values
    return md, np.asarray(filtered_ai, dtype=float), np.asarray(filtered_logs.Vp.values, dtype=float)


def _regularize_depth_curve(depth: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=float)
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(depth) & np.isfinite(values)
    if int(valid.sum()) < 2:
        raise ValueError("Too few finite samples.")
    depth = depth[valid]
    values = values[valid]
    order = np.argsort(depth)
    depth = depth[order]
    values = values[order]
    unique_depth, unique_idx = np.unique(depth, return_index=True)
    unique_values = values[unique_idx]
    if unique_depth.size < 2:
        raise ValueError("Too few unique depth samples.")
    return unique_depth, unique_values


def _lowpass_depth_ai(
    depth: np.ndarray,
    ai: np.ndarray,
    *,
    cutoff_wavelength_m: float,
    order: int,
    buffer_mode: str,
) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    ai = np.asarray(ai, dtype=np.float64)
    dz_values = np.diff(depth)
    dz_values = dz_values[np.isfinite(dz_values) & (dz_values > 0.0)]
    if dz_values.size == 0:
        raise ValueError("Cannot resolve positive depth sample step.")
    dz = float(np.median(dz_values))
    fs = 1.0 / dz
    highcut = 1.0 / float(cutoff_wavelength_m)
    if highcut >= 0.5 * fs:
        raise ValueError(f"cutoff_wavelength_m={cutoff_wavelength_m} m is above Nyquist for dz={dz:.6g} m.")
    if ai.size <= max(3, int(order)):
        return ai.astype(np.float32)
    pad = min(max(1, 3 * int(order)), ai.size - 1)
    padded = np.pad(ai, (pad, pad), mode=buffer_mode)  # type: ignore
    filtered = apply_butter_lowpass_filter(
        padded,
        highcut,
        fs,
        order=int(order),
        zero_phase=True,
    )[pad : pad + ai.size]
    return np.clip(filtered, 1e-6, None).astype(np.float32)


def _ai_to_reflectivity(ai_twt: np.ndarray) -> np.ndarray:
    ai = np.asarray(ai_twt, dtype=np.float64)
    ref = np.zeros_like(ai, dtype=np.float32)
    if ai.size < 2:
        return ref
    upper = ai[:-1]
    lower = ai[1:]
    ref[1:] = ((lower - upper) / (lower + upper + 1e-10)).astype(np.float32)
    return ref


def _select_recommended_cutoff(
    aggregate: pd.DataFrame,
    *,
    corr_tolerance: float,
    nmae_tolerance: float,
) -> dict[str, Any]:
    ok = aggregate[np.isfinite(aggregate["median_corr"]) & np.isfinite(aggregate["median_nmae"])].copy()
    if ok.empty:
        return {"recommended_cutoff_wavelength_m": None, "reason": "No finite aggregate metrics."}
    best_corr = float(ok["median_corr"].max())
    best_nmae = float(ok.loc[ok["median_corr"].idxmax(), "median_nmae"])
    plateau = ok[
        (ok["median_corr"] >= best_corr - float(corr_tolerance))
        & (ok["median_nmae"] <= best_nmae + float(nmae_tolerance))
    ]
    if plateau.empty:
        chosen = ok.loc[ok["median_corr"].idxmax()]
        reason = "No plateau candidate passed both tolerances; selected max median_corr."
    else:
        chosen = plateau.loc[plateau["cutoff_wavelength_m"].idxmax()]
        reason = (
            "Selected the largest wavelength on the near-best waveform-fit plateau "
            "to keep stage-1 conservative."
        )
    return {
        "recommended_cutoff_wavelength_m": float(chosen["cutoff_wavelength_m"]),
        "recommended_median_corr": float(chosen["median_corr"]),
        "recommended_median_nmae": float(chosen["median_nmae"]),
        "best_median_corr": best_corr,
        "best_corr_cutoff_wavelength_m": float(ok.loc[ok["median_corr"].idxmax(), "cutoff_wavelength_m"]),
        "best_corr_median_nmae": best_nmae,
        "corr_tolerance": float(corr_tolerance),
        "nmae_tolerance": float(nmae_tolerance),
        "reason": reason,
    }


def _plot_aggregate(
    aggregate: pd.DataFrame,
    *,
    figure_path: Path,
    current_cutoff: float | None,
    recommended_cutoff: float | None,
) -> None:
    plot_df = aggregate.sort_values("cutoff_wavelength_m")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = plot_df["cutoff_wavelength_m"]
    axes[0].plot(x, plot_df["median_corr"], marker="o", color="tab:blue", label="median corr")
    axes[0].fill_between(x, plot_df["p25_corr"], plot_df["p75_corr"], color="tab:blue", alpha=0.15, label="p25-p75")
    axes[0].set_xlabel("Lowpass cutoff wavelength (m)")
    axes[0].set_ylabel("Correlation")
    axes[0].set_title("Waveform correlation")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(x, plot_df["median_nmae"], marker="o", color="tab:orange", label="median nmae")
    axes[1].fill_between(x, plot_df["p25_nmae"], plot_df["p75_nmae"], color="tab:orange", alpha=0.15, label="p25-p75")
    axes[1].set_xlabel("Lowpass cutoff wavelength (m)")
    axes[1].set_ylabel("NMAE")
    axes[1].set_title("Normalized MAE")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    for ax in axes:
        if current_cutoff is not None and np.isfinite(current_cutoff):
            ax.axvline(float(current_cutoff), color="black", lw=1.0, ls=":", label="current")
        if recommended_cutoff is not None and np.isfinite(recommended_cutoff):
            ax.axvline(float(recommended_cutoff), color="tab:red", lw=1.2, ls="--", label="recommended")
    _save_fig(figure_path)


def _plot_well_metrics(well_df: pd.DataFrame, figure_path: Path) -> None:
    plot_df = well_df.sort_values("cutoff_wavelength_m")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(plot_df["cutoff_wavelength_m"], plot_df["corr"], marker="o", color="tab:blue")
    axes[0].set_xlabel("Lowpass cutoff wavelength (m)")
    axes[0].set_ylabel("Correlation")
    axes[0].set_title(str(plot_df["well_name"].iloc[0]))
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(plot_df["cutoff_wavelength_m"], plot_df["nmae"], marker="o", color="tab:orange")
    axes[1].set_xlabel("Lowpass cutoff wavelength (m)")
    axes[1].set_ylabel("NMAE")
    axes[1].set_title("Cutoff sweep")
    axes[1].grid(True, alpha=0.25)
    _save_fig(figure_path)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = cfg.get("frequency_split_diagnosis_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'frequency_split_diagnosis_depth' section in config.")

    data_root = resolve_relative_path(str(cfg.get("data_root", "data")), root=REPO_ROOT)
    output_root = resolve_relative_path(str(cfg.get("output_root", "scripts/output")), root=REPO_ROOT)
    batch_cfg = cfg.get("wavelet_batch_synthetic_depth", {})
    well_constraints_cfg = cfg.get("well_constraints_depth", {})

    source_batch_dir = resolve_relative_path(str(script_cfg["source_batch_dir"]), root=REPO_ROOT)
    shifted_las_dir = resolve_relative_path(
        str(script_cfg.get("shifted_las_dir", source_batch_dir / "shifted_las")),
        root=source_batch_dir,
    )
    metrics_path = resolve_relative_path(
        str(script_cfg.get("metrics_file", source_batch_dir / "wavelet_batch_metrics.csv")),
        root=source_batch_dir,
    )
    seismic_file = resolve_relative_path(str(cfg["segy"]["file"]), root=data_root)
    well_heads_file = resolve_relative_path(str(cfg["well"]["well_heads_file"]), root=data_root)
    source_auto_tie_dir = resolve_relative_path(str(batch_cfg["source_auto_tie_dir"]), root=REPO_ROOT)
    source_well_name = str(batch_cfg["source_well_name"])
    wavelet_path = source_auto_tie_dir / f"wavelet_201ms_{source_well_name}.csv"
    for path in [shifted_las_dir, metrics_path, seismic_file, well_heads_file, wavelet_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"frequency_split_diagnosis_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dirs = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "well_figures": output_dir / "figures" / "wells",
        "metadata": output_dir / "metadata",
    }
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    cutoffs = [float(v) for v in script_cfg.get("candidate_cutoff_wavelength_m", [])]
    if not cutoffs:
        raise ValueError("frequency_split_diagnosis_depth.candidate_cutoff_wavelength_m must not be empty.")
    cutoffs = sorted(set(cutoffs))
    if any(value <= 0.0 or not np.isfinite(value) for value in cutoffs):
        raise ValueError("All candidate cutoff wavelengths must be positive finite values.")
    filter_order = int(script_cfg.get("filter_order", 6))
    buffer_mode = str(script_cfg.get("filter_buffer_mode", "reflect")).strip().lower()
    if buffer_mode not in {"reflect", "symmetric", "edge"}:
        raise ValueError("filter_buffer_mode must be reflect, symmetric, or edge.")
    las_ai_source = str(script_cfg.get("las_ai_source", "filtered_shifted_las")).strip().lower()
    min_batch_corr = script_cfg.get("min_batch_corr")
    min_batch_corr = None if min_batch_corr is None else float(min_batch_corr)
    corr_tolerance = float(script_cfg.get("selection_corr_tolerance", 0.02))
    nmae_tolerance = float(script_cfg.get("selection_nmae_tolerance", 0.03))

    wavelet_time_s, wavelet_amp = load_wavelet_csv(wavelet_path)
    wavelet_dt_s = infer_wavelet_dt(wavelet_time_s)
    wavelet_active_half_support_s = compute_wavelet_active_half_support_s(wavelet_time_s, wavelet_amp)
    log_filter_params = _load_auto_tie_log_filter_params(cfg)

    segy_cfg = cfg["segy"]
    segy_options = {
        "iline": segy_cfg["iline_byte"],
        "xline": segy_cfg["xline_byte"],
        "istep": segy_cfg["istep"],
        "xstep": segy_cfg["xstep"],
    }
    survey = open_survey(seismic_file, seismic_type="segy", segy_options=segy_options)
    well_heads = import_well_heads_petrel(well_heads_file)
    well_heads = well_heads.assign(Name_norm=well_heads["Name"].astype(str).str.upper())
    batch_metrics = pd.read_csv(metrics_path)
    batch_metrics = batch_metrics.assign(well_name_norm=batch_metrics["well_name"].astype(str).str.upper())
    modeler = ConvModeler()

    print("=== Frequency Split Diagnosis (Depth) ===")
    print(f"Shifted LAS dir: {shifted_las_dir}")
    print(f"Batch metrics: {metrics_path}")
    print(f"Wavelet: {wavelet_path}")
    print(f"Cutoffs (m): {cutoffs}")
    print(f"LAS AI source: {las_ai_source}")
    print(f"Output dir: {output_dir}")

    rows: list[dict[str, Any]] = []
    for las_path in sorted(shifted_las_dir.glob("*.las")):
        well_name = las_path.stem
        well_norm = well_name.upper()
        head_match = well_heads.loc[well_heads["Name_norm"] == well_norm]
        metric_match = batch_metrics.loc[batch_metrics["well_name_norm"] == well_norm]
        if head_match.empty:
            rows.append({"well_name": well_name, "status": "failed", "error": "well head not matched"})
            continue
        if not metric_match.empty and min_batch_corr is not None:
            batch_corr = float(metric_match.iloc[0].get("corr", np.nan))
            if not np.isfinite(batch_corr) or batch_corr < min_batch_corr:
                rows.append({"well_name": well_name, "status": "skipped", "error": "below min_batch_corr"})
                continue

        try:
            head = head_match.iloc[0]
            kb_m = float(head["Well datum value"])
            md, ai, vp = _read_shifted_las_curves(
                las_path,
                log_filter_params=log_filter_params,
                las_ai_source=las_ai_source,
            )
            tvdss = np.asarray(md, dtype=float) - kb_m
            depth_ai, ai_values = _regularize_depth_curve(tvdss, ai)
            depth_vp, vp_values = _regularize_depth_curve(tvdss, vp)

            seismic_depth_trace = survey.import_seismic_at_well(
                well_x=float(head["Surface X"]),
                well_y=float(head["Surface Y"]),
                domain="depth",
            )
            seis_depth = seismic_depth_trace.basis.astype(float)
            seis_amp = interpolate_nans(seismic_depth_trace.values, method="linear")

            overlap_z_min = max(float(np.nanmin(depth_vp)), float(np.nanmin(seis_depth)))
            overlap_z_max = min(float(np.nanmax(depth_vp)), float(np.nanmax(seis_depth)))
            if overlap_z_max <= overlap_z_min:
                raise ValueError("No TVDSS overlap between shifted LAS and seismic trace.")
            win_mask = (depth_vp >= overlap_z_min) & (depth_vp <= overlap_z_max)
            if int(win_mask.sum()) < 10:
                raise ValueError(f"Too few VP samples in overlap window: {int(win_mask.sum())}.")
            tdt_df = grid.build_local_tdt_from_vp(
                tvdss_m=depth_vp[win_mask],
                vp_mps=vp_values[win_mask],
                md_m=depth_vp[win_mask] + kb_m,
            )
            twt_seis, seis_twt = depth_curve_to_twt(seis_depth, seis_amp, tdt_df, wavelet_dt_s)
            t_min = float(twt_seis[0])
            t_max = float(twt_seis[-1])
            twt_s = np.arange(t_min, t_max + 0.5 * wavelet_dt_s, wavelet_dt_s)
            if twt_s.size < 10:
                raise ValueError(f"Too few common TWT samples: {twt_s.size}")
            seismic_norm = zscore_trace(np.interp(twt_s, twt_seis, seis_twt))
            eval_mask = make_eval_mask(twt_s, wavelet_active_half_support_s)

            for cutoff in cutoffs:
                low_ai_depth = _lowpass_depth_ai(
                    depth_ai,
                    ai_values,
                    cutoff_wavelength_m=cutoff,
                    order=filter_order,
                    buffer_mode=buffer_mode,
                )
                ref_twt, low_ai_twt = depth_curve_to_twt(depth_ai, low_ai_depth, tdt_df, wavelet_dt_s)
                ai_on_twt = np.interp(twt_s, ref_twt, low_ai_twt, left=np.nan, right=np.nan)
                valid_ai = np.isfinite(ai_on_twt) & (ai_on_twt > 0.0)
                reflectivity = np.zeros_like(twt_s, dtype=np.float32)
                reflectivity[valid_ai] = _ai_to_reflectivity(ai_on_twt[valid_ai])
                synthetic_raw = modeler(wavelet_amp, reflectivity)
                metric = metrics_for_synthetic(seismic_norm, synthetic_raw, eval_mask & valid_ai)
                rows.append(
                    {
                        "well_name": well_name,
                        "status": "ok",
                        "cutoff_wavelength_m": cutoff,
                        "corr": metric["corr"],
                        "nmae": metric["nmae"],
                        "scale": metric["scale"],
                        "signed_ls_scale": metric["signed_ls_scale"],
                        "n_eval_samples": metric["n_eval_samples"],
                        "las_ai_source": las_ai_source,
                        "filter_order": filter_order,
                        "filter_buffer_mode": buffer_mode,
                    }
                )
            print(f"OK: {well_name}")
        except Exception as exc:
            rows.append({"well_name": well_name, "status": "failed", "error": str(exc)})
            print(f"FAILED: {well_name}: {exc}")

    metrics_df = pd.DataFrame(rows)
    metrics_csv = output_dir / "cutoff_sweep_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    ok = metrics_df.loc[metrics_df["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError(f"No successful cutoff sweep rows; inspect {metrics_csv}.")

    aggregate = (
        ok.groupby("cutoff_wavelength_m")
        .agg(
            n_wells=("well_name", "nunique"),
            median_corr=("corr", "median"),
            mean_corr=("corr", "mean"),
            p25_corr=("corr", lambda x: float(np.nanpercentile(x, 25))),
            p75_corr=("corr", lambda x: float(np.nanpercentile(x, 75))),
            median_nmae=("nmae", "median"),
            mean_nmae=("nmae", "mean"),
            p25_nmae=("nmae", lambda x: float(np.nanpercentile(x, 25))),
            p75_nmae=("nmae", lambda x: float(np.nanpercentile(x, 75))),
        )
        .reset_index()
        .sort_values("cutoff_wavelength_m")
    )
    aggregate_csv = output_dir / "cutoff_sweep_aggregate.csv"
    aggregate.to_csv(aggregate_csv, index=False)

    recommendation = _select_recommended_cutoff(
        aggregate,
        corr_tolerance=corr_tolerance,
        nmae_tolerance=nmae_tolerance,
    )
    current_cutoff = well_constraints_cfg.get("frequency_split_cutoff_wavelength_m")
    current_cutoff = None if current_cutoff is None else float(current_cutoff)
    aggregate_figure = output_dirs["figures"] / "cutoff_sweep_aggregate.png"
    _plot_aggregate(
        aggregate,
        figure_path=aggregate_figure,
        current_cutoff=current_cutoff,
        recommended_cutoff=recommendation.get("recommended_cutoff_wavelength_m"),
    )

    well_figure_paths: dict[str, str] = {}
    for well_name, well_df in ok.groupby("well_name"):
        safe = sanitize_filename(str(well_name))
        path = output_dirs["well_figures"] / f"cutoff_sweep_{safe}.png"
        _plot_well_metrics(well_df, path)
        well_figure_paths[str(well_name)] = repo_relative_path(path, root=REPO_ROOT)

    summary = {
        "created_at": datetime.now().isoformat(),
        "source_script": Path(__file__).name,
        "source_batch_dir": repo_relative_path(source_batch_dir, root=REPO_ROOT),
        "shifted_las_dir": repo_relative_path(shifted_las_dir, root=REPO_ROOT),
        "metrics_file": repo_relative_path(metrics_path, root=REPO_ROOT),
        "wavelet_path": repo_relative_path(wavelet_path, root=REPO_ROOT),
        "candidate_cutoff_wavelength_m": cutoffs,
        "current_well_constraints_cutoff_wavelength_m": current_cutoff,
        "las_ai_source": las_ai_source,
        "filter_order": filter_order,
        "filter_buffer_mode": buffer_mode,
        "log_filter_params": log_filter_params,
        "selection": recommendation,
        "outputs": {
            "metrics_csv": repo_relative_path(metrics_csv, root=REPO_ROOT),
            "aggregate_csv": repo_relative_path(aggregate_csv, root=REPO_ROOT),
            "aggregate_figure": repo_relative_path(aggregate_figure, root=REPO_ROOT),
            "well_figures": well_figure_paths,
        },
    }
    summary_path = output_dirs["metadata"] / "cutoff_recommendation.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved aggregate: {aggregate_csv}")
    print(f"Saved recommendation: {summary_path}")
    print(f"Saved figures: {output_dirs['figures']}")
    print(f"Recommended cutoff wavelength: {recommendation.get('recommended_cutoff_wavelength_m')}")


if __name__ == "__main__":
    main()
