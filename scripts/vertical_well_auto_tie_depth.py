"""Depth-domain auto well tie wavelet extraction.

Reads a LAS curve and depth-domain well-location seismic trace, builds a local
time-depth table from Vp, runs ``autotie.tie_v1``, then crops the resulting
wavelet to a target length centered at 0 ms with L2 energy normalization.

Usage::

    python scripts/vertical_well_auto_tie_depth.py --well NW11
    python scripts/vertical_well_auto_tie_depth.py --config experiments/common_depth.yaml --well NW11
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

from cup.utils.io import load_yaml_config, resolve_relative_path, sanitize_filename  # noqa: E402

if TYPE_CHECKING:
    from wtie.modeling.modeling import ConvModeler
    from wtie.processing.grid import LogSet, Reflectivity, Seismic, Wavelet

matplotlib.use("Agg")

plt.rcParams["figure.dpi"] = 120
pd.set_option("display.max_columns", 30)


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
        help="Well name (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to scripts/output/vertical_well_auto_tie_depth_<timestamp>.",
    )
    return parser.parse_args()


def build_output_tree(output_dir: Path, well_name: str) -> dict[str, Path]:
    dirs = {
        "wavelet_raw": output_dir / "wavelet_raw",
        "depth_match": output_dir / "depth_match",
        "synthetic_qc": output_dir / "synthetic_qc",
        "figures": output_dir / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    safe = sanitize_filename(well_name)
    return {
        "root": output_dir,
        "wavelet_raw_csv": dirs["wavelet_raw"] / f"auto_well_tie_wavelet_raw_{safe}.csv",
        "local_tdt_csv": dirs["depth_match"] / f"local_tdt_md_{safe}.csv",
        "seismic_twt_csv": dirs["depth_match"] / f"seismic_twt_from_depth_{safe}.csv",
        "synthetic_qc_raw_csv": dirs["synthetic_qc"] / f"auto_well_tie_synthetic_qc_raw_{safe}.csv",
        "synthetic_qc_cropped_csv": dirs["synthetic_qc"] / f"auto_well_tie_synthetic_qc_cropped_201ms_{safe}.csv",
        "wavelet_final_csv": output_dir / f"wavelet_201ms_{safe}.csv",
        "summary_json": output_dir / f"run_summary_{safe}.json",
        "fig_01_depth_time_match": dirs["figures"] / f"qc_01_depth_time_match_{safe}.png",
        "fig_02_optimization_objective": dirs["figures"] / f"qc_02_optimization_objective_{safe}.png",
        "fig_03a_tie_window_raw": dirs["figures"] / f"qc_03a_tie_window_raw_{safe}.png",
        "fig_03b_cropped_synthetic_vs_seismic": dirs["figures"] / f"qc_03b_cropped_synthetic_vs_seismic_{safe}.png",
        "fig_04_td_table": dirs["figures"] / f"qc_04_td_table_{safe}.png",
        "fig_05_wavelet_raw_vs_cropped": dirs["figures"] / f"qc_05_wavelet_raw_vs_cropped_{safe}.png",
    }


# =============================================================================
# JSON serialization
# =============================================================================


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# =============================================================================
# Auto-tie utilities
# =============================================================================


def build_auto_tie_search_space(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": "logs_median_size",
            "type": "choice",
            "values": config["logs_median_size_values"],
            "value_type": "int",
            "is_ordered": True,
            "sort_values": True,
        },
        {
            "name": "logs_median_threshold",
            "type": "range",
            "bounds": config["logs_median_threshold_bounds"],
            "value_type": "float",
        },
        {
            "name": "logs_std",
            "type": "range",
            "bounds": config["logs_std_bounds"],
            "value_type": "float",
        },
        {
            "name": "table_t_shift",
            "type": "range",
            "bounds": config["table_t_shift_bounds"],
            "value_type": "float",
        },
    ]


# =============================================================================
# Depth / TWT conversion
# =============================================================================


def depth_trace_to_twt(
    depth_tvdss: np.ndarray,
    amp: np.ndarray,
    tdt_df: pd.DataFrame,
    dt_s: float,
) -> "Seismic":
    from wtie.processing.grid import Seismic
    from wtie.processing.logs import interpolate_nans

    depth_tvdss = np.asarray(depth_tvdss, dtype=float)
    amp = interpolate_nans(amp, method="linear")
    z_tdt = tdt_df["tvdss_m"].to_numpy(dtype=float)
    t_tdt = tdt_df["twt_s"].to_numpy(dtype=float)
    z_min = max(float(depth_tvdss.min()), float(z_tdt[0]))
    z_max = min(float(depth_tvdss.max()), float(z_tdt[-1]))
    if z_max <= z_min:
        raise ValueError("Depth trace and local TDT do not overlap.")
    dz = float(np.median(np.diff(z_tdt)))
    z_regular = np.arange(z_min, z_max + 0.5 * dz, dz)
    amp_z = np.interp(z_regular, depth_tvdss, amp)
    twt_z = np.interp(z_regular, z_tdt, t_tdt)
    twt_regular = np.arange(twt_z[0], twt_z[-1] + 0.5 * dt_s, dt_s)
    amp_t = np.interp(twt_regular, twt_z, amp_z)
    return Seismic(amp_t, twt_regular, "twt", name="Depth-derived seismic")


def clip_logset_by_md(
    logset: "LogSet",
    md_min: float,
    md_max: float,
) -> "LogSet":
    from wtie.processing.grid import Log, LogSet

    mask = (logset.basis >= md_min) & (logset.basis <= md_max)
    if np.count_nonzero(mask) < 10:
        raise ValueError("Too few log samples in requested MD window.")
    logs = {}
    for key, log in logset.Logs.items():
        logs[key] = Log(
            log.values[mask],
            log.basis[mask],
            "md",
            name=log.name,
            unit=log.unit,
            allow_nan=log.allow_nan,
        )
    return LogSet(logs)


# =============================================================================
# Wavelet post-processing
# =============================================================================


def crop_wavelet_center_energy_normalize(
    wavelet: "Wavelet",
    target_ms: float,
) -> tuple["Wavelet", dict[str, Any]]:
    from wtie.processing.grid import Wavelet

    dt = float(wavelet.sampling_rate)
    target_s = target_ms / 1000.0
    n_target = int(round(target_s / dt))
    if n_target % 2 == 0:
        n_target += 1
    n_target = min(n_target, wavelet.size)
    if n_target % 2 == 0:
        n_target -= 1
    center_idx = int(np.argmin(np.abs(wavelet.basis)))
    half = n_target // 2
    start = max(0, center_idx - half)
    end = start + n_target
    if end > wavelet.size:
        end = wavelet.size
        start = end - n_target
    values = np.asarray(wavelet.values[start:end], dtype=float).copy()
    basis = np.asarray(wavelet.basis[start:end], dtype=float).copy()
    energy = float(np.sqrt(np.sum(values**2)))
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("Cannot energy-normalize a zero-energy wavelet.")
    values_norm = values / energy
    cropped = Wavelet(values_norm, basis, name="Auto well tie wavelet cropped 201ms energy-normalized")
    info = {
        "target_ms": target_ms,
        "dt_s": dt,
        "original_samples": int(wavelet.size),
        "cropped_samples": int(cropped.size),
        "cropped_span_ms": float((basis[-1] - basis[0]) * 1000.0),
        "cropped_nominal_ms": float(cropped.size * dt * 1000.0),
        "pre_normalization_l2_energy": energy,
        "post_normalization_l2_energy": float(np.sqrt(np.sum(values_norm**2))),
    }
    return cropped, info


def scaled_synthetic_for_qc(
    modeler: "ConvModeler",
    wavelet: "Wavelet",
    reflectivity: "Reflectivity",
    seismic: "Seismic",
) -> tuple[np.ndarray, float, float, float]:
    synthetic_raw = modeler(wavelet.values, reflectivity.values)
    seis = np.asarray(seismic.values, dtype=float)
    seis = seis - np.mean(seis)
    seis_std = np.std(seis)
    if seis_std <= 0:
        raise ValueError("Seismic trace has zero standard deviation.")
    seis_norm = seis / seis_std
    scale = float(np.dot(seis_norm, synthetic_raw) / max(np.dot(synthetic_raw, synthetic_raw), 1e-12))
    synthetic = scale * synthetic_raw
    corr = float(np.corrcoef(seis_norm, synthetic)[0, 1]) if np.std(synthetic) > 0 else np.nan
    nmae = float(np.sum(np.abs(seis_norm - synthetic)) / np.sum(np.abs(seis_norm)))
    return synthetic, scale, corr, nmae


# =============================================================================
# Core logic
# =============================================================================


def run_auto_tie(
    *,
    well_name: str,
    las_file: Path,
    las_vp_unit: str,
    las_rho_unit: str,
    kb_m: float,
    well_x: float,
    well_y: float,
    survey: Any,
    wavelet_extractor: Any,
    dt_s: float,
    search_space_config: dict[str, Any],
    search_params: dict[str, Any],
    wavelet_scaling_params: dict[str, Any],
    target_crop_ms: float,
    output: dict[str, Path],
) -> None:
    from cup.petrel.load import load_vp_rho_logset_from_las
    from wtie.modeling.modeling import ConvModeler
    from wtie.optimize import autotie
    from wtie.processing import grid
    from wtie.processing.logs import interpolate_nans
    from wtie.utils import viz
    from wtie.utils.datasets.utils import InputSet

    modeler = ConvModeler()

    # ── 1) Load well data and well-location seismic trace ──

    logset_md_full = load_vp_rho_logset_from_las(las_file, vp_unit=las_vp_unit, rho_unit=las_rho_unit)
    seismic_depth_trace = survey.import_seismic_at_well(well_x=well_x, well_y=well_y, domain="depth")
    seis_depth = seismic_depth_trace.basis.astype(float)
    seis_amp = interpolate_nans(seismic_depth_trace.values, method="linear")

    print(f"Well XY=({well_x:.2f}, {well_y:.2f}), KB={kb_m:.2f} m")
    print(f"Network expected dt={dt_s * 1000:.1f} ms")
    print(f"Depth trace: z=[{seis_depth[0]:.2f}, {seis_depth[-1]:.2f}] m")

    # ── 2) Overlap window and local time-depth table ──

    md_full = logset_md_full.basis.astype(float)
    tvdss_full = md_full - kb_m

    overlap_z_min = max(float(tvdss_full.min()), float(seis_depth.min()))
    overlap_z_max = min(float(tvdss_full.max()), float(seis_depth.max()))
    if overlap_z_max <= overlap_z_min:
        raise ValueError("Well TVDSS and seismic depth trace do not overlap.")
    overlap_md_min = overlap_z_min + kb_m
    overlap_md_max = overlap_z_max + kb_m

    logset_md = clip_logset_by_md(logset_md_full, overlap_md_min, overlap_md_max)
    md_win = logset_md.basis.astype(float)
    tvdss_win = md_win - kb_m
    tdt_df = grid.build_local_tdt_from_vp(
        tvdss_m=tvdss_win,
        vp_mps=interpolate_nans(logset_md.Vp.values, method="linear"),
        md_m=md_win,
    )
    local_tdt_md = grid.TimeDepthTable(
        twt=tdt_df["twt_s"].to_numpy(),
        md=tdt_df["md_m"].to_numpy(),
    )
    seismic_twt = depth_trace_to_twt(seis_depth, seis_amp, tdt_df, dt_s)

    inputs = InputSet(
        logset_md=logset_md,
        seismic=seismic_twt,
        table=local_tdt_md,
        wellpath=None,  # type: ignore[arg-type]
    )

    print(f"Overlap TVDSS window: {overlap_z_min:.2f} - {overlap_z_max:.2f} m")
    print(f"Overlap MD window: {overlap_md_min:.2f} - {overlap_md_max:.2f} m")
    print(f"Seismic TWT window: {seismic_twt.basis[0]:.4f} - {seismic_twt.basis[-1]:.4f} s")

    # Save depth-time match artifacts
    tdt_df.to_csv(output["local_tdt_csv"], index=False)
    pd.DataFrame({"twt_s": seismic_twt.basis, "seismic": seismic_twt.values}).to_csv(
        output["seismic_twt_csv"], index=False
    )

    # ── 3) Run autotie.tie_v1 ──

    search_space = build_auto_tie_search_space(search_space_config)
    search_params_full = {
        "num_iters": search_params["num_iters"],
        "similarity_std": search_params["similarity_std"],
        "suppress_runtime_warnings": True,
        "show_all_warnings": False,
    }

    outputs = autotie.tie_v1(
        inputs,
        wavelet_extractor,
        modeler,
        wavelet_scaling_params,
        search_params=search_params_full,
        search_space=search_space,  # type: ignore
        stretch_and_squeeze_params=None,  # type: ignore
    )

    raw_wavelet = outputs.wavelet
    raw_wavelet_df = pd.DataFrame({"time_s": raw_wavelet.basis, "amplitude": raw_wavelet.values})
    raw_wavelet_df.to_csv(output["wavelet_raw_csv"], index=False)

    raw_tie_df = pd.DataFrame(
        {
            "twt_s": outputs.seismic.basis,
            "seismic": outputs.seismic.values,
            "reflectivity": outputs.r.values,
            "synthetic_raw_wavelet": outputs.synth_seismic.values,
        }
    )
    raw_tie_df.to_csv(output["synthetic_qc_raw_csv"], index=False)

    print(
        f"Raw auto-tie wavelet samples={raw_wavelet.size}, "  # type: ignore
        f"dt={raw_wavelet.sampling_rate * 1000:.1f} ms, "
        f"span={(raw_wavelet.basis[-1] - raw_wavelet.basis[0]) * 1000:.1f} ms"
    )

    # ── 4) Crop to target length and energy-normalize ──

    cropped_wavelet, crop_info = crop_wavelet_center_energy_normalize(raw_wavelet, target_crop_ms)  # type: ignore
    cropped_wavelet_df = pd.DataFrame({"time_s": cropped_wavelet.basis, "amplitude": cropped_wavelet.values})
    cropped_wavelet_df.to_csv(output["wavelet_final_csv"], index=False)

    cropped_synthetic, cropped_scale, cropped_corr, cropped_nmae = scaled_synthetic_for_qc(
        modeler=modeler,
        wavelet=cropped_wavelet,
        reflectivity=outputs.r,  # type: ignore
        seismic=outputs.seismic,  # type: ignore
    )

    seis_norm = outputs.seismic.values - np.mean(outputs.seismic.values)
    seis_norm = seis_norm / np.std(seis_norm)
    cropped_qc_df = pd.DataFrame(
        {
            "twt_s": outputs.seismic.basis,
            "seismic_norm": seis_norm,
            "reflectivity": outputs.r.values,
            "synthetic_cropped_scaled": cropped_synthetic,
            "residual": seis_norm - cropped_synthetic,
        }
    )
    cropped_qc_df.to_csv(output["synthetic_qc_cropped_csv"], index=False)

    best_parameters, best_values = outputs.ax_client.get_best_parameters()  # type: ignore
    best_objective_means, best_objective_covariances = best_values  # type: ignore

    summary: dict[str, Any] = {
        "well_name": well_name,
        "kb_m": kb_m,
        "overlap_tvdss_min_m": overlap_z_min,
        "overlap_tvdss_max_m": overlap_z_max,
        "auto_tie_search_params": search_params_full,
        "auto_tie_search_space_config": to_jsonable(search_space_config),
        "wavelet_scaling_params": wavelet_scaling_params,
        "auto_tie_best_parameters": to_jsonable(best_parameters),
        "auto_tie_best_objective_means": to_jsonable(best_objective_means),
        "auto_tie_best_objective_covariances": to_jsonable(best_objective_covariances),
        "raw_wavelet_samples": int(raw_wavelet.size),  # type: ignore
        "raw_wavelet_dt_s": float(raw_wavelet.sampling_rate),
        "raw_wavelet_span_ms": float((raw_wavelet.basis[-1] - raw_wavelet.basis[0]) * 1000.0),
        "crop_info": crop_info,
        "cropped_synthetic_scale": cropped_scale,
        "cropped_synthetic_corr": cropped_corr,
        "cropped_synthetic_nmae": cropped_nmae,
    }
    output["summary_json"].write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(crop_info)
    print(f"Cropped synthetic: corr={cropped_corr:.3f}, nmae={cropped_nmae:.3f}, scale={cropped_scale:.3g}")

    # ── 5) QC figures ──

    # Fig 01: depth/time match
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    axes[0].plot(logset_md.Vp.values, logset_md.basis - kb_m, lw=0.8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Vp (m/s)")
    axes[0].set_ylabel("TVDSS (m)")
    axes[0].set_title("Well Vp in overlap")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(seis_amp, seis_depth, lw=0.8, color="tab:gray")
    axes[1].axhspan(overlap_z_min, overlap_z_max, color="tab:green", alpha=0.15)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Amplitude")
    axes[1].set_title("Depth-domain well trace")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(tdt_df["twt_s"] * 1000.0, tdt_df["tvdss_m"], lw=1.1)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Relative TWT (ms)")
    axes[2].set_title("Local time-depth table")
    axes[2].grid(True, alpha=0.25)
    _save_fig(output["fig_01_depth_time_match"])

    # Fig 02: optimization objective
    fig, ax = outputs.plot_optimization_objective(figsize=(6, 3))
    _save_fig(output["fig_02_optimization_objective"])

    # Fig 03a: tie window raw
    _scale = 120000
    fig, axes = outputs.plot_tie_window(wiggle_scale=_scale, figsize=(12.0, 7.5))
    _save_fig(output["fig_03a_tie_window_raw"])

    # Fig 03b: cropped synthetic overlay
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    t_ms = outputs.seismic.basis * 1000.0
    axes[0].plot(outputs.r.values, t_ms, lw=0.8, color="tab:purple")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Reflectivity")
    axes[0].set_ylabel("Relative TWT (ms)")
    axes[0].set_title("Reflectivity")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(seis_norm, t_ms, lw=0.9, label="Seismic", color="black")
    axes[1].plot(cropped_synthetic, t_ms, lw=0.9, label="Synthetic", color="tab:red", alpha=0.85)
    axes[1].set_xlabel("Normalized amplitude")
    axes[1].set_title(f"Auto-tie cropped: corr={cropped_corr:.3f}, nmae={cropped_nmae:.3f}")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(seis_norm - cropped_synthetic, t_ms, lw=0.9, color="tab:gray")
    axes[2].axvline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[2].set_xlabel("Residual")
    axes[2].set_title("Residual")
    axes[2].grid(True, alpha=0.25)
    _save_fig(output["fig_03b_cropped_synthetic_vs_seismic"])

    # Fig 04: TD table comparison
    fig, ax = viz.plot_td_table(inputs.table, plot_params={"label": "initial"})
    viz.plot_td_table(outputs.table, plot_params={"label": "optimized"}, fig_axes=(fig, ax))
    ax.legend(loc="best")
    _save_fig(output["fig_04_td_table"])

    # Fig 05: wavelet raw vs cropped
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(
        raw_wavelet.basis * 1000.0,
        raw_wavelet.values / np.max(np.abs(raw_wavelet.values)),
        lw=1.1,
        label="raw auto-tie",
    )
    axes[0].plot(
        cropped_wavelet.basis * 1000.0,
        cropped_wavelet.values / np.max(np.abs(cropped_wavelet.values)),
        lw=1.2,
        label=f"cropped {target_crop_ms:g} ms + energy norm",
    )
    axes[0].axvline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_title("Wavelet time domain")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    freq_raw = np.fft.rfftfreq(raw_wavelet.size, d=raw_wavelet.sampling_rate)  # type: ignore
    spec_raw = np.abs(np.fft.rfft(raw_wavelet.values))
    freq_crop = np.fft.rfftfreq(cropped_wavelet.size, d=cropped_wavelet.sampling_rate)
    spec_crop = np.abs(np.fft.rfft(cropped_wavelet.values))
    axes[1].plot(freq_raw, spec_raw / np.max(spec_raw), lw=1.1, label="raw auto-tie")
    axes[1].plot(freq_crop, spec_crop / np.max(spec_crop), lw=1.2, label=f"cropped {target_crop_ms:g} ms")
    axes[1].set_xlim(0, min(125, freq_raw[-1]))
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_title("Normalized amplitude spectrum")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)
    _save_fig(output["fig_05_wavelet_raw_vs_cropped"])

    # Print best parameters
    print("\nAuto-tie best parameters:")
    for key in ["logs_median_size", "logs_median_threshold", "logs_std", "table_t_shift"]:
        value = best_parameters.get(key)
        if key == "table_t_shift" and value is not None:
            print(f"  {key}: {value:.6f} s ({value * 1000.0:.2f} ms)")  # type: ignore
        else:
            print(f"  {key}: {value}")
    print(f"  objective means: {best_objective_means}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    data_root = resolve_relative_path(cfg.get("data_root", "data"), root=REPO_ROOT)

    script_cfg = cfg.get("vertical_well_auto_tie_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'vertical_well_auto_tie_depth' section in config.")

    well_name = args.well or script_cfg.get("well_name")
    if not well_name:
        raise ValueError("well_name not set in config or --well.")

    target_crop_ms = float(script_cfg["target_crop_ms"])
    las_vp_unit = str(script_cfg.get("las_vp_unit", "us/m"))
    las_rho_unit = str(script_cfg.get("las_rho_unit", "g/cm3"))

    # Resolve paths
    las_dir = resolve_relative_path(str(script_cfg["las_dir"]), root=data_root)
    las_file = las_dir / f"{well_name}.las"
    if not las_file.exists():
        raise FileNotFoundError(f"LAS file not found: {las_file}")

    well_heads_file = resolve_relative_path(str(cfg["well"]["well_heads_file"]), root=data_root)
    seismic_file = resolve_relative_path(str(cfg["segy"]["file"]), root=data_root)
    tutorial_model = resolve_relative_path(str(script_cfg["tutorial_model"]), root=data_root)
    tutorial_params = resolve_relative_path(str(script_cfg["tutorial_params"]), root=data_root)

    for p in [las_file, well_heads_file, seismic_file, tutorial_model, tutorial_params]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    # Output dir
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / cfg.get("output_root", "scripts/output")
        output_dir = output_root / f"vertical_well_auto_tie_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir

    output = build_output_tree(output_dir, well_name)

    print("=== Depth Auto Well Tie ===")
    print(f"Well name: {well_name}")
    print(f"LAS file: {las_file}")
    print(f"Seismic file: {seismic_file}")
    print(f"Output dir: {output_dir}")
    print(f"Target crop: {target_crop_ms} ms")

    # ── Import heavy dependencies ──

    from cup.petrel.load import import_well_heads_petrel
    from cup.seismic.survey import open_survey
    from wtie.utils.datasets import tutorial as tutorial_mod

    # ── Well head lookup ──

    segy_cfg = cfg["segy"]
    segy_options = {
        "iline": segy_cfg["iline_byte"],
        "xline": segy_cfg["xline_byte"],
        "istep": segy_cfg["istep"],
        "xstep": segy_cfg["xstep"],
    }

    # Column names are fixed by import_well_heads_petrel:
    # Name, Surface X, Surface Y, Well datum value, ...
    well_heads_df = import_well_heads_petrel(well_heads_file)
    well_row = well_heads_df.loc[well_heads_df["Name"] == well_name]
    if well_row.empty:
        raise ValueError(f"Well '{well_name}' not found in well heads file.")
    if well_row.shape[0] != 1:
        raise ValueError(f"Well '{well_name}' duplicated in well heads: {well_row.shape[0]}")
    well_row = well_row.iloc[0]
    kb_m = float(well_row["Well datum value"])
    well_x = float(well_row["Surface X"])
    well_y = float(well_row["Surface Y"])

    # ── Network ──

    with open(tutorial_params, "r", encoding="utf-8") as f:
        training_parameters = yaml.load(f, Loader=yaml.Loader)
    wavelet_extractor = tutorial_mod.load_wavelet_extractor(training_parameters, tutorial_model)
    dt_s = float(wavelet_extractor.expected_sampling)

    # ── Survey ──

    survey = open_survey(seismic_file, seismic_type="segy", segy_options=segy_options)
    geometry_depth = survey.query_geometry(domain="depth")
    il_float, xl_float = survey.coord_to_line(well_x, well_y)
    if not (geometry_depth["inline_min"] <= il_float <= geometry_depth["inline_max"]):
        raise ValueError(f"Inline {il_float} outside survey range.")
    if not (geometry_depth["xline_min"] <= xl_float <= geometry_depth["xline_max"]):
        raise ValueError(f"Crossline {xl_float} outside survey range.")

    # ── Run ──

    wavelet_scaling_params = {
        "wavelet_min_scale": script_cfg["wavelet_scaling"]["min_scale"],
        "wavelet_max_scale": script_cfg["wavelet_scaling"]["max_scale"],
        "num_iters": script_cfg["wavelet_scaling"]["num_iters"],
    }

    run_auto_tie(
        well_name=well_name,
        las_file=las_file,
        las_vp_unit=las_vp_unit,
        las_rho_unit=las_rho_unit,
        kb_m=kb_m,
        well_x=well_x,
        well_y=well_y,
        survey=survey,
        wavelet_extractor=wavelet_extractor,
        dt_s=dt_s,
        search_space_config=script_cfg["search_space"],
        search_params=script_cfg["search_params"],
        wavelet_scaling_params=wavelet_scaling_params,
        target_crop_ms=target_crop_ms,
        output=output,
    )

    # ── Output manifest ──

    print("\n=== Outputs ===")
    for key, path in output.items():
        if key != "root" and path.exists():
            print(f"  {path}")


if __name__ == "__main__":
    main()
