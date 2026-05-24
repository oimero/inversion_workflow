"""Run time-domain well auto-tie route 1: vertical wells with TDT.

The script builds a full tie plan for all inventory wells, then executes only
enabled routes.  The first implementation enables ``vertical_with_tdt`` by
default and leaves later routes as explicit skipped/rejected plan rows.

Usage::

    python scripts/well_auto_tie.py
    python scripts/well_auto_tie.py --config experiments/common.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

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

from cup.seismic.survey import open_survey
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sanitize_filename, write_json
from cup.well.assets import build_file_lookup
from cup.well.depth_time import (
    build_vp_rho_logset_from_preprocessed_las,
    read_time_depth_table,
    validate_time_depth_table,
    write_time_depth_table_csv,
)
from cup.well.tie import TieRoute, WellTiePlan, WellTieResult, build_tie_plan, plans_dataframe, results_dataframe


# =============================================================================
# CLI and config
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--well", type=str, default=None, help="Optional single well execution filter.")
    return parser.parse_args()


def _merge_dict_defaults(config: dict[str, Any], key: str, defaults: dict[str, Any]) -> None:
    value = config.get(key)
    if value is None:
        config[key] = dict(defaults)
        return
    if not isinstance(value, dict):
        raise ValueError(f"well_auto_tie.{key} must be a mapping.")
    merged = dict(defaults)
    merged.update(value)
    config[key] = merged


def _script_config(cfg: dict[str, Any]) -> dict[str, Any]:
    script_cfg = dict(cfg.get("well_auto_tie") or {})
    _merge_dict_defaults(
        script_cfg,
        "source_runs",
        {
            "mode": "latest",
            "well_inventory_dir": None,
            "las_curve_screen_dir": None,
            "log_preprocess_dir": None,
            "well_trajectory_qc_dir": None,
        },
    )
    script_cfg.setdefault("inventory_file", None)
    script_cfg.setdefault("curve_screen_file", None)
    script_cfg.setdefault("preprocess_status_file", None)
    script_cfg.setdefault("preprocessed_las_dir", None)
    script_cfg.setdefault("trajectory_qc_file", None)
    script_cfg.setdefault("time_depth_dir", "time_depth_table")
    script_cfg.setdefault("well_trace_dir", "all_well_trace")
    _merge_dict_defaults(
        script_cfg,
        "seismic",
        {"file": "raw/obn-clipped-240-912-872-1544.zgy", "type": "zgy"},
    )
    script_cfg.setdefault("enabled_routes", ["vertical_with_tdt"])
    script_cfg.setdefault("tutorial_model", "tutorial/trained_net_state_dict.pt")
    script_cfg.setdefault("tutorial_params", "tutorial/network_parameters.yaml")
    script_cfg.setdefault("target_crop_ms", 201.0)
    _merge_dict_defaults(
        script_cfg,
        "search_space",
        {
            "logs_median_size_values": [51, 71, 91, 111],
            "logs_median_threshold_bounds": [0.5, 3.0],
            "logs_std_bounds": [20, 50],
            "table_t_shift_bounds": [-0.030, 0.030],
        },
    )
    _merge_dict_defaults(script_cfg, "search_params", {"num_iters": 60, "similarity_std": 0.02})
    _merge_dict_defaults(
        script_cfg,
        "wavelet_scaling",
        {"min_scale": 50000, "max_scale": 500000, "num_iters": 60},
    )
    _merge_dict_defaults(
        script_cfg,
        "reject",
        {"allow_near_outside": False, "min_tie_samples": 64, "min_valid_log_fraction": 0.7},
    )
    _merge_dict_defaults(script_cfg, "coarse_anchor", {"enabled": True, "apply_to_routes": []})
    return script_cfg


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _resolve_repo_path(value: str | Path) -> Path:
    return resolve_relative_path(value, root=REPO_ROOT)


def _resolve_data_path(value: str | Path, *, data_root: Path) -> Path:
    return resolve_relative_path(value, root=data_root)


def _resolve_output_dir(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    if args.output_dir is not None:
        return _resolve_repo_path(args.output_dir)
    output_root = _resolve_repo_path(str(cfg.get("output_root", "scripts/output")))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"well_auto_tie_{timestamp}"


def _discover_latest_dir(cfg: dict[str, Any], prefix: str) -> Path:
    output_root = _resolve_repo_path(str(cfg.get("output_root", "scripts/output")))
    all_candidates = [path for path in output_root.glob(f"{prefix}_*") if path.is_dir()]
    timestamped = [path for path in all_candidates if re.fullmatch(rf"{re.escape(prefix)}_\d{{8}}_\d{{6}}", path.name)]
    candidates = sorted(timestamped or all_candidates, key=lambda path: path.name)
    if not candidates:
        raise FileNotFoundError(f"No {prefix}_* output directory found under output_root.")
    return candidates[-1]


def _resolve_inputs(cfg: dict[str, Any], script_cfg: dict[str, Any]) -> dict[str, Path | None]:
    source_runs = dict(script_cfg.get("source_runs") or {})

    inventory_dir = (
        _resolve_repo_path(source_runs["well_inventory_dir"])
        if source_runs.get("well_inventory_dir") is not None
        else _discover_latest_dir(cfg, "well_inventory")
    )
    screen_dir = (
        _resolve_repo_path(source_runs["las_curve_screen_dir"])
        if source_runs.get("las_curve_screen_dir") is not None
        else _discover_latest_dir(cfg, "las_curve_screen")
    )
    preprocess_dir = (
        _resolve_repo_path(source_runs["log_preprocess_dir"])
        if source_runs.get("log_preprocess_dir") is not None
        else _discover_latest_dir(cfg, "log_preprocess")
    )

    trajectory_qc_dir = None
    if source_runs.get("well_trajectory_qc_dir") is not None:
        trajectory_qc_dir = _resolve_repo_path(source_runs["well_trajectory_qc_dir"])
    else:
        try:
            trajectory_qc_dir = _discover_latest_dir(cfg, "well_trajectory_qc")
        except FileNotFoundError:
            trajectory_qc_dir = None

    inventory_file = (
        _resolve_repo_path(script_cfg["inventory_file"])
        if script_cfg.get("inventory_file") is not None
        else inventory_dir / "well_inventory.csv"
    )
    curve_screen_file = (
        _resolve_repo_path(script_cfg["curve_screen_file"])
        if script_cfg.get("curve_screen_file") is not None
        else screen_dir / "well_curve_screen.csv"
    )
    preprocess_status_file = (
        _resolve_repo_path(script_cfg["preprocess_status_file"])
        if script_cfg.get("preprocess_status_file") is not None
        else preprocess_dir / "well_preprocess_status.csv"
    )
    trajectory_qc_file = (
        _resolve_repo_path(script_cfg["trajectory_qc_file"])
        if script_cfg.get("trajectory_qc_file") is not None
        else (trajectory_qc_dir / "well_trajectory_qc.csv" if trajectory_qc_dir is not None else None)
    )

    return {
        "inventory_file": inventory_file,
        "curve_screen_file": curve_screen_file,
        "preprocess_status_file": preprocess_status_file,
        "preprocessed_las_dir": (
            _resolve_repo_path(script_cfg["preprocessed_las_dir"])
            if script_cfg.get("preprocessed_las_dir") is not None
            else preprocess_dir / "preprocessed_las"
        ),
        "trajectory_qc_file": trajectory_qc_file if trajectory_qc_file is not None and trajectory_qc_file.exists() else None,
    }


def _segy_options(seismic_cfg: dict[str, Any]) -> dict[str, int]:
    mapping = {
        "iline": "iline",
        "xline": "xline",
        "istep": "istep",
        "xstep": "xstep",
        "iline_byte": "iline",
        "xline_byte": "xline",
    }
    options: dict[str, int] = {}
    for key, target in mapping.items():
        value = seismic_cfg.get(key)
        if value is not None:
            options[target] = int(value)
    return options


def _series_value_counts(series: pd.Series) -> dict[str, int]:
    if series.empty:
        return {}
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).sort_index().items()}


# =============================================================================
# Output helpers
# =============================================================================


def _build_output_paths(output_dir: Path, well_name: str) -> dict[str, Path]:
    safe = sanitize_filename(well_name)
    figure_dir = output_dir / "figures" / safe
    return {
        "wavelet": output_dir / "wavelets" / f"wavelet_201ms_{safe}.csv",
        "initial_tdt": output_dir / "time_depth" / f"initial_tdt_{safe}.csv",
        "optimized_tdt": output_dir / "time_depth" / f"optimized_tdt_{safe}.csv",
        "synthetic_qc": output_dir / "synthetic_qc" / f"tie_qc_{safe}.csv",
        "seismic_trace": output_dir / "seismic_trace" / f"seismic_trace_{safe}.csv",
        "figure_dir": figure_dir,
        "fig_objective": figure_dir / "optimization_objective.png",
        "fig_tdt": figure_dir / "time_depth_table.png",
        "fig_tie": figure_dir / "synthetic_match.png",
        "fig_wavelet": figure_dir / "wavelet.png",
    }


def _ensure_output_dirs(output_dir: Path) -> None:
    for name in ["wavelets", "time_depth", "synthetic_qc", "seismic_trace", "figures"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def _save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
    except RuntimeError:
        pass
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _build_auto_tie_search_space(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "name": "logs_median_size",
            "type": "choice",
            "values": list(config["logs_median_size_values"]),
            "value_type": "int",
            "is_ordered": True,
            "sort_values": True,
        },
        {
            "name": "logs_median_threshold",
            "type": "range",
            "bounds": list(config["logs_median_threshold_bounds"]),
            "value_type": "float",
        },
        {"name": "logs_std", "type": "range", "bounds": list(config["logs_std_bounds"]), "value_type": "float"},
        {
            "name": "table_t_shift",
            "type": "range",
            "bounds": list(config["table_t_shift_bounds"]),
            "value_type": "float",
        },
    ]


def _crop_wavelet_center_energy_normalize(wavelet: Any, target_ms: float) -> tuple[Any, dict[str, Any]]:
    from wtie.processing.grid import Wavelet

    dt = float(wavelet.sampling_rate)
    target_s = float(target_ms) / 1000.0
    n_target = int(round(target_s / dt))
    if n_target % 2 == 0:
        n_target += 1
    n_target = min(n_target, int(wavelet.size))
    if n_target % 2 == 0:
        n_target -= 1
    center_idx = int(np.argmin(np.abs(wavelet.basis)))
    half = n_target // 2
    start = max(0, center_idx - half)
    end = start + n_target
    if end > int(wavelet.size):
        end = int(wavelet.size)
        start = end - n_target
    values = np.asarray(wavelet.values[start:end], dtype=np.float64).copy()
    basis = np.asarray(wavelet.basis[start:end], dtype=np.float64).copy()
    energy = float(np.sqrt(np.sum(values**2)))
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError("Cannot normalize a zero-energy wavelet.")
    cropped = Wavelet(values / energy, basis, name="Auto well tie wavelet cropped energy-normalized")
    return cropped, {
        "target_ms": float(target_ms),
        "dt_s": dt,
        "original_samples": int(wavelet.size),
        "cropped_samples": int(cropped.size),
        "pre_normalization_l2_energy": energy,
    }


def _scaled_synthetic_metrics(modeler: Any, wavelet: Any, reflectivity: Any, seismic: Any) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    synthetic_raw = np.asarray(modeler(wavelet.values, reflectivity.values), dtype=np.float64)
    seismic_values = np.asarray(seismic.values, dtype=np.float64)
    seismic_norm = seismic_values - float(np.nanmean(seismic_values))
    std = float(np.nanstd(seismic_norm))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("Seismic trace has zero standard deviation.")
    seismic_norm = seismic_norm / std
    denom = max(float(np.dot(synthetic_raw, synthetic_raw)), 1e-12)
    scale = float(np.dot(seismic_norm, synthetic_raw) / denom)
    synthetic = scale * synthetic_raw
    corr = float(np.corrcoef(seismic_norm, synthetic)[0, 1]) if np.std(synthetic) > 0 else np.nan
    nmae = float(np.sum(np.abs(seismic_norm - synthetic)) / max(np.sum(np.abs(seismic_norm)), 1e-12))
    return seismic_norm, synthetic, corr, nmae, scale


def _compute_table_synthetic_metrics(logset_md: Any, seismic: Any, table: Any, wavelet: Any, modeler: Any, dt_s: float) -> tuple[float, float]:
    from wtie.optimize import tie as tie_ops

    logset_twt = tie_ops.convert_logs_from_md_to_twt(logset_md, None, table, dt_s)
    reflectivity = tie_ops.compute_reflectivity(logset_twt, angle_range=seismic.angle_range)
    seismic_match, reflectivity_match = tie_ops.match_seismic_and_reflectivity(seismic, reflectivity)
    _, _, corr, nmae, _ = _scaled_synthetic_metrics(modeler, wavelet, reflectivity_match, seismic_match)
    return corr, nmae


def _write_qc_figures(paths: dict[str, Path], inputs_table: Any, outputs: Any, cropped_wavelet: Any, seismic_norm: np.ndarray, synthetic: np.ndarray) -> None:
    from wtie.utils import viz

    try:
        fig, _ = outputs.plot_optimization_objective(figsize=(6, 3))
        _save_current_figure(paths["fig_objective"])
    except Exception:
        plt.close("all")

    fig, ax = viz.plot_td_table(inputs_table, plot_params={"label": "initial"})
    viz.plot_td_table(outputs.table, plot_params={"label": "optimized"}, fig_axes=(fig, ax))
    ax.legend(loc="best")
    _save_current_figure(paths["fig_tdt"])

    t_ms = np.asarray(outputs.seismic.basis, dtype=np.float64) * 1000.0
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    axes[0].plot(outputs.r.values, t_ms, lw=0.8, color="tab:purple")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Reflectivity")
    axes[0].set_ylabel("TWT (ms)")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(seismic_norm, t_ms, lw=0.9, label="Seismic", color="black")
    axes[1].plot(synthetic, t_ms, lw=0.9, label="Synthetic", color="tab:red")
    axes[1].set_xlabel("Normalized amplitude")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(seismic_norm - synthetic, t_ms, lw=0.9, color="tab:gray")
    axes[2].axvline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[2].set_xlabel("Residual")
    axes[2].grid(True, alpha=0.25)
    _save_current_figure(paths["fig_tie"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(outputs.wavelet.basis * 1000.0, outputs.wavelet.values / np.max(np.abs(outputs.wavelet.values)), label="raw")
    axes[0].plot(
        cropped_wavelet.basis * 1000.0,
        cropped_wavelet.values / np.max(np.abs(cropped_wavelet.values)),
        label="cropped",
    )
    axes[0].axvline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[0].set_xlabel("Time (ms)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)
    freq = np.fft.rfftfreq(cropped_wavelet.size, d=cropped_wavelet.sampling_rate)
    spec = np.abs(np.fft.rfft(cropped_wavelet.values))
    axes[1].plot(freq, spec / max(float(spec.max()), 1e-12))
    axes[1].set_xlim(0, min(125, float(freq[-1])))
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].grid(True, alpha=0.25)
    _save_current_figure(paths["fig_wavelet"])


# =============================================================================
# Route execution
# =============================================================================


def _run_vertical_with_tdt(
    *,
    plan: WellTiePlan,
    survey: Any,
    wavelet_extractor: Any,
    modeler: Any,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[WellTieResult, dict[str, Any]]:
    from wtie.optimize import autotie
    from wtie.utils.datasets.utils import InputSet

    if plan.surface_x is None or plan.surface_y is None:
        raise ValueError("Plan has no valid surface XY.")

    paths = _build_output_paths(output_dir, plan.well_name)
    paths["figure_dir"].mkdir(parents=True, exist_ok=True)
    input_las = _resolve_repo_path(plan.input_las)
    time_depth_file = _resolve_repo_path(plan.time_depth_file)

    logset_md = build_vp_rho_logset_from_preprocessed_las(input_las)
    table = read_time_depth_table(time_depth_file, domain="md")
    overlap = validate_time_depth_table(table, logset_md.basis, min_overlap_samples=int(config["reject"]["min_tie_samples"]))
    seismic = survey.import_seismic_at_well(plan.surface_x, plan.surface_y, domain="time")
    pd.DataFrame({"twt_s": seismic.basis, "seismic": seismic.values}).to_csv(paths["seismic_trace"], index=False)
    write_time_depth_table_csv(table, paths["initial_tdt"])

    wavelet_scaling_params = {
        "wavelet_min_scale": config["wavelet_scaling"]["min_scale"],
        "wavelet_max_scale": config["wavelet_scaling"]["max_scale"],
        "num_iters": config["wavelet_scaling"]["num_iters"],
    }
    search_params = {
        "num_iters": config["search_params"]["num_iters"],
        "similarity_std": config["search_params"]["similarity_std"],
        "suppress_runtime_warnings": True,
        "show_all_warnings": False,
    }
    outputs = autotie.tie_v1(
        InputSet(logset_md=logset_md, seismic=seismic, table=table, wellpath=None),  # type: ignore[arg-type]
        wavelet_extractor,
        modeler,
        wavelet_scaling_params,
        search_params=search_params,
        search_space=_build_auto_tie_search_space(config["search_space"]),
        stretch_and_squeeze_params=None,
    )

    cropped_wavelet, crop_info = _crop_wavelet_center_energy_normalize(outputs.wavelet, float(config["target_crop_ms"]))
    pd.DataFrame({"time_s": cropped_wavelet.basis, "amplitude": cropped_wavelet.values}).to_csv(paths["wavelet"], index=False)
    write_time_depth_table_csv(outputs.table, paths["optimized_tdt"])

    seismic_norm, synthetic, optimized_corr, optimized_nmae, synthetic_scale = _scaled_synthetic_metrics(
        modeler, cropped_wavelet, outputs.r, outputs.seismic
    )
    initial_corr, initial_nmae = _compute_table_synthetic_metrics(
        logset_md, seismic, table, cropped_wavelet, modeler, float(wavelet_extractor.expected_sampling)
    )

    qc_df = pd.DataFrame(
        {
            "twt_s": outputs.seismic.basis,
            "seismic_norm": seismic_norm,
            "reflectivity": outputs.r.values,
            "synthetic_cropped_scaled": synthetic,
            "residual": seismic_norm - synthetic,
        }
    )
    qc_df.to_csv(paths["synthetic_qc"], index=False)
    _write_qc_figures(paths, table, outputs, cropped_wavelet, seismic_norm, synthetic)

    best_params, _ = outputs.ax_client.get_best_parameters()
    best_shift_ms = float(best_params.get("table_t_shift", 0.0)) * 1000.0
    result = WellTieResult(
        well_name=plan.well_name,
        route=plan.route,
        tie_status="success",
        initial_corr=initial_corr,
        optimized_corr=optimized_corr,
        optimized_nmae=optimized_nmae,
        best_table_shift_ms=best_shift_ms,
        wavelet_file=repo_relative_path(paths["wavelet"], root=REPO_ROOT),
        optimized_tdt_file=repo_relative_path(paths["optimized_tdt"], root=REPO_ROOT),
        qc_figure_dir=repo_relative_path(paths["figure_dir"], root=REPO_ROOT),
        reasons="",
    )
    extra = {
        "initial_nmae": initial_nmae,
        "synthetic_scale": synthetic_scale,
        "crop_info": crop_info,
        "overlap": overlap,
        "best_parameters": best_params,
        "synthetic_qc_file": repo_relative_path(paths["synthetic_qc"], root=REPO_ROOT),
        "seismic_trace_file": repo_relative_path(paths["seismic_trace"], root=REPO_ROOT),
        "initial_tdt_file": repo_relative_path(paths["initial_tdt"], root=REPO_ROOT),
    }
    return result, extra


# =============================================================================
# Main
# =============================================================================


def _load_wavelet_extractor(model_path: Path, params_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"tutorial_model does not exist: {model_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"tutorial_params does not exist: {params_path}")
    from wtie.utils.datasets import tutorial as tutorial_mod

    with params_path.open("r", encoding="utf-8") as fp:
        training_parameters = yaml.load(fp, Loader=yaml.Loader)
    return tutorial_mod.load_wavelet_extractor(training_parameters, model_path)


def _write_wavelet_inventory(results: Sequence[WellTieResult], output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.tie_status != "success" or not result.wavelet_file:
            continue
        wavelet_path = _resolve_repo_path(result.wavelet_file)
        df = pd.read_csv(wavelet_path)
        dt_s = float(np.median(np.diff(df["time_s"].to_numpy(dtype=np.float64)))) if len(df) > 1 else np.nan
        rows.append(
            {
                "source_well": result.well_name,
                "route": result.route,
                "wavelet_file": result.wavelet_file,
                "dt_s": dt_s,
                "n_samples": int(len(df)),
                "tie_corr": result.optimized_corr,
                "tie_nmae": result.optimized_nmae,
                "usable_as_candidate": True,
                "reasons": "",
            }
        )
    columns = [
        "source_well",
        "route",
        "wavelet_file",
        "dt_s",
        "n_samples",
        "tie_corr",
        "tie_nmae",
        "usable_as_candidate",
        "reasons",
    ]
    wavelet_df = pd.DataFrame.from_records(rows, columns=columns)
    wavelet_df.to_csv(output_dir / "wavelet_inventory.csv", index=False)
    return wavelet_df


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = _script_config(cfg)
    data_root = _resolve_repo_path(str(cfg.get("data_root", "data")))
    output_dir = _resolve_output_dir(args, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_output_dirs(output_dir)

    paths = _resolve_inputs(cfg, script_cfg)
    inventory_df = pd.read_csv(paths["inventory_file"])
    curve_screen_df = pd.read_csv(paths["curve_screen_file"])
    preprocess_df = pd.read_csv(paths["preprocess_status_file"])
    trajectory_df = pd.read_csv(paths["trajectory_qc_file"]) if paths["trajectory_qc_file"] is not None else None

    time_depth_dir = _resolve_data_path(script_cfg["time_depth_dir"], data_root=data_root)
    well_trace_dir = _resolve_data_path(script_cfg["well_trace_dir"], data_root=data_root)
    time_depth_lookup = build_file_lookup(time_depth_dir.iterdir() if time_depth_dir.exists() else [], asset_label=str(time_depth_dir))
    trace_lookup = build_file_lookup(well_trace_dir.iterdir() if well_trace_dir.exists() else [], asset_label=str(well_trace_dir))

    plans = build_tie_plan(
        inventory_df=inventory_df,
        preprocess_df=preprocess_df,
        trajectory_df=trajectory_df,
        time_depth_lookup=time_depth_lookup,
        trace_lookup=trace_lookup,
        enabled_routes=script_cfg["enabled_routes"],
        allow_near_outside=_as_bool(script_cfg["reject"]["allow_near_outside"]),
    )
    if args.well:
        wanted = args.well.strip().casefold()
        plans = [plan for plan in plans if plan.well_name.casefold() == wanted]
        if not plans:
            raise ValueError(f"Well {args.well!r} was not found in tie plan.")

    plan_df = plans_dataframe(plans)
    plan_path = output_dir / "well_tie_plan.csv"
    plan_df.to_csv(plan_path, index=False)

    seismic_cfg = dict(script_cfg["seismic"])
    seismic_file = _resolve_data_path(seismic_cfg["file"], data_root=data_root)
    survey = open_survey(
        seismic_file,
        seismic_type=str(seismic_cfg.get("type", "segy")),
        segy_options=_segy_options(seismic_cfg) or None,
    )

    model_path = _resolve_data_path(script_cfg["tutorial_model"], data_root=data_root)
    params_path = _resolve_data_path(script_cfg["tutorial_params"], data_root=data_root)

    results: list[WellTieResult] = []
    result_extras: dict[str, Any] = {}
    planned_to_run = [plan for plan in plans if plan.route_status == "planned" and plan.route == TieRoute.VERTICAL_WITH_TDT.value]
    if planned_to_run:
        from wtie.modeling.modeling import ConvModeler

        wavelet_extractor = _load_wavelet_extractor(model_path, params_path)
        modeler = ConvModeler()
        for plan in planned_to_run:
            try:
                result, extra = _run_vertical_with_tdt(
                    plan=plan,
                    survey=survey,
                    wavelet_extractor=wavelet_extractor,
                    modeler=modeler,
                    config=script_cfg,
                    output_dir=output_dir,
                )
                results.append(result)
                result_extras[plan.well_name] = extra
            except Exception as exc:
                results.append(
                    WellTieResult(
                        well_name=plan.well_name,
                        route=plan.route,
                        tie_status="failed",
                        reasons=str(exc),
                    )
                )

    metrics_df = results_dataframe(results)
    metrics_path = output_dir / "well_tie_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    rejected_df = plan_df.loc[plan_df["route_status"] == "rejected"].copy()
    rejected_path = output_dir / "rejected_wells.csv"
    rejected_df.to_csv(rejected_path, index=False)
    wavelet_df = _write_wavelet_inventory(results, output_dir)

    run_summary = {
        "script": "well_auto_tie.py",
        "config_file": repo_relative_path(args.config, root=REPO_ROOT),
        "inputs": {
            "inventory_file": repo_relative_path(paths["inventory_file"], root=REPO_ROOT),
            "curve_screen_file": repo_relative_path(paths["curve_screen_file"], root=REPO_ROOT),
            "preprocess_status_file": repo_relative_path(paths["preprocess_status_file"], root=REPO_ROOT),
            "trajectory_qc_file": repo_relative_path(paths["trajectory_qc_file"], root=REPO_ROOT)
            if paths["trajectory_qc_file"] is not None
            else None,
            "time_depth_dir": repo_relative_path(time_depth_dir, root=REPO_ROOT),
            "seismic_file": repo_relative_path(seismic_file, root=REPO_ROOT),
            "tutorial_model": repo_relative_path(model_path, root=REPO_ROOT),
            "tutorial_params": repo_relative_path(params_path, root=REPO_ROOT),
        },
        "enabled_routes": list(script_cfg["enabled_routes"]),
        "input_row_counts": {
            "inventory": int(len(inventory_df)),
            "curve_screen": int(len(curve_screen_df)),
            "preprocess_status": int(len(preprocess_df)),
            "trajectory_qc": int(len(trajectory_df)) if trajectory_df is not None else 0,
        },
        "route_counts": _series_value_counts(plan_df["route"]),
        "route_status_counts": _series_value_counts(plan_df["route_status"]),
        "tie_status_counts": _series_value_counts(metrics_df["tie_status"]) if "tie_status" in metrics_df.columns else {},
        "planned_run_count": int(len(planned_to_run)),
        "successful_tie_count": int((metrics_df["tie_status"] == "success").sum()) if not metrics_df.empty else 0,
        "wavelet_count": int(len(wavelet_df)),
        "result_extras": result_extras,
        "paths": {
            "well_tie_plan": repo_relative_path(plan_path, root=REPO_ROOT),
            "well_tie_metrics": repo_relative_path(metrics_path, root=REPO_ROOT),
            "rejected_wells": repo_relative_path(rejected_path, root=REPO_ROOT),
            "wavelet_inventory": repo_relative_path(output_dir / "wavelet_inventory.csv", root=REPO_ROOT),
        },
    }
    write_json(output_dir / "run_summary.json", run_summary)
    print(
        f"Wrote auto-tie plan for {len(plan_df)} wells to {repo_relative_path(output_dir, root=REPO_ROOT)}; "
        f"executed {len(planned_to_run)} vertical_with_tdt wells."
    )


if __name__ == "__main__":
    main()
