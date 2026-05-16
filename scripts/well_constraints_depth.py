"""Build depth-domain well-derived constraints for GINN workflows.

The script samples shifted LAS AI logs on the depth GINN/LFM axis and writes two
related artifacts from the same well QC pass:

* ``well_resolution_prior_depth.npz`` for stage-2 enhancement.
* ``log_ai_anchor_depth.npz`` for stage-1 log-AI anchor supervision.

Usage::

    python scripts/well_constraints_depth.py
    python scripts/well_constraints_depth.py --config experiments/common_depth.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lasio
import numpy as np
import pandas as pd

# =============================================================================
# Bootstrap
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.petrel.load import import_well_heads_petrel
from cup.seismic.survey import open_survey
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path
from enhance.prior import (
    WellResolutionPriorBundle,
    save_well_resolution_prior_npz,
    summarize_well_resolution_prior,
)
from ginn.anchor import build_log_ai_anchor_bundle, save_log_ai_anchor_npz
from ginn_depth.data import load_lfm_depth_npz
from wtie.optimize import tie as tie_utils
from wtie.optimize.logs import filter_log
from wtie.processing import grid
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
        help="Output directory. Defaults to <output_root>/well_constraints_depth_<timestamp>.",
    )
    return parser.parse_args()


# LAS helpers


def _normalize_name(name: object) -> str:
    return str(name).strip()


def _find_curve(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    normalized = {str(col).strip().upper(): str(col) for col in df.columns}
    for name in names:
        hit = normalized.get(name.upper())
        if hit is not None:
            return hit
    return None


def _read_ai_curves_from_shifted_las(
    path: Path,
    *,
    log_filter_params: dict[str, float | int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    las = lasio.read(path)
    df = las.df()
    md = df.index.to_numpy(dtype=float)
    ai_col = _find_curve(df, ("AI",))
    vp_col = _find_curve(df, ("VP_MPS", "VP"))
    rho_col = _find_curve(df, ("RHO_GCC", "RHO"))

    if ai_col is not None:
        raw_ai = df[ai_col].to_numpy(dtype=float)
    elif vp_col is not None and rho_col is not None:
        raw_ai = df[vp_col].to_numpy(dtype=float) * df[rho_col].to_numpy(dtype=float)
    else:
        raise ValueError("LAS must contain AI or both VP_MPS/VP and RHO_GCC/RHO curves.")

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
        source = f"filtered({vp_col}*{rho_col})"
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
        source = f"filtered({ai_col})"
    else:
        raise ValueError("LAS must contain AI or both VP_MPS/VP and RHO_GCC/RHO curves.")

    return md, raw_ai, np.asarray(filtered_ai, dtype=float), source


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


def _resolve_frequency_split_params(script_cfg: dict[str, Any], lfm_metadata: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(script_cfg.get("frequency_split_enabled", True))
    buffer_mode = str(script_cfg.get("frequency_split_buffer_mode", "reflect")).strip().lower()
    if buffer_mode not in {"reflect", "symmetric", "edge"}:
        raise ValueError("well_constraints_depth.frequency_split_buffer_mode must be reflect, symmetric, or edge.")
    if not enabled:
        return {
            "enabled": False,
            "cutoff_wavelength_m": None,
            "filter_order": None,
            "buffer_mode": buffer_mode,
            "frequency_split_parameter_source": "disabled",
            "cutoff_source": "disabled",
            "filter_order_source": "disabled",
        }

    cutoff_value = script_cfg.get("frequency_split_cutoff_wavelength_m")
    order_value = script_cfg.get("frequency_split_filter_order")
    cutoff_source = "config"
    order_source = "config"
    if cutoff_value is None:
        cutoff_value = lfm_metadata.get("filter_cutoff_wavelength_m")
        cutoff_source = "lfm_metadata"
    if order_value is None:
        order_value = lfm_metadata.get("filter_order")
        order_source = "lfm_metadata"
    if cutoff_value is None or order_value is None:
        raise ValueError(
            "Missing frequency split parameters: set well_constraints_depth.frequency_split_cutoff_wavelength_m "
            "and frequency_split_filter_order, or provide filter_cutoff_wavelength_m/filter_order in AI LFM metadata."
        )

    cutoff = float(cutoff_value)
    order = int(order_value)
    if cutoff <= 0.0 or not np.isfinite(cutoff):
        raise ValueError(f"frequency split cutoff wavelength must be positive, got {cutoff_value!r}.")
    if order <= 0 or order % 2 != 0:
        raise ValueError(f"frequency split filter_order must be a positive even integer, got {order_value!r}.")
    parameter_source = "config" if cutoff_source == "config" or order_source == "config" else "lfm_metadata"
    return {
        "enabled": True,
        "cutoff_wavelength_m": cutoff,
        "filter_order": order,
        "buffer_mode": buffer_mode,
        "frequency_split_parameter_source": parameter_source,
        "cutoff_source": cutoff_source,
        "filter_order_source": order_source,
    }


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return []
    padded = np.concatenate([[False], values, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, changes.size, 2)]


def _split_log_ai_frequency_bands(
    depth: np.ndarray,
    ai: np.ndarray,
    mask: np.ndarray,
    split_params: dict[str, Any],
) -> dict[str, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float64).reshape(-1)
    ai = np.asarray(ai, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1) & np.isfinite(depth) & np.isfinite(ai) & (ai > 0.0)
    if depth.shape != ai.shape or ai.shape != mask.shape:
        raise ValueError("depth, ai, and mask must have matching 1D shapes for frequency split.")

    well_log_ai = np.zeros_like(ai, dtype=np.float32)
    well_low_ai = np.zeros_like(ai, dtype=np.float32)
    well_low_log_ai = np.zeros_like(ai, dtype=np.float32)
    well_high_log_ai = np.zeros_like(ai, dtype=np.float32)
    well_log_ai[mask] = np.log(np.clip(ai[mask], 1e-6, None)).astype(np.float32)

    if not bool(split_params.get("enabled", True)):
        well_low_ai[mask] = ai[mask].astype(np.float32)
    else:
        if depth.size > 1:
            dz_values = np.diff(depth[mask]) if np.count_nonzero(mask) > 1 else np.diff(depth)
        else:
            dz_values = np.asarray([], dtype=np.float64)
        dz_values = dz_values[np.isfinite(dz_values) & (dz_values > 0.0)]
        if dz_values.size == 0:
            raise ValueError("Cannot resolve positive sample step for frequency split.")
        dz = float(np.median(dz_values))
        fs = 1.0 / dz
        highcut = 1.0 / float(split_params["cutoff_wavelength_m"])
        if highcut >= 0.5 * fs:
            raise ValueError(
                f"frequency split cutoff wavelength={split_params['cutoff_wavelength_m']} m is above Nyquist "
                f"for dz={dz:.6g} m."
            )
        order = int(split_params["filter_order"])
        pad_samples = max(1, 3 * order)
        buffer_mode = str(split_params.get("buffer_mode", "reflect"))
        for start, stop in _true_runs(mask):
            values = ai[start:stop].astype(np.float64)
            if values.size <= max(3, order):
                filtered = values
            else:
                run_pad = min(pad_samples, values.size - 1)
                padded = np.pad(values, (run_pad, run_pad), mode=buffer_mode)  # type: ignore
                filtered = apply_butter_lowpass_filter(
                    padded,
                    highcut,
                    fs,
                    order=order,
                    zero_phase=True,
                )[run_pad : run_pad + values.size]
            well_low_ai[start:stop] = np.clip(filtered, 1e-6, None).astype(np.float32)

    low_valid = mask & np.isfinite(well_low_ai) & (well_low_ai > 0.0)
    well_low_log_ai[low_valid] = np.log(np.clip(well_low_ai[low_valid], 1e-6, None)).astype(np.float32)
    well_high_log_ai[low_valid] = (well_log_ai[low_valid] - well_low_log_ai[low_valid]).astype(np.float32)
    return {
        "well_log_ai": well_log_ai,
        "well_low_ai": well_low_ai,
        "well_low_log_ai": well_low_log_ai,
        "well_high_log_ai": well_high_log_ai,
    }


def _confidence_from_corr(corr: object, *, floor: float, span: float, mode: str = "corr") -> float:
    mode = str(mode).strip().lower()
    if mode == "uniform":
        return 1.0
    if mode != "corr":
        raise ValueError("well_constraints_depth.confidence_weight_mode must be 'corr' or 'uniform'.")
    corr_value = _finite_float_or_nan(corr)
    if not np.isfinite(corr_value):
        return 0.0
    if span <= 0.0:
        raise ValueError(f"confidence_corr_span must be positive, got {span}.")
    return float(np.clip((corr_value - floor) / span, 0.0, 1.0))


def _finite_float_or_nan(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _interpolate_ai_to_depth_axis(
    md: np.ndarray,
    ai: np.ndarray,
    kb_m: float,
    depth_axis_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tvdss = np.asarray(md, dtype=float) - float(kb_m)
    ai = np.asarray(ai, dtype=float)
    valid = np.isfinite(tvdss) & np.isfinite(ai) & (ai > 0.0)
    if int(valid.sum()) < 2:
        raise ValueError("Too few finite positive AI samples after MD->TVDSS conversion.")

    tvdss = tvdss[valid]
    ai = ai[valid]
    order = np.argsort(tvdss)
    tvdss = tvdss[order]
    ai = ai[order]
    unique_tvdss, unique_idx = np.unique(tvdss, return_index=True)
    unique_ai = ai[unique_idx]
    if unique_tvdss.size < 2:
        raise ValueError("Too few unique TVDSS samples.")

    ai_on_axis = np.interp(depth_axis_m, unique_tvdss, unique_ai, left=np.nan, right=np.nan)
    mask = np.isfinite(ai_on_axis) & (ai_on_axis > 0.0)
    if int(mask.sum()) < 2:
        raise ValueError("Too few AI samples overlap the depth LFM axis.")
    return ai_on_axis.astype(np.float32), mask


def _prepare_highres_ai(md: np.ndarray, ai: np.ndarray, kb_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return finite positive shifted well AI on its native TVDSS axis."""
    tvdss = np.asarray(md, dtype=float) - float(kb_m)
    ai = np.asarray(ai, dtype=float)
    valid = np.isfinite(tvdss) & np.isfinite(ai) & (ai > 0.0)
    if int(valid.sum()) < 2:
        raise ValueError("Too few finite positive AI samples after MD->TVDSS conversion.")

    tvdss = tvdss[valid]
    ai = ai[valid]
    order = np.argsort(tvdss)
    tvdss = tvdss[order]
    ai = ai[order]
    unique_tvdss, unique_idx = np.unique(tvdss, return_index=True)
    unique_ai = ai[unique_idx]
    if unique_tvdss.size < 2:
        raise ValueError("Too few unique high-resolution TVDSS samples.")
    return unique_tvdss.astype(np.float32), unique_ai.astype(np.float32)


def _build_highres_prior_trace(
    highres_depth: np.ndarray,
    highres_ai: np.ndarray,
    depth_axis_m: np.ndarray,
    lfm_ai: np.ndarray,
    split_params: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Build high-resolution well/LFM residual arrays for one well."""
    lfm_interp = np.interp(highres_depth, depth_axis_m, lfm_ai, left=np.nan, right=np.nan)
    mask = np.isfinite(highres_depth) & np.isfinite(highres_ai) & (highres_ai > 0.0)
    mask = mask & np.isfinite(lfm_interp) & (lfm_interp > 0.0)
    if int(mask.sum()) < 2:
        raise ValueError("Too few high-resolution AI samples overlap finite positive LFM samples.")

    split = _split_log_ai_frequency_bands(highres_depth, highres_ai, mask, split_params)
    well_log_ai = split["well_log_ai"]
    lfm_log_ai = np.zeros_like(highres_ai, dtype=np.float32)
    residual_log_ai = np.zeros_like(highres_ai, dtype=np.float32)
    lfm_log_ai[mask] = np.log(np.clip(lfm_interp[mask], 1e-6, None)).astype(np.float32)
    residual_log_ai[mask] = (well_log_ai[mask] - lfm_log_ai[mask]).astype(np.float32)
    return {
        "highres_depth": highres_depth.astype(np.float32, copy=False),
        "highres_well_ai": np.where(mask, highres_ai, 0.0).astype(np.float32),
        "highres_well_log_ai": well_log_ai,
        "highres_well_low_ai": split["well_low_ai"],
        "highres_well_low_log_ai": split["well_low_log_ai"],
        "highres_well_high_log_ai": split["well_high_log_ai"],
        "highres_lfm_log_ai": lfm_log_ai,
        "highres_residual_log_ai": residual_log_ai,
        "highres_well_mask": mask.astype(bool),
    }


def _pad_highres_records(
    records: list[dict[str, Any]], key: str, *, dtype: np.dtype, fill_value: float | bool
) -> np.ndarray:
    """Pad variable-length high-resolution well arrays to a rectangular bundle field."""
    max_len = max(int(np.asarray(row[key]).size) for row in records)
    padded = np.full((len(records), max_len), fill_value, dtype=dtype)
    for row_idx, row in enumerate(records):
        values = np.asarray(row[key], dtype=dtype).reshape(-1)
        padded[row_idx, : values.size] = values
    return padded


def _median_highres_sample_step(highres_depth: np.ndarray, highres_mask: np.ndarray) -> float | None:
    steps: list[np.ndarray] = []
    for depth, mask in zip(highres_depth, highres_mask):
        valid_depth = np.asarray(depth, dtype=np.float64)[np.asarray(mask, dtype=bool)]
        if valid_depth.size > 1:
            diff = np.diff(valid_depth)
            diff = diff[np.isfinite(diff) & (diff > 0.0)]
            if diff.size:
                steps.append(diff)
    if not steps:
        return None
    return float(np.median(np.concatenate(steps)))


def _nearest_flat_index(ai_lfm: Any, iline: float, xline: float) -> tuple[int, int, int, float, float]:
    il_idx = int(np.argmin(np.abs(ai_lfm.ilines - float(iline))))
    xl_idx = int(np.argmin(np.abs(ai_lfm.xlines - float(xline))))
    flat_idx = il_idx * int(ai_lfm.xlines.size) + xl_idx
    return flat_idx, il_idx, xl_idx, float(ai_lfm.ilines[il_idx]), float(ai_lfm.xlines[xl_idx])


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    script_cfg = cfg.get("well_constraints_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'well_constraints_depth' section in config.")

    data_root = resolve_relative_path(str(cfg.get("data_root", "data")), root=REPO_ROOT)
    output_root = resolve_relative_path(str(cfg.get("output_root", "scripts/output")), root=REPO_ROOT)

    source_batch_dir = resolve_relative_path(str(script_cfg["source_batch_dir"]), root=REPO_ROOT)
    shifted_las_dir = resolve_relative_path(
        str(script_cfg.get("shifted_las_dir", source_batch_dir / "shifted_las")),
        root=source_batch_dir,
    )
    metrics_path = resolve_relative_path(
        str(script_cfg.get("metrics_file", source_batch_dir / "wavelet_batch_metrics.csv")),
        root=source_batch_dir,
    )
    ai_lfm_file = resolve_relative_path(str(script_cfg["ai_lfm_file"]), root=REPO_ROOT)
    seismic_file = resolve_relative_path(str(cfg["segy"]["file"]), root=data_root)
    well_heads_file = resolve_relative_path(str(cfg["well"]["well_heads_file"]), root=data_root)

    for path in [seismic_file, well_heads_file, shifted_las_dir, metrics_path, ai_lfm_file]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"well_constraints_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prior_path = output_dir / "well_resolution_prior_depth.npz"
    anchor_path = output_dir / "log_ai_anchor_depth.npz"
    qc_path = output_dir / "well_constraints_depth_qc.csv"
    run_summary_path = output_dir / "run_summary.json"

    confidence_corr_floor = float(script_cfg.get("confidence_corr_floor", 0.3))
    confidence_corr_span = float(script_cfg.get("confidence_corr_span", 0.4))
    confidence_weight_mode = str(script_cfg.get("confidence_weight_mode", "corr")).strip().lower()
    if confidence_weight_mode not in {"corr", "uniform"}:
        raise ValueError("well_constraints_depth.confidence_weight_mode must be 'corr' or 'uniform'.")
    las_ai_source = str(script_cfg.get("las_ai_source", "filtered_shifted_las")).strip().lower()
    if las_ai_source not in {"filtered_shifted_las", "raw_shifted_las"}:
        raise ValueError("well_constraints_depth.las_ai_source must be 'filtered_shifted_las' or 'raw_shifted_las'.")
    log_filter_params = _load_auto_tie_log_filter_params(cfg)

    segy_cfg = cfg["segy"]
    segy_options = {
        "iline": segy_cfg["iline_byte"],
        "xline": segy_cfg["xline_byte"],
        "istep": segy_cfg["istep"],
        "xstep": segy_cfg["xstep"],
    }

    ai_lfm = load_lfm_depth_npz(ai_lfm_file)
    frequency_split = _resolve_frequency_split_params(script_cfg, dict(ai_lfm.metadata))
    depth_axis_m = ai_lfm.samples.astype(np.float32)

    print("=== Well Constraints (Depth) ===")
    print(f"Shifted LAS dir: {shifted_las_dir}")
    print(f"Batch metrics: {metrics_path}")
    print(f"AI LFM: {ai_lfm_file}")
    print(f"LAS AI source: {las_ai_source} ({log_filter_params})")
    print(f"Frequency split: {frequency_split}")
    print(f"Output dir: {output_dir}")

    well_heads_df = import_well_heads_petrel(well_heads_file)
    metrics_df = pd.read_csv(metrics_path)
    survey = open_survey(seismic_file, seismic_type="segy", segy_options=segy_options)

    heads_by_name = {_normalize_name(row["Name"]): row for _, row in well_heads_df.iterrows()}
    heads_by_upper = {_normalize_name(row["Name"]).upper(): row for _, row in well_heads_df.iterrows()}
    metrics_by_name = {_normalize_name(row["well_name"]): row for _, row in metrics_df.iterrows()}
    metrics_by_upper = {_normalize_name(row["well_name"]).upper(): row for _, row in metrics_df.iterrows()}

    las_paths = sorted(shifted_las_dir.glob("*.las"))
    if not las_paths:
        raise ValueError(f"No shifted LAS files found in {shifted_las_dir}")

    records: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for las_path in las_paths:
        well_name = las_path.stem
        well_name_upper = well_name.upper()
        qc: dict[str, Any] = {
            "well_name": well_name,
            "las_path": repo_relative_path(las_path, root=REPO_ROOT),
            "status": "skipped",
        }
        try:
            head = heads_by_name.get(well_name)
            if head is None:
                head = heads_by_upper.get(well_name_upper)
            if head is None:
                raise ValueError("Missing well head row.")
            kb_m = float(head["Well datum value"])
            well_x = float(head["Surface X"])
            well_y = float(head["Surface Y"])
            if not np.isfinite(kb_m) or not np.isfinite(well_x) or not np.isfinite(well_y):
                raise ValueError("Non-finite KB or surface coordinates.")

            iline, xline = survey.coord_to_line(well_x, well_y)
            flat_idx, il_idx, xl_idx, nearest_inline, nearest_xline = _nearest_flat_index(ai_lfm, iline, xline)
            md, raw_ai, filtered_ai, ai_source = _read_ai_curves_from_shifted_las(
                las_path,
                log_filter_params=log_filter_params,
            )
            constraint_ai = filtered_ai if las_ai_source == "filtered_shifted_las" else raw_ai
            constraint_ai_source = (
                ai_source if las_ai_source == "filtered_shifted_las" else ai_source.replace("filtered(", "raw(", 1)
            )
            ai_on_axis, mask = _interpolate_ai_to_depth_axis(md, constraint_ai, kb_m, depth_axis_m)
            raw_ai_on_axis, raw_mask = _interpolate_ai_to_depth_axis(md, raw_ai, kb_m, depth_axis_m)
            lfm_ai = np.asarray(ai_lfm.trace_by_index(il_idx, xl_idx), dtype=np.float32)
            highres_depth, highres_ai = _prepare_highres_ai(md, constraint_ai, kb_m)
            highres_trace = _build_highres_prior_trace(
                highres_depth,
                highres_ai,
                depth_axis_m,
                lfm_ai,
                frequency_split,
            )
            lfm_valid = np.isfinite(lfm_ai) & (lfm_ai > 0.0)
            mask = mask & lfm_valid
            if int(mask.sum()) < 2:
                raise ValueError("Too few AI samples overlap finite positive LFM samples.")

            split_trace = _split_log_ai_frequency_bands(depth_axis_m, ai_on_axis, mask, frequency_split)
            log_ai = split_trace["well_log_ai"]
            well_low_ai = split_trace["well_low_ai"]
            well_low_log_ai = split_trace["well_low_log_ai"]
            well_high_log_ai = split_trace["well_high_log_ai"]
            lfm_log_ai = np.zeros_like(lfm_ai, dtype=np.float32)
            residual_log_ai = np.zeros_like(ai_on_axis, dtype=np.float32)
            lfm_log_ai[mask] = np.log(np.clip(lfm_ai[mask], 1e-6, None)).astype(np.float32)
            residual_log_ai[mask] = (log_ai[mask] - lfm_log_ai[mask]).astype(np.float32)
            raw_filtered_mask = mask & raw_mask & np.isfinite(raw_ai_on_axis) & np.isfinite(ai_on_axis)
            raw_filtered_diff = raw_ai_on_axis[raw_filtered_mask] - ai_on_axis[raw_filtered_mask]

            metric = metrics_by_name.get(well_name)
            if metric is None:
                metric = metrics_by_upper.get(well_name_upper)
            corr = np.nan if metric is None else _finite_float_or_nan(metric.get("corr", np.nan))
            confidence = _confidence_from_corr(
                corr,
                floor=confidence_corr_floor,
                span=confidence_corr_span,
                mode=confidence_weight_mode,
            )
            weight = np.zeros_like(ai_on_axis, dtype=np.float32)
            weight[mask] = confidence

            records.append(
                {
                    "well_name": well_name,
                    "flat_idx": int(flat_idx),
                    "inline": float(nearest_inline),
                    "xline": float(nearest_xline),
                    "log_ai": log_ai,
                    "well_low_ai": well_low_ai,
                    "well_low_log_ai": well_low_log_ai,
                    "well_high_log_ai": well_high_log_ai,
                    "lfm_log_ai": lfm_log_ai,
                    "residual_log_ai": residual_log_ai,
                    "ai": np.nan_to_num(ai_on_axis, nan=0.0).astype(np.float32),
                    "lfm_ai": np.where(lfm_valid, lfm_ai, 0.0).astype(np.float32),
                    "mask": mask.astype(bool),
                    "weight": weight,
                    **highres_trace,
                }
            )
            qc.update(
                {
                    "status": "ok",
                    "ai_source": constraint_ai_source,
                    "las_ai_source": las_ai_source,
                    "log_filter_params_json": json.dumps(log_filter_params, ensure_ascii=False),
                    "raw_minus_filtered_ai_mean": (
                        float(np.mean(raw_filtered_diff)) if raw_filtered_diff.size else np.nan
                    ),
                    "raw_minus_filtered_ai_rmse": (
                        float(np.sqrt(np.mean(raw_filtered_diff**2))) if raw_filtered_diff.size else np.nan
                    ),
                    "frequency_split_enabled": bool(frequency_split["enabled"]),
                    "frequency_split_cutoff_wavelength_m": frequency_split["cutoff_wavelength_m"],
                    "frequency_split_filter_order": frequency_split["filter_order"],
                    "frequency_split_parameter_source": frequency_split["frequency_split_parameter_source"],
                    "well_high_log_ai_abs_p95": (
                        float(np.percentile(np.abs(well_high_log_ai[mask]), 95.0)) if int(mask.sum()) else np.nan
                    ),
                    "kb_m": kb_m,
                    "well_x": well_x,
                    "well_y": well_y,
                    "resolved_inline": float(iline),
                    "resolved_xline": float(xline),
                    "nearest_inline": float(nearest_inline),
                    "nearest_xline": float(nearest_xline),
                    "flat_idx": int(flat_idx),
                    "corr": corr,
                    "confidence_weight_mode": confidence_weight_mode,
                    "confidence": confidence,
                    "valid_samples": int(mask.sum()),
                    "highres_valid_samples": int(highres_trace["highres_well_mask"].sum()),
                }
            )
        except Exception as exc:
            qc.update({"status": "failed", "error": str(exc)})
        qc_rows.append(qc)

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(qc_path, index=False)
    print(qc_df.to_string(index=False))

    if not records:
        raise ValueError(f"No successful wells; inspect {qc_path} before building the prior.")

    duplicates = [flat for flat, count in Counter(row["flat_idx"] for row in records).items() if count > 1]
    if duplicates:
        duplicate_wells = [row["well_name"] for row in records if row["flat_idx"] in duplicates]
        raise ValueError(f"Duplicate nearest traces are not supported in v2: {duplicates}, wells={duplicate_wells}")

    sample_step = float(np.median(np.diff(depth_axis_m))) if depth_axis_m.size > 1 else None
    well_mask = np.stack([row["mask"] for row in records]).astype(bool)
    well_weight = np.stack([row["weight"] for row in records]).astype(np.float32)
    well_low_ai = np.stack([row["well_low_ai"] for row in records]).astype(np.float32)
    well_low_log_ai = np.stack([row["well_low_log_ai"] for row in records]).astype(np.float32)
    well_high_log_ai = np.stack([row["well_high_log_ai"] for row in records]).astype(np.float32)
    residual_log_ai = np.stack([row["residual_log_ai"] for row in records]).astype(np.float32)
    highres_depth = _pad_highres_records(records, "highres_depth", dtype=np.float32, fill_value=0.0)  # type: ignore
    highres_well_ai = _pad_highres_records(records, "highres_well_ai", dtype=np.float32, fill_value=0.0)  # type: ignore
    highres_well_log_ai = _pad_highres_records(records, "highres_well_log_ai", dtype=np.float32, fill_value=0.0)  # type: ignore
    highres_well_low_ai = _pad_highres_records(records, "highres_well_low_ai", dtype=np.float32, fill_value=0.0)  # type: ignore
    highres_well_low_log_ai = _pad_highres_records(
        records,
        "highres_well_low_log_ai",
        dtype=np.float32,  # type: ignore
        fill_value=0.0,
    )
    highres_well_high_log_ai = _pad_highres_records(
        records,
        "highres_well_high_log_ai",
        dtype=np.float32,  # type: ignore
        fill_value=0.0,
    )
    highres_lfm_log_ai = _pad_highres_records(records, "highres_lfm_log_ai", dtype=np.float32, fill_value=0.0)  # type: ignore
    highres_residual_log_ai = _pad_highres_records(
        records,
        "highres_residual_log_ai",
        dtype=np.float32,  # type: ignore
        fill_value=0.0,
    )
    highres_well_mask = _pad_highres_records(records, "highres_well_mask", dtype=bool, fill_value=False)  # type: ignore
    summary = summarize_well_resolution_prior(
        residual_log_ai,
        well_mask,
        sample_step=sample_step,
        well_weight=well_weight,
    )
    highres_summary = summarize_well_resolution_prior(
        highres_residual_log_ai,
        highres_well_mask,
        sample_step=_median_highres_sample_step(highres_depth, highres_well_mask),
    )
    summary["highres_residual"] = highres_summary["residual"]
    summary["highres_event_density"] = highres_summary["event_density"]
    summary["highres_reflectivity"] = highres_summary["reflectivity"]
    summary["highres_spectrum"] = highres_summary["spectrum"]
    summary["well_high_log_ai"] = summarize_well_resolution_prior(
        well_high_log_ai,
        well_mask,
        sample_step=sample_step,
        well_weight=well_weight,
    )["residual"]
    summary["highres_well_high_log_ai"] = summarize_well_resolution_prior(
        highres_well_high_log_ai,
        highres_well_mask,
        sample_step=_median_highres_sample_step(highres_depth, highres_well_mask),
    )["residual"]

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": Path(__file__).name,
        "artifact_family": "well_constraints_depth",
        "artifact_role": "well_resolution_prior",
        "used_by_stage": "stage_2_enhancement",
        "path_style": "repo_relative",
        "paired_anchor_path": repo_relative_path(anchor_path, root=REPO_ROOT),
        "shifted_las_dir": repo_relative_path(shifted_las_dir, root=REPO_ROOT),
        "metrics_path": repo_relative_path(metrics_path, root=REPO_ROOT),
        "well_heads_file": repo_relative_path(well_heads_file, root=REPO_ROOT),
        "ai_lfm_file": repo_relative_path(ai_lfm_file, root=REPO_ROOT),
        "las_ai_source": las_ai_source,
        "log_filter_params": log_filter_params,
        "frequency_split": frequency_split,
        "frequency_split_enabled": bool(frequency_split["enabled"]),
        "frequency_split_parameter_source": frequency_split["frequency_split_parameter_source"],
        "confidence_weight_mode": confidence_weight_mode,
        "confidence_formula": (
            "1.0 for every successful well"
            if confidence_weight_mode == "uniform"
            else f"clip((corr - {confidence_corr_floor}) / {confidence_corr_span}, 0, 1)"
        ),
        "qc_path": repo_relative_path(qc_path, root=REPO_ROOT),
        "n_input_las": int(len(las_paths)),
        "n_successful_wells": int(len(records)),
        "highres_max_samples": int(highres_depth.shape[1]),
        "status_counts": qc_df["status"].value_counts(dropna=False).to_dict(),
    }

    bundle = WellResolutionPriorBundle(
        sample_domain="depth",
        sample_unit="m",
        samples=depth_axis_m.astype(np.float32),
        flat_indices=np.array([row["flat_idx"] for row in records], dtype=np.int64),
        well_log_ai=np.stack([row["log_ai"] for row in records]).astype(np.float32),
        lfm_log_ai=np.stack([row["lfm_log_ai"] for row in records]).astype(np.float32),
        residual_log_ai=residual_log_ai,
        well_ai=np.stack([row["ai"] for row in records]).astype(np.float32),
        well_low_ai=well_low_ai,
        well_low_log_ai=well_low_log_ai,
        well_high_log_ai=well_high_log_ai,
        lfm_ai=np.stack([row["lfm_ai"] for row in records]).astype(np.float32),
        well_mask=well_mask,
        well_weight=well_weight,
        highres_depth=highres_depth,
        highres_well_ai=highres_well_ai,
        highres_well_log_ai=highres_well_log_ai,
        highres_well_low_ai=highres_well_low_ai,
        highres_well_low_log_ai=highres_well_low_log_ai,
        highres_well_high_log_ai=highres_well_high_log_ai,
        highres_lfm_log_ai=highres_lfm_log_ai,
        highres_residual_log_ai=highres_residual_log_ai,
        highres_well_mask=highres_well_mask,
        well_names=np.array([row["well_name"] for row in records]),
        inline=np.array([row["inline"] for row in records], dtype=np.float32),
        xline=np.array([row["xline"] for row in records], dtype=np.float32),
        summary=summary,
        metadata=metadata,
    )

    anchor_metadata = {
        "created_at_utc": metadata["created_at_utc"],
        "source_script": Path(__file__).name,
        "artifact_family": "well_constraints_depth",
        "artifact_role": "log_ai_anchor",
        "used_by_stage": "stage_1_ginn_anchor",
        "path_style": "repo_relative",
        "paired_prior_path": repo_relative_path(prior_path, root=REPO_ROOT),
        "anchor_source": "shifted_las_wells",
        "shifted_las_dir": repo_relative_path(shifted_las_dir, root=REPO_ROOT),
        "metrics_path": repo_relative_path(metrics_path, root=REPO_ROOT),
        "well_heads_file": repo_relative_path(well_heads_file, root=REPO_ROOT),
        "ai_lfm_file": repo_relative_path(ai_lfm_file, root=REPO_ROOT),
        "las_ai_source": metadata["las_ai_source"],
        "log_filter_params": metadata["log_filter_params"],
        "frequency_split": metadata["frequency_split"],
        "frequency_split_enabled": metadata["frequency_split_enabled"],
        "frequency_split_parameter_source": metadata["frequency_split_parameter_source"],
        "anchor_target_band": "lowpass_lfm_matched" if frequency_split["enabled"] else "fullband",
        "confidence_weight_mode": metadata["confidence_weight_mode"],
        "confidence_formula": metadata["confidence_formula"],
        "qc_path": repo_relative_path(qc_path, root=REPO_ROOT),
        "n_input_las": int(len(las_paths)),
        "n_successful_anchors": int(len(records)),
        "status_counts": qc_df["status"].value_counts(dropna=False).to_dict(),
        "reserved_anchor_types": ["well", "facies_control"],
    }
    anchor_bundle = build_log_ai_anchor_bundle(
        sample_domain="depth",
        sample_unit="m",
        samples=depth_axis_m.astype(np.float32),
        flat_indices=np.array([row["flat_idx"] for row in records], dtype=np.int64),
        target_ai=well_low_ai,
        anchor_mask=well_mask,
        anchor_weight=well_weight,
        anchor_names=np.array([row["well_name"] for row in records]),
        anchor_types=np.array(["well"] * len(records)),
        inline=np.array([row["inline"] for row in records], dtype=np.float32),
        xline=np.array([row["xline"] for row in records], dtype=np.float32),
        metadata=anchor_metadata,
    )

    save_well_resolution_prior_npz(prior_path, bundle)
    save_log_ai_anchor_npz(anchor_path, anchor_bundle)
    with run_summary_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "created_at_utc": metadata["created_at_utc"],
                "source_script": Path(__file__).name,
                "artifact_family": "well_constraints_depth",
                "shared_source": {
                    "path_style": "repo_relative",
                    "shifted_las_dir": repo_relative_path(shifted_las_dir, root=REPO_ROOT),
                    "metrics_path": repo_relative_path(metrics_path, root=REPO_ROOT),
                    "well_heads_file": repo_relative_path(well_heads_file, root=REPO_ROOT),
                    "ai_lfm_file": repo_relative_path(ai_lfm_file, root=REPO_ROOT),
                    "las_ai_source": metadata["las_ai_source"],
                    "log_filter_params": metadata["log_filter_params"],
                    "frequency_split": metadata["frequency_split"],
                    "frequency_split_enabled": metadata["frequency_split_enabled"],
                    "frequency_split_parameter_source": metadata["frequency_split_parameter_source"],
                    "qc_path": repo_relative_path(qc_path, root=REPO_ROOT),
                    "n_input_las": int(len(las_paths)),
                    "n_successful_wells": int(len(records)),
                    "status_counts": qc_df["status"].value_counts(dropna=False).to_dict(),
                },
                "artifacts": {
                    "stage_2_enhancement_prior": {
                        "path": repo_relative_path(prior_path, root=REPO_ROOT),
                        "schema": "ginn_well_resolution_prior_v3",
                        "contains_residual_log_ai": True,
                        "contains_highres_residual_log_ai": True,
                        "contains_well_low_high_log_ai": True,
                        "summary": summary,
                    },
                    "stage_1_log_ai_anchor": {
                        "path": repo_relative_path(anchor_path, root=REPO_ROOT),
                        "schema": "ginn_log_ai_anchor_v1",
                        "contains_residual_log_ai": False,
                        "anchor_target_band": anchor_metadata["anchor_target_band"],
                        "reserved_anchor_types": ["well", "facies_control"],
                        "summary": anchor_bundle.summary,
                    },
                },
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved stage-2 well-resolution prior: {prior_path}")
    print(f"Saved stage-1 log-AI anchor: {anchor_path}")
    print(f"Saved QC: {qc_path}")
    print(f"Saved run summary: {run_summary_path}")


if __name__ == "__main__":
    main()
