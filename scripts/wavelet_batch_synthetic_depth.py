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

from cup.utils.io import (
    load_yaml_config,
    repo_relative_path,
    resolve_relative_path,
    sanitize_filename,
)
from cup.config.workflow import WorkflowConfig
from cup.seismic.survey import segy_options_from_config

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
        default=Path("experiments/common/common.yaml"),
        help="Main workflow config YAML containing wavelet_batch_synthetic_depth.",
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


def load_interpolated_standard_vp_rho(las_file: Path):
    """Adapt current DT_USM/RHO_GCC LAS files to the legacy all-gap-filled path."""
    from cup.well.las import load_standard_vp_rho_logs
    from wtie.processing import grid
    from wtie.processing.logs import interpolate_nans

    source = load_standard_vp_rho_logs(las_file).logs
    md = np.asarray(source.basis, dtype=float)
    vp = interpolate_nans(np.asarray(source.Vp.values, dtype=float), method="linear")
    rho = interpolate_nans(np.asarray(source.Rho.values, dtype=float), method="linear")
    return grid.LogSet(
        {
            "Vp": grid.Log(vp, md, "md", name="Vp", unit="m/s", allow_nan=False),
            "Rho": grid.Log(rho, md, "md", name="Rho", unit="g/cm3", allow_nan=False),
        }
    )


def zscore_trace(values: np.ndarray) -> np.ndarray:
    """Legacy whole-trace z-score used by this one-off script."""
    from wtie.processing.logs import interpolate_nans

    finite = interpolate_nans(np.asarray(values, dtype=float), method="linear")
    scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Trace has non-positive standard deviation.")
    return (finite - float(np.mean(finite))) / scale


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
) -> dict[str, Any]:
    mask = np.isfinite(seismic_norm) & np.isfinite(synthetic_raw)
    if int(mask.sum()) < 5:
        return {
            "scale": np.nan,
            "signed_ls_scale": np.nan,
            "corr": np.nan,
            "nmae": np.nan,
            "n_score_samples": int(mask.sum()),
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
            "n_score_samples": int(mask.sum()),
        }
    scale = abs(float(signed_scale))
    synthetic_scaled = scale * synthetic_raw
    nmae = float(np.sum(np.abs(seismic_norm[mask] - synthetic_scaled[mask])) / np.sum(np.abs(seismic_norm[mask])))
    return {
        "scale": scale,
        "signed_ls_scale": float(signed_scale),
        "corr": corr,
        "nmae": nmae,
        "n_score_samples": int(mask.sum()),
    }


def evaluate_shift(
    *,
    twt_s: np.ndarray,
    seismic_norm: np.ndarray,
    ref_twt_s: np.ndarray,
    ref_values: np.ndarray,
    wavelet_amp: np.ndarray,
    shift_s: float,
    modeler: Any,
) -> dict[str, Any]:
    shifted_time = twt_s - shift_s
    shifted_ref = np.interp(shifted_time, ref_twt_s, ref_values, left=0.0, right=0.0)
    synthetic_raw = modeler(wavelet_amp, shifted_ref)
    metric = metrics_for_synthetic(seismic_norm, synthetic_raw)
    metric.update({"shift_s": float(shift_s)})
    return {
        "metrics": metric,
        "reflectivity_shifted": shifted_ref,
        "synthetic_raw": synthetic_raw,
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


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in zip(changes[0::2], changes[1::2])]


def _source_null_value(las: Any, default: float = -999.25) -> float:
    try:
        value = float(las.well["NULL"].value)
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _finite_curve_values(values: np.ndarray, null_value: float) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(values, dtype=float).copy()
    invalid = ~np.isfinite(out)
    if np.isfinite(null_value):
        invalid |= np.isclose(out, null_value, rtol=0.0, atol=1e-9)
    out[invalid] = np.nan
    return out, ~invalid


def _depth_shift_at_tvdss(
    tvdss_m: np.ndarray,
    depth_shift_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    if depth_shift_df.empty:
        raise ValueError("Depth shift curve is empty; cannot export shifted LAS.")

    tvdss_m = np.asarray(tvdss_m, dtype=float)
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

    finite_tvdss = np.isfinite(tvdss_m)
    below = finite_tvdss & (tvdss_m < unique_shift_z[0])
    above = finite_tvdss & (tvdss_m > unique_shift_z[-1])
    depth_shift_at_log = np.interp(
        tvdss_m,
        unique_shift_z,
        unique_shift_m,
        left=unique_shift_m[0],
        right=unique_shift_m[-1],
    )
    total = int(np.count_nonzero(finite_tvdss))
    extrapolated = int(np.count_nonzero(below) + np.count_nonzero(above))
    stats = {
        "shift_support_tvdss_min_m": float(unique_shift_z[0]),
        "shift_support_tvdss_max_m": float(unique_shift_z[-1]),
        "shift_extrapolated_top_count": int(np.count_nonzero(below)),
        "shift_extrapolated_bottom_count": int(np.count_nonzero(above)),
        "shift_extrapolated_sample_count": extrapolated,
        "shift_extrapolated_total_count": total,
        "shift_extrapolated_fraction": float(extrapolated / total) if total else np.nan,
    }
    return depth_shift_at_log, stats


def _regular_md_from_shifted(
    source_md_m: np.ndarray,
    shifted_md_m: np.ndarray,
) -> tuple[np.ndarray, float]:
    finite_md = np.isfinite(source_md_m)
    if int(finite_md.sum()) < 2:
        raise ValueError("Input LAS has too few finite MD samples.")
    md_step_m = float(np.nanmedian(np.diff(source_md_m[finite_md])))
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
    return regular_md, md_step_m


def _interpolate_curve_preserving_gaps(
    shifted_md_m: np.ndarray,
    values: np.ndarray,
    regular_md: np.ndarray,
    *,
    curve_name: str,
    require_two_samples: bool = False,
) -> np.ndarray:
    shifted_md_m = np.asarray(shifted_md_m, dtype=float)
    values = np.asarray(values, dtype=float)
    regular_md = np.asarray(regular_md, dtype=float)
    out = np.full(regular_md.shape, np.nan, dtype=float)
    valid = np.isfinite(shifted_md_m) & np.isfinite(values)
    if int(valid.sum()) < 2:
        if require_two_samples:
            raise ValueError(f"Curve {curve_name} has too few finite shifted samples.")
        return out

    for start, stop in _true_runs(valid):
        if stop - start < 2:
            continue
        x = shifted_md_m[start:stop]
        y = values[start:stop]
        local = np.isfinite(x) & np.isfinite(y)
        if int(local.sum()) < 2:
            continue
        x = x[local]
        y = y[local]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        unique_x, unique_idx = np.unique(x, return_index=True)
        unique_y = y[unique_idx]
        if unique_x.size < 2:
            continue
        inside = (regular_md >= unique_x[0]) & (regular_md <= unique_x[-1])
        out[inside] = np.interp(regular_md[inside], unique_x, unique_y)
    if require_two_samples and int(np.count_nonzero(np.isfinite(out))) < 2:
        raise ValueError(f"Curve {curve_name} has too few finite shifted output samples.")
    return out


def _curve_index_by_mnemonic(las: Any, mnemonic: str) -> int | None:
    target = mnemonic.strip().casefold()
    for idx, curve in enumerate(las.curves):
        if str(curve.mnemonic).strip().casefold() == target:
            return idx
    return None


def _append_las_curve_with_nulls(
    las: Any,
    mnemonic: str,
    values: np.ndarray,
    *,
    unit: str,
    descr: str,
    null_value: float,
) -> None:
    out = np.asarray(values, dtype=float).copy()
    out[~np.isfinite(out)] = float(null_value)
    las.append_curve(str(mnemonic), out, unit=str(unit or ""), descr=str(descr or mnemonic))


def export_shifted_preprocessed_las(
    source_las: Path,
    output_las: Path,
    *,
    kb_m: float,
    depth_shift_df: pd.DataFrame,
) -> tuple[Path, dict[str, Any]]:
    """Shift a Step-3 preprocessed LAS while preserving all numeric curves and gaps."""
    import lasio

    source = lasio.read(str(source_las))
    if len(source.curves) < 2:
        raise ValueError(f"LAS file has no data curves: {source_las}")
    data = np.asarray(source.data, dtype=float)
    if data.ndim != 2 or data.shape[1] != len(source.curves):
        raise ValueError(f"LAS data shape does not match curve headers: {source_las}")

    null_value = _source_null_value(source)
    md_m, md_valid = _finite_curve_values(data[:, 0], null_value)
    if int(np.count_nonzero(md_valid)) < 2:
        raise ValueError(f"LAS has too few finite MD samples: {source_las}")
    tvdss_m = md_m - float(kb_m)
    depth_shift_m, shift_stats = _depth_shift_at_tvdss(tvdss_m, depth_shift_df)
    shifted_md_m = tvdss_m + depth_shift_m + float(kb_m)
    regular_md, md_step_m = _regular_md_from_shifted(md_m, shifted_md_m)

    out = lasio.LASFile()
    for item in source.well:
        out.well[item.mnemonic] = item
    for item in source.params:
        out.params[item.mnemonic] = item
    out.well["NULL"].value = float(null_value)
    out.well["STRT"].value = float(regular_md[0])
    out.well["STOP"].value = float(regular_md[-1])
    out.well["STEP"].value = float(md_step_m)

    index_curve = source.curves[0]
    out.append_curve(
        str(index_curve.mnemonic),
        regular_md,
        unit=str(index_curve.unit or "m"),
        descr=str(index_curve.descr or "Measured depth"),
    )

    dt_idx = _curve_index_by_mnemonic(source, "DT_USM")
    rho_idx = _curve_index_by_mnemonic(source, "RHO_GCC")
    ai_idx = _curve_index_by_mnemonic(source, "AI")
    recomputed_ai: np.ndarray | None = None
    if dt_idx is not None and rho_idx is not None:
        dt_values, _ = _finite_curve_values(data[:, dt_idx], null_value)
        rho_values, _ = _finite_curve_values(data[:, rho_idx], null_value)
        shifted_dt = _interpolate_curve_preserving_gaps(
            shifted_md_m,
            dt_values,
            regular_md,
            curve_name="DT_USM",
        )
        shifted_rho = _interpolate_curve_preserving_gaps(
            shifted_md_m,
            rho_values,
            regular_md,
            curve_name="RHO_GCC",
        )
        valid_ai = np.isfinite(shifted_dt) & (shifted_dt > 0.0) & np.isfinite(shifted_rho)
        recomputed_ai = np.full(regular_md.shape, np.nan, dtype=float)
        recomputed_ai[valid_ai] = (1_000_000.0 / shifted_dt[valid_ai]) * shifted_rho[valid_ai]

    exported_curves: list[str] = []
    for idx, curve in enumerate(source.curves[1:], start=1):
        mnemonic = str(curve.mnemonic)
        if idx == ai_idx and recomputed_ai is not None:
            values = recomputed_ai
        else:
            source_values, _ = _finite_curve_values(data[:, idx], null_value)
            values = _interpolate_curve_preserving_gaps(
                shifted_md_m,
                source_values,
                regular_md,
                curve_name=mnemonic,
            )
        _append_las_curve_with_nulls(
            out,
            mnemonic,
            values,
            unit=str(curve.unit or ""),
            descr=str(curve.descr or mnemonic),
            null_value=null_value,
        )
        exported_curves.append(mnemonic)

    if ai_idx is None and recomputed_ai is not None:
        _append_las_curve_with_nulls(
            out,
            "AI",
            recomputed_ai,
            unit="m/s*g/cm3",
            descr="AI",
            null_value=null_value,
        )
        exported_curves.append("AI")

    output_las = Path(output_las)
    output_las.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(output_las), version=2.0, wrap=False, fmt="%.6f")
    stats = {
        **shift_stats,
        "source_las": repo_relative_path(source_las, root=REPO_ROOT),
        "exported_las": repo_relative_path(output_las, root=REPO_ROOT),
        "exported_curve_count": len(exported_curves),
        "exported_curves": exported_curves,
        "ai_recomputed": bool(recomputed_ai is not None),
        "output_md_start_m": float(regular_md[0]),
        "output_md_end_m": float(regular_md[-1]),
        "output_md_step_m": float(md_step_m),
        "output_sample_count": int(regular_md.size),
    }
    return output_las, stats


def build_shifted_filtered_logset_for_export(
    filtered_logset_md: Any,
    *,
    kb_m: float,
    depth_shift_df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from wtie.processing.grid import Log

    md_m = np.asarray(filtered_logset_md.basis, dtype=float)
    tvdss_m = md_m - float(kb_m)
    depth_shift_m, shift_stats = _depth_shift_at_tvdss(tvdss_m, depth_shift_df)
    shifted_md_m = tvdss_m + depth_shift_m + float(kb_m)
    regular_md, md_step_m = _regular_md_from_shifted(md_m, shifted_md_m)

    vp_mps = _interpolate_curve_preserving_gaps(
        shifted_md_m,
        np.asarray(filtered_logset_md.Vp.values, dtype=float),
        regular_md,
        curve_name="filtered VP_MPS",
        require_two_samples=True,
    )
    rho_gcc = _interpolate_curve_preserving_gaps(
        shifted_md_m,
        np.asarray(filtered_logset_md.Rho.values, dtype=float),
        regular_md,
        curve_name="filtered RHO_GCC",
        require_two_samples=True,
    )
    valid_dt = np.isfinite(vp_mps) & (vp_mps > 0.0)
    dt_usm = np.full(regular_md.shape, np.nan, dtype=float)
    dt_usm[valid_dt] = 1_000_000.0 / vp_mps[valid_dt]
    ai = vp_mps * rho_gcc
    stats = {
        **shift_stats,
        "exported_curve_count": 3,
        "exported_curves": ["DT_USM", "RHO_GCC", "AI"],
        "ai_recomputed": True,
        "output_md_start_m": float(regular_md[0]),
        "output_md_end_m": float(regular_md[-1]),
        "output_md_step_m": float(md_step_m),
        "output_sample_count": int(regular_md.size),
    }

    return (
        {
            "DT_USM": Log(dt_usm, regular_md, "md", name="DT_USM", unit="us/m", allow_nan=True),
            "RHO_GCC": Log(rho_gcc, regular_md, "md", name="RHO_GCC", unit="g/cm3", allow_nan=True),
            "AI": Log(ai, regular_md, "md", name="AI", unit="m/s*g/cm3", allow_nan=True),
        },
        stats,
    )


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(valid)) < 2:
        return np.nan
    aa = a[valid]
    bb = b[valid]
    if float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def save_r1_style_synthetic_qc(
    *,
    well_name: str,
    output_path: Path,
    twt_s: np.ndarray,
    filtered_logset_twt: Any,
    reflectivity_shifted: np.ndarray,
    seismic_norm: np.ndarray,
    synthetic_scaled: np.ndarray,
    best_shift_s: float,
) -> Path:
    from cup.seismic.viz import plot_well_waveform_qc
    from wtie.optimize.similarity import dynamic_normalized_xcorr, normalized_xcorr
    from wtie.processing import grid

    twt_s = np.asarray(twt_s, dtype=float)
    filtered_ai_basis = np.asarray(filtered_logset_twt.basis, dtype=float)
    filtered_ai = np.asarray(filtered_logset_twt.Vp.values, dtype=float) * np.asarray(
        filtered_logset_twt.Rho.values,
        dtype=float,
    )
    filtered_ai_on_twt = np.interp(twt_s, filtered_ai_basis, filtered_ai, left=np.nan, right=np.nan)

    valid = (
        np.isfinite(twt_s)
        & np.isfinite(reflectivity_shifted)
        & np.isfinite(seismic_norm)
        & np.isfinite(synthetic_scaled)
    )
    runs = _true_runs(valid)
    if not runs:
        raise ValueError("No valid samples for R1-style synthetic QC figure.")
    start, stop = max(runs, key=lambda item: item[1] - item[0])
    if stop - start < 8:
        raise ValueError(f"Too few valid samples for R1-style synthetic QC figure: {stop - start}")
    sl = slice(start, stop)

    basis = twt_s[sl]
    selected_ai = grid.Log(filtered_ai_on_twt[sl], basis, "twt", name="Filtered AI")
    reflectivity = grid.Reflectivity(
        np.asarray(reflectivity_shifted, dtype=float)[sl],
        basis,
        "twt",
        name="Shifted reflectivity",
    )
    synthetic_trace = grid.Seismic(
        np.asarray(synthetic_scaled, dtype=float)[sl],
        basis,
        "twt",
        name="Synthetic",
    )
    observed_trace = grid.Seismic(
        np.asarray(seismic_norm, dtype=float)[sl],
        basis,
        "twt",
        name="Seismic",
    )
    xcorr_values = normalized_xcorr(observed_trace.values, synthetic_trace.values)
    xcorr_basis = synthetic_trace.sampling_rate * np.arange(
        -(synthetic_trace.size - 1),
        synthetic_trace.size,
    )
    xcorr = grid.XCorr(xcorr_values, xcorr_basis, "tlag", name="XCorr")
    dxcorr = dynamic_normalized_xcorr(observed_trace, synthetic_trace)
    corr = _safe_corr(observed_trace.values, synthetic_trace.values)
    title = (
        f"Depth Step 5 synthetic QC | {well_name} | relative TWT | "
        f"corr={corr:.3f}, best shift={best_shift_s * 1000.0:.1f} ms"
    )
    fig, _axes = plot_well_waveform_qc(
        [selected_ai],
        reflectivity,
        synthetic_trace,
        observed_trace,
        xcorr,
        dxcorr,
        figsize=(12.0, 7.5),
        synthetic_ai=selected_ai,
        title=title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


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
    shift_values_s: np.ndarray,
    auto_tie_log_filter_params: dict[str, Any],
    modeler: Any,
    output_dirs: dict[str, Path],
) -> dict[str, Any]:
    from cup.well.las import export_logset_to_las
    from wtie.optimize import tie as tie_utils
    from wtie.processing import grid
    from wtie.processing.logs import interpolate_nans

    name = sanitize_filename(well_name)
    row: dict[str, Any] = {"well_name": well_name, "status": "failed", "error": ""}

    il_float, xl_float = survey.line_geometry.coord_to_line(well_x, well_y)
    if not (geometry_depth["inline_min"] <= il_float <= geometry_depth["inline_max"]):
        raise ValueError(f"Inline outside survey range: {il_float}")
    if not (geometry_depth["xline_min"] <= xl_float <= geometry_depth["xline_max"]):
        raise ValueError(f"Crossline outside survey range: {xl_float}")

    logset_md = load_interpolated_standard_vp_rho(las_file)
    md_m = logset_md.basis.astype(float)
    tvdss_m = md_m - kb_m
    vp_mps = interpolate_nans(logset_md.Vp.values, method="linear")

    seismic_depth_trace = survey.read_trace_at_xy(well_x, well_y, domain="depth")
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
                "n_score_samples": metric["n_score_samples"],
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
        }
    )

    scan_path = output_dirs["shift_scans"] / f"shift_scan_{name}.csv"
    qc_path = output_dirs["synthetic_qc"] / f"synthetic_qc_{name}.csv"
    depth_shift_path = output_dirs["depth_shift_curves"] / f"depth_shift_curve_{name}.csv"
    scan_df.to_csv(scan_path, index=False)
    qc_df.to_csv(qc_path, index=False)
    depth_shift_df.to_csv(depth_shift_path, index=False)

    # Export depth-shifted LAS products for downstream Synthoseis calibration.
    shifted_preprocessed_las_path, shifted_preprocessed_stats = export_shifted_preprocessed_las(
        las_file,
        output_dirs["shifted_preprocessed_las"] / f"{name}.las",
        kb_m=kb_m,
        depth_shift_df=depth_shift_df,
    )
    shifted_filtered_logset_for_export, shifted_filtered_stats = build_shifted_filtered_logset_for_export(
        filtered_logset_md,
        kb_m=kb_m,
        depth_shift_df=depth_shift_df,
    )
    shifted_filtered_las_path = export_logset_to_las(
        well_name,
        shifted_filtered_logset_for_export,
        output_dirs["shifted_filtered_las"] / f"{name}.las",
        curve_names=["DT_USM", "RHO_GCC", "AI"],
        template_las=las_file,
    )

    # Per-well figures
    synthetic_fig_path = save_r1_style_synthetic_qc(
        well_name=well_name,
        output_path=output_dirs["figures"] / f"qc_{name}_synthetic_vs_seismic.png",
        twt_s=twt_s,
        filtered_logset_twt=logset_twt,
        reflectivity_shifted=best["reflectivity_shifted"],
        seismic_norm=seismic_norm,
        synthetic_scaled=best_synthetic_scaled,
        best_shift_s=best_shift_s,
    )

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
            "n_score_samples": int(best_metrics["n_score_samples"]),
            "median_vp_mps": median_vp_mps,
            "median_depth_shift_m": median_depth_shift,
            "mean_depth_shift_m": mean_depth_shift,
            "p10_depth_shift_m": p10_depth_shift,
            "p90_depth_shift_m": p90_depth_shift,
            "approx_depth_shift_m": approx_depth_shift_m,
            "synthetic_qc_path": repo_relative_path(qc_path, root=REPO_ROOT),
            "shift_scan_path": repo_relative_path(scan_path, root=REPO_ROOT),
            "depth_shift_curve_path": repo_relative_path(depth_shift_path, root=REPO_ROOT),
            "shifted_preprocessed_las_path": repo_relative_path(shifted_preprocessed_las_path, root=REPO_ROOT),
            "shifted_filtered_las_path": repo_relative_path(shifted_filtered_las_path, root=REPO_ROOT),
            "shift_extrapolated_fraction": shifted_preprocessed_stats["shift_extrapolated_fraction"],
            "shift_extrapolated_top_count": shifted_preprocessed_stats["shift_extrapolated_top_count"],
            "shift_extrapolated_bottom_count": shifted_preprocessed_stats["shift_extrapolated_bottom_count"],
            "shifted_preprocessed_curve_count": shifted_preprocessed_stats["exported_curve_count"],
            "shifted_filtered_curve_count": shifted_filtered_stats["exported_curve_count"],
            "synthetic_fig_path": repo_relative_path(synthetic_fig_path, root=REPO_ROOT),
            "shift_fig_path": repo_relative_path(output_dirs["figures"] / f"qc_{name}_shift_scan.png", root=REPO_ROOT),
        }
    )
    return row


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    args = parse_args()

    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    workflow = WorkflowConfig.from_mapping(cfg)
    if workflow.seismic.domain != "depth" or workflow.seismic.depth_basis != "tvdss":
        raise ValueError(
            "wavelet_batch_synthetic_depth requires seismic.domain='depth' "
            "and seismic.depth_basis='tvdss'."
        )
    data_root = resolve_relative_path(workflow.data_root, root=REPO_ROOT)

    script_cfg = cfg.get("wavelet_batch_synthetic_depth", {})
    if not script_cfg:
        raise ValueError("Missing 'wavelet_batch_synthetic_depth' section in config.")

    las_vp_unit = str(script_cfg.get("las_vp_unit", "us/m"))
    las_rho_unit = str(script_cfg.get("las_rho_unit", "g/cm3"))

    # ── Resolve input paths ──

    las_dir_str = str(script_cfg["las_dir"])
    las_dir = resolve_relative_path(las_dir_str, root=REPO_ROOT)
    well_heads_file = resolve_relative_path(workflow.assets.well_heads_file, root=data_root)
    seismic_file = resolve_relative_path(workflow.seismic.file, root=data_root)
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
        output_root = resolve_relative_path(workflow.output_root, root=REPO_ROOT)
        output_dir = output_root / f"wavelet_batch_synthetic_depth_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir

    output_dirs = {
        "root": output_dir,
        "synthetic_qc": output_dir / "synthetic_qc",
        "shift_scans": output_dir / "shift_scans",
        "depth_shift_curves": output_dir / "depth_shift_curves",
        "shifted_preprocessed_las": output_dir / "shifted_preprocessed_las",
        "shifted_filtered_las": output_dir / "shifted_filtered_las",
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

    from cup.seismic.wavelet import (
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
    print(f"Wavelet full half-support={wavelet_full_half_s * 1000:.1f} ms")
    print(f"Shift scan: {shift_values_s[0] * 1000:.1f} to {shift_values_s[-1] * 1000:.1f} ms, n={shift_values_s.size}")
    if np.isfinite(source_expected_shift_s):
        print(f"Source auto-tie table_t_shift: {source_expected_shift_s * 1000:.3f} ms")
    print(f"Auto-tie log filter params: {auto_tie_log_filter_params}")

    # ── Load shared resources ──

    from cup.petrel.load import import_well_heads_petrel
    from cup.seismic.survey import open_survey
    from wtie.modeling.modeling import ConvModeler

    well_heads_df = import_well_heads_petrel(well_heads_file)

    seismic_cfg = workflow.seismic.as_dict()
    segy_options = (
        segy_options_from_config(seismic_cfg) or None
        if workflow.seismic.type == "segy"
        else None
    )
    survey = open_survey(
        seismic_file,
        seismic_type=workflow.seismic.type,
        segy_options=segy_options,
    )
    geometry_depth = survey.describe_geometry(domain="depth")
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
    run_summary_path = output_dir / "run_summary.json"

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

    ok_count = int((metrics_df["status"] == "ok").sum()) if "status" in metrics_df.columns else 0
    failed_count = int((metrics_df["status"] != "ok").sum()) if "status" in metrics_df.columns else len(metrics_df)
    summary_payload = {
        "schema_version": "wavelet_batch_synthetic_depth_v2",
        "script": "wavelet_batch_synthetic_depth.py",
        "status": "success" if ok_count > 0 else "failed",
        "completion_status": "complete" if failed_count == 0 else "partial" if ok_count > 0 else "failed",
        "sample_domain": "depth",
        "sample_unit": "m",
        "depth_basis": "tvdss",
        "output_dir": repo_relative_path(output_dir, root=REPO_ROOT),
        "inputs": {
            "config": repo_relative_path(args.config, root=REPO_ROOT) if args.config.exists() else str(args.config),
            "las_dir": repo_relative_path(las_dir, root=REPO_ROOT),
            "source_auto_tie_dir": repo_relative_path(source_auto_tie_dir, root=REPO_ROOT),
            "source_well_name": source_well_name,
            "wavelet_file": repo_relative_path(wavelet_path, root=REPO_ROOT),
            "seismic_file": repo_relative_path(seismic_file, root=REPO_ROOT),
        },
        "outputs": {
            "metrics_csv": repo_relative_path(metrics_path, root=REPO_ROOT),
            "synthetic_qc_dir": repo_relative_path(output_dirs["synthetic_qc"], root=REPO_ROOT),
            "shift_scans_dir": repo_relative_path(output_dirs["shift_scans"], root=REPO_ROOT),
            "depth_shift_curves_dir": repo_relative_path(output_dirs["depth_shift_curves"], root=REPO_ROOT),
            "shifted_preprocessed_las_dir": repo_relative_path(output_dirs["shifted_preprocessed_las"], root=REPO_ROOT),
            "shifted_filtered_las_dir": repo_relative_path(output_dirs["shifted_filtered_las"], root=REPO_ROOT),
            "figures_dir": repo_relative_path(output_dirs["figures"], root=REPO_ROOT),
        },
        "las_contract": {
            "shifted_preprocessed_las": {
                "role": "depth-shifted full Step-3 preprocessed LAS for Synthoseis full_log_ai",
                "source": "wavelet_batch_synthetic_depth.las_dir",
                "curves": "all numeric source curves; AI recomputed from DT_USM/RHO_GCC when available",
                "filtering": "none",
                "gap_policy": "preserve finite segments; do not bridge source null gaps",
            },
            "shifted_filtered_las": {
                "role": "depth-shifted filtered LAS for Synthoseis filtered_log_ai/background fit",
                "source": "Step-5 filtered logset used by synthetic shift scan",
                "curves": ["DT_USM", "RHO_GCC", "AI"],
                "filtering": dict(auto_tie_log_filter_params),
                "gap_policy": "same as Step-5 synthetic scan filtered logset",
            },
            "depth_shift_boundary_policy": "edge extrapolate outside depth_shift_curve support and record counts/fraction",
        },
        "counts": {
            "requested_wells": len(well_names),
            "successful_wells": ok_count,
            "failed_wells": failed_count,
        },
    }
    run_summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {run_summary_path}")

    # ── Manifest ──

    print(f"\n=== Outputs ===")
    print(f"Summary: {metrics_path}")
    print(f"Run summary: {run_summary_path}")
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
                "shifted_preprocessed_las_path",
                "shifted_filtered_las_path",
                "synthetic_fig_path",
                "shift_fig_path",
            ]:
                print(f"    {row[key]}")

    failed = metrics_df.loc[metrics_df["status"] != "ok", ["well_name", "error"]]
    if not failed.empty:
        details = "; ".join(
            f"{row.well_name}: {row.error}" for row in failed.itertuples(index=False)
        )
        raise RuntimeError(f"Depth batch completed with failed wells: {details}")


if __name__ == "__main__":
    main()
