"""Run deterministic post-stack impedance inversion as a GINN baseline.

Bypass experiment placed after step 7 (LFM) and before step 8 (GINN training).
Produces a non-neural, physically-constrained baseline for comparison with
GINN predictions.

Usage::

    python scripts/deterministic_inversion.py
    python scripts/deterministic_inversion.py --config experiments/common.yaml
    python scripts/deterministic_inversion.py --output-dir scripts/output/deterministic_inversion_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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

from cup.petrel.load import import_interpretation_petrel
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.seismic.target_zone import TargetZone
from cup.seismic.viz import impedance_qc_metrics, plot_well_impedance_qc
from cup.utils.config import deep_merge_dict
from cup.utils.io import (
    latest_run,
    load_yaml_config,
    repo_relative_path,
    resolve_relative_path,
    sanitize_filename,
    to_json_compatible,
    write_json,
)
from cup.well.assets import normalize_well_name
from cup.seismic.wavelet import load_wavelet_csv, validate_wavelet_dt
from ginn.anchor import load_log_ai_anchor_npz
from wtie.processing import grid

logger = logging.getLogger(__name__)

# =============================================================================
# Default configuration
# =============================================================================

DEFAULT_CONFIG: dict[str, Any] = {
    "source_runs": {
        "mode": "latest",
        "lfm_precomputed_dir": None,
        "wavelet_generation_dir": None,
        "well_constraints_dir": None,
    },
    "seismic": {
        "file": None,
        "type": None,
    },
    "ai_lfm_file": None,
    "wavelet_file": None,
    "log_ai_anchor_file": None,
    "boundary_extension_samples": 30,
    "epsR": 0.20,
    "damp": 0.03,
    "iter_lim": 100,
    "show_solver": True,
    "export_volume": True,
    "export_segy": False,
    "export_zgy": True,
    "zgy_inline_chunk_size": 16,
    "qc_wells": True,
    "slice_mode": "inline",
    "slice_index": None,
    "clip_percentiles": [1.0, 99.0],
}


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <output_root>/deterministic_inversion_<timestamp>.",
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


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "min": None, "p01": None, "median": None, "p99": None, "max": None, "mean": None}
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _axis_values(geometry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_il = int(geometry["n_il"])
    n_xl = int(geometry["n_xl"])
    n_sample = int(geometry["n_sample"])
    ilines = float(geometry["inline_min"]) + np.arange(n_il, dtype=np.float64) * float(geometry["inline_step"])
    xlines = float(geometry["xline_min"]) + np.arange(n_xl, dtype=np.float64) * float(geometry["xline_step"])
    samples = float(geometry["sample_min"]) + np.arange(n_sample, dtype=np.float64) * float(geometry["sample_step"])
    return ilines, xlines, samples


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


def _robust_limits(*arrays: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(arr, dtype=np.float32).ravel() for arr in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot compute robust limits from empty finite values.")
    lo, hi = np.percentile(values, percentiles)
    if np.isclose(lo, hi):
        pad = max(abs(float(lo)) * 0.01, 1.0)
        lo = float(lo) - pad
        hi = float(hi) + pad
    return float(lo), float(hi)


def _save_npz(
    path: Path,
    *,
    volume: np.ndarray,
    geometry: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    ilines, xlines, samples = _axis_values(geometry)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        volume=volume.astype(np.float32),
        ilines=ilines.astype(np.float32),
        xlines=xlines.astype(np.float32),
        samples=samples.astype(np.float32),
        geometry_json=json.dumps(to_json_compatible(geometry), ensure_ascii=False),
        metadata_json=json.dumps(to_json_compatible(metadata), ensure_ascii=False),
    )


def _zgy_corners_from_survey(survey: Any, ilines: np.ndarray, xlines: np.ndarray) -> tuple[tuple[float, float], ...]:
    il0 = float(ilines[0])
    iln = float(ilines[-1])
    xl0 = float(xlines[0])
    xln = float(xlines[-1])
    geometry = survey.line_geometry
    return (
        tuple(float(v) for v in geometry.line_to_coord(il0, xl0)),
        tuple(float(v) for v in geometry.line_to_coord(iln, xl0)),
        tuple(float(v) for v in geometry.line_to_coord(il0, xln)),
        tuple(float(v) for v in geometry.line_to_coord(iln, xln)),
    )


def _try_write_zgy(
    zgy_file: Path,
    *,
    volume: np.ndarray,
    geometry: dict[str, Any],
    seismic_file: Path,
    seismic_type: str,
    inline_chunk_size: int,
) -> str:
    if seismic_type.lower() != "zgy":
        return "skipped_non_zgy_source"
    try:
        from pyzgy.write import SeismicWriter

        ilines, xlines, samples = _axis_values(geometry)
        if samples.size < 2:
            raise ValueError("ZGY export requires at least two samples.")
        sample_step_s = float(np.median(np.diff(samples)))
        if not np.allclose(np.diff(samples), sample_step_s, rtol=1e-6, atol=1e-9):
            raise ValueError("ZGY export requires a regular sample axis.")
        inline_inc = float(np.median(np.diff(ilines))) if ilines.size > 1 else 0.0
        xline_inc = float(np.median(np.diff(xlines))) if xlines.size > 1 else 0.0

        survey = open_survey(seismic_file, seismic_type=seismic_type)
        corners = _zgy_corners_from_survey(survey, ilines, xlines)

        zgy_file.parent.mkdir(parents=True, exist_ok=True)
        if zgy_file.exists():
            zgy_file.unlink()
        chunk = max(1, int(inline_chunk_size))
        export_volume = np.asarray(volume, dtype=np.float32)
        with SeismicWriter(
            zgy_file,
            tuple(int(v) for v in export_volume.shape),
            float(samples[0]) * 1000.0,
            sample_step_s * 1000.0,
            (float(ilines[0]), float(xlines[0])),
            (inline_inc, xline_inc),
            corners=corners,
        ) as writer:
            for il_start in range(0, export_volume.shape[0], chunk):
                il_end = min(export_volume.shape[0], il_start + chunk)
                writer.write_subvolume(export_volume[il_start:il_end], il_start, 0, 0)
        return "written"
    except Exception as exc:
        return f"failed:{exc}"


def _json_scalar_to_dict(value: np.ndarray) -> dict[str, Any]:
    payload = np.asarray(value).item()
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return json.loads(str(payload))


def _is_missing_config_value(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or text.casefold() in {"none", "null", "nan"}


def _apply_seismic_fallback(script_cfg: dict[str, Any], common_cfg: dict[str, Any]) -> None:
    seismic_cfg = dict(script_cfg.get("seismic") or {})
    fallback = dict((common_cfg.get("lfm_precomputed") or {}).get("seismic") or {})
    if _is_missing_config_value(seismic_cfg.get("file")) and not _is_missing_config_value(fallback.get("file")):
        seismic_cfg["file"] = fallback["file"]
    if _is_missing_config_value(seismic_cfg.get("type")) and not _is_missing_config_value(fallback.get("type")):
        seismic_cfg["type"] = fallback["type"]
    script_cfg["seismic"] = seismic_cfg


def _resolve_run_file(
    *,
    configured_file: Any,
    configured_dir: Any,
    output_root: Path,
    prefix: str,
    filename: str,
    required: bool,
) -> Path | None:
    if not _is_missing_config_value(configured_file):
        return resolve_relative_path(configured_file, root=REPO_ROOT)

    if not _is_missing_config_value(configured_dir):
        run_dir = resolve_relative_path(configured_dir, root=REPO_ROOT)
    else:
        try:
            run_dir = latest_run(output_root, prefix, filename)
        except FileNotFoundError:
            if required:
                raise
            return None

    path = run_dir / filename
    if required and not path.exists():
        raise FileNotFoundError(f"Required deterministic inversion input does not exist: {path}")
    return path if path.exists() else None


# =============================================================================
# Segment 1: Load inputs
# =============================================================================


def _load_seismic(script_cfg: dict[str, Any], data_root: Path) -> tuple[np.ndarray, Any, Path, str, dict[str, Any]]:
    """Open seismic survey and read the full volume.

    Returns (volume, survey, seismic_file, seismic_type, geometry).
    """
    from cup.petrel.load import import_seismic

    seismic_cfg = dict(script_cfg.get("seismic") or {})
    raw_file = seismic_cfg.get("file")
    if _is_missing_config_value(raw_file):
        raise ValueError("deterministic_inversion.seismic.file must be set in config.")
    seismic_file = resolve_relative_path(raw_file, root=data_root)
    seismic_type_value = seismic_cfg.get("type")
    seismic_type = "segy" if _is_missing_config_value(seismic_type_value) else str(seismic_type_value).strip().lower()

    segy_opts = None
    if seismic_type == "segy":
        segy_opts = segy_options_from_config(dict(script_cfg.get("segy") or {}))

    volume = import_seismic(
        seismic_file,
        seismic_type=seismic_type,
        iline=segy_opts.get("iline") if segy_opts else None,
        xline=segy_opts.get("xline") if segy_opts else None,
        istep=segy_opts.get("istep") if segy_opts else None,
        xstep=segy_opts.get("xstep") if segy_opts else None,
    )
    survey = open_survey(seismic_file, seismic_type=seismic_type, segy_options=segy_opts)
    geometry = survey.describe_geometry(domain="time")
    logger.info("Loaded seismic: shape=%s, file=%s", volume.shape, seismic_file)
    return volume, survey, seismic_file, seismic_type, geometry


def _load_lfm_npz(lfm_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Load the step-7 LFM NPZ.

    Returns (lfm_volume, samples, metadata, geometry).
    """
    if not lfm_path.exists():
        raise FileNotFoundError(f"AI LFM NPZ not found: {lfm_path}")
    if lfm_path.suffix.lower() != ".npz":
        raise ValueError(f"AI LFM file must be .npz, got {lfm_path.suffix}")

    with np.load(lfm_path, allow_pickle=False) as archive:
        if "volume" not in archive.files:
            raise ValueError(f"AI LFM NPZ must contain 'volume' key, got {archive.files}")
        volume = np.asarray(archive["volume"], dtype=np.float32)

        metadata = {}
        if "metadata_json" in archive.files:
            metadata = _json_scalar_to_dict(archive["metadata_json"])

        geometry = {}
        if "geometry_json" in archive.files:
            geometry = _json_scalar_to_dict(archive["geometry_json"])

        samples = None
        if "samples" in archive.files:
            samples = np.asarray(archive["samples"], dtype=np.float64)

    if samples is None:
        raise ValueError("AI LFM NPZ must contain 'samples' axis.")
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError(f"AI LFM NPZ 'samples' must be a non-empty 1D axis, got shape={samples.shape}.")
    logger.info(
        "Loaded LFM: shape=%s, samples=[%.6f, ..., %.6f] s",
        volume.shape,
        float(samples[0]),
        float(samples[-1]),
    )
    return volume, samples, metadata, geometry


def _load_wavelet(wavelet_path: Path, seismic_dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Load the step-5 selected wavelet and validate against seismic dt."""
    if not wavelet_path.exists():
        raise FileNotFoundError(f"Wavelet CSV not found: {wavelet_path}")
    time_s, amplitude = load_wavelet_csv(wavelet_path)
    validate_wavelet_dt(time_s, seismic_dt_s)
    logger.info(
        "Loaded wavelet: %d samples, dt=%.6f s, peak=%.4f",
        time_s.size,
        float(np.median(np.diff(time_s))),
        float(np.max(np.abs(amplitude))),
    )
    return time_s, amplitude


def _validate_inputs(
    seismic: np.ndarray,
    lfm: np.ndarray,
    seismic_samples: np.ndarray,
    lfm_samples: np.ndarray,
    wavelet_dt_s: float,
    seismic_dt_s: float,
) -> None:
    """Run all contract validations before inversion."""
    if seismic.shape != lfm.shape:
        raise ValueError(
            f"Seismic shape {seismic.shape} does not match LFM shape {lfm.shape}."
        )

    if lfm_samples.shape != (seismic.shape[2],):
        raise ValueError(
            f"LFM samples length {lfm_samples.size} does not match seismic n_sample={seismic.shape[2]}."
        )

    if not np.allclose(lfm_samples, seismic_samples, rtol=0.0, atol=1e-6):
        max_diff = float(np.max(np.abs(lfm_samples - seismic_samples)))
        raise ValueError(
            f"LFM samples axis does not match seismic time axis (max_abs_diff={max_diff:.6g} s)."
        )

    if not np.isclose(wavelet_dt_s, seismic_dt_s, rtol=1e-5, atol=1e-9):
        raise ValueError(
            f"Wavelet dt ({wavelet_dt_s:.6f} s) does not match seismic dt ({seismic_dt_s:.6f} s)."
        )

    if np.any(~np.isfinite(lfm)) or np.any(lfm <= 0.0):
        raise ValueError("LFM must be finite and positive everywhere.")

    logger.info("Input validation passed: shape=%s, dt=%.6f s", seismic.shape, seismic_dt_s)


# =============================================================================
# Segment 2: Reconstruct target window
# =============================================================================


def _build_target_zone(
    lfm_metadata: dict[str, Any],
    geometry: dict[str, Any],
    data_root: Path,
) -> TargetZone:
    """Reconstruct target zone from LFM metadata horizons."""
    hz_list = lfm_metadata.get("horizons")
    if not hz_list or len(hz_list) < 2:
        raise ValueError(
            "AI LFM NPZ metadata must contain at least two sorted horizons. "
            "Fix the step-7 LFM output before running deterministic inversion."
        )

    tl_meta = lfm_metadata.get("target_layer", {})

    raw_horizons: dict[str, pd.DataFrame] = {}
    for idx, hz_entry in enumerate(hz_list):
        hz_file_rel = hz_entry.get("file")
        if hz_file_rel is None:
            raise ValueError(f"Horizon {idx} in LFM metadata is missing 'file' field.")
        hz_file = resolve_relative_path(hz_file_rel, root=REPO_ROOT)
        if not hz_file.exists():
            raise FileNotFoundError(f"Horizon file not found: {hz_file} (resolved from {hz_file_rel})")
        raw_horizons[f"horizon_{idx}"] = import_interpretation_petrel(hz_file)

    # Build TargetZone with metadata QC params
    target_layer = TargetZone(
        raw_horizon_dfs=raw_horizons,
        geometry=geometry,
        horizon_names=list(raw_horizons.keys()),
        min_thickness=tl_meta.get("min_thickness"),
        nearest_distance_limit=tl_meta.get("nearest_distance_limit"),
        outlier_threshold=tl_meta.get("outlier_threshold"),
        outlier_min_neighbor_count=tl_meta.get("outlier_min_neighbor_count", 2),
    )
    logger.info(
        "Reconstructed target zone: %d horizons, valid_traces=%d",
        len(target_layer.horizon_names),
        int(np.count_nonzero(target_layer.valid_control_mask)),
    )
    return target_layer


def _find_target_window(
    target_layer: TargetZone,
    boundary_extension_samples: int,
) -> tuple[int, int, np.ndarray]:
    """Find the global time window covering all valid target zone samples.

    Returns (window_start, window_end, target_mask) where target_mask
    has shape (n_il, n_xl, n_sample).
    """
    mask = target_layer.to_mask(use_valid_control_mask=False)
    n_il, n_xl, n_sample = mask.shape

    if not np.any(mask):
        raise ValueError("Target layer mask is empty — no valid samples found.")

    # Find global time range of valid samples
    sample_has_mask = np.any(mask, axis=(0, 1))
    valid_indices = np.flatnonzero(sample_has_mask)
    if valid_indices.size == 0:
        raise ValueError("Target layer mask covers no time samples.")

    t_min_idx = int(valid_indices[0])
    t_max_idx = int(valid_indices[-1])

    window_start = max(0, t_min_idx - int(boundary_extension_samples))
    window_end = min(n_sample, t_max_idx + int(boundary_extension_samples) + 1)

    if window_start >= window_end:
        raise ValueError(
            f"Target window is empty: start={window_start}, end={window_end}, "
            f"n_sample={n_sample}, boundary_extension={boundary_extension_samples}"
        )

    logger.info(
        "Target window: samples [%d, %d), size=%d, target_range=[%d, %d], "
        "boundary_extension=%d",
        window_start,
        window_end,
        window_end - window_start,
        t_min_idx,
        t_max_idx,
        boundary_extension_samples,
    )
    return window_start, window_end, mask


# =============================================================================
# Segment 3 & 4: Build operators and solve inversion
# =============================================================================


def _build_operators(
    wavelet_amplitude: np.ndarray,
    nt_window: int,
    n_inline: int,
    n_xline: int,
) -> tuple[Any, Any]:
    """Build forward modelling operator and spatial regularization operator.

    Returns (Pop, Reg_spatial).
    """
    try:
        from pylops.avo.poststack import PoststackLinearModelling
        from pylops import Laplacian
    except ImportError as exc:
        raise ImportError(
            "PyLops is required for deterministic inversion. "
            "Install with: pip install pylops"
        ) from exc

    wavelet = np.asarray(wavelet_amplitude, dtype=np.float64)
    if wavelet.size == 0:
        raise ValueError("Wavelet is empty.")

    Pop = PoststackLinearModelling(
        wavelet,
        nt0=nt_window,
        spatdims=(n_inline, n_xline),
        explicit=False,
        kind="forward",
    )
    logger.info("Built PoststackLinearModelling: nt0=%d, spatdims=(%d, %d)", nt_window, n_inline, n_xline)

    Reg_spatial = Laplacian(
        (nt_window, n_inline, n_xline),
        axes=(1, 2),
        dtype=np.float64,
    )
    logger.info("Built spatial Laplacian: shape=(%d, %d, %d), axes=(1, 2)", nt_window, n_inline, n_xline)

    return Pop, Reg_spatial


def _solve_inversion(
    Pop: Any,
    Reg_spatial: Any,
    d_obs: np.ndarray,
    m0: np.ndarray,
    epsR: float,
    damp: float,
    iter_lim: int,
    show_solver: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the regularized least-squares inversion.

    Returns (m_inv, solver_info).
    """
    try:
        from pylops.optimization.leastsquares import regularized_inversion
    except ImportError as exc:
        raise ImportError(
            "PyLops is required for deterministic inversion. "
            "Install with: pip install pylops"
        ) from exc

    logger.info(
        "Starting regularized inversion: epsR=%.4f, damp=%.4f, iter_lim=%d, n_params=%d",
        epsR,
        damp,
        iter_lim,
        int(m0.size),
    )

    xinv, istop, itn, r1norm, r2norm = regularized_inversion(
        Pop,
        d_obs.ravel().astype(np.float64),
        [Reg_spatial],
        x0=m0.ravel().astype(np.float64),
        epsRs=[float(epsR)],
        damp=float(damp),
        iter_lim=int(iter_lim),
        show=bool(show_solver),
    )

    r1_values = np.atleast_1d(np.asarray(r1norm, dtype=np.float64))
    r2_values = np.atleast_1d(np.asarray(r2norm, dtype=np.float64))
    cost_initial = (
        float(r1_values[0] + r2_values[0])
        if r1_values.size > 0 and r2_values.size > 0
        else float("nan")
    )
    cost_final_data = float(r1_values[-1]) if r1_values.size > 0 else float("nan")
    cost_final_reg = float(r2_values[-1]) if r2_values.size > 0 else float("nan")

    solver_info: dict[str, Any] = {
        "istop": int(istop),
        "niter": int(itn),
        "cost_initial": cost_initial,
        "cost_final_data_misfit": cost_final_data,
        "cost_final_regularization": cost_final_reg,
        "r1norm": [float(v) for v in r1_values],
        "r2norm": [float(v) for v in r2_values],
    }

    logger.info(
        "Inversion finished: istop=%d, niter=%d, cost_data=%.6g, cost_reg=%.6g",
        int(istop),
        int(itn),
        cost_final_data,
        cost_final_reg,
    )
    return xinv, solver_info


# =============================================================================
# Segment 5: Export & QC
# =============================================================================


def _plot_deterministic_slice(
    path: Path,
    *,
    det_volume: np.ndarray,
    lfm_volume: np.ndarray,
    mask_volume: np.ndarray,
    window_start: int,
    window_end: int,
    geometry: dict[str, Any],
    slice_mode: str,
    slice_index: int,
    clip_percentiles: tuple[float, float],
) -> None:
    det_section = _extract_section(det_volume, slice_mode, slice_index)
    lfm_section = _extract_section(lfm_volume, slice_mode, slice_index)
    mask_section = _extract_section(mask_volume.astype(np.float32, copy=False), slice_mode, slice_index)
    diff_section = det_section - lfm_section

    shared_vmin, shared_vmax = _robust_limits(det_section, lfm_section, percentiles=clip_percentiles)
    finite_diff = diff_section[np.isfinite(diff_section)]
    diff_abs = float(np.percentile(np.abs(finite_diff), clip_percentiles[1])) if finite_diff.size else 1.0
    diff_abs = max(diff_abs, 1.0)

    _, _, samples = _axis_values(geometry)
    if slice_mode == "inline":
        _, xlines, _ = _axis_values(geometry)
        extent = [float(xlines[0]), float(xlines[-1]), float(samples[-1]), float(samples[0])]
        xlabel = "Xline"
    else:
        ilines, _, _ = _axis_values(geometry)
        extent = [float(ilines[0]), float(ilines[-1]), float(samples[-1]), float(samples[0])]
        xlabel = "Inline"

    fig, axes = plt.subplots(1, 4, figsize=(20, 7), constrained_layout=True)

    im0 = axes[0].imshow(
        det_section, cmap="viridis", aspect="auto", origin="upper",
        extent=extent, vmin=shared_vmin, vmax=shared_vmax,
    )
    axes[0].set_title(f"Deterministic AI | {slice_mode}={slice_index}")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("TWT (s)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(
        lfm_section, cmap="viridis", aspect="auto", origin="upper",
        extent=extent, vmin=shared_vmin, vmax=shared_vmax,
    )
    axes[1].set_title("LFM")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("TWT (s)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    im2 = axes[2].imshow(
        diff_section, cmap="seismic", aspect="auto", origin="upper",
        extent=extent, vmin=-diff_abs, vmax=diff_abs,
    )
    axes[2].set_title("Deterministic - LFM")
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel("TWT (s)")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)

    im3 = axes[3].imshow(
        mask_section, cmap="gray_r", aspect="auto", origin="upper",
        extent=extent, vmin=0.0, vmax=1.0,
    )
    axes[3].set_title("Target layer mask")
    axes[3].set_xlabel(xlabel)
    axes[3].set_ylabel("TWT (s)")
    fig.colorbar(im3, ax=axes[3], shrink=0.85)

    # Mark the inversion window boundaries
    window_twt_top = float(samples[window_start])
    window_twt_bot = float(samples[window_end - 1])
    for ax in axes[:3]:
        ax.axhline(window_twt_top, color="cyan", lw=1.0, ls="--", alpha=0.7)
        ax.axhline(window_twt_bot, color="cyan", lw=1.0, ls="--", alpha=0.7)

    _save_fig(path)


def _write_well_qc(
    *,
    output_dir: Path,
    det_volume: np.ndarray,
    lfm_volume: np.ndarray,
    anchor_file: Path | None,
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Generate well QC comparing deterministic AI vs anchor data and LFM."""
    well_qc_dir = output_dir / "well_qc"
    trace_dir = well_qc_dir / "traces"
    figure_dir = well_qc_dir / "figures"
    trace_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = well_qc_dir / "well_qc_metrics.csv"

    if anchor_file is None:
        metrics_df = pd.DataFrame([{"status": "skipped", "reason": "no_log_ai_anchor_file"}])
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        return {
            "well_qc_dir": repo_relative_path(well_qc_dir, root=REPO_ROOT),
            "metrics": repo_relative_path(metrics_path, root=REPO_ROOT),
            "n_wells_qc": 0,
        }

    if not anchor_file.exists():
        logger.warning("Anchor file not found: %s — skipping well QC.", anchor_file)
        metrics_df = pd.DataFrame([{"status": "skipped", "reason": f"anchor_file_not_found:{anchor_file}"}])
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        return {
            "well_qc_dir": repo_relative_path(well_qc_dir, root=REPO_ROOT),
            "metrics": repo_relative_path(metrics_path, root=REPO_ROOT),
            "n_wells_qc": 0,
        }

    anchor = load_log_ai_anchor_npz(anchor_file)
    n_sample = int(geometry["n_sample"])
    det_flat = np.asarray(det_volume, dtype=np.float32).reshape(-1, n_sample)
    lfm_flat = np.asarray(lfm_volume, dtype=np.float32).reshape(-1, n_sample)
    anchor_samples = np.asarray(anchor.samples, dtype=np.float64)
    metrics_rows: list[dict[str, Any]] = []

    for anchor_row, flat_idx_value in enumerate(np.asarray(anchor.flat_indices, dtype=np.int64)):
        flat_idx = int(flat_idx_value)
        anchor_name = str(np.asarray(anchor.anchor_names).astype(str)[anchor_row])
        safe = sanitize_filename(f"{flat_idx}_{anchor_name}")
        trace_path = trace_dir / f"well_qc_{safe}.csv"
        figure_path = figure_dir / f"well_qc_{safe}.png"

        if flat_idx < 0 or flat_idx >= det_flat.shape[0]:
            metrics_rows.append({
                "well_name": anchor_name, "flat_idx": flat_idx,
                "status": "failed", "error": "flat_idx_out_of_range",
            })
            continue

        target_ai = np.asarray(anchor.target_ai[anchor_row], dtype=np.float64)
        det_ai = np.asarray(det_flat[flat_idx], dtype=np.float64)
        lfm_ai = np.asarray(lfm_flat[flat_idx], dtype=np.float64)
        mask = (
            np.asarray(anchor.anchor_mask[anchor_row], dtype=bool)
            & np.isfinite(target_ai)
            & np.isfinite(det_ai)
            & np.isfinite(lfm_ai)
        )

        trace_df = pd.DataFrame({
            "twt_s": anchor_samples,
            "anchor_target_ai": target_ai,
            "deterministic_ai": det_ai,
            "lfm_ai": lfm_ai,
            "anchor_mask": np.asarray(anchor.anchor_mask[anchor_row], dtype=bool),
            "valid_for_metrics": mask,
        })
        trace_df.to_csv(trace_path, index=False, encoding="utf-8-sig")

        # Plot: full-band anchor + LFM + deterministic AI
        # Note: well anchor data is low-frequency; full-band well AI may be absent.
        anchor_trace = grid.Log(target_ai, anchor_samples, "twt", name="Anchor target AI")
        det_trace = grid.Log(det_ai, anchor_samples, "twt", name="Deterministic AI")
        lfm_trace = grid.Log(lfm_ai, anchor_samples, "twt", name="LFM")

        fig, _axes = plot_well_impedance_qc(
            full_ai=anchor_trace,
            low_ai=lfm_trace,
            model_ai=det_trace,
            mask=mask,
            title=f"Deterministic inversion well QC | {anchor_name}",
            model_label="Deterministic AI",
        )
        fig.savefig(figure_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        qc_metrics = impedance_qc_metrics(model_ai=det_trace, low_ai=lfm_trace, full_ai=anchor_trace, mask=mask)
        qc_metrics = {
            key.replace("vs_full_", "vs_anchor_"): value
            for key, value in qc_metrics.items()
        }

        metrics_rows.append({
            "well_name": anchor_name,
            "flat_idx": flat_idx,
            "inline": float(np.asarray(anchor.inline, dtype=np.float64)[anchor_row]),
            "xline": float(np.asarray(anchor.xline, dtype=np.float64)[anchor_row]),
            "status": "ok" if np.any(mask) else "no_valid_samples",
            "trace_csv": repo_relative_path(trace_path, root=REPO_ROOT),
            "figure": repo_relative_path(figure_path, root=REPO_ROOT),
            **qc_metrics,
        })

    metrics_df = pd.DataFrame.from_records(metrics_rows)
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    ok = metrics_df.loc[metrics_df["status"] == "ok"] if "status" in metrics_df else pd.DataFrame()
    summary: dict[str, Any] = {
        "well_qc_dir": repo_relative_path(well_qc_dir, root=REPO_ROOT),
        "metrics": repo_relative_path(metrics_path, root=REPO_ROOT),
        "n_wells_qc": int(len(ok)),
    }
    if not ok.empty and "vs_anchor_mae" in ok:
        vs_col = "vs_anchor_mae"
        if vs_col in ok.columns:
            summary["mean_vs_anchor_mae"] = float(ok[vs_col].mean())
        vs_col = "vs_anchor_rmse"
        if vs_col in ok.columns:
            summary["mean_vs_anchor_rmse"] = float(ok[vs_col].mean())
        vs_col = "vs_anchor_corr"
        if vs_col in ok.columns:
            summary["mean_vs_anchor_corr"] = float(ok[vs_col].mean())
    return summary


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    common_cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    if "deterministic_inversion" not in common_cfg:
        raise ValueError("Missing 'deterministic_inversion' section in config.")
    script_cfg = deep_merge_dict(DEFAULT_CONFIG, dict(common_cfg.get("deterministic_inversion") or {}))
    _apply_seismic_fallback(script_cfg, common_cfg)

    data_root = REPO_ROOT / str(common_cfg.get("data_root", "data"))
    output_root = resolve_relative_path(common_cfg.get("output_root", "scripts/output"), root=REPO_ROOT)
    if args.output_dir is None:
        output_dir = output_root / f"deterministic_inversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = resolve_relative_path(args.output_dir, root=REPO_ROOT)

    figure_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    well_qc_dir = output_dir / "well_qc"
    for path in (output_dir, figure_dir, metadata_dir, well_qc_dir):
        path.mkdir(parents=True, exist_ok=True)

    # ── Resolve required input paths ──
    source_cfg = dict(script_cfg.get("source_runs") or {})
    mode = str(source_cfg.get("mode", "latest")).strip().casefold()
    if mode != "latest":
        raise ValueError(f"deterministic_inversion.source_runs.mode only supports 'latest', got {mode!r}.")

    ai_lfm_path = _resolve_run_file(
        configured_file=script_cfg.get("ai_lfm_file"),
        configured_dir=source_cfg.get("lfm_precomputed_dir"),
        output_root=output_root,
        prefix="lfm_precomputed",
        filename="ai_lfm_time.npz",
        required=True,
    )
    wavelet_path = _resolve_run_file(
        configured_file=script_cfg.get("wavelet_file"),
        configured_dir=source_cfg.get("wavelet_generation_dir"),
        output_root=output_root,
        prefix="wavelet_generation",
        filename="selected_wavelet.csv",
        required=True,
    )
    anchor_path = _resolve_run_file(
        configured_file=script_cfg.get("log_ai_anchor_file"),
        configured_dir=source_cfg.get("well_constraints_dir"),
        output_root=output_root,
        prefix="well_constraints",
        filename="log_ai_anchor_time.npz",
        required=False,
    )
    if anchor_path is None:
        logger.warning("No log_ai_anchor_time.npz found — well QC will be skipped.")
    elif not anchor_path.exists():
        logger.warning("Anchor file does not exist: %s — well QC will be skipped.", anchor_path)
        anchor_path = None
    if ai_lfm_path is None or wavelet_path is None:
        raise RuntimeError("Internal error: required deterministic inversion input path resolved to None.")

    # ── Segment 1: Load inputs ──
    logger.info("=== Segment 1: Loading inputs ===")
    seismic, survey, seismic_file, seismic_type, geometry = _load_seismic(script_cfg, data_root)
    seismic_samples = (
        float(geometry["sample_min"])
        + np.arange(int(geometry["n_sample"]), dtype=np.float64) * float(geometry["sample_step"])
    )
    seismic_dt_s = float(geometry["sample_step"])

    lfm, lfm_samples, lfm_metadata, lfm_geometry = _load_lfm_npz(ai_lfm_path)
    wavelet_time_s, wavelet_amplitude = _load_wavelet(wavelet_path, seismic_dt_s)
    wavelet_dt_s = float(np.median(np.diff(wavelet_time_s)))

    _validate_inputs(seismic, lfm, seismic_samples, lfm_samples, wavelet_dt_s, seismic_dt_s)

    # ── Segment 2: Reconstruct target window ──
    logger.info("=== Segment 2: Reconstructing target window ===")
    target_layer = _build_target_zone(lfm_metadata, geometry, data_root)
    boundary_ext = int(script_cfg["boundary_extension_samples"])
    window_start, window_end, full_mask = _find_target_window(target_layer, boundary_ext)

    nt_window = window_end - window_start
    n_inline, n_xline = seismic.shape[0], seismic.shape[1]
    logger.info("Window dimensions: nt=%d, n_inline=%d, n_xline=%d", nt_window, n_inline, n_xline)

    # Crop to window
    seis_window = seismic[:, :, window_start:window_end].astype(np.float64)
    lfm_window = lfm[:, :, window_start:window_end].astype(np.float64)
    mask_window = full_mask[:, :, window_start:window_end]
    window_samples = seismic_samples[window_start:window_end]

    if not np.any(np.isfinite(seis_window)):
        raise ValueError("Seismic window contains no finite values.")
    if not np.any(np.isfinite(lfm_window)):
        raise ValueError("LFM window contains no finite values.")

    # ── Normalize seismic ──
    # RMS normalization within the target layer mask for auditability
    valid_seis = seis_window[mask_window]
    if valid_seis.size == 0:
        raise ValueError("No valid seismic samples within target layer mask window.")
    seis_rms = float(np.sqrt(np.mean(valid_seis.astype(np.float64) ** 2)))
    if seis_rms <= 0.0:
        raise ValueError(f"Seismic RMS within target window is non-positive: {seis_rms}.")
    d_obs = np.moveaxis(seis_window / seis_rms, -1, 0).astype(np.float64, copy=False)
    logger.info("Seismic normalization: target_mask_rms=%.6f, n_valid=%d", seis_rms, int(valid_seis.size))

    # ── Segment 3: Build operators ──
    logger.info("=== Segment 3: Building operators ===")
    Pop, Reg_spatial = _build_operators(wavelet_amplitude, nt_window, n_inline, n_xline)

    # ── Segment 4: Solve inversion ──
    logger.info("=== Segment 4: Solving inversion ===")
    m0_log = np.moveaxis(np.log(np.maximum(lfm_window, 1e-6)), -1, 0).astype(np.float64, copy=False)

    epsR = float(script_cfg["epsR"])
    damp = float(script_cfg["damp"])
    iter_lim = int(script_cfg["iter_lim"])
    show_solver = bool(script_cfg.get("show_solver", True))

    xinv_log, solver_info = _solve_inversion(
        Pop, Reg_spatial, d_obs, m0_log,
        epsR=epsR, damp=damp, iter_lim=iter_lim, show_solver=show_solver,
    )

    # Transform back to AI
    ai_window_time_first = np.exp(xinv_log.reshape(nt_window, n_inline, n_xline)).astype(np.float32)
    ai_window = np.moveaxis(ai_window_time_first, 0, -1)

    # Validate output
    if np.any(~np.isfinite(ai_window)):
        n_bad = int(np.sum(~np.isfinite(ai_window)))
        raise ValueError(
            f"Deterministic inversion produced {n_bad} non-finite AI values. "
            f"Increase damp (current={damp}) or epsR (current={epsR})."
        )
    n_negative = int(np.sum(ai_window <= 0.0))
    if n_negative > 0:
        logger.warning(
            "Deterministic inversion produced %d non-positive AI values. "
            "Consider increasing damp (current=%.4f) or epsR (current=%.4f).",
            n_negative, damp, epsR,
        )
        ai_window = np.maximum(ai_window, 1e-6)

    # Assemble full volume: LFM outside window, inverted AI inside window
    ai_full = lfm.astype(np.float32).copy()
    ai_full[:, :, window_start:window_end] = ai_window

    # ── Segment 5: Export & QC ──
    logger.info("=== Segment 5: Export & QC ===")

    npz_path = output_dir / "deterministic_ai_full.npz"
    zgy_path = output_dir / "deterministic_ai_full.zgy"

    # Statistics
    det_stats = _stats(ai_full)
    lfm_stats = _stats(lfm)
    diff_stats_data = (ai_full.astype(np.float64) - lfm.astype(np.float64)).ravel()
    diff_stats_data = diff_stats_data[np.isfinite(diff_stats_data)]
    diff_stats = _stats(diff_stats_data) if diff_stats_data.size else {"n": 0}

    prediction_stats = {
        "deterministic_ai": det_stats,
        "lfm_ai": lfm_stats,
        "deterministic_minus_lfm": diff_stats,
        "target_mask_coverage": float(full_mask.mean()),
        "window_start_sample": window_start,
        "window_end_sample": window_end,
    }

    # Metadata
    clip_values = tuple(float(v) for v in script_cfg.get("clip_percentiles", [1.0, 99.0]))
    if len(clip_values) != 2:
        raise ValueError("clip_percentiles must contain exactly two values.")

    slice_mode = str(script_cfg.get("slice_mode", "inline"))
    raw_slice_index = script_cfg.get("slice_index")
    slice_index = None if raw_slice_index is None else int(raw_slice_index)
    resolved_slice_index = _resolve_slice_index(slice_mode, slice_index, geometry)

    metadata: dict[str, Any] = {
        "artifact": "deterministic_ai_full.npz",
        "source_script": Path(__file__).name,
        "experiment_type": "baseline/bypass",
        "ai_lfm_file": repo_relative_path(ai_lfm_path, root=REPO_ROOT),
        "wavelet_file": repo_relative_path(wavelet_path, root=REPO_ROOT),
        "log_ai_anchor_file": None if anchor_path is None else repo_relative_path(anchor_path, root=REPO_ROOT),
        "seismic_file": repo_relative_path(seismic_file, root=REPO_ROOT),
        "seismic_type": seismic_type,
        "config": script_cfg,
        "solver": {
            "epsR": epsR,
            "damp": damp,
            "iter_lim": iter_lim,
            **solver_info,
        },
        "normalization": {
            "method": "target_layer_rms",
            "seis_rms": float(seis_rms),
            "n_valid_samples": int(valid_seis.size),
        },
        "window": {
            "start_sample": window_start,
            "end_sample": window_end,
            "start_twt_s": float(window_samples[0]),
            "end_twt_s": float(window_samples[-1]),
            "boundary_extension_samples": boundary_ext,
        },
        "target_layer": lfm_metadata.get("target_layer", {}),
        "horizons": lfm_metadata.get("horizons", []),
        "prediction_stats": prediction_stats,
    }

    # Slice QC figure
    slice_figure_path = figure_dir / f"{slice_mode}_{resolved_slice_index:04d}_deterministic_vs_lfm.png"
    _plot_deterministic_slice(
        slice_figure_path,
        det_volume=ai_full,
        lfm_volume=lfm,
        mask_volume=full_mask,
        window_start=window_start,
        window_end=window_end,
        geometry=geometry,
        slice_mode=slice_mode,
        slice_index=resolved_slice_index,
        clip_percentiles=clip_values,
    )

    # Well QC
    well_qc_summary: dict[str, Any] = {"well_qc_dir": None, "n_wells_qc": 0}
    if bool(script_cfg.get("qc_wells", True)):
        well_qc_summary = _write_well_qc(
            output_dir=output_dir,
            det_volume=ai_full,
            lfm_volume=lfm,
            anchor_file=anchor_path,
            geometry=geometry,
        )

    # Save NPZ
    _save_npz(npz_path, volume=ai_full, geometry=geometry, metadata=metadata)

    # ZGY export
    zgy_status = "disabled"
    if bool(script_cfg.get("export_zgy", True)):
        zgy_status = _try_write_zgy(
            zgy_path,
            volume=ai_full,
            geometry=geometry,
            seismic_file=seismic_file,
            seismic_type=seismic_type,
            inline_chunk_size=int(script_cfg.get("zgy_inline_chunk_size", 16)),
        )

    # Run summary
    outputs = {
        "deterministic_ai_full_npz": repo_relative_path(npz_path, root=REPO_ROOT),
        "deterministic_ai_full_zgy": repo_relative_path(zgy_path, root=REPO_ROOT) if zgy_status == "written" else None,
        "slice_figure": repo_relative_path(slice_figure_path, root=REPO_ROOT),
        "well_qc_dir": well_qc_summary.get("well_qc_dir"),
        "well_qc_metrics": well_qc_summary.get("metrics"),
    }
    metadata["outputs"] = outputs
    metadata["zgy_export_status"] = zgy_status

    summary = {
        "config": script_cfg,
        "input_paths": {
            "ai_lfm_file": repo_relative_path(ai_lfm_path, root=REPO_ROOT),
            "wavelet_file": repo_relative_path(wavelet_path, root=REPO_ROOT),
            "log_ai_anchor_file": None if anchor_path is None else repo_relative_path(anchor_path, root=REPO_ROOT),
            "seismic_file": repo_relative_path(seismic_file, root=REPO_ROOT),
        },
        "geometry": geometry,
        "solver": metadata["solver"],
        "normalization": metadata["normalization"],
        "window": metadata["window"],
        "prediction_stats": prediction_stats,
        "outputs": outputs,
        "well_qc": well_qc_summary,
        "zgy_export_status": zgy_status,
    }
    write_json(metadata_dir / "run_summary.json", summary)

    # Final print
    print("=== Deterministic Inversion ===")
    print(f"Output: {output_dir}")
    print(f"Window: samples [{window_start}, {window_end}), nt={nt_window}")
    print(f"epsR={epsR}, damp={damp}, iter_lim={iter_lim}")
    print(f"Solver: istop={solver_info['istop']}, niter={solver_info['niter']}")
    print(f"NPZ: {npz_path}")
    print(f"ZGY export: {zgy_status}")
    print(f"Well QC: {well_qc_summary.get('n_wells_qc', 0)} wells")


if __name__ == "__main__":
    main()
