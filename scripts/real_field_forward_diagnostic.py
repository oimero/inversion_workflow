"""Run R1 fixed forward diagnostics for R0 real-field zero-shot outputs."""

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
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.seismic.viz import plot_well_waveform_qc
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sha256_file, write_json
from ginn_v2.real_field import (
    diagnostic_metrics,
    forward_log_ai,
    load_selected_wavelet,
    load_zero_shot_predictions,
    phase_shift_scan,
)
from wtie.optimize.similarity import dynamic_normalized_xcorr, normalized_xcorr
from wtie.processing import grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--zero-shot-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
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


def _align_forward_arrays(
    observed: np.ndarray,
    synthetic: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(observed, dtype=np.float64)[:, 1:]
    syn = np.asarray(synthetic, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)[:, 1:]
    if obs.shape != syn.shape or mask.shape != syn.shape:
        raise ValueError(f"Forward diagnostic shape mismatch: obs={obs.shape}, syn={syn.shape}, mask={mask.shape}")
    return obs, syn, mask


def _spatial_rows(*, model_role: str, observed: np.ndarray, synthetic: np.ndarray, valid: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for lateral_idx in range(observed.shape[0]):
        metrics = diagnostic_metrics(
            observed=observed[lateral_idx : lateral_idx + 1],
            synthetic=synthetic[lateral_idx : lateral_idx + 1],
            valid_mask=valid[lateral_idx : lateral_idx + 1],
        )
        rows.append({"model_role": model_role, "lateral_index": lateral_idx, **metrics})
    return rows


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) <= 0.0 or np.std(b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _finite_stats(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(data)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    finite = data[valid]
    if finite.size == 0:
        return {
            "n_valid": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "rms": float("nan"),
            "p01": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
        }
    return {
        "n_valid": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "rms": float(np.sqrt(np.mean(finite * finite))),
        "p01": float(np.quantile(finite, 0.01)),
        "p10": float(np.quantile(finite, 0.10)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p99": float(np.quantile(finite, 0.99)),
    }


def _normalization_for_role(predictions: dict[str, dict[str, object]], role: str) -> dict[str, object]:
    payload = predictions.get(role)
    if payload is None:
        return {}
    summary = payload.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("normalization"), dict):
        return dict(summary["normalization"])
    return {}


def _normalization_stats(normalization: dict[str, object], key: str) -> tuple[float, float]:
    item = normalization.get(key)
    if not isinstance(item, dict):
        return float("nan"), float("nan")
    return float(item.get("mean", float("nan"))), float(item.get("std", float("nan")))


def _configured_bands(run_cfg: dict) -> list[dict[str, float | str]]:
    spectral = dict(run_cfg.get("spectral_qc") or {})
    bands = spectral.get("bands") or [
        {"name": "lowfreq", "low_hz": 0.0, "high_hz": 10.0},
        {"name": "observable_band", "low_hz": 10.0, "high_hz": 20.0},
        {"name": "highfreq_or_nullspace", "low_hz": 20.0, "high_hz": 80.0},
    ]
    out = []
    for item in bands:
        if not isinstance(item, dict):
            raise ValueError("Each R1 spectral band must be a mapping.")
        out.append(
            {
                "name": str(item["name"]),
                "low_hz": float(item.get("low_hz", 0.0)),
                "high_hz": float(item["high_hz"]) if item.get("high_hz") is not None else float("inf"),
            }
        )
    return out


def _band_rms(values: np.ndarray, valid: np.ndarray, *, dt_s: float, low_hz: float, high_hz: float) -> float:
    data = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(data)
    if data.ndim != 1 or int(np.count_nonzero(mask)) < 8:
        return float("nan")
    prepared = np.where(mask, data - float(np.mean(data[mask])), 0.0)
    spectrum = np.fft.rfft(prepared)
    freqs = np.fft.rfftfreq(data.size, d=float(dt_s))
    band = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(band):
        return 0.0
    filtered = np.zeros_like(spectrum)
    filtered[band] = spectrum[band]
    component = np.fft.irfft(filtered, n=data.size)
    return float(np.sqrt(np.mean(component[mask] ** 2)))


def _band_component_1d(values: np.ndarray, valid: np.ndarray, *, dt_s: float, low_hz: float, high_hz: float) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(data)
    if data.ndim != 1:
        raise ValueError("_band_component_1d expects a 1-D array.")
    prepared = np.zeros_like(data, dtype=np.float64)
    if int(np.count_nonzero(mask)) >= 8:
        prepared = np.where(mask, data - float(np.mean(data[mask])), 0.0)
    spectrum = np.fft.rfft(prepared)
    freqs = np.fft.rfftfreq(data.size, d=float(dt_s))
    band = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    filtered = np.zeros_like(spectrum)
    if np.any(band):
        filtered[band] = spectrum[band]
    return np.fft.irfft(filtered, n=data.size)


def _band_pair_metrics(
    *,
    reference: np.ndarray,
    model: np.ndarray,
    valid: np.ndarray,
    dt_s: float,
    low_hz: float,
    high_hz: float,
) -> dict[str, float | int]:
    ref = np.asarray(reference, dtype=np.float64)
    pred = np.asarray(model, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(ref) & np.isfinite(pred)
    if int(np.count_nonzero(mask)) < 8:
        return {"n_valid": int(np.count_nonzero(mask)), "rmse": float("nan"), "corr": float("nan")}
    ref_band = _band_component_1d(ref, mask, dt_s=dt_s, low_hz=low_hz, high_hz=high_hz)
    pred_band = _band_component_1d(pred, mask, dt_s=dt_s, low_hz=low_hz, high_hz=high_hz)
    residual = pred_band[mask] - ref_band[mask]
    return {
        "n_valid": int(np.count_nonzero(mask)),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "corr": _safe_corr(ref_band[mask], pred_band[mask]),
    }


def _band_rms_2d(values: np.ndarray, valid: np.ndarray, *, dt_s: float, low_hz: float, high_hz: float) -> float:
    data = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(data)
    if data.ndim != 2 or not np.any(mask):
        return float("nan")
    prepared = np.zeros_like(data, dtype=np.float64)
    for idx in range(data.shape[0]):
        row_mask = mask[idx]
        if int(np.count_nonzero(row_mask)) < 8:
            continue
        prepared[idx] = np.where(row_mask, data[idx] - float(np.mean(data[idx, row_mask])), 0.0)
    spectrum = np.fft.rfft(prepared, axis=1)
    freqs = np.fft.rfftfreq(data.shape[1], d=float(dt_s))
    band = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(band):
        return 0.0
    filtered = np.zeros_like(spectrum)
    filtered[:, band] = spectrum[:, band]
    component = np.fft.irfft(filtered, n=data.shape[1], axis=1)
    return float(np.sqrt(np.mean(component[mask] ** 2)))


def _read_wavelet_csv(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if "amplitude" not in frame:
        raise ValueError(f"Wavelet CSV lacks amplitude column: {path}")
    return frame["amplitude"].to_numpy(dtype=np.float64)


def _load_wavelet_scenarios(wavelet_dir: Path, nominal: np.ndarray, nominal_meta: dict[str, object], run_cfg: dict) -> list[dict[str, object]]:
    scan_cfg = dict(run_cfg.get("diagnostic_scan") or {})
    limit = int(scan_cfg.get("candidate_wavelet_limit", 0) or 0)
    scenarios: list[dict[str, object]] = [
        {
            "wavelet_scenario_id": "nominal_selected",
            "source_well": str((nominal_meta.get("selected_source_well") or "optimized_consensus")),
            "candidate_score": float("nan"),
            "wavelet_path": str(nominal_meta["selected_wavelet_csv"]),
            "wavelet_sha256": str(nominal_meta["wavelet_sha256"]),
            "wavelet": nominal,
        }
    ]
    aggregate_path = wavelet_dir / "wavelet_candidate_aggregate.csv"
    selected_summary_path = wavelet_dir / "selected_wavelet_summary.json"
    if not aggregate_path.is_file() or not selected_summary_path.is_file():
        return scenarios
    aggregate = pd.read_csv(aggregate_path)
    with selected_summary_path.open("r", encoding="utf-8") as handle:
        selected_summary = json.load(handle)
    source_auto_tie = resolve_relative_path(str(selected_summary.get("source_auto_tie_dir") or ""), root=REPO_ROOT)
    if not source_auto_tie.is_dir():
        return scenarios
    aggregate = aggregate.sort_values("score", ascending=False)
    if limit > 0:
        aggregate = aggregate.head(limit)
    for _, row in aggregate.iterrows():
        source_well = str(row.get("source_well") or "")
        candidate = str(row.get("candidate_wavelet") or source_well)
        if not source_well:
            continue
        path = source_auto_tie / "wavelets" / f"wavelet_201ms_{source_well}.csv"
        if not path.is_file():
            continue
        wavelet = _read_wavelet_csv(path)
        if wavelet.size != nominal.size:
            continue
        scenarios.append(
            {
                "wavelet_scenario_id": candidate,
                "source_well": source_well,
                "candidate_score": float(pd.to_numeric(row.get("score"), errors="coerce")),
                "wavelet_path": repo_relative_path(path, root=REPO_ROOT),
                "wavelet_sha256": sha256_file(path),
                "wavelet": wavelet,
            }
        )
    return scenarios


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
    tdt_twt_sorted = tdt_twt[finite_tdt][order_tdt]
    tdt_md_sorted = tdt_md[finite_tdt][order_tdt]
    md_at_twt = np.interp(twt_s, tdt_twt_sorted, tdt_md_sorted, left=np.nan, right=np.nan)
    order_ai = np.argsort(md_axis[finite_ai])
    md_sorted = md_axis[finite_ai][order_ai]
    ai_sorted = ai[finite_ai][order_ai]
    ai_at_twt = np.interp(md_at_twt, md_sorted, ai_sorted, left=np.nan, right=np.nan)
    return np.where(ai_at_twt > 0.0, np.log(ai_at_twt), np.nan)


def _trace_distance_to_section(
    *,
    well_inline: float,
    well_xline: float,
    well_x_m: float,
    well_y_m: float,
    section_ilines: np.ndarray,
    section_xlines: np.ndarray,
    section_x_m: np.ndarray,
    section_y_m: np.ndarray,
) -> tuple[int, float]:
    del well_inline, well_xline, section_ilines, section_xlines
    distances = np.hypot(section_x_m - float(well_x_m), section_y_m - float(well_y_m))
    index = int(np.nanargmin(distances))
    return index, float(distances[index])


def _load_zero_shot_line_geometry(zero_shot_dir: Path):
    summary_path = zero_shot_dir / "real_field_zero_shot_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"real_field_zero_shot_summary.json not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    config_text = str(summary.get("config_file") or "").strip()
    if not config_text:
        raise ValueError("R1 XY well QC requires zero-shot summary config_file.")
    config_path = resolve_relative_path(config_text, root=REPO_ROOT)
    cfg = load_yaml_config(config_path)
    r0_cfg = dict(cfg.get("real_field_zero_shot") or {})
    inputs = dict(r0_cfg.get("real_field_inputs") or {})
    data_root = resolve_relative_path(cfg.get("data_root", "data"), root=REPO_ROOT)
    seismic_text = str(inputs.get("seismic_file") or "").strip()
    if not seismic_text:
        raise ValueError("R1 XY well QC requires real_field_zero_shot.real_field_inputs.seismic_file.")
    seismic_path = resolve_relative_path(seismic_text, root=data_root)
    seismic_type = str(inputs.get("seismic_type", inputs.get("type", "zgy"))).casefold()
    segy_options = segy_options_from_config(dict(inputs.get("segy_options") or {})) if seismic_type == "segy" else None
    survey = open_survey(seismic_path, seismic_type, segy_options=segy_options)
    return survey.line_geometry


def _line_xy_arrays(line_geometry, ilines: np.ndarray, xlines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(
        [line_geometry.line_to_coord(float(il), float(xl)) for il, xl in zip(ilines, xlines)],
        dtype=np.float64,
    )
    return xy[:, 0], xy[:, 1]


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
    required = {"well_name", "inline_float", "xline_float"}
    if not required.issubset(inventory.columns):
        return {}
    positions: dict[str, tuple[float, float]] = {}
    for _, row in inventory.iterrows():
        well = str(row.get("well_name", ""))
        inline = _coerce_float(row.get("inline_float"))
        xline = _coerce_float(row.get("xline_float"))
        if well and np.isfinite(inline) and np.isfinite(xline):
            positions[well] = (inline, xline)
    return positions


def _load_well_inventory_info(cfg: dict) -> dict[str, dict[str, object]]:
    inventory_text = str(cfg.get("well_inventory_file") or "").strip()
    if not inventory_text:
        return {}
    inventory_path = resolve_relative_path(inventory_text, root=REPO_ROOT)
    if not inventory_path.is_file():
        return {}
    inventory = pd.read_csv(inventory_path)
    if "well_name" not in inventory.columns:
        return {}
    info: dict[str, dict[str, object]] = {}
    for _, row in inventory.iterrows():
        well = str(row.get("well_name", ""))
        if not well:
            continue
        info[well] = {
            "inline_float": _coerce_float(row.get("inline_float")),
            "xline_float": _coerce_float(row.get("xline_float")),
            "wellbore_class": str(row.get("wellbore_class", "")),
        }
    return info


def _well_ai_metrics(*, well_log_ai: np.ndarray, pred_log_ai: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(well_log_ai) & np.isfinite(pred_log_ai)
    n_valid = int(np.count_nonzero(mask))
    if n_valid < 8:
        return {"well_ai_status": "insufficient_valid_samples", "well_ai_n_valid": n_valid}
    residual = pred_log_ai[mask] - well_log_ai[mask]
    return {
        "well_ai_status": "ok",
        "well_ai_n_valid": n_valid,
        "well_ai_rmse": float(np.sqrt(np.mean(residual * residual))),
        "well_ai_bias": float(np.mean(residual)),
        "well_ai_corr": _safe_corr(well_log_ai[mask], pred_log_ai[mask]),
        "well_ai_pred_mean": float(np.mean(pred_log_ai[mask])),
        "well_ai_log_mean": float(np.mean(well_log_ai[mask])),
    }


def _build_well_forward_qc(
    *,
    run_cfg: dict,
    zero_shot_dir: Path,
    output_dir: Path,
    predictions: dict[str, dict[str, object]],
    synthetic_dir: Path,
    wavelet: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, str]]:
    cfg = dict(run_cfg.get("well_qc") or {})
    if not bool(cfg.get("enabled", True)):
        frame = pd.DataFrame(columns=["status", "note"]).assign(
            status=["disabled"],
            note=["well_qc.enabled=false"],
        )
        return frame, {}
    well_auto_tie_text = str(cfg.get("well_auto_tie_dir") or "").strip()
    if not well_auto_tie_text:
        raise ValueError("real_field_forward_diagnostic.well_qc.well_auto_tie_dir must be explicit when well_qc is enabled.")
    well_auto_tie_dir = resolve_relative_path(
        well_auto_tie_text,
        root=REPO_ROOT,
    )
    metrics_path = well_auto_tie_dir / "well_tie_metrics.csv"
    if not metrics_path.is_file():
        frame = pd.DataFrame(columns=["status", "note"]).assign(
            status=["not_implemented"],
            note=[f"well_tie_metrics.csv not found: {metrics_path}"],
        )
        return frame, {}
    ties = pd.read_csv(metrics_path)
    ties = ties[ties["tie_status"].astype(str).eq("success")].copy()
    if ties.empty:
        frame = pd.DataFrame(columns=["status", "note"]).assign(
            status=["no_successful_ties"],
            note=["No successful well-auto-tie rows."],
        )
        return frame, {}
    first_arrays = next(iter(predictions.values()))["arrays"]
    section_ilines = np.asarray(first_arrays["ilines"], dtype=np.float64)
    section_xlines = np.asarray(first_arrays["xlines"], dtype=np.float64)
    twt_model = np.asarray(first_arrays["twt_s"], dtype=np.float64)
    if "max_line_distance" in cfg:
        raise ValueError("real_field_forward_diagnostic.well_qc.max_line_distance is retired; use max_xy_distance_m.")
    if cfg.get("max_xy_distance_m") is None:
        raise ValueError("real_field_forward_diagnostic.well_qc.max_xy_distance_m must be explicit.")
    max_xy_distance_m = float(cfg.get("max_xy_distance_m"))
    include_deviated_wells = bool(cfg.get("include_deviated_wells", False))
    inventory_positions = _load_well_positions(cfg)
    inventory_info = _load_well_inventory_info(cfg)
    line_geometry = _load_zero_shot_line_geometry(zero_shot_dir)
    section_x_m, section_y_m = _line_xy_arrays(line_geometry, section_ilines, section_xlines)
    bands = _configured_bands(run_cfg)
    dt_s = float(np.median(np.diff(twt_model))) if twt_model.size > 1 else 0.002
    rows: list[dict[str, object]] = []
    figure_outputs: dict[str, str] = {}
    figures = output_dir / "figures" / "wells"
    figures.mkdir(parents=True, exist_ok=True)
    for _, tie in ties.iterrows():
        well_name = str(tie.get("well_name", ""))
        well_inline = _coerce_float(tie.get("inline_float"))
        well_xline = _coerce_float(tie.get("xline_float"))
        well_info = inventory_info.get(well_name, {})
        wellbore_class = str(well_info.get("wellbore_class", ""))
        position_source = "well_tie_metrics"
        if (not np.isfinite(well_inline) or not np.isfinite(well_xline)) and well_name in inventory_positions:
            well_inline, well_xline = inventory_positions[well_name]
            position_source = "well_inventory"
        base = {
            "well_name": well_name,
            "wellbore_class": wellbore_class,
            "well_inline": well_inline if np.isfinite(well_inline) else float("nan"),
            "well_xline": well_xline if np.isfinite(well_xline) else float("nan"),
            "well_position_source": position_source,
            "tie_window_start_s": tie.get("tie_window_start_s", ""),
            "tie_window_end_s": tie.get("tie_window_end_s", ""),
        }
        if (not include_deviated_wells) and wellbore_class.casefold() == "deviated":
            rows.append(
                {
                    **base,
                    "model_role": "",
                    "status": "skipped_deviated_well_for_section_qc",
                    "include_deviated_wells": include_deviated_wells,
                }
            )
            continue
        if not np.isfinite(well_inline) or not np.isfinite(well_xline):
            rows.append({**base, "model_role": "", "status": "skipped_missing_well_position"})
            continue
        try:
            well_x_m, well_y_m = line_geometry.line_to_coord(float(well_inline), float(well_xline))
        except Exception as exc:
            rows.append({**base, "model_role": "", "status": "skipped_well_xy_projection_failed", "reason": str(exc)})
            continue
        trace_idx, distance = _trace_distance_to_section(
            well_inline=float(well_inline),
            well_xline=float(well_xline),
            well_x_m=float(well_x_m),
            well_y_m=float(well_y_m),
            section_ilines=section_ilines,
            section_xlines=section_xlines,
            section_x_m=section_x_m,
            section_y_m=section_y_m,
        )
        base.update(
            {
                "well_x_m": float(well_x_m),
                "well_y_m": float(well_y_m),
                "nearest_section_trace": trace_idx,
                "nearest_section_inline": float(section_ilines[trace_idx]),
                "nearest_section_xline": float(section_xlines[trace_idx]),
                "nearest_section_x_m": float(section_x_m[trace_idx]),
                "nearest_section_y_m": float(section_y_m[trace_idx]),
                "section_xy_distance_m": distance,
            }
        )
        if distance > max_xy_distance_m:
            rows.append(
                {
                    **base,
                    "model_role": "",
                    "status": "skipped_outside_section_support",
                    "max_xy_distance_m": max_xy_distance_m,
                }
            )
            continue
        try:
            well_log_ai = _log_ai_at_twt(
                las_path=resolve_relative_path(str(tie["filtered_las_file"]), root=REPO_ROOT),
                tdt_path=resolve_relative_path(str(tie["optimized_tdt_file"]), root=REPO_ROOT),
                twt_s=twt_model,
            )
        except Exception as exc:
            rows.append({**base, "model_role": "", "status": "skipped_well_log_projection_failed", "reason": str(exc)})
            continue
        first_synthetic_path = next(
            (synthetic_dir / f"zero_shot_{role}.npz" for role in predictions if (synthetic_dir / f"zero_shot_{role}.npz").is_file()),
            None,
        )
        if first_synthetic_path is not None:
            with np.load(first_synthetic_path, allow_pickle=True) as syn_arrays:
                observed = np.asarray(syn_arrays["observed_seismic_forward_axis"], dtype=np.float64)[trace_idx : trace_idx + 1]
                twt_forward = np.asarray(syn_arrays["twt_s_forward_axis"], dtype=np.float64)
                valid_forward = np.asarray(syn_arrays["valid_mask_forward_axis"], dtype=bool)[trace_idx : trace_idx + 1]
            filtered_synthetic = forward_log_ai(well_log_ai[None, :], wavelet)
            tie_start = pd.to_numeric(tie.get("tie_window_start_s"), errors="coerce")
            tie_end = pd.to_numeric(tie.get("tie_window_end_s"), errors="coerce")
            if np.isfinite(tie_start) and np.isfinite(tie_end):
                window = (twt_forward >= float(tie_start)) & (twt_forward <= float(tie_end))
                valid_forward = valid_forward & window[None, :]
            filtered_waveform = diagnostic_metrics(
                observed=observed,
                synthetic=filtered_synthetic,
                valid_mask=valid_forward,
            )
            rows.append(
                {
                    **base,
                    "model_role": "filtered_las",
                    "status": "ok",
                    "well_ai_status": "reference",
                    "well_ai_n_valid": int(np.count_nonzero(np.isfinite(well_log_ai))),
                    "well_ai_rmse": 0.0,
                    "well_ai_bias": 0.0,
                    "well_ai_corr": 1.0,
                    **{f"waveform_{key}": value for key, value in filtered_waveform.items()},
                }
            )
        lfm_log_ai = np.asarray(first_arrays["lfm_input"], dtype=np.float64)[trace_idx]
        lfm_valid = (
            np.asarray(first_arrays["valid_mask_model"], dtype=bool)[trace_idx]
            & np.isfinite(lfm_log_ai)
            & np.isfinite(well_log_ai)
        )
        lfm_ai_metrics = _well_ai_metrics(well_log_ai=well_log_ai, pred_log_ai=lfm_log_ai, valid=lfm_valid)
        lfm_band_metrics: dict[str, object] = {}
        for band in bands:
            name = str(band["name"])
            metrics = _band_pair_metrics(
                reference=well_log_ai,
                model=lfm_log_ai,
                valid=lfm_valid,
                dt_s=dt_s,
                low_hz=float(band["low_hz"]),
                high_hz=float(band["high_hz"]),
            )
            lfm_band_metrics[f"well_ai_{name}_n_valid"] = metrics["n_valid"]
            lfm_band_metrics[f"well_ai_{name}_rmse"] = metrics["rmse"]
            lfm_band_metrics[f"well_ai_{name}_corr"] = metrics["corr"]
        rows.append(
            {
                **base,
                "model_role": "lfm_input",
                "status": "ok" if lfm_ai_metrics.get("well_ai_status") == "ok" else lfm_ai_metrics.get("well_ai_status"),
                **lfm_ai_metrics,
                **lfm_band_metrics,
            }
        )
        for role, payload in predictions.items():
            arrays = payload["arrays"]
            pred = np.asarray(arrays["stitched_pred_log_ai"], dtype=np.float64)[trace_idx]
            valid = np.asarray(arrays["valid_mask_model"], dtype=bool)[trace_idx] & (np.asarray(arrays["stitching_weight"])[trace_idx] > 0.0)
            ai_metrics = _well_ai_metrics(well_log_ai=well_log_ai, pred_log_ai=pred, valid=valid)
            local_spectrum = {}
            residual = pred - well_log_ai
            pred_delta = np.asarray(arrays["pred_delta_vs_lfm"], dtype=np.float64)[trace_idx]
            for band in bands:
                name = str(band["name"])
                local_spectrum[f"well_ai_residual_{name}_rms"] = _band_rms(
                    residual,
                    valid,
                    dt_s=dt_s,
                    low_hz=float(band["low_hz"]),
                    high_hz=float(band["high_hz"]),
                )
                local_spectrum[f"pred_delta_{name}_rms"] = _band_rms(
                    pred_delta,
                    valid,
                    dt_s=dt_s,
                    low_hz=float(band["low_hz"]),
                    high_hz=float(band["high_hz"]),
                )
                pair_metrics = _band_pair_metrics(
                    reference=well_log_ai,
                    model=pred,
                    valid=valid,
                    dt_s=dt_s,
                    low_hz=float(band["low_hz"]),
                    high_hz=float(band["high_hz"]),
                )
                local_spectrum[f"well_ai_{name}_n_valid"] = pair_metrics["n_valid"]
                local_spectrum[f"well_ai_{name}_rmse"] = pair_metrics["rmse"]
                local_spectrum[f"well_ai_{name}_corr"] = pair_metrics["corr"]
            synthetic_path = synthetic_dir / f"zero_shot_{role}.npz"
            waveform_metrics: dict[str, object] = {"waveform_status": "synthetic_not_found"}
            if synthetic_path.is_file():
                syn_arrays = np.load(synthetic_path, allow_pickle=True)
                observed = np.asarray(syn_arrays["observed_seismic_forward_axis"], dtype=np.float64)[trace_idx : trace_idx + 1]
                synthetic = np.asarray(syn_arrays["synthetic_seismic"], dtype=np.float64)[trace_idx : trace_idx + 1]
                valid_display = np.asarray(syn_arrays["valid_mask_forward_axis"], dtype=bool)[trace_idx : trace_idx + 1]
                valid_forward = np.asarray(syn_arrays["valid_mask_forward_axis"], dtype=bool)[trace_idx : trace_idx + 1]
                twt_forward = np.asarray(syn_arrays["twt_s_forward_axis"], dtype=np.float64)
                tie_start = pd.to_numeric(tie.get("tie_window_start_s"), errors="coerce")
                tie_end = pd.to_numeric(tie.get("tie_window_end_s"), errors="coerce")
                if np.isfinite(tie_start) and np.isfinite(tie_end):
                    window = (twt_forward >= float(tie_start)) & (twt_forward <= float(tie_end))
                    valid_forward = valid_forward & window[None, :]
                    valid_display = valid_display & window[None, :]
                waveform = diagnostic_metrics(observed=observed, synthetic=synthetic, valid_mask=valid_forward)
                waveform_metrics = {f"waveform_{key}": value for key, value in waveform.items()}
            rows.append(
                {
                    **base,
                    "model_role": role,
                    "status": "ok" if ai_metrics.get("well_ai_status") == "ok" else ai_metrics.get("well_ai_status"),
                    **ai_metrics,
                    **local_spectrum,
                    **waveform_metrics,
                }
            )
        figure_outputs.update(
            _plot_well_qc(
                figures_dir=figures,
                well_name=well_name,
                twt_model=twt_model,
                well_log_ai=well_log_ai,
                trace_idx=trace_idx,
                predictions=predictions,
                synthetic_dir=synthetic_dir,
                tie_window_start_s=_coerce_float(tie.get("tie_window_start_s")),
                tie_window_end_s=_coerce_float(tie.get("tie_window_end_s")),
            )
        )
    return pd.DataFrame.from_records(rows), figure_outputs


def _plot_well_qc(
    *,
    figures_dir: Path,
    well_name: str,
    twt_model: np.ndarray,
    well_log_ai: np.ndarray,
    trace_idx: int,
    predictions: dict[str, dict[str, object]],
    synthetic_dir: Path,
    tie_window_start_s: float,
    tie_window_end_s: float,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    twt_model = np.asarray(twt_model, dtype=np.float64)
    well_log_ai = np.asarray(well_log_ai, dtype=np.float64)
    if twt_model.size < 2:
        return outputs
    twt_forward_model = twt_model[1:]
    for role, payload in predictions.items():
        arrays = payload["arrays"]
        synthetic_path = synthetic_dir / f"zero_shot_{role}.npz"
        if not synthetic_path.is_file():
            continue
        pred_log_ai = np.asarray(arrays["stitched_pred_log_ai"], dtype=np.float64)[trace_idx]
        pred_filled = _fill_nonfinite_1d(pred_log_ai)
        reflectivity = np.tanh(0.5 * (pred_filled[1:] - pred_filled[:-1]))
        with np.load(synthetic_path, allow_pickle=True) as syn_arrays:
            twt_forward = np.asarray(syn_arrays["twt_s_forward_axis"], dtype=np.float64)
            observed = np.asarray(syn_arrays["observed_seismic_forward_axis"], dtype=np.float64)[trace_idx]
            synthetic = np.asarray(syn_arrays["synthetic_seismic"], dtype=np.float64)[trace_idx]
            display_mask = np.asarray(syn_arrays["valid_mask_forward_axis"], dtype=bool)[trace_idx]
        if twt_forward.shape != observed.shape or twt_forward.shape != synthetic.shape:
            continue
        if reflectivity.shape != twt_forward.shape:
            if reflectivity.shape == twt_forward_model.shape and np.allclose(twt_forward, twt_forward_model, equal_nan=True):
                pass
            else:
                continue
        plot_mask = display_mask & np.isfinite(observed) & np.isfinite(synthetic) & np.isfinite(reflectivity)
        if np.isfinite(tie_window_start_s) and np.isfinite(tie_window_end_s):
            plot_mask &= (twt_forward >= float(tie_window_start_s)) & (twt_forward <= float(tie_window_end_s))
        run = _largest_true_run(plot_mask)
        if run is None:
            continue
        start, stop = run
        if stop - start < 8:
            continue
        sl = slice(start, stop)
        basis = twt_forward[sl]
        pred_ai = np.exp(pred_filled[1:][sl])
        filtered_ai = np.exp(well_log_ai[1:][sl])
        synthetic_trace = grid.Seismic(synthetic[sl], basis, "twt", name="Synthetic")
        observed_trace = grid.Seismic(observed[sl], basis, "twt", name="Seismic")
        reflectivity_trace = grid.Reflectivity(reflectivity[sl], basis, "twt", name="Reflectivity")
        pred_ai_trace = grid.Log(pred_ai, basis, "twt", name=f"{role} AI")
        filtered_ai_trace = grid.Log(filtered_ai, basis, "twt", name="Filtered LAS AI")
        xcorr_values = normalized_xcorr(observed_trace.values, synthetic_trace.values)
        xcorr_basis = synthetic_trace.sampling_rate * np.arange(-(synthetic_trace.size - 1), synthetic_trace.size)
        xcorr = grid.XCorr(xcorr_values, xcorr_basis, "tlag", name="XCorr")
        dxcorr = dynamic_normalized_xcorr(observed_trace, synthetic_trace)
        corr = _safe_corr(observed_trace.values, synthetic_trace.values)
        title = f"R1 well QC | {well_name} | {role} | corr={corr:.3f}"
        fig, _axes = plot_well_waveform_qc(
            [pred_ai_trace, filtered_ai_trace],
            reflectivity_trace,
            synthetic_trace,
            observed_trace,
            xcorr,
            dxcorr,
            figsize=(12.0, 7.5),
            synthetic_ai=pred_ai_trace,
            title=title,
        )
        path = figures_dir / f"well_forward_qc_{well_name}_{role}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs[f"{well_name}/{role}"] = repo_relative_path(path, root=REPO_ROOT)
    return outputs


def _fill_nonfinite_1d(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if np.all(finite):
        return out
    if not np.any(finite):
        return np.zeros_like(out)
    x = np.arange(out.size, dtype=np.float64)
    out[~finite] = np.interp(x[~finite], x[finite], out[finite])
    return out


def _largest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    best_start = -1
    best_stop = -1
    current_start: int | None = None
    for index, flag in enumerate(values):
        if flag and current_start is None:
            current_start = index
        elif not flag and current_start is not None:
            if index - current_start > best_stop - best_start:
                best_start, best_stop = current_start, index
            current_start = None
    if current_start is not None and values.size - current_start > best_stop - best_start:
        best_start, best_stop = current_start, values.size
    if best_start < 0:
        return None
    return best_start, best_stop


def _plot_forward_qc(
    *,
    output_dir: Path,
    model_role: str,
    observed: np.ndarray,
    synthetic: np.ndarray,
    display_valid: np.ndarray,
    diagnostic_valid: np.ndarray,
) -> str:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    display_mask = np.asarray(display_valid, dtype=bool)
    residual = np.where(display_mask, observed - synthetic, np.nan)
    panels = [
        ("observed", np.where(display_mask, observed, np.nan), "seismic"),
        ("synthetic", np.where(display_mask, synthetic, np.nan), "seismic"),
        ("residual", residual, "residual"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (_, values, title) in zip(axes, panels):
        finite = values[np.isfinite(values)]
        vmax = float(np.quantile(np.abs(finite), 0.99)) if finite.size else 1.0
        image = ax.imshow(values.T, aspect="auto", origin="upper", cmap="seismic", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("lateral trace")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("forward TWT sample")
    fig.suptitle(f"R1 forward QC: {model_role}")
    fig.tight_layout()
    path = figures / f"{model_role}_observed_synthetic_residual.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return repo_relative_path(path, root=REPO_ROOT)


def _plot_scan_qc(output_dir: Path, decomposition: pd.DataFrame) -> str:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "phase_shift_gain_scan.png"
    if decomposition.empty:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    phase_frame = decomposition[
        decomposition["scan_type"].eq("phase") & np.isfinite(decomposition["residual_rms_scaled"])
    ]
    for role, group in phase_frame.groupby("model_role"):
        axes[0].plot(group["phase_deg"], group["residual_rms_scaled"], marker="o", label=role)
    axes[0].set_title("Phase scan")
    axes[0].set_xlabel("phase deg")
    axes[0].set_ylabel("scaled residual RMS")
    if phase_frame.empty:
        axes[0].text(0.5, 0.5, "no valid positive-scale phase rows", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].legend(fontsize=8)
    shift_frame = decomposition[
        decomposition["scan_type"].eq("fractional_shift") & np.isfinite(decomposition["residual_rms_scaled"])
    ]
    for role, group in shift_frame.groupby("model_role"):
        axes[1].plot(group["fractional_shift_samples"], group["residual_rms_scaled"], marker="o", label=role)
    axes[1].set_title("Fractional shift scan")
    axes[1].set_xlabel("shift samples")
    if shift_frame.empty:
        axes[1].text(0.5, 0.5, "no valid positive-scale shift rows", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return repo_relative_path(path, root=REPO_ROOT)


def _plot_spatial_qc(output_dir: Path, spatial: pd.DataFrame) -> str:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / "spatial_residual_qc.png"
    if spatial.empty:
        return ""
    fig, ax = plt.subplots(figsize=(8, 4))
    finite = spatial[np.isfinite(spatial["residual_rms_scaled"])]
    for role, group in finite.groupby("model_role"):
        ax.plot(group["lateral_index"], group["residual_rms_scaled"], label=role)
    ax.set_xlabel("lateral trace")
    ax.set_ylabel("scaled residual RMS")
    ax.set_title("Spatial residual pattern")
    if finite.empty:
        ax.text(0.5, 0.5, "no valid positive-scale spatial rows", ha="center", va="center", transform=ax.transAxes)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return repo_relative_path(path, root=REPO_ROOT)


def _write_forward_band_residual_qc(
    output_dir: Path,
    *,
    run_cfg: dict,
    dt_s: float,
    observed_forward: np.ndarray,
    synthetic_by_role: dict[str, np.ndarray],
    valid_forward: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, str]]:
    bands = _configured_bands(run_cfg)
    rows: list[dict[str, object]] = []
    for role, synthetic in synthetic_by_role.items():
        residual = observed_forward - synthetic
        full_residual = _band_rms_2d(residual, valid_forward, dt_s=dt_s, low_hz=0.0, high_hz=float("inf"))
        full_observed = _band_rms_2d(observed_forward, valid_forward, dt_s=dt_s, low_hz=0.0, high_hz=float("inf"))
        full_synthetic = _band_rms_2d(synthetic, valid_forward, dt_s=dt_s, low_hz=0.0, high_hz=float("inf"))
        for band in bands:
            low = float(band["low_hz"])
            high = float(band["high_hz"])
            obs_rms = _band_rms_2d(observed_forward, valid_forward, dt_s=dt_s, low_hz=low, high_hz=high)
            syn_rms = _band_rms_2d(synthetic, valid_forward, dt_s=dt_s, low_hz=low, high_hz=high)
            res_rms = _band_rms_2d(residual, valid_forward, dt_s=dt_s, low_hz=low, high_hz=high)
            rows.append(
                {
                    "model_role": role,
                    "band": band["name"],
                    "low_hz": low,
                    "high_hz": high,
                    "observed_band_rms": obs_rms,
                    "synthetic_band_rms": syn_rms,
                    "residual_band_rms": res_rms,
                    "residual_to_observed_ratio": res_rms / obs_rms if np.isfinite(obs_rms) and obs_rms > 0 else float("nan"),
                    "band_residual_energy_fraction_of_full": (
                        res_rms / full_residual if np.isfinite(full_residual) and full_residual > 0 else float("nan")
                    ),
                    "fullband_observed_rms": full_observed,
                    "fullband_synthetic_rms": full_synthetic,
                    "fullband_residual_rms": full_residual,
                }
            )
    frame = pd.DataFrame.from_records(rows)
    path = output_dir / "forward_band_residual_qc.csv"
    frame.to_csv(path, index=False)
    figures: dict[str, str] = {}
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "forward_band_residual_qc.png"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for role, group in frame.groupby("model_role"):
        ax.plot(group["band"], group["residual_to_observed_ratio"], marker="o", label=role)
    ax.set_ylabel("residual band RMS / observed band RMS")
    ax.set_title("Forward residual by seismic band")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    figures["forward_band_residual_qc"] = repo_relative_path(fig_path, root=REPO_ROOT)
    return frame, figures


def _write_ai_plausibility_qc(
    output_dir: Path,
    *,
    run_cfg: dict,
    dt_s: float,
    predictions: dict[str, dict[str, object]],
    lfm: np.ndarray,
    valid: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, str]]:
    bands = _configured_bands(run_cfg)
    rows: list[dict[str, object]] = []
    valid = np.asarray(valid, dtype=bool)
    lfm = np.asarray(lfm, dtype=np.float64)
    rows.append({"model_role": "lfm_only", "signal": "lfm_log_ai", **_finite_stats(lfm, valid)})
    for role, payload in predictions.items():
        arrays = payload["arrays"]
        pred = np.asarray(arrays["stitched_pred_log_ai"], dtype=np.float64)
        delta = np.asarray(arrays["pred_delta_vs_lfm"], dtype=np.float64)
        role_valid = valid & np.asarray(arrays["valid_mask_model"], dtype=bool) & (np.asarray(arrays["stitching_weight"]) > 0.0)
        normalization = _normalization_for_role(predictions, role)
        delta_mean, delta_std = _normalization_stats(normalization, "delta")
        lfm_mean, lfm_std = _normalization_stats(normalization, "lfm")
        rows.append(
            {
                "model_role": role,
                "signal": "pred_log_ai",
                **_finite_stats(pred, role_valid),
                "synthetic_train_mean": float("nan"),
                "synthetic_train_std": float("nan"),
            }
        )
        delta_row = {
            "model_role": role,
            "signal": "pred_delta_vs_lfm",
            **_finite_stats(delta, role_valid),
            "synthetic_train_mean": delta_mean,
            "synthetic_train_std": delta_std,
        }
        delta_row["real_to_synthetic_std_ratio"] = (
            delta_row["std"] / delta_std if np.isfinite(delta_std) and delta_std > 0 else float("nan")
        )
        rows.append(delta_row)
        lfm_row = {
            "model_role": role,
            "signal": "lfm_input_for_role",
            **_finite_stats(lfm, role_valid),
            "synthetic_train_mean": lfm_mean,
            "synthetic_train_std": lfm_std,
        }
        lfm_row["real_to_synthetic_std_ratio"] = (
            lfm_row["std"] / lfm_std if np.isfinite(lfm_std) and lfm_std > 0 else float("nan")
        )
        rows.append(lfm_row)
        full_delta = _band_rms_2d(delta, role_valid, dt_s=dt_s, low_hz=0.0, high_hz=float("inf"))
        full_pred = _band_rms_2d(pred, role_valid, dt_s=dt_s, low_hz=0.0, high_hz=float("inf"))
        for band in bands:
            low = float(band["low_hz"])
            high = float(band["high_hz"])
            delta_band = _band_rms_2d(delta, role_valid, dt_s=dt_s, low_hz=low, high_hz=high)
            pred_band = _band_rms_2d(pred, role_valid, dt_s=dt_s, low_hz=low, high_hz=high)
            rows.append(
                {
                    "model_role": role,
                    "signal": "pred_delta_vs_lfm_band",
                    "band": band["name"],
                    "low_hz": low,
                    "high_hz": high,
                    "band_rms": delta_band,
                    "fullband_rms": full_delta,
                    "energy_ratio": delta_band / full_delta if np.isfinite(full_delta) and full_delta > 0 else float("nan"),
                    "synthetic_train_std": delta_std,
                    "band_rms_to_synthetic_delta_std": (
                        delta_band / delta_std if np.isfinite(delta_std) and delta_std > 0 else float("nan")
                    ),
                }
            )
            rows.append(
                {
                    "model_role": role,
                    "signal": "pred_log_ai_band",
                    "band": band["name"],
                    "low_hz": low,
                    "high_hz": high,
                    "band_rms": pred_band,
                    "fullband_rms": full_pred,
                    "energy_ratio": pred_band / full_pred if np.isfinite(full_pred) and full_pred > 0 else float("nan"),
                }
            )
    if "lateral" in predictions and "no_lateral" in predictions:
        lat = predictions["lateral"]["arrays"]
        no = predictions["no_lateral"]["arrays"]
        diff = np.asarray(lat["stitched_pred_log_ai"], dtype=np.float64) - np.asarray(no["stitched_pred_log_ai"], dtype=np.float64)
        diff_valid = (
            valid
            & np.asarray(lat["valid_mask_model"], dtype=bool)
            & np.asarray(no["valid_mask_model"], dtype=bool)
            & (np.asarray(lat["stitching_weight"]) > 0.0)
            & (np.asarray(no["stitching_weight"]) > 0.0)
        )
        full = _band_rms_2d(diff, diff_valid, dt_s=dt_s, low_hz=0.0, high_hz=float("inf"))
        rows.append({"model_role": "lateral_minus_no_lateral", "signal": "pred_log_ai_difference", **_finite_stats(diff, diff_valid)})
        for band in bands:
            band_rms = _band_rms_2d(
                diff,
                diff_valid,
                dt_s=dt_s,
                low_hz=float(band["low_hz"]),
                high_hz=float(band["high_hz"]),
            )
            rows.append(
                {
                    "model_role": "lateral_minus_no_lateral",
                    "signal": "pred_log_ai_difference_band",
                    "band": band["name"],
                    "low_hz": band["low_hz"],
                    "high_hz": band["high_hz"],
                    "band_rms": band_rms,
                    "fullband_rms": full,
                    "energy_ratio": band_rms / full if np.isfinite(full) and full > 0 else float("nan"),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    path = output_dir / "ai_plausibility_qc.csv"
    frame.to_csv(path, index=False)
    figures: dict[str, str] = {}
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    band_frame = frame[frame["signal"].astype(str).str.endswith("_band", na=False)].copy()
    fig_path = fig_dir / "ai_band_energy_qc.png"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (role, signal), group in band_frame.groupby(["model_role", "signal"]):
        if str(signal) != "pred_delta_vs_lfm_band":
            continue
        ax.plot(group["band"], group["energy_ratio"], marker="o", label=str(role))
    ax.set_ylabel("band RMS / fullband RMS")
    ax.set_title("AI pred_delta_vs_lfm band energy")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    figures["ai_band_energy_qc"] = repo_relative_path(fig_path, root=REPO_ROOT)
    return frame, figures


def _first_row(frame: pd.DataFrame, *, well_name: str, role: str) -> pd.Series | None:
    subset = frame[
        frame["well_name"].astype(str).eq(str(well_name))
        & frame["model_role"].astype(str).eq(str(role))
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def _num(row: pd.Series | None, key: str) -> float:
    if row is None or key not in row:
        return float("nan")
    return float(pd.to_numeric(row.get(key), errors="coerce"))


def _text(row: pd.Series | None, key: str) -> str:
    if row is None or key not in row or pd.isna(row.get(key)):
        return ""
    return str(row.get(key))


def _classify_well_model(
    *,
    lfm_rmse: float,
    model_rmse: float,
    lfm_corr: float,
    model_corr: float,
    model_waveform_corr: float,
    filtered_waveform_status: str,
    filtered_scale_status: str,
) -> tuple[str, str]:
    filtered_weak = filtered_waveform_status != "ok" or (
        filtered_scale_status and filtered_scale_status != "ok"
    )
    waveform_high = np.isfinite(model_waveform_corr) and model_waveform_corr >= 0.9
    rmse_improved = np.isfinite(lfm_rmse) and np.isfinite(model_rmse) and model_rmse < lfm_rmse
    corr_improved = np.isfinite(lfm_corr) and np.isfinite(model_corr) and model_corr > lfm_corr
    if filtered_weak:
        return "filtered_las_weak_reference", "filtered LAS synthetic is invalid or requires non-positive scale"
    if rmse_improved and corr_improved:
        return "model_improves_ai", "model improves both RMSE and correlation relative to LFM"
    if (not rmse_improved) and corr_improved:
        return "shape_improves_bias_worse", "model improves correlation but worsens/fullband RMSE relative to LFM"
    if (not rmse_improved) and (not corr_improved) and waveform_high:
        return "waveform_good_ai_worse", "model waveform is strong but AI is worse than LFM by RMSE/correlation"
    if rmse_improved and (not corr_improved):
        return "bias_improves_shape_worse", "model improves RMSE but not correlation relative to LFM"
    return "mixed_or_insufficient", "comparison is mixed or lacks finite metrics"


def _write_well_closure_tables(
    output_dir: Path,
    *,
    well_frame: pd.DataFrame,
    run_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    bands = _configured_bands(run_cfg)
    comparison_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    roles = ["lateral", "no_lateral"]
    wells = sorted(str(value) for value in well_frame["well_name"].dropna().unique())
    for well_name in wells:
        lfm = _first_row(well_frame, well_name=well_name, role="lfm_input")
        filtered = _first_row(well_frame, well_name=well_name, role="filtered_las")
        if lfm is None:
            continue
        for role in ["lfm_input", *roles]:
            row = _first_row(well_frame, well_name=well_name, role=role)
            if row is None:
                continue
            for band in bands:
                name = str(band["name"])
                band_rows.append(
                    {
                        "well_name": well_name,
                        "model_role": role,
                        "band": name,
                        "low_hz": float(band["low_hz"]),
                        "high_hz": float(band["high_hz"]),
                        "well_ai_band_n_valid": _num(row, f"well_ai_{name}_n_valid"),
                        "well_ai_band_rmse": _num(row, f"well_ai_{name}_rmse"),
                        "well_ai_band_corr": _num(row, f"well_ai_{name}_corr"),
                    }
                )
        for role in roles:
            model = _first_row(well_frame, well_name=well_name, role=role)
            if model is None:
                continue
            lfm_rmse = _num(lfm, "well_ai_rmse")
            model_rmse = _num(model, "well_ai_rmse")
            lfm_corr = _num(lfm, "well_ai_corr")
            model_corr = _num(model, "well_ai_corr")
            waveform_corr = _num(model, "waveform_residual_corr_scaled")
            filtered_waveform_status = _text(filtered, "waveform_status")
            filtered_scale_status = _text(filtered, "waveform_scale_status")
            classification, explanation = _classify_well_model(
                lfm_rmse=lfm_rmse,
                model_rmse=model_rmse,
                lfm_corr=lfm_corr,
                model_corr=model_corr,
                model_waveform_corr=waveform_corr,
                filtered_waveform_status=filtered_waveform_status,
                filtered_scale_status=filtered_scale_status,
            )
            comparison_rows.append(
                {
                    "well_name": well_name,
                    "model_role": role,
                    "lfm_rmse": lfm_rmse,
                    "model_rmse": model_rmse,
                    "rmse_delta_model_minus_lfm": model_rmse - lfm_rmse if np.isfinite(model_rmse) and np.isfinite(lfm_rmse) else float("nan"),
                    "lfm_corr": lfm_corr,
                    "model_corr": model_corr,
                    "corr_delta_model_minus_lfm": model_corr - lfm_corr if np.isfinite(model_corr) and np.isfinite(lfm_corr) else float("nan"),
                    "model_waveform_corr_scaled": waveform_corr,
                    "model_waveform_residual_rms_scaled": _num(model, "waveform_residual_rms_scaled"),
                    "filtered_las_waveform_status": filtered_waveform_status,
                    "filtered_las_scale_status": filtered_scale_status,
                    "filtered_las_waveform_corr_scaled": _num(filtered, "waveform_residual_corr_scaled"),
                    "classification": classification,
                    "classification_explanation": explanation,
                }
            )
    comparison = pd.DataFrame.from_records(comparison_rows)
    bands_frame = pd.DataFrame.from_records(band_rows)
    comparison_path = output_dir / "well_ai_comparison_summary.csv"
    bands_path = output_dir / "well_ai_band_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    bands_frame.to_csv(bands_path, index=False)
    figures: dict[str, str] = {}
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not comparison.empty:
        fig_path = fig_dir / "well_ai_comparison_summary.png"
        counts = comparison.groupby(["model_role", "classification"]).size().reset_index(name="count")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        pivot = counts.pivot(index="classification", columns="model_role", values="count").fillna(0)
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("well-model count")
        ax.set_title("Well AI closure classification")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
        figures["well_ai_comparison_summary"] = repo_relative_path(fig_path, root=REPO_ROOT)
    if not bands_frame.empty:
        fig_path = fig_dir / "well_ai_band_comparison.png"
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for role, group in bands_frame.groupby("model_role"):
            agg = group.groupby("band")["well_ai_band_rmse"].median().reset_index()
            ax.plot(agg["band"], agg["well_ai_band_rmse"], marker="o", label=role)
        ax.set_ylabel("median well-side band RMSE")
        ax.set_title("Well AI band comparison vs filtered LAS")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
        figures["well_ai_band_comparison"] = repo_relative_path(fig_path, root=REPO_ROOT)
    return comparison, bands_frame, figures


def _red_flags(
    *,
    zero_shot_dir: Path,
    metrics: pd.DataFrame,
    run_cfg: dict,
) -> list[dict[str, object]]:
    thresholds = dict(run_cfg.get("red_flag_thresholds") or {})
    seismic_ood_gt5 = float(thresholds.get("seismic_ood_fraction_abs_gt5", 0.05))
    nullspace_ratio_threshold = float(thresholds.get("lateral_nullspace_energy_ratio", 0.5))
    flags: list[dict[str, object]] = []
    input_qc_path = zero_shot_dir / "model_input_qc.csv"
    if input_qc_path.is_file():
        qc = pd.read_csv(input_qc_path)
        seismic = qc[qc["input"].astype(str).eq("seismic")]
        if not seismic.empty:
            value = float(pd.to_numeric(seismic.iloc[0].get("fraction_abs_normalized_gt_5"), errors="coerce"))
            if np.isfinite(value) and value > seismic_ood_gt5:
                flags.append(
                    {
                        "flag": "real_input_seismic_ood",
                        "severity": "red",
                        "value": value,
                        "threshold": seismic_ood_gt5,
                        "meaning": "Real seismic normalized by synthetic train statistics is far outside the training domain.",
                    }
                )
    for _, row in metrics.iterrows():
        status = str(row.get("status", ""))
        if status != "ok":
            flags.append(
                {
                    "flag": "forward_metric_status_not_ok",
                    "severity": "red",
                    "model_role": row.get("model_role", ""),
                    "status": status,
                }
            )
        scale_status = str(row.get("scale_status", ""))
        if scale_status and scale_status != "ok":
            flags.append(
                {
                    "flag": "scale_status_not_ok",
                    "severity": "red",
                    "model_role": row.get("model_role", ""),
                    "scale_status": scale_status,
                }
            )
    band_path = zero_shot_dir / "lateral_difference_band_qc.csv"
    if band_path.is_file():
        band = pd.read_csv(band_path)
        null = band[band["band"].astype(str).eq("highfreq_or_nullspace")]
        if not null.empty:
            ratio = float(pd.to_numeric(null.iloc[0].get("energy_ratio"), errors="coerce"))
            if np.isfinite(ratio) and ratio > nullspace_ratio_threshold:
                flags.append(
                    {
                        "flag": "lateral_difference_concentrated_in_nullspace",
                        "severity": "red",
                        "value": ratio,
                        "threshold": nullspace_ratio_threshold,
                    }
                )
    return flags


def main() -> None:
    args = parse_args()
    cfg_path = resolve_relative_path(args.config, root=REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    run_cfg = dict(cfg.get("real_field_forward_diagnostic") or {})
    if not run_cfg:
        raise ValueError("experiments/common.yaml lacks real_field_forward_diagnostic section.")
    output_root = resolve_relative_path(cfg.get("output_root", "scripts/output"), root=REPO_ROOT)
    zero_shot_dir = (
        resolve_relative_path(args.zero_shot_dir, root=REPO_ROOT)
        if args.zero_shot_dir is not None
        else resolve_relative_path(run_cfg.get("zero_shot_dir"), root=REPO_ROOT)
    )
    output_dir = _timestamped_output("real_field_forward_diagnostic", args.output_dir, output_root=output_root)
    output_dir.mkdir(parents=True, exist_ok=False)

    predictions = load_zero_shot_predictions(zero_shot_dir)
    first = next(iter(predictions.values()))["arrays"]
    observed = np.asarray(first["seismic_input"], dtype=np.float64)
    lfm = np.asarray(first["lfm_input"], dtype=np.float64)
    valid = np.asarray(first["valid_mask_model"], dtype=bool)
    twt_s = np.asarray(first["twt_s"], dtype=np.float64)
    dt_s = float(np.median(np.diff(twt_s))) if twt_s.size > 1 else 0.002

    source_runs = dict(run_cfg.get("source_runs") or {})
    wavelet_dir_text = source_runs.get("wavelet_generation_dir")
    if not wavelet_dir_text:
        raise ValueError("R1 requires explicit source_runs.wavelet_generation_dir.")
    wavelet_dir = resolve_relative_path(wavelet_dir_text, root=REPO_ROOT)
    wavelet, wavelet_meta = load_selected_wavelet(wavelet_dir)
    wavelet_scenarios = _load_wavelet_scenarios(wavelet_dir, wavelet, wavelet_meta, run_cfg)

    scan_cfg = dict(run_cfg.get("diagnostic_scan") or {})
    phase_values = [float(v) for v in scan_cfg.get("phase_deg", [-20, -10, 0, 10, 20])]
    shift_values = [float(v) for v in scan_cfg.get("fractional_shift_samples", [-1.0, -0.5, 0.0, 0.5, 1.0])]
    boundary = dict(run_cfg.get("boundary") or {})
    if boundary.get("forward_diagnostic_crop_s") is not None:
        raise ValueError(
            "forward_diagnostic_crop_s is retired; R1 diagnostics now use the full valid mask."
        )
    if boundary.get("forward_diagnostic_erosion_s") not in (None, 0, 0.0):
        raise ValueError("forward_diagnostic_erosion_s is disabled in R1; remove it or set it to 0.")

    impedance_inputs: list[tuple[str, np.ndarray, str]] = [("lfm_only", lfm, "lfm_input")]
    for role, payload in predictions.items():
        arrays = payload["arrays"]
        impedance_inputs.append((f"zero_shot_{role}", np.asarray(arrays["stitched_pred_log_ai"], dtype=np.float64), role))

    metric_rows = []
    wavelet_rows = []
    scan_frames = []
    spatial_rows = []
    figure_outputs: dict[str, str] = {}
    synthetic_by_role: dict[str, np.ndarray] = {}
    synthetic_dir = output_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=False)
    for model_role, log_ai, source_role in impedance_inputs:
        synthetic = forward_log_ai(log_ai, wavelet)
        synthetic_by_role[model_role] = synthetic
        obs_crop, syn_crop, valid_crop = _align_forward_arrays(
            observed,
            synthetic,
            valid,
        )
        metrics = diagnostic_metrics(observed=obs_crop, synthetic=syn_crop, valid_mask=valid_crop)
        metric_rows.append(
            {
                "model_role": model_role,
                "source_role": source_role,
                "forward_operator_id": "real_field_log_ai_tanh_reflectivity_nominal_wavelet_v1",
                "reflectivity_hang_point": "r[j]=tanh((x[j]-x[j-1])/2), aligned to observed[:, 1:]",
                "diagnostic_window": "full_valid_mask",
                **metrics,
            }
        )
        figure_outputs[model_role] = _plot_forward_qc(
            output_dir=output_dir,
            model_role=model_role,
            observed=observed[:, 1:],
            synthetic=synthetic,
            display_valid=valid[:, 1:],
            diagnostic_valid=valid_crop,
        )
        np.savez_compressed(
            synthetic_dir / f"{model_role}.npz",
            synthetic_seismic=synthetic.astype(np.float32),
            observed_seismic_forward_axis=observed[:, 1:].astype(np.float32),
            valid_mask_forward_axis=valid[:, 1:].astype(bool),
            twt_s_forward_axis=twt_s[1:],
        )
        scan = phase_shift_scan(
            observed=obs_crop,
            synthetic=syn_crop,
            valid_mask=valid_crop,
            phase_degrees=phase_values,
            fractional_shift_samples=shift_values,
        )
        scan.insert(0, "model_role", model_role)
        scan_frames.append(scan)
        spatial_rows.extend(_spatial_rows(model_role=model_role, observed=obs_crop, synthetic=syn_crop, valid=valid_crop))
        for scenario in wavelet_scenarios:
            scenario_wavelet = np.asarray(scenario["wavelet"], dtype=np.float64)
            scenario_synthetic = forward_log_ai(log_ai, scenario_wavelet)
            obs_wave, syn_wave, valid_wave = _align_forward_arrays(
                observed,
                scenario_synthetic,
                valid,
            )
            scenario_metrics = diagnostic_metrics(observed=obs_wave, synthetic=syn_wave, valid_mask=valid_wave)
            wavelet_rows.append(
                {
                    "model_role": model_role,
                    "source_role": source_role,
                    "wavelet_scenario_id": scenario["wavelet_scenario_id"],
                    "source_well": scenario["source_well"],
                    "candidate_score": scenario["candidate_score"],
                    "wavelet_path": scenario["wavelet_path"],
                    "wavelet_sha256": scenario["wavelet_sha256"],
                    "diagnostic_window": "full_valid_mask",
                    **scenario_metrics,
                }
            )

    common_forward_valid = valid[:, 1:].astype(bool)
    forward_band_frame, forward_band_figures = _write_forward_band_residual_qc(
        output_dir,
        run_cfg=run_cfg,
        dt_s=dt_s,
        observed_forward=observed[:, 1:],
        synthetic_by_role=synthetic_by_role,
        valid_forward=common_forward_valid,
    )
    ai_plausibility_frame, ai_plausibility_figures = _write_ai_plausibility_qc(
        output_dir,
        run_cfg=run_cfg,
        dt_s=dt_s,
        predictions=predictions,
        lfm=lfm,
        valid=valid,
    )
    figure_outputs.update(forward_band_figures)
    figure_outputs.update(ai_plausibility_figures)

    metrics_path = output_dir / "forward_diagnostic_metrics.csv"
    metrics_frame = pd.DataFrame.from_records(metric_rows)
    metrics_frame.to_csv(metrics_path, index=False)
    decomposition_path = output_dir / "residual_decomposition.csv"
    decomposition_frame = pd.concat(scan_frames, ignore_index=True)
    decomposition_frame.to_csv(decomposition_path, index=False)
    wavelet_sensitivity_path = output_dir / "wavelet_sensitivity.csv"
    pd.DataFrame.from_records(wavelet_rows).to_csv(wavelet_sensitivity_path, index=False)
    spatial_path = output_dir / "spatial_residual_qc.csv"
    spatial_frame = pd.DataFrame.from_records(spatial_rows)
    spatial_frame.to_csv(spatial_path, index=False)
    well_path = output_dir / "well_forward_diagnostic.csv"
    well_frame, well_figure_outputs = _build_well_forward_qc(
        run_cfg=run_cfg,
        zero_shot_dir=zero_shot_dir,
        output_dir=output_dir,
        predictions=predictions,
        synthetic_dir=synthetic_dir,
        wavelet=wavelet,
    )
    well_frame.to_csv(well_path, index=False)
    well_comparison_frame, well_band_frame, well_closure_figures = _write_well_closure_tables(
        output_dir,
        well_frame=well_frame,
        run_cfg=run_cfg,
    )
    figure_outputs.update(well_closure_figures)
    figure_outputs["phase_shift_gain_scan"] = _plot_scan_qc(output_dir, decomposition_frame)
    figure_outputs["spatial_residual_qc"] = _plot_spatial_qc(output_dir, spatial_frame)
    if well_figure_outputs:
        figure_outputs["well_forward_qc"] = well_figure_outputs
    red_flags = _red_flags(zero_shot_dir=zero_shot_dir, metrics=metrics_frame, run_cfg=run_cfg)
    recommended_next_state = (
        "return_to_input_preparation_or_synthetic_diagnostic"
        if red_flags
        else "r2_calibration_only_candidate"
    )

    summary = {
        "schema_version": "real_field_forward_diagnostic_summary_v1",
        "status": "ok",
        "config_file": repo_relative_path(cfg_path, root=REPO_ROOT),
        "zero_shot_dir": repo_relative_path(zero_shot_dir, root=REPO_ROOT),
        "wavelet": wavelet_meta,
        "forward_contract": {
            "reflectivity": "r[j] = tanh((logAI[j] - logAI[j-1]) / 2)",
            "convolution": "numpy.convolve(trace_reflectivity, wavelet, mode='same')",
            "observed_alignment": "observed[:, 1:]",
            "time_alignment_mode": "drop_first_observed_sample_to_match_N_minus_1_reflectivity_axis",
            "discarded_samples_for_forward_axis": 1,
            "synthetic_twt_axis": "twt_s[1:] on full valid mask",
            "observed_twt_axis_after_alignment": "twt_s[1:] on full valid mask",
            "dt_s": dt_s,
            "diagnostic_window": "full_valid_mask",
            "forward_diagnostic_erosion": "disabled",
            "phase_scan_deg": phase_values,
            "fractional_shift_scan_samples": shift_values,
            "wavelet_scenario_count": len(wavelet_scenarios),
        },
        "outputs": {
            "forward_diagnostic_metrics": repo_relative_path(metrics_path, root=REPO_ROOT),
            "well_forward_diagnostic": repo_relative_path(well_path, root=REPO_ROOT),
            "residual_decomposition": repo_relative_path(decomposition_path, root=REPO_ROOT),
            "wavelet_sensitivity": repo_relative_path(wavelet_sensitivity_path, root=REPO_ROOT),
            "spatial_residual_qc": repo_relative_path(spatial_path, root=REPO_ROOT),
            "forward_band_residual_qc": repo_relative_path(output_dir / "forward_band_residual_qc.csv", root=REPO_ROOT),
            "ai_plausibility_qc": repo_relative_path(output_dir / "ai_plausibility_qc.csv", root=REPO_ROOT),
            "well_ai_comparison_summary": repo_relative_path(output_dir / "well_ai_comparison_summary.csv", root=REPO_ROOT),
            "well_ai_band_comparison": repo_relative_path(output_dir / "well_ai_band_comparison.csv", root=REPO_ROOT),
            "synthetic_dir": repo_relative_path(synthetic_dir, root=REPO_ROOT),
            "figures": figure_outputs,
        },
        "red_flags": red_flags,
        "recommended_next_state": recommended_next_state,
        "wavelet_sha256": wavelet_meta["wavelet_sha256"],
        "zero_shot_summary_sha256": (
            sha256_file(zero_shot_dir / "real_field_zero_shot_summary.json")
            if (zero_shot_dir / "real_field_zero_shot_summary.json").is_file()
            else ""
        ),
        "code_version_or_git_commit": str(run_cfg.get("code_version_or_git_commit") or _git_commit()),
    }
    write_json(output_dir / "real_field_forward_diagnostic_summary.json", summary)
    print("=== Real Field Forward Diagnostic ===")
    print(f"Output: {output_dir}")
    print(f"Inputs: {len(impedance_inputs)}")
    print("Status: ok")


if __name__ == "__main__":
    main()
