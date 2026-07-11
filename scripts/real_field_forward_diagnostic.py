"""Run R1 fixed forward diagnostics for R0 real-field zero-shot outputs.

Usage::

    python scripts/real_field_forward_diagnostic.py
    python scripts/real_field_forward_diagnostic.py --config experiments/common/common.yaml
    python scripts/real_field_forward_diagnostic.py --zero-shot-dir scripts/output/real_field_zero_shot_<timestamp>
    python scripts/real_field_forward_diagnostic.py --output-dir scripts/output/real_field_forward_diagnostic_test
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
src_text = str(SRC_DIR)
sys.path = [path for path in sys.path if path != src_text]
sys.path.insert(0, src_text)

from cup.seismic.viz import plot_well_waveform_qc
from cup.seismic.survey import open_survey, segy_options_from_config
from cup.config.sources import assert_same_path, load_summary, resolve_source_file_from_run, resolve_source_run
from cup.physics.numpy_backend import forward_depth, forward_time, velocity_from_ai
from cup.synthetic.contracts import FORWARD_MODEL_INPUTS_SCHEMA_VERSION
from cup.well.anchor import sample_volume_trilinear
from cup.well.real_field_controls import load_well_control_set
from cup.utils.io import (
    CONTRACT_FINGERPRINT_SCHEMA,
    contract_fingerprint_sha256,
    load_yaml_config,
    published_contract_reference,
    repo_relative_path,
    resolve_relative_path,
    write_json,
)
from cup.utils.statistics import radius_connected_components
from ginn_v2.contracts import FORWARD_DIAGNOSTIC_SCHEMA_VERSION, ZERO_SHOT_SUMMARY_SCHEMA_VERSION
from ginn_v2.real_field import (
    diagnostic_metrics,
    load_selected_wavelet,
    load_zero_shot_predictions,
    phase_shift_scan,
)
from wtie.optimize.similarity import dynamic_normalized_xcorr, normalized_xcorr
from wtie.processing import grid


DEFAULT_COMMON_CONFIG = Path("experiments/common/common.yaml")
SCHEMA_VERSION = FORWARD_DIAGNOSTIC_SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_COMMON_CONFIG)
    parser.add_argument("--zero-shot-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _resolve_output_dir(prefix: str, explicit: Path | None, *, output_root: Path) -> Path:
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
    obs = np.asarray(observed, dtype=np.float64)
    syn = np.asarray(synthetic, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if obs.shape != syn.shape or mask.shape != syn.shape:
        raise ValueError(f"Forward diagnostic shape mismatch: obs={obs.shape}, syn={syn.shape}, mask={mask.shape}")
    return obs, syn, mask


def _spatial_rows(*, model_role: str, observed: np.ndarray, synthetic: np.ndarray, valid: np.ndarray) -> list[dict[str, object]]:
    rows = []
    if observed.ndim == 2:
        for lateral_idx in range(observed.shape[0]):
            metrics = diagnostic_metrics(
                observed=observed[lateral_idx : lateral_idx + 1],
                synthetic=synthetic[lateral_idx : lateral_idx + 1],
                valid_mask=valid[lateral_idx : lateral_idx + 1],
            )
            rows.append({"model_role": model_role, "spatial_axis": "lateral", "lateral_index": lateral_idx, **metrics})
        return rows
    if observed.ndim == 3:
        for inline_idx in range(observed.shape[0]):
            metrics = diagnostic_metrics(
                observed=observed[inline_idx : inline_idx + 1, :, :],
                synthetic=synthetic[inline_idx : inline_idx + 1, :, :],
                valid_mask=valid[inline_idx : inline_idx + 1, :, :],
            )
            rows.append({"model_role": model_role, "spatial_axis": "inline", "spatial_index": inline_idx, **metrics})
        for xline_idx in range(observed.shape[1]):
            metrics = diagnostic_metrics(
                observed=observed[:, xline_idx : xline_idx + 1, :],
                synthetic=synthetic[:, xline_idx : xline_idx + 1, :],
                valid_mask=valid[:, xline_idx : xline_idx + 1, :],
            )
            rows.append({"model_role": model_role, "spatial_axis": "xline", "spatial_index": xline_idx, **metrics})
        return rows
    raise ValueError(f"Unsupported spatial QC dimensionality: {observed.ndim}")
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


def _configured_bands(run_cfg: dict, *, dt_s: float = 0.002) -> list[dict[str, float | str | bool]]:
    spectral = dict(run_cfg.get("spectral_qc") or {})
    bands = spectral.get("bands")
    if bands is None:
        nyquist = 0.5 / float(dt_s)
        configured_max = float(run_cfg.get("diagnostic_max_hz", 80.0))
        high = min(configured_max, 0.45 * nyquist)
        if not high > 0.0:
            raise ValueError("Cannot build default spectral bands with non-positive diagnostic max frequency.")
        return [
            {"name": "lowfreq", "low_hz": 0.0, "high_hz": 0.2 * high, "manual_spectral_band_override": False},
            {"name": "observable_band", "low_hz": 0.2 * high, "high_hz": 0.4 * high, "manual_spectral_band_override": False},
            {"name": "highfreq_or_nullspace", "low_hz": 0.4 * high, "high_hz": high, "manual_spectral_band_override": False},
        ]
    reason = str(spectral.get("manual_override_reason") or "").strip()
    if not reason:
        raise ValueError("Manual spectral_qc.bands requires spectral_qc.manual_override_reason.")
    out = []
    for item in bands:
        if not isinstance(item, dict):
            raise ValueError("Each R1 spectral band must be a mapping.")
        out.append(
            {
                "name": str(item["name"]),
                "low_hz": float(item.get("low_hz", 0.0)),
                "high_hz": float(item["high_hz"]) if item.get("high_hz") is not None else float("inf"),
                "manual_spectral_band_override": True,
                "manual_override_reason": reason,
            }
        )
    return out


def _observability_evidence(run_cfg: dict, bands: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    source_runs = dict(run_cfg.get("source_runs") or {})
    obs_dir_text = str(run_cfg.get("observability_dir") or source_runs.get("forward_observability_dir") or "").strip()
    missing = {
        str(band["name"]): {
            "observability_evidence_status": "missing",
            "dominant_evidence_status": "",
            "operator_support_summary": "",
            "detectability_ratio_p25_range": "",
            "observability_source_run": "",
        }
        for band in bands
    }
    if not obs_dir_text:
        return missing
    obs_dir = resolve_relative_path(obs_dir_text, root=REPO_ROOT)
    path = obs_dir / "frequency_evidence_bands.csv"
    if not path.is_file():
        return missing
    frame = pd.read_csv(path)
    if "frequency_hz" not in frame:
        return missing
    out = {}
    for band in bands:
        name = str(band["name"])
        low = float(band["low_hz"])
        high = float(band["high_hz"])
        scoped = frame[
            pd.to_numeric(frame["frequency_hz"], errors="coerce").ge(low)
            & pd.to_numeric(frame["frequency_hz"], errors="coerce").lt(high)
        ].copy()
        if scoped.empty:
            out[name] = {
                **missing[name],
                "observability_evidence_status": "empty_band_overlap",
                "observability_source_run": repo_relative_path(obs_dir, root=REPO_ROOT),
            }
            continue
        status_col = "dominant_evidence_status" if "dominant_evidence_status" in scoped else "empirical_status"
        support_col = "operator_support_summary" if "operator_support_summary" in scoped else "operator_support_class"
        ratio_col = "detectability_ratio_p25" if "detectability_ratio_p25" in scoped else "cluster_p25_detectability_ratio"
        ratios = pd.to_numeric(scoped.get(ratio_col, pd.Series(dtype=float)), errors="coerce").dropna()
        out[name] = {
            "observability_evidence_status": "present",
            "dominant_evidence_status": str(scoped[status_col].mode().iloc[0]) if status_col in scoped and not scoped[status_col].dropna().empty else "",
            "operator_support_summary": str(scoped[support_col].mode().iloc[0]) if support_col in scoped and not scoped[support_col].dropna().empty else "",
            "detectability_ratio_p25_range": (
                f"{float(ratios.min()):.6g}..{float(ratios.max()):.6g}" if not ratios.empty else ""
            ),
            "observability_source_run": repo_relative_path(obs_dir, root=REPO_ROOT),
        }
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
    if data.ndim < 2 or not np.any(mask):
        return float("nan")
    flat = data.reshape((-1, data.shape[-1]))
    flat_mask = mask.reshape((-1, data.shape[-1]))
    prepared = np.zeros_like(flat, dtype=np.float64)
    for idx in range(flat.shape[0]):
        row_mask = flat_mask[idx]
        if int(np.count_nonzero(row_mask)) < 8:
            continue
        prepared[idx] = np.where(row_mask, flat[idx] - float(np.mean(flat[idx, row_mask])), 0.0)
    spectrum = np.fft.rfft(prepared, axis=1)
    freqs = np.fft.rfftfreq(flat.shape[1], d=float(dt_s))
    band = (freqs >= float(low_hz)) & (freqs < float(high_hz))
    if not np.any(band):
        return 0.0
    filtered = np.zeros_like(spectrum)
    filtered[:, band] = spectrum[:, band]
    component = np.fft.irfft(filtered, n=flat.shape[1], axis=1)
    return float(np.sqrt(np.mean(component[flat_mask] ** 2)))


def _read_wavelet_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    missing = sorted({"time_s", "amplitude"} - set(frame.columns))
    if missing:
        raise ValueError(f"Wavelet CSV lacks columns {missing}: {path}")
    return (
        frame["time_s"].to_numpy(dtype=np.float64),
        frame["amplitude"].to_numpy(dtype=np.float64),
    )


def _load_wavelet_scenarios(
    wavelet_dir: Path,
    nominal_time_s: np.ndarray,
    nominal: np.ndarray,
    nominal_meta: dict[str, object],
    run_cfg: dict,
) -> list[dict[str, object]]:
    scan_cfg = dict(run_cfg.get("diagnostic_scan") or {})
    limit = int(scan_cfg.get("candidate_wavelet_limit", 0) or 0)
    scenarios: list[dict[str, object]] = [
        {
            "wavelet_scenario_id": "nominal_selected",
            "source_well": str((nominal_meta.get("selected_source_well") or "optimized_consensus")),
            "candidate_score": float("nan"),
            "wavelet_path": str(nominal_meta["selected_wavelet_csv"]),
            "wavelet_time_s": nominal_time_s,
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
        candidate_time_s, wavelet = _read_wavelet_csv(path)
        if wavelet.size != nominal.size or not np.array_equal(candidate_time_s, nominal_time_s):
            continue
        scenarios.append(
            {
                "wavelet_scenario_id": candidate,
                "source_well": source_well,
                "candidate_score": float(pd.to_numeric(row.get("score"), errors="coerce")),
                "wavelet_path": repo_relative_path(path, root=REPO_ROOT),
                "wavelet_time_s": candidate_time_s,
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
    summary = _load_zero_shot_summary(zero_shot_dir)
    config_text = str(summary.get("config_file") or "").strip()
    if not config_text:
        raise ValueError("R1 XY well QC requires zero-shot summary config_file.")
    config_path = resolve_relative_path(config_text, root=REPO_ROOT)
    cfg = load_yaml_config(config_path)
    data_root = resolve_relative_path(cfg.get("data_root", "data"), root=REPO_ROOT)
    seismic = dict(cfg.get("seismic") or {})
    seismic_text = str(seismic.get("file") or "").strip()
    if not seismic_text:
        raise ValueError("R1 XY well QC requires top-level seismic.file.")
    seismic_path = resolve_relative_path(seismic_text, root=data_root)
    seismic_type = str(seismic.get("type", "zgy")).casefold()
    segy_options = segy_options_from_config(seismic) if seismic_type == "segy" else None
    survey = open_survey(seismic_path, seismic_type, segy_options=segy_options)
    return survey.line_geometry


def _load_zero_shot_summary(zero_shot_dir: Path) -> dict:
    return load_summary(
        zero_shot_dir / "real_field_zero_shot_summary.json",
        schema_version=ZERO_SHOT_SUMMARY_SCHEMA_VERSION,
        allowed_status={"needs_forward_diagnostic", "ok"},
        label="real_field_zero_shot_summary.json",
    )


def _resolve_zero_shot_dir(run_cfg: dict, *, output_root: Path, cli_value: Path | None) -> Path:
    return resolve_source_run(
        cli_value if cli_value is not None else run_cfg.get("zero_shot_dir"),
        output_root=output_root,
        prefix="real_field_zero_shot",
        required_files=["real_field_zero_shot_summary.json"],
        root=REPO_ROOT,
        label="real_field_zero_shot",
        summary_file="real_field_zero_shot_summary.json",
        schema_version=ZERO_SHOT_SUMMARY_SCHEMA_VERSION,
        allowed_status={"needs_forward_diagnostic", "ok"},
    )


def _resolve_wavelet_dir(run_cfg: dict, *, zero_shot_dir: Path, output_root: Path) -> Path:
    source_runs = dict(run_cfg.get("source_runs") or {})
    explicit = str(source_runs.get("wavelet_generation_dir") or "").strip()
    summary_path = zero_shot_dir / "real_field_zero_shot_summary.json"
    recorded = ""
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        recorded = str((summary.get("source_runs") or {}).get("wavelet_generation_dir") or "").strip()
    if explicit:
        wavelet_dir = resolve_source_run(
            explicit,
            output_root=output_root,
            prefix="wavelet_generation",
            required_files=["selected_wavelet.csv", "selected_wavelet_summary.json"],
            root=REPO_ROOT,
            label="wavelet_generation",
        )
        if recorded:
            assert_same_path(
                wavelet_dir,
                recorded,
                root=REPO_ROOT,
                message="R1 wavelet_generation_dir override does not match R0 summary source_runs.",
            )
        return wavelet_dir
    if recorded:
        return resolve_source_run(
            recorded,
            output_root=output_root,
            prefix="wavelet_generation",
            required_files=["selected_wavelet.csv", "selected_wavelet_summary.json"],
            root=REPO_ROOT,
            label="wavelet_generation",
        )
    return resolve_source_run(
        None,
        output_root=output_root,
        prefix="wavelet_generation",
        required_files=["selected_wavelet.csv", "selected_wavelet_summary.json"],
        root=REPO_ROOT,
        label="wavelet_generation",
    )


def _resolve_well_auto_tie_dir(
    run_cfg: dict,
    *,
    zero_shot_summary: Mapping[str, object],
    output_root: Path,
) -> Path:
    well_cfg = dict(run_cfg.get("well_qc") or {})
    explicit = str(well_cfg.get("well_auto_tie_dir") or "").strip()
    r0_source_runs = dict(zero_shot_summary.get("source_runs") or {})
    recorded = str(r0_source_runs.get("well_auto_tie_dir") or "").strip()
    if explicit:
        well_auto_tie_dir = resolve_source_run(
            explicit,
            output_root=output_root,
            prefix="well_auto_tie",
            required_files=["well_tie_metrics.csv", "well_tie_plan.csv", "wavelet_inventory.csv"],
            root=REPO_ROOT,
            label="well_auto_tie",
        )
        if recorded:
            assert_same_path(
                well_auto_tie_dir,
                recorded,
                root=REPO_ROOT,
                message="R1 well_qc.well_auto_tie_dir override does not match R0/LFM source_runs.",
            )
        return well_auto_tie_dir
    if recorded:
        return resolve_source_run(
            recorded,
            output_root=output_root,
            prefix="well_auto_tie",
            required_files=["well_tie_metrics.csv", "well_tie_plan.csv", "wavelet_inventory.csv"],
            root=REPO_ROOT,
            label="well_auto_tie",
        )
    return resolve_source_run(
        None,
        output_root=output_root,
        prefix="well_auto_tie",
        required_files=["well_tie_metrics.csv", "well_tie_plan.csv", "wavelet_inventory.csv"],
        root=REPO_ROOT,
        label="well_auto_tie",
    )


def _resolve_well_inventory_file(
    run_cfg: dict,
    *,
    zero_shot_summary: Mapping[str, object],
    output_root: Path,
) -> Path:
    well_cfg = dict(run_cfg.get("well_qc") or {})
    explicit = str(well_cfg.get("well_inventory_file") or "").strip()
    recorded = ""
    if explicit:
        inventory_file = resolve_source_file_from_run(
            explicit,
            output_root=output_root,
            prefix="well_inventory",
            file_name="well_inventory.csv",
            root=REPO_ROOT,
            label="well_inventory_file",
        )
        if recorded:
            assert_same_path(
                inventory_file,
                recorded,
                root=REPO_ROOT,
                message="R1 well_qc.well_inventory_file override does not match LFM summary inputs.",
            )
        return inventory_file
    if recorded:
        return resolve_source_file_from_run(
            recorded,
            output_root=output_root,
            prefix="well_inventory",
            file_name="well_inventory.csv",
            root=REPO_ROOT,
            label="well_inventory_file",
        )
    return resolve_source_file_from_run(
        None,
        output_root=output_root,
        prefix="well_inventory",
        file_name="well_inventory.csv",
        root=REPO_ROOT,
        label="well_inventory_file",
    )


def _prepare_well_qc_sources(
    run_cfg: dict,
    *,
    zero_shot_summary: Mapping[str, object],
    output_root: Path,
) -> dict:
    prepared = dict(run_cfg)
    well_cfg = dict(prepared.get("well_qc") or {})
    if bool(well_cfg.get("enabled", True)):
        if not str(well_cfg.get("well_auto_tie_dir") or "").strip():
            well_dir = _resolve_well_auto_tie_dir(
                prepared,
                zero_shot_summary=zero_shot_summary,
                output_root=output_root,
            )
            well_cfg["well_auto_tie_dir"] = repo_relative_path(well_dir, root=REPO_ROOT)
        if not str(well_cfg.get("well_inventory_file") or "").strip():
            inventory_file = _resolve_well_inventory_file(
                prepared,
                zero_shot_summary=zero_shot_summary,
                output_root=output_root,
            )
            well_cfg["well_inventory_file"] = repo_relative_path(inventory_file, root=REPO_ROOT)
    prepared["well_qc"] = well_cfg
    return prepared


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
            "surface_x": _coerce_float(row.get("surface_x")),
            "surface_y": _coerce_float(row.get("surface_y")),
            "wellbore_class": str(row.get("wellbore_class", "")),
        }
    return info


def _well_spatial_clusters(
    inventory_info: Mapping[str, Mapping[str, object]],
    run_cfg: dict,
    *,
    well_names: set[str],
) -> dict[str, dict[str, object]]:
    wells = []
    xy = []
    for well_name, info in inventory_info.items():
        if str(well_name) not in well_names:
            continue
        x = _coerce_float(info.get("surface_x"))
        y = _coerce_float(info.get("surface_y"))
        if not (well_name and np.isfinite(x) and np.isfinite(y)):
            continue
        wells.append(str(well_name))
        xy.append((x, y))
    if not wells:
        return {}
    spatial_cfg = dict(run_cfg.get("spatial_debias") or {})
    radius_m = float(spatial_cfg.get("cluster_radius_m", 600.0))
    labels = radius_connected_components(np.asarray(xy, dtype=np.float64), radius_m)
    sizes: dict[int, int] = {}
    for label in labels:
        sizes[int(label)] = sizes.get(int(label), 0) + 1
    return {
        well: {
            "spatial_cluster_id": int(label),
            "spatial_cluster_size": int(sizes[int(label)]),
        }
        for well, label in zip(wells, labels)
    }


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
        "well_ai_bias_median": float(np.median(residual)),
        "calibration_bias_filtered_minus_pred_median": float(np.median(well_log_ai[mask] - pred_log_ai[mask])),
        "well_ai_corr": _safe_corr(well_log_ai[mask], pred_log_ai[mask]),
        "well_ai_pred_mean": float(np.mean(pred_log_ai[mask])),
        "well_ai_log_mean": float(np.mean(well_log_ai[mask])),
    }


def _axis_index_float(axis: np.ndarray, value: float) -> float:
    values = np.asarray(axis, dtype=np.float64)
    if values.size < 2:
        raise ValueError("Axis must contain at least two samples.")
    step = float(np.median(np.diff(values)))
    if step == 0.0 or not np.isfinite(step):
        raise ValueError("Axis step is invalid.")
    return (float(value) - float(values[0])) / step


def _sample_volume_bilinear(
    volume: np.ndarray,
    *,
    ilines: np.ndarray,
    xlines: np.ndarray,
    twt_s: np.ndarray,
    inline_values: np.ndarray,
    xline_values: np.ndarray,
    sample_twt_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(volume, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"Volume sampling expects [inline, xline, twt], got {data.shape}.")
    inline_values = np.asarray(inline_values, dtype=np.float64).reshape(-1)
    xline_values = np.asarray(xline_values, dtype=np.float64).reshape(-1)
    sample_twt_s = np.asarray(sample_twt_s, dtype=np.float64).reshape(-1)
    if not (inline_values.size == xline_values.size == sample_twt_s.size):
        raise ValueError("Volume sampling coordinate arrays must have the same length.")
    out = np.full(sample_twt_s.shape, np.nan, dtype=np.float64)
    inside = np.zeros(sample_twt_s.shape, dtype=bool)
    for idx, (inline_value, xline_value, twt_value) in enumerate(zip(inline_values, xline_values, sample_twt_s)):
        if not (np.isfinite(inline_value) and np.isfinite(xline_value) and np.isfinite(twt_value)):
            continue
        i_float = _axis_index_float(ilines, float(inline_value))
        j_float = _axis_index_float(xlines, float(xline_value))
        i0 = int(np.floor(i_float))
        j0 = int(np.floor(j_float))
        i1 = i0 + 1
        j1 = j0 + 1
        if i0 < 0 or j0 < 0 or i1 >= data.shape[0] or j1 >= data.shape[1]:
            continue
        wi = float(i_float - i0)
        wj = float(j_float - j0)
        accum = 0.0
        weight_sum = 0.0
        for i, j, weight in (
            (i0, j0, (1.0 - wi) * (1.0 - wj)),
            (i0, j1, (1.0 - wi) * wj),
            (i1, j0, wi * (1.0 - wj)),
            (i1, j1, wi * wj),
        ):
            if weight <= 0.0:
                continue
            trace = data[i, j, :]
            finite = np.isfinite(trace) & np.isfinite(twt_s)
            if int(np.count_nonzero(finite)) < 2:
                continue
            value = float(np.interp(float(twt_value), twt_s[finite], trace[finite], left=np.nan, right=np.nan))
            if not np.isfinite(value):
                continue
            accum += weight * value
            weight_sum += weight
        if weight_sum > 0.0:
            out[idx] = accum / weight_sum
            inside[idx] = True
    return out, inside


def _well_sampling_coordinates(
    *,
    tie: pd.Series,
    inventory: Mapping[str, object],
    twt_s: np.ndarray,
    include_deviated: bool,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    well = str(tie.get("well_name", "")).strip()
    wellbore_class = str(inventory.get("wellbore_class") or "")
    plan_text = str(tie.get("optimized_trace_sample_plan_file") or "").strip()
    if plan_text and plan_text.casefold() != "nan":
        if not include_deviated and wellbore_class.casefold() == "deviated":
            raise ValueError("deviated well disabled by config")
        plan_path = resolve_relative_path(plan_text, root=REPO_ROOT)
        plan = pd.read_csv(plan_path)
        inside = plan["survey_position"].astype(str).eq("inside")
        plan = plan[inside].copy()
        if plan.empty:
            raise ValueError(f"Trace sample plan has no inside samples: {plan_path}")
        return (
            "volume_trace_plan_bilinear",
            pd.to_numeric(plan["inline_float"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(plan["xline_float"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(plan["twt_s"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(plan["x_m"], errors="coerce").to_numpy(dtype=np.float64) if "x_m" in plan else np.full(plan.shape[0], np.nan, dtype=np.float64),
            pd.to_numeric(plan["y_m"], errors="coerce").to_numpy(dtype=np.float64) if "y_m" in plan else np.full(plan.shape[0], np.nan, dtype=np.float64),
            wellbore_class,
        )
    inline_value = _coerce_float(inventory.get("inline_float"))
    xline_value = _coerce_float(inventory.get("xline_float"))
    if not (np.isfinite(inline_value) and np.isfinite(xline_value)):
        raise ValueError(f"Missing inline/xline for well {well}")
    x_value = _coerce_float(inventory.get("surface_x"))
    y_value = _coerce_float(inventory.get("surface_y"))
    return (
        "volume_vertical_bilinear",
        np.full(twt_s.shape, inline_value, dtype=np.float64),
        np.full(twt_s.shape, xline_value, dtype=np.float64),
        np.asarray(twt_s, dtype=np.float64),
        np.full(twt_s.shape, x_value, dtype=np.float64),
        np.full(twt_s.shape, y_value, dtype=np.float64),
        wellbore_class or "vertical",
    )


def _build_volume_well_forward_qc(
    *,
    run_cfg: dict,
    output_dir: Path,
    predictions: dict[str, dict[str, object]],
    synthetic_by_role: dict[str, np.ndarray],
    wavelet_time_s: np.ndarray,
    wavelet: np.ndarray,
    observed: np.ndarray,
    lfm: np.ndarray,
    valid: np.ndarray,
    ilines: np.ndarray,
    xlines: np.ndarray,
    twt_s: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    well_cfg = dict(run_cfg.get("well_qc") or {})
    if not bool(well_cfg.get("enabled", False)):
        return pd.DataFrame.from_records([{"status": "disabled_by_config"}]), pd.DataFrame(), {}
    well_auto_tie_text = str(well_cfg.get("well_auto_tie_dir") or "").strip()
    if not well_auto_tie_text:
        raise ValueError("real_field_forward_diagnostic.well_qc.well_auto_tie_dir must be explicit when well_qc is enabled.")
    well_auto_tie_dir = resolve_relative_path(well_auto_tie_text, root=REPO_ROOT)
    metrics = pd.read_csv(well_auto_tie_dir / "well_tie_metrics.csv")
    metrics = metrics[metrics["tie_status"].astype(str).eq("success")].copy()
    inventory_info = _load_well_inventory_info(well_cfg)
    spatial_clusters = _well_spatial_clusters(
        inventory_info,
        run_cfg,
        well_names={str(value) for value in metrics["well_name"].dropna().unique()},
    )
    include_deviated = bool(well_cfg.get("include_deviated_wells", True))
    dt_s = float(np.median(np.diff(twt_s))) if twt_s.size > 1 else 0.002
    bands = _configured_bands(run_cfg, dt_s=dt_s)
    rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    figures: dict[str, str] = {}
    figures_dir = output_dir / "figures" / "wells"
    figures_dir.mkdir(parents=True, exist_ok=True)
    first_arrays = next(iter(predictions.values()))["arrays"]
    for _, tie in metrics.iterrows():
        well_name = str(tie.get("well_name", "")).strip()
        inventory = inventory_info.get(well_name, {})
        base_common = {"well_name": well_name, "sampling_mode": "volume"}
        try:
            sample_method, sample_ilines, sample_xlines, sample_twt, sample_x_m, sample_y_m, wellbore_class = _well_sampling_coordinates(
                tie=tie,
                inventory=inventory,
                twt_s=twt_s,
                include_deviated=include_deviated,
            )
            well_log_ai = _log_ai_at_twt(
                las_path=resolve_relative_path(str(tie["filtered_las_file"]), root=REPO_ROOT),
                tdt_path=resolve_relative_path(str(tie["optimized_tdt_file"]), root=REPO_ROOT),
                twt_s=sample_twt,
            )
        except Exception as exc:
            rows.append({**base_common, "model_role": "", "wellbore_class": str(inventory.get("wellbore_class", "")), "status": "skipped_sampling_failed", "reason": str(exc)})
            continue
        base = {
            **base_common,
            "wellbore_class": wellbore_class,
            "sample_method": sample_method,
            "n_well_samples": int(sample_twt.size),
            "twt_start_s": float(np.nanmin(sample_twt)) if sample_twt.size else np.nan,
            "twt_stop_s": float(np.nanmax(sample_twt)) if sample_twt.size else np.nan,
            "section_xy_distance_m": 0.0,
            "nearest_section_trace": -1,
        }
        lfm_trace, lfm_inside = _sample_volume_bilinear(
            lfm,
            ilines=ilines,
            xlines=xlines,
            twt_s=twt_s,
            inline_values=sample_ilines,
            xline_values=sample_xlines,
            sample_twt_s=sample_twt,
        )
        valid_trace, valid_inside = _sample_volume_bilinear(
            valid.astype(float),
            ilines=ilines,
            xlines=xlines,
            twt_s=twt_s,
            inline_values=sample_ilines,
            xline_values=sample_xlines,
            sample_twt_s=sample_twt,
        )
        model_valid = valid_inside & lfm_inside & (valid_trace > 0.5)
        cluster_info = spatial_clusters.get(well_name, {})
        spatial_cluster_id = cluster_info.get("spatial_cluster_id", -1)
        spatial_cluster_size = cluster_info.get("spatial_cluster_size", 1)
        filtered_synthetic = forward_time(
            _fill_nonfinite_1d(well_log_ai)[None, :], wavelet_time_s, wavelet
        )[0]
        forward_twt = sample_twt
        forward_ilines = sample_ilines
        forward_xlines = sample_xlines
        observed_forward, observed_inside = _sample_volume_bilinear(
            observed,
            ilines=ilines,
            xlines=xlines,
            twt_s=twt_s,
            inline_values=forward_ilines,
            xline_values=forward_xlines,
            sample_twt_s=forward_twt,
        )
        forward_valid_base = observed_inside & model_valid
        filtered_waveform = diagnostic_metrics(
            observed=observed_forward[None, :],
            synthetic=filtered_synthetic[None, :],
            valid_mask=(forward_valid_base & np.isfinite(filtered_synthetic))[None, :],
        )
        figures.update(
            _plot_volume_well_qc_case(
                figures_dir=figures_dir,
                well_name=well_name,
                role="filtered_las",
                selected_log_ai=well_log_ai,
                filtered_log_ai=well_log_ai,
                sample_twt=sample_twt,
                observed=observed_forward,
                synthetic=filtered_synthetic,
                valid_forward=forward_valid_base & np.isfinite(filtered_synthetic),
                tie_window_start_s=_coerce_float(tie.get("tie_window_start_s")),
                tie_window_end_s=_coerce_float(tie.get("tie_window_end_s")),
            )
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
                "well_ai_bias_median": 0.0,
                "calibration_bias_filtered_minus_pred_median": 0.0,
                "well_ai_corr": 1.0,
                **{f"waveform_{key}": value for key, value in filtered_waveform.items()},
            }
        )
        lfm_ai_metrics = _well_ai_metrics(well_log_ai=well_log_ai, pred_log_ai=lfm_trace, valid=model_valid)
        rows.append({**base, "model_role": "lfm_input", "status": "ok" if lfm_ai_metrics.get("well_ai_status") == "ok" else lfm_ai_metrics.get("well_ai_status"), **lfm_ai_metrics})
        for role, payload in predictions.items():
            arrays = payload["arrays"]
            pred_trace, pred_inside = _sample_volume_bilinear(
                np.asarray(arrays["stitched_pred_log_ai"], dtype=np.float64),
                ilines=ilines,
                xlines=xlines,
                twt_s=twt_s,
                inline_values=sample_ilines,
                xline_values=sample_xlines,
                sample_twt_s=sample_twt,
            )
            weight_trace, _ = _sample_volume_bilinear(
                np.asarray(arrays["stitching_weight"], dtype=np.float64),
                ilines=ilines,
                xlines=xlines,
                twt_s=twt_s,
                inline_values=sample_ilines,
                xline_values=sample_xlines,
                sample_twt_s=sample_twt,
            )
            role_valid = model_valid & pred_inside & (weight_trace > 0.0)
            ai_metrics = _well_ai_metrics(well_log_ai=well_log_ai, pred_log_ai=pred_trace, valid=role_valid)
            r0_valid_mask = valid_inside & pred_inside & (valid_trace > 0.5)
            well_twt_support_valid = lfm_inside & np.isfinite(well_log_ai)
            finite_for_fit = (
                role_valid
                & well_twt_support_valid
                & np.isfinite(lfm_trace)
                & np.isfinite(pred_trace)
                & np.isfinite(well_log_ai)
            )
            for sample_index, (
                sample_twt_value,
                inline_value,
                xline_value,
                x_value,
                y_value,
                filtered_value,
                lfm_value,
                pred_value,
                r0_mask_value,
                weight_value,
                support_value,
                fit_value,
            ) in enumerate(
                zip(
                    sample_twt,
                    sample_ilines,
                    sample_xlines,
                    sample_x_m,
                    sample_y_m,
                    well_log_ai,
                    lfm_trace,
                    pred_trace,
                    r0_valid_mask,
                    weight_trace,
                    well_twt_support_valid,
                    finite_for_fit,
                )
            ):
                reason = "ok"
                if not bool(support_value):
                    reason = "well_twt_support_invalid"
                elif not bool(r0_mask_value):
                    reason = "r0_valid_mask_false"
                elif not (np.isfinite(weight_value) and weight_value > 0.0):
                    reason = "r0_blend_weight_nonpositive"
                elif not (np.isfinite(filtered_value) and np.isfinite(lfm_value) and np.isfinite(pred_value)):
                    reason = "nonfinite_ai_sample"
                sample_rows.append(
                    {
                        "well_name": well_name,
                        "sample_index": int(sample_index),
                        "twt_s": float(sample_twt_value),
                        "inline": float(inline_value),
                        "xline": float(xline_value),
                        "x_m": float(x_value),
                        "y_m": float(y_value),
                        "spatial_cluster_id": spatial_cluster_id,
                        "spatial_cluster_size": spatial_cluster_size,
                        "filtered_log_ai": float(filtered_value) if np.isfinite(filtered_value) else np.nan,
                        "lfm_log_ai": float(lfm_value) if np.isfinite(lfm_value) else np.nan,
                        "model_role": role,
                        "r0_pred_log_ai": float(pred_value) if np.isfinite(pred_value) else np.nan,
                        "r0_valid_mask": bool(r0_mask_value),
                        "r0_blend_weight": float(weight_value) if np.isfinite(weight_value) else np.nan,
                        "well_twt_support_valid": bool(support_value),
                        "valid_for_fit": bool(fit_value),
                        "valid_reason": reason if not bool(fit_value) else "ok",
                        "sampling_mode": "volume",
                        "sample_method": sample_method,
                        "wellbore_class": wellbore_class,
                    }
                )
            local_spectrum: dict[str, object] = {}
            pred_delta_trace, _ = _sample_volume_bilinear(
                np.asarray(arrays["pred_delta_vs_lfm"], dtype=np.float64),
                ilines=ilines,
                xlines=xlines,
                twt_s=twt_s,
                inline_values=sample_ilines,
                xline_values=sample_xlines,
                sample_twt_s=sample_twt,
            )
            residual = pred_trace - well_log_ai
            for band in bands:
                name = str(band["name"])
                pair = _band_pair_metrics(
                    reference=well_log_ai,
                    model=pred_trace,
                    valid=role_valid,
                    dt_s=dt_s,
                    low_hz=float(band["low_hz"]),
                    high_hz=float(band["high_hz"]),
                )
                local_spectrum[f"well_ai_{name}_n_valid"] = pair["n_valid"]
                local_spectrum[f"well_ai_{name}_rmse"] = pair["rmse"]
                local_spectrum[f"well_ai_{name}_corr"] = pair["corr"]
                local_spectrum[f"well_ai_residual_{name}_rms"] = _band_rms(residual, role_valid, dt_s=dt_s, low_hz=float(band["low_hz"]), high_hz=float(band["high_hz"]))
                local_spectrum[f"pred_delta_{name}_rms"] = _band_rms(pred_delta_trace, role_valid, dt_s=dt_s, low_hz=float(band["low_hz"]), high_hz=float(band["high_hz"]))
            synthetic_key = f"zero_shot_{role}"
            synthetic_trace = np.full(forward_twt.shape, np.nan, dtype=np.float64)
            synthetic_inside = np.zeros(forward_twt.shape, dtype=bool)
            if synthetic_key in synthetic_by_role:
                synthetic_trace, synthetic_inside = _sample_volume_bilinear(
                    synthetic_by_role[synthetic_key],
                    ilines=ilines,
                    xlines=xlines,
                    twt_s=twt_s,
                    inline_values=forward_ilines,
                    xline_values=forward_xlines,
                    sample_twt_s=forward_twt,
                )
            waveform = diagnostic_metrics(
                observed=observed_forward[None, :],
                synthetic=synthetic_trace[None, :],
                valid_mask=(forward_valid_base & synthetic_inside)[None, :],
            )
            figures.update(
                _plot_volume_well_qc_case(
                    figures_dir=figures_dir,
                    well_name=well_name,
                    role=role,
                    selected_log_ai=pred_trace,
                    filtered_log_ai=well_log_ai,
                    sample_twt=sample_twt,
                    observed=observed_forward,
                    synthetic=synthetic_trace,
                    valid_forward=forward_valid_base & synthetic_inside,
                    tie_window_start_s=_coerce_float(tie.get("tie_window_start_s")),
                    tie_window_end_s=_coerce_float(tie.get("tie_window_end_s")),
                )
            )
            rows.append(
                {
                    **base,
                    "model_role": role,
                    "status": "ok" if ai_metrics.get("well_ai_status") == "ok" else ai_metrics.get("well_ai_status"),
                    **ai_metrics,
                    **local_spectrum,
                    **{f"waveform_{key}": value for key, value in waveform.items()},
                }
            )
    frame = pd.DataFrame.from_records(rows)
    samples = pd.DataFrame.from_records(sample_rows)
    return frame, samples, figures


def _plot_volume_well_qc_case(
    *,
    figures_dir: Path,
    well_name: str,
    role: str,
    selected_log_ai: np.ndarray,
    filtered_log_ai: np.ndarray,
    sample_twt: np.ndarray,
    observed: np.ndarray,
    synthetic: np.ndarray,
    valid_forward: np.ndarray,
    tie_window_start_s: float,
    tie_window_end_s: float,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    sample_twt = np.asarray(sample_twt, dtype=np.float64)
    selected_log_ai = np.asarray(selected_log_ai, dtype=np.float64)
    filtered_log_ai = np.asarray(filtered_log_ai, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    synthetic = np.asarray(synthetic, dtype=np.float64)
    valid_forward = np.asarray(valid_forward, dtype=bool)
    if sample_twt.size < 2 or observed.shape != synthetic.shape or observed.shape != sample_twt.shape:
        return outputs
    selected_filled = _fill_nonfinite_1d(selected_log_ai)
    reflectivity = np.tanh(0.5 * (selected_filled[1:] - selected_filled[:-1]))
    observed_plot = observed[1:]
    synthetic_plot = synthetic[1:]
    valid_plot = valid_forward[1:]
    plot_mask = (
        valid_plot
        & np.isfinite(observed_plot)
        & np.isfinite(synthetic_plot)
        & np.isfinite(reflectivity)
    )
    twt_forward = sample_twt[1:]
    if np.isfinite(tie_window_start_s) and np.isfinite(tie_window_end_s):
        plot_mask &= (twt_forward >= float(tie_window_start_s)) & (twt_forward <= float(tie_window_end_s))
    run = _largest_true_run(plot_mask)
    if run is None:
        return outputs
    start, stop = run
    if stop - start < 8:
        return outputs
    sl = slice(start, stop)
    basis = twt_forward[sl]
    selected_ai = np.exp(selected_filled[1:][sl])
    filtered_ai = np.exp(_fill_nonfinite_1d(filtered_log_ai)[1:][sl])
    synthetic_trace = grid.Seismic(synthetic_plot[sl], basis, "twt", name="Synthetic")
    observed_trace = grid.Seismic(observed_plot[sl], basis, "twt", name="Seismic")
    reflectivity_trace = grid.Reflectivity(reflectivity[sl], basis, "twt", name="Reflectivity")
    selected_ai_trace = grid.Log(selected_ai, basis, "twt", name=f"{role} AI")
    traces: list[grid.Log] = [selected_ai_trace]
    if role != "filtered_las":
        traces.append(grid.Log(filtered_ai, basis, "twt", name="Filtered LAS AI"))
    xcorr_values = normalized_xcorr(observed_trace.values, synthetic_trace.values)
    xcorr_basis = synthetic_trace.sampling_rate * np.arange(-(synthetic_trace.size - 1), synthetic_trace.size)
    xcorr = grid.XCorr(xcorr_values, xcorr_basis, "tlag", name="XCorr")
    dxcorr = dynamic_normalized_xcorr(observed_trace, synthetic_trace)
    corr = _safe_corr(observed_trace.values, synthetic_trace.values)
    title = f"R1 volume well QC | {well_name} | {role} | corr={corr:.3f}"
    fig, _axes = plot_well_waveform_qc(
        traces,
        reflectivity_trace,
        synthetic_trace,
        observed_trace,
        xcorr,
        dxcorr,
        figsize=(12.0, 7.5),
        synthetic_ai=selected_ai_trace,
        title=title,
    )
    path = figures_dir / f"well_forward_qc_{well_name}_{role}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    outputs[f"{well_name}/{role}"] = repo_relative_path(path, root=REPO_ROOT)
    return outputs


def _build_well_forward_qc(
    *,
    run_cfg: dict,
    zero_shot_dir: Path,
    output_dir: Path,
    predictions: dict[str, dict[str, object]],
    synthetic_dir: Path,
    wavelet_time_s: np.ndarray,
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
    twt_model = np.asarray(first_arrays["samples"], dtype=np.float64)
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
    dt_s = float(np.median(np.diff(twt_model))) if twt_model.size > 1 else 0.002
    bands = _configured_bands(run_cfg, dt_s=dt_s)
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
            filtered_synthetic = forward_time(
                _fill_nonfinite_1d(well_log_ai)[None, :],
                wavelet_time_s,
                wavelet,
            )
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
        if twt_forward.shape != twt_model.shape or not np.allclose(
            twt_forward, twt_model, equal_nan=True
        ):
            continue
        observed_plot = observed[1:]
        synthetic_plot = synthetic[1:]
        display_plot = display_mask[1:]
        twt_plot = twt_forward[1:]
        plot_mask = (
            display_plot
            & np.isfinite(observed_plot)
            & np.isfinite(synthetic_plot)
            & np.isfinite(reflectivity)
        )
        if np.isfinite(tie_window_start_s) and np.isfinite(tie_window_end_s):
            plot_mask &= (twt_plot >= float(tie_window_start_s)) & (twt_plot <= float(tie_window_end_s))
        run = _largest_true_run(plot_mask)
        if run is None:
            continue
        start, stop = run
        if stop - start < 8:
            continue
        sl = slice(start, stop)
        basis = twt_plot[sl]
        pred_ai = np.exp(pred_filled[1:][sl])
        filtered_ai = np.exp(well_log_ai[1:][sl])
        synthetic_trace = grid.Seismic(synthetic_plot[sl], basis, "twt", name="Synthetic")
        observed_trace = grid.Seismic(observed_plot[sl], basis, "twt", name="Seismic")
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
    sample_domain: str,
) -> str:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    display_mask = np.asarray(display_valid, dtype=bool)
    residual = np.where(display_mask, observed - synthetic, np.nan)
    observed_plot = _central_display_panel(np.where(display_mask, observed, np.nan))
    synthetic_plot = _central_display_panel(np.where(display_mask, synthetic, np.nan))
    residual_plot = _central_display_panel(residual)
    panels = [
        ("observed", observed_plot, "seismic"),
        ("synthetic", synthetic_plot, "seismic"),
        ("residual", residual_plot, "residual"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (_, values, title) in zip(axes, panels):
        finite = values[np.isfinite(values)]
        vmax = float(np.quantile(np.abs(finite), 0.99)) if finite.size else 1.0
        image = ax.imshow(values.T, aspect="auto", origin="upper", cmap="seismic", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("lateral trace")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("TVDSS sample" if sample_domain == "depth" else "TWT sample")
    fig.suptitle(f"R1 forward QC: {model_role}")
    fig.tight_layout()
    path = figures / f"{model_role}_observed_synthetic_residual.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return repo_relative_path(path, root=REPO_ROOT)


def _central_display_panel(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values)
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return data[data.shape[0] // 2, :, :]
    raise ValueError(f"Cannot display forward QC array with ndim={data.ndim}.")


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
    x_col = "lateral_index" if "lateral_index" in finite.columns and finite["lateral_index"].notna().any() else "spatial_index"
    group_cols = ["model_role"]
    if "spatial_axis" in finite.columns:
        group_cols.append("spatial_axis")
    for key, group in finite.groupby(group_cols):
        label = " / ".join(str(part) for part in (key if isinstance(key, tuple) else (key,)))
        ax.plot(group[x_col], group["residual_rms_scaled"], label=label)
    ax.set_xlabel(x_col)
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
    bands = _configured_bands(run_cfg, dt_s=dt_s)
    evidence = _observability_evidence(run_cfg, bands)
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
                    "manual_spectral_band_override": bool(band.get("manual_spectral_band_override", False)),
                    "manual_override_reason": str(band.get("manual_override_reason", "")),
                    **evidence.get(str(band["name"]), {}),
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
    bands = _configured_bands(run_cfg, dt_s=dt_s)
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
    for comparison in list(run_cfg.get("comparisons") or []):
        comparison_id = str(comparison["comparison_id"])
        left_id = str(comparison["left"])
        right_id = str(comparison["right"])
        if left_id not in predictions or right_id not in predictions:
            raise ValueError(f"R1 comparison {comparison_id} references unavailable experiments.")
        lat = predictions[left_id]["arrays"]
        no = predictions[right_id]["arrays"]
        diff = np.asarray(lat["stitched_pred_log_ai"], dtype=np.float64) - np.asarray(no["stitched_pred_log_ai"], dtype=np.float64)
        diff_valid = (
            valid
            & np.asarray(lat["valid_mask_model"], dtype=bool)
            & np.asarray(no["valid_mask_model"], dtype=bool)
            & (np.asarray(lat["stitching_weight"]) > 0.0)
            & (np.asarray(no["stitching_weight"]) > 0.0)
        )
        full = _band_rms_2d(diff, diff_valid, dt_s=dt_s, low_hz=0.0, high_hz=float("inf"))
        rows.append({"model_role": comparison_id, "signal": "pred_log_ai_difference", "left_experiment_id": left_id, "right_experiment_id": right_id, **_finite_stats(diff, diff_valid)})
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
                    "model_role": comparison_id,
                    "left_experiment_id": left_id,
                    "right_experiment_id": right_id,
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
    roles = [
        str(role)
        for role in sorted(well_frame["model_role"].dropna().unique())
        if str(role) not in {"", "filtered_las", "lfm_input"}
    ]
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


def _finite_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    padded = np.pad(values.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    return [
        (int(start), int(stop))
        for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
        if int(stop) - int(start) >= 2
    ]


def _forward_depth_masked(
    log_ai: np.ndarray,
    valid_mask: np.ndarray,
    *,
    tvdss_m: np.ndarray,
    wavelet_time_s: np.ndarray,
    wavelet: np.ndarray,
    relation: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(log_ai, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    if values.shape != valid.shape or values.shape[-1] != tvdss_m.size:
        raise ValueError("Depth R1 logAI/mask/TVDSS shape mismatch.")
    output = np.full(values.shape, np.nan, dtype=np.float64)
    output_valid = np.zeros(values.shape, dtype=bool)
    flat_values = values.reshape((-1, values.shape[-1]))
    flat_valid = valid.reshape((-1, valid.shape[-1]))
    flat_output = output.reshape((-1, output.shape[-1]))
    flat_output_valid = output_valid.reshape((-1, output_valid.shape[-1]))
    for trace_index, (trace, trace_valid) in enumerate(zip(flat_values, flat_valid)):
        for start, stop in _finite_true_runs(trace_valid):
            segment = trace[start:stop]
            ai = np.exp(segment)
            velocity = velocity_from_ai(
                ai,
                a=float(relation["a"]),
                b=float(relation["b"]),
            )
            synthetic = forward_depth(
                segment,
                velocity,
                tvdss_m[start:stop],
                wavelet_time_s,
                wavelet,
            )
            flat_output[trace_index, start:stop] = synthetic
            flat_output_valid[trace_index, start:stop] = True
    return output, output_valid


def _phase_rotate_wavelet(wavelet: np.ndarray, phase_deg: float) -> np.ndarray:
    values = np.asarray(wavelet, dtype=np.float64)
    spectrum = np.fft.fft(values)
    n = values.size
    multiplier = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        multiplier[0] = multiplier[n // 2] = 1.0
        multiplier[1 : n // 2] = 2.0
    else:
        multiplier[0] = 1.0
        multiplier[1 : (n + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * multiplier)
    return np.real(analytic * np.exp(1j * np.deg2rad(float(phase_deg))))


def _time_shift_wavelet(
    time_s: np.ndarray,
    wavelet: np.ndarray,
    shift_s: float,
) -> np.ndarray:
    return np.interp(
        np.asarray(time_s, dtype=np.float64) - float(shift_s),
        np.asarray(time_s, dtype=np.float64),
        np.asarray(wavelet, dtype=np.float64),
        left=0.0,
        right=0.0,
    )


def _depth_shift_volume(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    tvdss_m: np.ndarray,
    shift_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(data)
    shifted = np.full(data.shape, np.nan, dtype=np.float64)
    shifted_valid = np.zeros(data.shape, dtype=bool)
    query = np.asarray(tvdss_m, dtype=np.float64) - float(shift_m)
    for source, source_valid, target, target_valid in zip(
        data.reshape((-1, data.shape[-1])),
        valid.reshape((-1, valid.shape[-1])),
        shifted.reshape((-1, shifted.shape[-1])),
        shifted_valid.reshape((-1, shifted_valid.shape[-1])),
    ):
        for start, stop in _finite_true_runs(source_valid):
            inside = (query >= tvdss_m[start]) & (query <= tvdss_m[stop - 1])
            target[inside] = np.interp(query[inside], tvdss_m[start:stop], source[start:stop])
            target_valid[inside] = True
    return shifted, shifted_valid


def _load_depth_forward_inputs(
    zero_shot_summary: Mapping[str, object],
) -> tuple[Path, dict[str, object], np.ndarray, np.ndarray]:
    reference = dict(zero_shot_summary.get("forward_model_inputs") or {})
    path = resolve_relative_path(str(reference.get("path") or ""), root=REPO_ROOT)
    if not path.is_file():
        raise FileNotFoundError(f"Depth R0 forward_model_inputs not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != FORWARD_MODEL_INPUTS_SCHEMA_VERSION:
        raise ValueError(
            f"Depth R1 expected {FORWARD_MODEL_INPUTS_SCHEMA_VERSION}, "
            f"got {payload.get('schema')!r}; rebuild rock-physics analysis."
        )
    if payload.get("sample_domain") != "depth" or payload.get("depth_basis") != "tvdss":
        raise ValueError("Depth R1 forward_model_inputs must declare depth/TVDSS.")
    relation = dict(payload.get("ai_velocity_relation") or {})
    if relation.get("vp_unit") != "m/s" or relation.get("ai_unit") != "m/s*g/cm3":
        raise ValueError("Depth R1 AI–Vp relation units are invalid.")
    wavelet_path = resolve_relative_path(
        str(dict(payload.get("wavelet") or {}).get("path") or ""),
        root=REPO_ROOT,
    )
    wavelet_time_s, wavelet = _read_wavelet_csv(wavelet_path)
    return path, relation, wavelet_time_s, wavelet


def _depth_well_qc(
    *,
    zero_shot_dir: Path,
    zero_shot_summary: Mapping[str, object],
    observed: np.ndarray,
    synthetic_by_role: Mapping[str, np.ndarray],
    predictions: Mapping[str, Mapping[str, object]],
    ilines: np.ndarray,
    xlines: np.ndarray,
    tvdss_m: np.ndarray,
) -> pd.DataFrame:
    controls_ref = dict(dict(zero_shot_summary.get("input_contracts") or {}).get("well_control_set") or {})
    controls_summary = resolve_relative_path(str(controls_ref.get("path") or ""), root=REPO_ROOT)
    controls = load_well_control_set(controls_summary.parent, repo_root=REPO_ROOT)
    if controls.sample_domain != "depth" or controls.depth_basis != "tvdss":
        raise ValueError("Depth R1 requires a depth/TVDSS WellControlSet.")
    is_volume = observed.ndim == 3
    line_geometry = _load_zero_shot_line_geometry(zero_shot_dir)
    section_xy = np.asarray(
        [line_geometry.line_to_coord(float(il), float(xl)) for il, xl in zip(ilines, xlines)],
        dtype=np.float64,
    ) if not is_volume else np.empty((0, 2), dtype=np.float64)
    rows: list[dict[str, object]] = []
    for control in controls.controls:
        indices = np.searchsorted(control.sample_axis.values, tvdss_m)
        indices = np.clip(indices, 0, control.sample_axis.values.size - 1)
        same_axis = np.isclose(control.sample_axis.values[indices], tvdss_m, rtol=0.0, atol=1e-8)
        well_log_ai = np.where(same_axis, control.log_ai.values[indices], np.nan)
        inline_values = control.inline_by_sample[indices]
        xline_values = control.xline_by_sample[indices]
        if is_volume:
            sample_kwargs = {
                "ilines": ilines,
                "xlines": xlines,
                "twt_s": tvdss_m,
                "inline_values": inline_values,
                "xline_values": xline_values,
                "sample_twt_s": tvdss_m,
            }
            observed_well, observed_inside = sample_volume_trilinear(observed, **sample_kwargs)
        else:
            well_xy = np.column_stack([control.x_m_by_sample[indices], control.y_m_by_sample[indices]])
            distances = np.linalg.norm(well_xy[:, None, :] - section_xy[None, :, :], axis=2)
            trace_indices = np.argmin(distances, axis=1)
            observed_well = observed[trace_indices, np.arange(tvdss_m.size)]
            observed_inside = np.isfinite(distances[np.arange(tvdss_m.size), trace_indices])
        for role, synthetic in synthetic_by_role.items():
            if is_volume:
                synthetic_well, synthetic_inside = sample_volume_trilinear(synthetic, **sample_kwargs)
            else:
                synthetic_well = synthetic[trace_indices, np.arange(tvdss_m.size)]
                synthetic_inside = observed_inside
            model_role = role.removeprefix("zero_shot_")
            pred_payload = predictions.get(model_role)
            if pred_payload is None:
                pred_values = np.full_like(well_log_ai, np.nan)
            else:
                pred = np.asarray(pred_payload["arrays"]["stitched_pred_log_ai"], dtype=np.float64)
                if is_volume:
                    pred_values, _ = sample_volume_trilinear(pred, **sample_kwargs)
                else:
                    pred_values = pred[trace_indices, np.arange(tvdss_m.size)]
            valid = (
                same_axis
                & control.valid_mask[indices]
                & observed_inside
                & synthetic_inside
                & np.isfinite(observed_well)
                & np.isfinite(synthetic_well)
            )
            metrics = diagnostic_metrics(
                observed=observed_well,
                synthetic=synthetic_well,
                valid_mask=valid,
            )
            ai_valid = valid & np.isfinite(well_log_ai) & np.isfinite(pred_values)
            rows.append(
                {
                    "well_name": control.well_name,
                    "model_role": role,
                    "sample_domain": "depth",
                    "sample_unit": "m",
                    "depth_basis": "tvdss",
                    "n_ai_valid": int(np.count_nonzero(ai_valid)),
                    "ai_rmse": float(np.sqrt(np.mean((pred_values[ai_valid] - well_log_ai[ai_valid]) ** 2)))
                    if np.any(ai_valid)
                    else float("nan"),
                    **metrics,
                }
            )
    return pd.DataFrame.from_records(rows)


def _run_depth_diagnostic(
    *,
    run_cfg: dict,
    cfg_path: Path,
    zero_shot_dir: Path,
    zero_shot_summary: Mapping[str, object],
    output_dir: Path,
) -> None:
    predictions = load_zero_shot_predictions(zero_shot_dir)
    first = next(iter(predictions.values()))["arrays"]
    observed = np.asarray(first["seismic_input"], dtype=np.float64)
    lfm = np.asarray(first["lfm_input"], dtype=np.float64)
    valid = np.asarray(first["valid_mask_model"], dtype=bool)
    ilines = np.asarray(first["ilines"], dtype=np.float64)
    xlines = np.asarray(first["xlines"], dtype=np.float64)
    tvdss_m = np.asarray(first["samples"], dtype=np.float64)
    if str(np.asarray(first["sample_domain"]).item()) != "depth":
        raise ValueError("Depth R1 prediction artifact does not declare sample_domain=depth.")
    if str(np.asarray(first["sample_unit"]).item()) != "m" or str(np.asarray(first["depth_basis"]).item()) != "tvdss":
        raise ValueError("Depth R1 prediction artifact must use metre TVDSS samples.")
    for role, payload in predictions.items():
        arrays = payload["arrays"]
        if not np.array_equal(np.asarray(arrays["samples"], dtype=np.float64), tvdss_m):
            raise ValueError(f"R1 model {role} uses a different TVDSS axis.")
    forward_inputs_path, relation, wavelet_time_s, wavelet = _load_depth_forward_inputs(zero_shot_summary)
    impedance_inputs: list[tuple[str, np.ndarray]] = [("lfm_only", lfm)]
    impedance_inputs.extend(
        (f"zero_shot_{role}", np.asarray(payload["arrays"]["stitched_pred_log_ai"], dtype=np.float64))
        for role, payload in predictions.items()
    )
    scan_cfg = dict(run_cfg.get("diagnostic_scan") or {})
    phase_values = [float(value) for value in scan_cfg.get("phase_deg", [-20, -10, 0, 10, 20])]
    time_shift_values = [float(value) for value in scan_cfg.get("wavelet_time_shift_s", [0.0])]
    depth_shift_values = [float(value) for value in scan_cfg.get("depth_static_m", [-10.0, -5.0, 0.0, 5.0, 10.0])]
    wrong_fields = [key for key in ("fractional_shift_samples", "diagnostic_max_hz") if key in scan_cfg or key in run_cfg]
    if wrong_fields:
        raise ValueError(f"Depth R1 rejects time-axis diagnostic fields: {wrong_fields}")
    metrics_rows: list[dict[str, object]] = []
    scan_rows: list[dict[str, object]] = []
    spatial_rows: list[dict[str, object]] = []
    synthetic_by_role: dict[str, np.ndarray] = {}
    figures: dict[str, str] = {}
    synthetic_dir = output_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=False)
    for role, log_ai in impedance_inputs:
        synthetic, forward_valid = _forward_depth_masked(
            log_ai,
            valid,
            tvdss_m=tvdss_m,
            wavelet_time_s=wavelet_time_s,
            wavelet=wavelet,
            relation=relation,
        )
        diagnostic_valid = valid & forward_valid & np.isfinite(observed)
        synthetic_by_role[role] = synthetic
        metrics_rows.append(
            {
                "model_role": role,
                "forward_operator_id": "cup.physics.numpy_backend.forward_depth",
                "sample_domain": "depth",
                "sample_unit": "m",
                "depth_basis": "tvdss",
                **diagnostic_metrics(observed=observed, synthetic=synthetic, valid_mask=diagnostic_valid),
            }
        )
        figures[role] = _plot_forward_qc(
            output_dir=output_dir,
            model_role=role,
            observed=observed,
            synthetic=synthetic,
            display_valid=diagnostic_valid,
            diagnostic_valid=diagnostic_valid,
            sample_domain="depth",
        )
        np.savez_compressed(
            synthetic_dir / f"{role}.npz",
            synthetic_seismic=synthetic.astype(np.float32),
            observed_seismic=observed.astype(np.float32),
            valid_mask=diagnostic_valid,
            samples=tvdss_m,
            sample_domain=np.asarray("depth"),
            sample_unit=np.asarray("m"),
            depth_basis=np.asarray("tvdss"),
        )
        spatial_rows.extend(
            _spatial_rows(
                model_role=role,
                observed=observed,
                synthetic=synthetic,
                valid=diagnostic_valid,
            )
        )
        for phase_deg in phase_values:
            candidate, candidate_valid = _forward_depth_masked(
                log_ai,
                valid,
                tvdss_m=tvdss_m,
                wavelet_time_s=wavelet_time_s,
                wavelet=_phase_rotate_wavelet(wavelet, phase_deg),
                relation=relation,
            )
            scan_rows.append(
                {
                    "model_role": role,
                    "scan_type": "wavelet_phase",
                    "phase_deg": phase_deg,
                    "wavelet_time_shift_s": 0.0,
                    "depth_static_m": 0.0,
                    **diagnostic_metrics(observed=observed, synthetic=candidate, valid_mask=valid & candidate_valid),
                }
            )
        for time_shift_s in time_shift_values:
            candidate, candidate_valid = _forward_depth_masked(
                log_ai,
                valid,
                tvdss_m=tvdss_m,
                wavelet_time_s=wavelet_time_s,
                wavelet=_time_shift_wavelet(wavelet_time_s, wavelet, time_shift_s),
                relation=relation,
            )
            scan_rows.append(
                {
                    "model_role": role,
                    "scan_type": "wavelet_time_shift",
                    "phase_deg": 0.0,
                    "wavelet_time_shift_s": time_shift_s,
                    "depth_static_m": 0.0,
                    **diagnostic_metrics(observed=observed, synthetic=candidate, valid_mask=valid & candidate_valid),
                }
            )
        for depth_static_m in depth_shift_values:
            candidate, candidate_valid = _depth_shift_volume(
                synthetic,
                diagnostic_valid,
                tvdss_m=tvdss_m,
                shift_m=depth_static_m,
            )
            scan_rows.append(
                {
                    "model_role": role,
                    "scan_type": "depth_static",
                    "phase_deg": 0.0,
                    "wavelet_time_shift_s": 0.0,
                    "depth_static_m": depth_static_m,
                    **diagnostic_metrics(observed=observed, synthetic=candidate, valid_mask=valid & candidate_valid),
                }
            )
    metrics_path = output_dir / "forward_diagnostic_metrics.csv"
    scan_path = output_dir / "residual_decomposition.csv"
    spatial_path = output_dir / "spatial_residual_qc.csv"
    well_path = output_dir / "well_forward_diagnostic.csv"
    pd.DataFrame.from_records(metrics_rows).to_csv(metrics_path, index=False)
    pd.DataFrame.from_records(scan_rows).to_csv(scan_path, index=False)
    pd.DataFrame.from_records(spatial_rows).to_csv(spatial_path, index=False)
    _depth_well_qc(
        zero_shot_dir=zero_shot_dir,
        zero_shot_summary=zero_shot_summary,
        observed=observed,
        synthetic_by_role=synthetic_by_role,
        predictions=predictions,
        ilines=ilines,
        xlines=xlines,
        tvdss_m=tvdss_m,
    ).to_csv(well_path, index=False)
    model_manifests = []
    for payload in predictions.values():
        model_dir = resolve_relative_path(str(payload["summary"]["model_run_dir"]), root=REPO_ROOT)
        with (model_dir / "model_run_manifest.json").open("r", encoding="utf-8") as handle:
            model_manifests.append(json.load(handle))
    rock_contracts = [dict(dict(item.get("input_contracts") or {}).get("rock_physics_analysis") or {}) for item in model_manifests]
    if (
        not rock_contracts
        or not str(rock_contracts[0].get("contract_fingerprint_sha256") or "")
        or any(item != rock_contracts[0] for item in rock_contracts[1:])
    ):
        raise ValueError("Depth R1 models do not share one rock-physics contract.")
    recorded_rock_summary = resolve_relative_path(
        str(rock_contracts[0].get("path") or ""), root=REPO_ROOT
    )
    if recorded_rock_summary.parent.resolve() != forward_inputs_path.parent.resolve():
        raise ValueError(
            "Depth R1 forward_model_inputs and model rock-physics contract "
            "come from different runs."
        )
    input_contracts = {
        "real_field_zero_shot": published_contract_reference(
            zero_shot_dir / "real_field_zero_shot_summary.json",
            root=REPO_ROOT,
            label=f"R0 run {zero_shot_dir}",
        ),
        "rock_physics_analysis": rock_contracts[0],
    }
    fingerprint = contract_fingerprint_sha256(
        contract_schema_version=SCHEMA_VERSION,
        semantics={
            "mode": "volume" if observed.ndim == 3 else "section",
            "sample_domain": "depth",
            "sample_unit": "m",
            "depth_basis": "tvdss",
            "forward_operator": "cup.physics.numpy_backend.forward_depth",
        },
        business_config={
            "phase_scan_deg": phase_values,
            "wavelet_time_shift_s": time_shift_values,
            "depth_static_m": depth_shift_values,
        },
        input_contracts=input_contracts,
        primary_artifacts={
            "forward_diagnostic_metrics": metrics_path,
            "well_forward_diagnostic": well_path,
            "residual_decomposition": scan_path,
            "spatial_residual_qc": spatial_path,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
        "contract_fingerprint_sha256": fingerprint,
        "input_contracts": input_contracts,
        "mode": "volume" if observed.ndim == 3 else "section",
        "sample_domain": "depth",
        "sample_unit": "m",
        "depth_basis": "tvdss",
        "axis_contract": {
            "n_sample": int(tvdss_m.size),
            "sample_min": float(tvdss_m[0]),
            "sample_max": float(tvdss_m[-1]),
            "sample_step": float(np.median(np.diff(tvdss_m))),
        },
        "forward_model_inputs": {"path": repo_relative_path(forward_inputs_path, root=REPO_ROOT)},
        "wavelet": {
            "time_unit": "s",
            "sample_count": int(wavelet.size),
            "dt_s": float(np.median(np.diff(wavelet_time_s))),
        },
        "forward_contract": {
            "operator": "cup.physics.numpy_backend.forward_depth",
            "alignment": "observed, synthetic and logAI share the N-point TVDSS axis",
            "wavelet_time_unit": "s",
            "phase_scan_deg": phase_values,
            "wavelet_time_shift_s": time_shift_values,
            "depth_static_m": depth_shift_values,
        },
        "outputs": {
            "forward_diagnostic_metrics": repo_relative_path(metrics_path, root=REPO_ROOT),
            "well_forward_diagnostic": repo_relative_path(well_path, root=REPO_ROOT),
            "residual_decomposition": repo_relative_path(scan_path, root=REPO_ROOT),
            "spatial_residual_qc": repo_relative_path(spatial_path, root=REPO_ROOT),
            "synthetic_dir": repo_relative_path(synthetic_dir, root=REPO_ROOT),
            "figures": figures,
        },
        "red_flags": [],
        "recommended_next_state": "depth_forward_closure_complete",
        "config_file": repo_relative_path(cfg_path, root=REPO_ROOT),
        "zero_shot_dir": repo_relative_path(zero_shot_dir, root=REPO_ROOT),
        "code_version_or_git_commit": str(run_cfg.get("code_version_or_git_commit") or _git_commit()),
    }
    write_json(output_dir / "real_field_forward_diagnostic_summary.json", summary)


def main() -> None:
    args = parse_args()
    cfg_path = resolve_relative_path(args.config, root=REPO_ROOT)
    cfg = load_yaml_config(cfg_path)
    run_cfg = dict(cfg.get("real_field_forward_diagnostic") or {})
    if not run_cfg:
        raise ValueError("experiments/common/common.yaml lacks real_field_forward_diagnostic section.")
    run_cfg["spatial_debias"] = dict(cfg.get("spatial_debias") or run_cfg.get("spatial_debias") or {})
    output_root = resolve_relative_path(cfg.get("output_root", "scripts/output"), root=REPO_ROOT)
    zero_shot_dir = _resolve_zero_shot_dir(run_cfg, output_root=output_root, cli_value=args.zero_shot_dir)
    zero_shot_summary = _load_zero_shot_summary(zero_shot_dir)
    run_cfg["comparisons"] = list(zero_shot_summary.get("comparisons") or [])
    sample_domain = str(zero_shot_summary.get("sample_domain") or "")
    if sample_domain not in {"time", "depth"}:
        raise ValueError("R0 summary must declare sample_domain=time or depth.")
    output_dir = _resolve_output_dir("real_field_forward_diagnostic", args.output_dir, output_root=output_root)
    output_dir.mkdir(parents=True, exist_ok=False)
    if sample_domain == "depth":
        if zero_shot_summary.get("sample_unit") != "m" or zero_shot_summary.get("depth_basis") != "tvdss":
            raise ValueError("Depth R0 summary must declare metre TVDSS samples.")
        _run_depth_diagnostic(
            run_cfg=run_cfg,
            cfg_path=cfg_path,
            zero_shot_dir=zero_shot_dir,
            zero_shot_summary=zero_shot_summary,
            output_dir=output_dir,
        )
        print("=== Real Field Forward Diagnostic ===")
        print(f"Output: {output_dir}")
        print("Domain: depth/TVDSS")
        print("Status: ok")
        return
    run_cfg = _prepare_well_qc_sources(
        run_cfg,
        zero_shot_summary=zero_shot_summary,
        output_root=output_root,
    )
    predictions = load_zero_shot_predictions(zero_shot_dir)
    first = next(iter(predictions.values()))["arrays"]
    observed = np.asarray(first["seismic_input"], dtype=np.float64)
    lfm = np.asarray(first["lfm_input"], dtype=np.float64)
    valid = np.asarray(first["valid_mask_model"], dtype=bool)
    ilines = np.asarray(first["ilines"], dtype=np.float64)
    xlines = np.asarray(first["xlines"], dtype=np.float64)
    output_mode = "volume" if observed.ndim == 3 else "section"
    twt_s = np.asarray(first["samples"], dtype=np.float64)
    dt_s = float(np.median(np.diff(twt_s))) if twt_s.size > 1 else 0.002

    wavelet_dir = _resolve_wavelet_dir(run_cfg, zero_shot_dir=zero_shot_dir, output_root=output_root)
    wavelet_time_s, wavelet, wavelet_meta = load_selected_wavelet(wavelet_dir)
    wavelet_scenarios = _load_wavelet_scenarios(
        wavelet_dir,
        wavelet_time_s,
        wavelet,
        wavelet_meta,
        run_cfg,
    )

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
        synthetic = forward_time(log_ai, wavelet_time_s, wavelet)
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
                "forward_operator_id": "cup.physics.forward_time_n_point_v1",
                "reflectivity_hang_point": "r[j]=tanh((x[j+1]-x[j])/2), event_twt=twt[j+1]",
                "diagnostic_window": "full_valid_mask",
                **metrics,
            }
        )
        figure_outputs[model_role] = _plot_forward_qc(
            output_dir=output_dir,
            model_role=model_role,
            observed=observed,
            synthetic=synthetic,
            display_valid=valid,
            diagnostic_valid=valid_crop,
            sample_domain="time",
        )
        np.savez_compressed(
            synthetic_dir / f"{model_role}.npz",
            synthetic_seismic=synthetic.astype(np.float32),
            observed_seismic_forward_axis=observed.astype(np.float32),
            valid_mask_forward_axis=valid.astype(bool),
            twt_s_forward_axis=twt_s,
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
            scenario_time_s = np.asarray(scenario["wavelet_time_s"], dtype=np.float64)
            scenario_wavelet = np.asarray(scenario["wavelet"], dtype=np.float64)
            scenario_synthetic = forward_time(log_ai, scenario_time_s, scenario_wavelet)
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
                    "diagnostic_window": "full_valid_mask",
                    **scenario_metrics,
                }
            )

    common_forward_valid = valid.astype(bool)
    forward_band_frame, forward_band_figures = _write_forward_band_residual_qc(
        output_dir,
        run_cfg=run_cfg,
        dt_s=dt_s,
        observed_forward=observed,
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
    well_samples_path = output_dir / "well_ai_samples.csv"
    if output_mode == "volume":
        well_frame, well_samples_frame, well_figure_outputs = _build_volume_well_forward_qc(
            run_cfg=run_cfg,
            output_dir=output_dir,
            predictions=predictions,
            synthetic_by_role=synthetic_by_role,
            wavelet_time_s=wavelet_time_s,
            wavelet=wavelet,
            observed=observed,
            lfm=lfm,
            valid=valid,
            ilines=ilines,
            xlines=xlines,
            twt_s=twt_s,
        )
    else:
        well_frame, well_figure_outputs = _build_well_forward_qc(
            run_cfg=run_cfg,
            zero_shot_dir=zero_shot_dir,
            output_dir=output_dir,
            predictions=predictions,
            synthetic_dir=synthetic_dir,
            wavelet_time_s=wavelet_time_s,
            wavelet=wavelet,
        )
        well_samples_frame = pd.DataFrame(
            columns=[
                "well_name",
                "sample_index",
                "twt_s",
                "inline",
                "xline",
                "x_m",
                "y_m",
                "spatial_cluster_id",
                "spatial_cluster_size",
                "filtered_log_ai",
                "lfm_log_ai",
                "model_role",
                "r0_pred_log_ai",
                "r0_valid_mask",
                "r0_blend_weight",
                "well_twt_support_valid",
                "valid_for_fit",
                "valid_reason",
                "sampling_mode",
                "sample_method",
                "wellbore_class",
            ]
        )
    well_frame.to_csv(well_path, index=False)
    well_samples_frame.to_csv(well_samples_path, index=False)
    if output_mode == "volume" and not {"well_name", "model_role"}.issubset(well_frame.columns):
        well_comparison_frame = pd.DataFrame()
        well_band_frame = pd.DataFrame()
        (output_dir / "well_ai_comparison_summary.csv").write_text("", encoding="utf-8")
        (output_dir / "well_ai_band_comparison.csv").write_text("", encoding="utf-8")
        well_closure_figures = {}
    else:
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
        else "future_sparse_well_adapter_candidate"
    )

    input_contracts = {
        "real_field_zero_shot": published_contract_reference(
            zero_shot_dir / "real_field_zero_shot_summary.json",
            root=REPO_ROOT,
            label=f"R0 run {zero_shot_dir}",
        ),
        "wavelet": published_contract_reference(
            wavelet_dir / "run_summary.json",
            root=REPO_ROOT,
            label=f"wavelet run {wavelet_dir}",
        ),
    }
    contract_fingerprint = contract_fingerprint_sha256(
        contract_schema_version=SCHEMA_VERSION,
        semantics={"mode": output_mode, "dt_s": dt_s, "forward_operator": "cup.physics.numpy_backend.forward_time"},
        business_config={
            "phase_scan_deg": phase_values,
            "fractional_shift_scan_samples": shift_values,
            "diagnostic_max_hz": run_cfg.get("diagnostic_max_hz"),
        },
        input_contracts=input_contracts,
        primary_artifacts={
            "forward_diagnostic_metrics": metrics_path,
            "well_forward_diagnostic": well_path,
            "residual_decomposition": decomposition_path,
            "wavelet_sensitivity": wavelet_sensitivity_path,
            "spatial_residual_qc": spatial_path,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "contract_fingerprint_schema": CONTRACT_FINGERPRINT_SCHEMA,
        "contract_fingerprint_sha256": contract_fingerprint,
        "input_contracts": input_contracts,
        "mode": output_mode,
        "sample_domain": "time",
        "sample_unit": "s",
        "depth_basis": None,
        "config_file": repo_relative_path(cfg_path, root=REPO_ROOT),
        "zero_shot_dir": repo_relative_path(zero_shot_dir, root=REPO_ROOT),
        "wavelet": wavelet_meta,
        "forward_contract": {
            "reflectivity": "r[j] = tanh((logAI[j+1] - logAI[j]) / 2)",
            "convolution": "cup.physics.numpy_backend.forward_time",
            "observed_alignment": "observed and synthetic share the N-point twt_s axis",
            "time_alignment_mode": "explicit_lower_interface_N_point_operator",
            "discarded_samples_for_forward_axis": 0,
            "synthetic_twt_axis": "twt_s on full valid mask",
            "observed_twt_axis_after_alignment": "twt_s on full valid mask",
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
            "well_ai_samples": repo_relative_path(well_samples_path, root=REPO_ROOT),
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
        "code_version_or_git_commit": str(run_cfg.get("code_version_or_git_commit") or _git_commit()),
    }
    write_json(output_dir / "real_field_forward_diagnostic_summary.json", summary)
    print("=== Real Field Forward Diagnostic ===")
    print(f"Output: {output_dir}")
    print(f"Inputs: {len(impedance_inputs)}")
    print("Status: ok")


if __name__ == "__main__":
    main()
