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

from cup.utils.io import load_yaml_config, repo_relative_path, resolve_relative_path, sha256_file, write_json
from ginn_v2.real_field import (
    diagnostic_metrics,
    forward_log_ai,
    load_selected_wavelet,
    load_zero_shot_predictions,
    phase_shift_scan,
)


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


def _erode_forward_arrays(
    observed: np.ndarray,
    synthetic: np.ndarray,
    valid: np.ndarray,
    *,
    erosion_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.asarray(observed, dtype=np.float64)[:, 1:]
    syn = np.asarray(synthetic, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)[:, 1:]
    if obs.shape != syn.shape or mask.shape != syn.shape:
        raise ValueError(f"Forward diagnostic shape mismatch: obs={obs.shape}, syn={syn.shape}, mask={mask.shape}")
    eroded = _erode_valid_runs(mask, int(erosion_samples))
    if not np.any(eroded):
        raise ValueError("forward_diagnostic_erosion_s removes all valid forward samples.")
    return obs, syn, eroded


def _erode_valid_runs(mask: np.ndarray, erosion_samples: int) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool)
    if erosion_samples <= 0:
        return valid.copy()
    out = np.zeros_like(valid, dtype=bool)
    for row_idx in range(valid.shape[0]):
        row = valid[row_idx]
        changes = np.diff(np.r_[False, row, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        for start, stop in zip(starts, stops):
            inner_start = int(start + erosion_samples)
            inner_stop = int(stop - erosion_samples)
            if inner_stop > inner_start:
                out[row_idx, inner_start:inner_stop] = True
    return out


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
    section_ilines: np.ndarray,
    section_xlines: np.ndarray,
) -> tuple[int, float]:
    distances = np.hypot(section_ilines - float(well_inline), section_xlines - float(well_xline))
    index = int(np.nanargmin(distances))
    return index, float(distances[index])


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
    max_line_distance = float(cfg.get("max_line_distance", 25.0))
    inventory_positions = _load_well_positions(cfg)
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
        position_source = "well_tie_metrics"
        if (not np.isfinite(well_inline) or not np.isfinite(well_xline)) and well_name in inventory_positions:
            well_inline, well_xline = inventory_positions[well_name]
            position_source = "well_inventory"
        base = {
            "well_name": well_name,
            "well_inline": well_inline if np.isfinite(well_inline) else float("nan"),
            "well_xline": well_xline if np.isfinite(well_xline) else float("nan"),
            "well_position_source": position_source,
            "tie_window_start_s": tie.get("tie_window_start_s", ""),
            "tie_window_end_s": tie.get("tie_window_end_s", ""),
        }
        if not np.isfinite(well_inline) or not np.isfinite(well_xline):
            rows.append({**base, "model_role": "", "status": "skipped_missing_well_position"})
            continue
        trace_idx, distance = _trace_distance_to_section(
            well_inline=float(well_inline),
            well_xline=float(well_xline),
            section_ilines=section_ilines,
            section_xlines=section_xlines,
        )
        base.update(
            {
                "nearest_section_trace": trace_idx,
                "nearest_section_inline": float(section_ilines[trace_idx]),
                "nearest_section_xline": float(section_xlines[trace_idx]),
                "section_line_distance": distance,
            }
        )
        if distance > max_line_distance:
            rows.append(
                {
                    **base,
                    "model_role": "",
                    "status": "skipped_outside_section_support",
                    "max_line_distance": max_line_distance,
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
            synthetic_path = synthetic_dir / f"zero_shot_{role}.npz"
            waveform_metrics: dict[str, object] = {"waveform_status": "synthetic_not_found"}
            if synthetic_path.is_file():
                syn_arrays = np.load(synthetic_path, allow_pickle=True)
                observed = np.asarray(syn_arrays["observed_seismic_forward_axis"], dtype=np.float64)[trace_idx : trace_idx + 1]
                synthetic = np.asarray(syn_arrays["synthetic_seismic"], dtype=np.float64)[trace_idx : trace_idx + 1]
                valid_display = np.asarray(syn_arrays["valid_mask_forward_axis"], dtype=bool)[trace_idx : trace_idx + 1]
                valid_key = "valid_mask_forward_eroded_axis" if "valid_mask_forward_eroded_axis" in syn_arrays.files else "valid_mask_forward_axis"
                valid_forward = np.asarray(syn_arrays[valid_key], dtype=bool)[trace_idx : trace_idx + 1]
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
            figure_path = figures / f"well_forward_qc_{well_name}_{role}.png"
            _plot_well_qc(
                path=figure_path,
                well_name=well_name,
                model_role=role,
                twt_model=twt_model,
                well_log_ai=well_log_ai,
                pred_log_ai=pred,
                trace_idx=trace_idx,
                synthetic_dir=synthetic_dir,
            )
            figure_outputs[f"{well_name}_{role}"] = repo_relative_path(figure_path, root=REPO_ROOT)
    return pd.DataFrame.from_records(rows), figure_outputs


def _plot_well_qc(
    *,
    path: Path,
    well_name: str,
    model_role: str,
    twt_model: np.ndarray,
    well_log_ai: np.ndarray,
    pred_log_ai: np.ndarray,
    trace_idx: int,
    synthetic_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    well_ai = np.exp(well_log_ai)
    pred_ai = np.exp(pred_log_ai)
    valid_ai = np.isfinite(well_ai)
    if np.any(valid_ai):
        axes[0].plot(well_ai, twt_model, label="filtered AI", color="0.55", linewidth=1.2)
    axes[0].plot(pred_ai, twt_model, label=f"{model_role} AI", color="red", linewidth=1.4)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("AI")
    axes[0].set_ylabel("TWT s")
    axes[0].set_title("AI")
    axes[0].legend(fontsize=8)

    pred_filled = _fill_nonfinite_1d(pred_log_ai)
    reflectivity = np.tanh(0.5 * (pred_filled[1:] - pred_filled[:-1]))
    twt_forward = twt_model[1:]
    axes[1].plot(reflectivity, twt_forward, color="red", linewidth=0.9)
    axes[1].axvline(0.0, color="black", linewidth=0.6)
    axes[1].set_xlabel("reflectivity")
    axes[1].set_title("Reflectivity")

    synthetic_path = synthetic_dir / f"zero_shot_{model_role}.npz"
    if synthetic_path.is_file():
        arrays = np.load(synthetic_path, allow_pickle=True)
        twt = arrays["twt_s_forward_axis"]
        observed = arrays["observed_seismic_forward_axis"][trace_idx]
        synthetic = arrays["synthetic_seismic"][trace_idx]
        valid_display = np.asarray(arrays["valid_mask_forward_axis"], dtype=bool)[trace_idx]
        valid_key = "valid_mask_forward_eroded_axis" if "valid_mask_forward_eroded_axis" in arrays.files else "valid_mask_forward_axis"
        valid_diagnostic = np.asarray(arrays[valid_key], dtype=bool)[trace_idx]
        observed_plot = np.where(valid_display, observed, np.nan)
        synthetic_plot = np.where(valid_display, synthetic, np.nan)
        residual_plot = np.where(valid_display, observed - synthetic, np.nan)
        axes[2].plot(observed_plot, twt, color="black", label="observed", linewidth=1.0)
        axes[2].plot(synthetic_plot, twt, color="red", label="synthetic", linewidth=1.0)
        axes[3].plot(residual_plot, twt, color="red", label="observed - synthetic", linewidth=1.0)
        diagnostic_n = int(np.count_nonzero(valid_diagnostic & np.isfinite(observed) & np.isfinite(synthetic)))
        if diagnostic_n < 8:
            axes[3].text(
                0.02,
                0.92,
                f"diagnostic n={diagnostic_n}",
                transform=axes[3].transAxes,
                fontsize=8,
            )
    axes[2].set_xlabel("seismic")
    axes[2].set_title("Synthetic")
    axes[2].legend(fontsize=8)
    axes[3].axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[3].set_xlabel("residual")
    axes[3].set_title("Residual")
    axes[3].legend(fontsize=8)
    fig.suptitle(f"{well_name} R1 well QC: {model_role}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _fill_nonfinite_1d(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(out)
    if np.all(finite):
        return out
    if not np.any(finite):
        return np.zeros_like(out, dtype=np.float64)
    coords = np.arange(out.size, dtype=np.float64)
    return np.interp(coords, coords[finite], out[finite])


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
    diagnostic_mask = np.asarray(diagnostic_valid, dtype=bool)
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
        if np.any(diagnostic_mask):
            ax.contour(diagnostic_mask.T.astype(float), levels=[0.5], colors="black", linewidths=0.4)
        ax.set_title(title)
        ax.set_xlabel("lateral trace")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("forward TWT sample")
    fig.suptitle(f"R1 forward QC: {model_role} (black outline = eroded diagnostic mask)")
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
            "forward_diagnostic_crop_s is retired; use forward_diagnostic_erosion_s "
            "or omit it to use the selected wavelet active half-support."
        )
    erosion_source = "configured"
    if boundary.get("forward_diagnostic_erosion_s") is not None:
        erosion_s = float(boundary.get("forward_diagnostic_erosion_s") or 0.0)
    else:
        erosion_s = float(wavelet_meta.get("active_half_support_s", 0.0) or 0.0)
        erosion_source = "wavelet_active_half_support"
    erosion_samples = int(np.ceil(erosion_s / dt_s)) if erosion_s > 0.0 else 0

    impedance_inputs: list[tuple[str, np.ndarray, str]] = [("lfm_only", lfm, "lfm_input")]
    for role, payload in predictions.items():
        arrays = payload["arrays"]
        impedance_inputs.append((f"zero_shot_{role}", np.asarray(arrays["stitched_pred_log_ai"], dtype=np.float64), role))

    metric_rows = []
    wavelet_rows = []
    scan_frames = []
    spatial_rows = []
    figure_outputs: dict[str, str] = {}
    synthetic_dir = output_dir / "synthetic"
    synthetic_dir.mkdir(parents=True, exist_ok=False)
    for model_role, log_ai, source_role in impedance_inputs:
        synthetic = forward_log_ai(log_ai, wavelet)
        obs_crop, syn_crop, valid_crop = _erode_forward_arrays(
            observed,
            synthetic,
            valid,
            erosion_samples=erosion_samples,
        )
        metrics = diagnostic_metrics(observed=obs_crop, synthetic=syn_crop, valid_mask=valid_crop)
        metric_rows.append(
            {
                "model_role": model_role,
                "source_role": source_role,
                "forward_operator_id": "real_field_log_ai_tanh_reflectivity_nominal_wavelet_v1",
                "reflectivity_hang_point": "r[j]=tanh((x[j]-x[j-1])/2), aligned to observed[:, 1:]",
                "erosion_samples_each_valid_run_side": erosion_samples,
                "erosion_s_each_valid_run_side": erosion_s,
                "erosion_source": erosion_source,
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
            valid_mask_forward_eroded_axis=valid_crop.astype(bool),
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
            obs_wave, syn_wave, valid_wave = _erode_forward_arrays(
                observed,
                scenario_synthetic,
                valid,
                erosion_samples=erosion_samples,
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
                    "erosion_samples_each_valid_run_side": erosion_samples,
                    "erosion_s_each_valid_run_side": erosion_s,
                    "erosion_source": erosion_source,
                    **scenario_metrics,
                }
            )

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
    )
    well_frame.to_csv(well_path, index=False)
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
            "synthetic_twt_axis": "twt_s[1:] before valid-run erosion",
            "observed_twt_axis_after_alignment": "twt_s[1:] before valid-run erosion",
            "dt_s": dt_s,
            "forward_diagnostic_erosion_s": erosion_s,
            "forward_diagnostic_erosion_samples": erosion_samples,
            "forward_diagnostic_erosion_source": erosion_source,
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
