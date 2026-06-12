"""Build time-domain point-control AI low-frequency model.

Usage::

    python scripts/lfm_precomputed.py
    python scripts/lfm_precomputed.py --config experiments/common.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cup.time_config import TimeWorkflowConfig
from cup.utils.config import deep_merge_dict
from cup.utils.io import (
    build_segy_textual_header,
    latest_run,
    load_yaml_config,
    repo_relative_path,
    resolve_artifact_path,
    resolve_relative_path,
    sanitize_filename,
    to_json_compatible,
    write_json,
)
from cup.seismic.geometry import nearest_sample_indices, validate_sample_indices
from cup.seismic.viz import (
    impedance_qc_metrics,
    plot_well_waveform_qc,
    sample_volume_at_points,
    waveform_qc_metrics,
)
from cup.well.assets import normalize_well_name
from cup.well.tie import load_saved_seismic_trace_csv
from cup.seismic.wavelet import compute_wavelet_active_half_support_s, load_wavelet_csv, validate_wavelet_dt
from wtie.modeling.modeling import ConvModeler
from wtie.optimize.similarity import dynamic_normalized_xcorr, normalized_xcorr
from wtie.processing import grid

matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 120


DEFAULT_CONFIG: dict[str, Any] = {
    "source_runs": {
        "well_constraints_dir": None,
    },
    "target_interval": {
        "horizons": ["interpre/H3-1", "interpre/H7-1"],
        "twt_unit": "auto",
    },
    "modeling": {
        "boundary_extension_samples": 50,
        "n_slices": 20,
        "variogram": "spherical",
        "exact": True,
        "nugget": 0.0,
        "post_slice_smoothing": False,
    },
    "export": {"export_volume": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/common.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
    except RuntimeError:
        pass
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def _resolve_source_dirs(script_cfg: dict[str, Any], output_root: Path) -> dict[str, Path]:
    source_cfg = dict(script_cfg.get("source_runs") or {})
    constraints_dir_value = source_cfg.get("well_constraints_dir")
    constraints_dir = (
        latest_run(output_root, "well_constraints", "lfm_control_points.csv")
        if constraints_dir_value in {None, ""}
        else resolve_relative_path(constraints_dir_value, root=REPO_ROOT)
    )
    return {"well_constraints_dir": constraints_dir}


def _resolve_segy_options(seismic_cfg: dict[str, Any]) -> dict[str, int] | None:
    from cup.seismic.survey import segy_options_from_config

    return segy_options_from_config(seismic_cfg) or None


def _open_survey(seismic_cfg: dict[str, Any], data_root: Path) -> tuple[Any, Path, str]:
    from cup.seismic.survey import open_survey

    seismic_file = resolve_relative_path(seismic_cfg.get("file"), root=data_root)
    seismic_type = str(seismic_cfg.get("type", "segy"))
    survey = open_survey(seismic_file, seismic_type=seismic_type, segy_options=_resolve_segy_options(seismic_cfg))
    return survey, seismic_file, seismic_type


def _normalize_horizon_twt_df(df: pd.DataFrame, *, unit: str) -> pd.DataFrame:
    unit_norm = str(unit or "auto").strip().casefold()
    if "interpretation" not in df.columns:
        return df.copy()
    out = df.copy()
    values = out["interpretation"].to_numpy(dtype=float, copy=True)
    finite = np.isfinite(values)
    if unit_norm in {"ms", "msec", "millisecond", "milliseconds"}:
        values[finite] = np.abs(values[finite]) / 1000.0
    elif unit_norm in {"s", "sec", "second", "seconds"}:
        values[finite] = np.abs(values[finite])
    elif unit_norm == "auto":
        if np.any(finite):
            abs_values = np.abs(values[finite])
            values[finite] = abs_values / 1000.0 if float(np.nanmax(abs_values)) > 20.0 else abs_values
    else:
        raise ValueError(f"Unsupported target_interval.twt_unit: {unit}")
    out["interpretation"] = values
    return out


def _build_target_layer(script_cfg: dict[str, Any], geometry: dict[str, Any], qc_dir: Path, data_root: Path) -> tuple[Any, list[dict[str, Any]]]:
    from cup.petrel.load import import_interpretation_petrel
    from cup.seismic.target_zone import TargetZone

    target_cfg = dict(script_cfg["target_interval"])
    horizon_values = target_cfg.get("horizons")
    if not isinstance(horizon_values, list) or len(horizon_values) < 2:
        raise ValueError("lfm_precomputed.target_interval.horizons must contain at least two horizon files.")
    horizon_files = [resolve_relative_path(value, root=data_root) for value in horizon_values]
    twt_unit = str(target_cfg.get("twt_unit", "auto"))
    raw_entries: list[tuple[float, str, Path, pd.DataFrame]] = []
    for index, horizon_file in enumerate(horizon_files):
        horizon_df = _normalize_horizon_twt_df(import_interpretation_petrel(horizon_file), unit=twt_unit)
        values = horizon_df["interpretation"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"Horizon contains no finite TWT values: {horizon_file}")
        raw_entries.append((float(np.mean(finite)), f"horizon_{index}", horizon_file, horizon_df))
    raw_entries.sort(key=lambda item: item[0])
    raw_horizons = {name: horizon_df for _, name, _, horizon_df in raw_entries}
    target_layer = TargetZone(
        raw_horizon_dfs=raw_horizons,
        geometry=geometry,
        horizon_names=list(raw_horizons.keys()),
        qc_output_dir=qc_dir,
        min_thickness=target_cfg.get("min_thickness"),
        nearest_distance_limit=target_cfg.get("nearest_distance_limit"),
        outlier_threshold=target_cfg.get("outlier_threshold"),
        outlier_min_neighbor_count=target_cfg.get("outlier_min_neighbor_count", 2),
    )
    horizons = []
    for mean_twt, name, path, _horizon_df in raw_entries:
        grid = target_layer.get_horizon_grid(name)
        horizons.append(
            {
                "file": repo_relative_path(path, root=REPO_ROOT),
                "mean_twt_s": float(np.nanmean(grid)),
                "input_mean_twt_s": float(mean_twt),
            }
        )
    return target_layer, horizons


def _to_control_points(rows: pd.DataFrame) -> list[Any]:
    from cup.seismic.lfm_time import LfmTimeControlPoint

    points = []
    for _, row in rows.iterrows():
        points.append(
            LfmTimeControlPoint(
                well_name=str(row["well_name"]),
                route=str(row["route"]),
                twt_s=float(row["twt_s"]),
                md_m=float(row["md_m"]),
                x_m=float(row["x_m"]),
                y_m=float(row["y_m"]),
                inline_float=float(row["inline_float"]),
                xline_float=float(row["xline_float"]),
                zone_name=str(row["zone_name"]),
                u_in_zone=float(row["u_in_zone"]),
                ai=float(row["ai"]),
                weight=float(row.get("weight", 1.0)),
                source=str(row.get("source", "")),
                flat_idx=None if pd.isna(row.get("flat_idx")) else int(row.get("flat_idx")),
            )
        )
    return points


def _plot_controls(control_df: pd.DataFrame, output_path: Path) -> None:
    if control_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 7.0), constrained_layout=True)
    scatter = ax.scatter(
        control_df["inline_float"],
        control_df["xline_float"],
        c=control_df["twt_s"],
        s=8,
        alpha=0.65,
        cmap="viridis",
    )
    ax.set_title("LFM layer control points")
    ax.set_xlabel("Inline")
    ax.set_ylabel("Xline")
    fig.colorbar(scatter, ax=ax, label="TWT (s)")
    _save_fig(output_path)


def _plot_lfm_result(result: Any, output_path: Path) -> None:
    ilines = result.ilines
    xlines = result.xlines
    samples = result.samples
    volume = result.volume
    i_il = len(ilines) // 2
    i_xl = len(xlines) // 2
    i_t = len(samples) // 2
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    im0 = axes[0].imshow(
        volume[i_il, :, :].T,
        aspect="auto",
        origin="upper",
        extent=[xlines[0], xlines[-1], samples[-1], samples[0]],
        cmap="viridis",
    )
    axes[0].set_title(f"AI LFM inline @ {ilines[i_il]:.0f}")
    axes[0].set_xlabel("Xline")
    axes[0].set_ylabel("TWT (s)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = axes[1].imshow(
        volume[:, i_xl, :].T,
        aspect="auto",
        origin="upper",
        extent=[ilines[0], ilines[-1], samples[-1], samples[0]],
        cmap="viridis",
    )
    axes[1].set_title(f"AI LFM xline @ {xlines[i_xl]:.0f}")
    axes[1].set_xlabel("Inline")
    axes[1].set_ylabel("TWT (s)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    im2 = axes[2].imshow(
        volume[:, :, i_t].T,
        aspect="auto",
        origin="lower",
        extent=[ilines[0], ilines[-1], xlines[0], xlines[-1]],
        cmap="viridis",
    )
    axes[2].set_title(f"AI LFM time slice @ {samples[i_t]:.3f} s")
    axes[2].set_xlabel("Inline")
    axes[2].set_ylabel("Xline")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)
    _save_fig(output_path)


def _reflectivity_from_ai(ai: np.ndarray) -> np.ndarray:
    values = np.asarray(ai, dtype=np.float64)
    out = np.zeros(values.shape, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0.0)
    pair_valid = valid[:-1] & valid[1:]
    upper = values[:-1][pair_valid]
    lower = values[1:][pair_valid]
    out[np.flatnonzero(pair_valid) + 1] = (lower - upper) / np.maximum(lower + upper, 1e-12)
    return out


def _resolve_lfm_qc_sources(constraints_summary: dict[str, Any]) -> tuple[Path, Path, pd.DataFrame]:
    source_dirs = dict(constraints_summary.get("source_dirs") or {})
    auto_value = source_dirs.get("well_auto_tie_dir")
    wavelet_value = source_dirs.get("wavelet_generation_dir")
    if not auto_value or not wavelet_value:
        raise ValueError("Sixth-step run_summary.json is missing well_auto_tie_dir or wavelet_generation_dir.")
    auto_dir = resolve_relative_path(auto_value, root=REPO_ROOT)
    wavelet_dir = resolve_relative_path(wavelet_value, root=REPO_ROOT)
    metrics_path = auto_dir / "well_tie_metrics.csv"
    wavelet_path = wavelet_dir / "selected_wavelet.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing fourth-step well_tie_metrics.csv: {metrics_path}")
    if not wavelet_path.exists():
        raise FileNotFoundError(f"Missing fifth-step selected_wavelet.csv: {wavelet_path}")
    return auto_dir, wavelet_path, pd.read_csv(metrics_path)


def _metric_row_for_well(metrics_df: pd.DataFrame, well_name: str) -> pd.Series:
    key = normalize_well_name(well_name)
    matches = metrics_df.loc[
        metrics_df["well_name"].map(lambda value: normalize_well_name(str(value))) == key
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one fourth-step metric row for {well_name!r}, found {len(matches)}.")
    row = matches.iloc[0]
    if str(row.get("tie_status", "")) != "success":
        raise ValueError(f"Fourth-step tie_status is not success for {well_name!r}.")
    return row


def _trajectory_coordinates(
    metric: pd.Series,
    *,
    auto_dir: Path,
    well_name: str,
    twt_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    plan_path = resolve_artifact_path(
        metric.get("optimized_trace_sample_plan_file"),
        root=REPO_ROOT,
        run_dir=auto_dir,
    )
    if plan_path is None or not plan_path.exists():
        raise FileNotFoundError(f"Missing optimized trajectory sample plan for deviated well {well_name!r}.")
    plan = pd.read_csv(plan_path)
    required = {"twt_s", "inline_float", "xline_float", "survey_position"}
    missing = required - set(plan.columns)
    if missing:
        raise ValueError(f"Trajectory sample plan is missing columns {sorted(missing)}: {plan_path}")
    plan = plan.loc[plan["survey_position"].astype(str).eq("inside")].sort_values("twt_s")
    if len(plan) < 2:
        raise ValueError(f"Trajectory sample plan has fewer than two inside samples: {plan_path}")
    plan_twt = plan["twt_s"].to_numpy(dtype=np.float64)
    if twt_s[0] < plan_twt[0] - 1e-9 or twt_s[-1] > plan_twt[-1] + 1e-9:
        raise ValueError("Requested LFM QC window extends outside the optimized trajectory sample plan.")
    return (
        np.interp(twt_s, plan_twt, plan["inline_float"].to_numpy(dtype=np.float64)),
        np.interp(twt_s, plan_twt, plan["xline_float"].to_numpy(dtype=np.float64)),
    )


def _lfm_qc_coordinates(
    group: pd.DataFrame,
    metric: pd.Series,
    *,
    auto_dir: Path,
    well_name: str,
    twt_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sources = set(group["source"].astype(str))
    if sources == {"deviated_trajectory"}:
        return _trajectory_coordinates(
            metric,
            auto_dir=auto_dir,
            well_name=well_name,
            twt_s=twt_s,
        )
    if sources == {"vertical_trace"}:
        return (
            np.full(twt_s.size, float(group["inline_float"].median())),
            np.full(twt_s.size, float(group["xline_float"].median())),
        )
    raise ValueError(f"Unsupported mixed LFM QC sources for {well_name!r}: {sorted(sources)}")


def _write_lfm_well_qc(
    control_df: pd.DataFrame,
    result: Any,
    output_dir: Path,
    *,
    constraints_summary: dict[str, Any],
) -> dict[str, Any]:
    well_qc_dir = output_dir / "well_qc"
    trace_dir = well_qc_dir / "traces"
    figure_dir = well_qc_dir / "figures"
    trace_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    qc_df = control_df.copy()
    sample_axis = np.asarray(result.samples, dtype=np.float64)
    qc_df["seismic_sample_index_used"] = validate_sample_indices(
        sample_axis,
        qc_df["twt_s"].to_numpy(dtype=np.float64),
        qc_df["seismic_sample_index"].to_numpy(dtype=np.float64),
        field_name="seismic_sample_index",
    )
    qc_df["lfm_ai_sampled"] = sample_volume_at_points(
        result.volume,
        result.geometry,
        qc_df["inline_float"].to_numpy(dtype=np.float64),
        qc_df["xline_float"].to_numpy(dtype=np.float64),
        qc_df["seismic_sample_index_used"].to_numpy(dtype=np.int64),
    )
    qc_df["diff_lfm_ai_minus_control"] = (
        qc_df["lfm_ai_sampled"].to_numpy(dtype=np.float64)
        - qc_df["ai"].to_numpy(dtype=np.float64)
    )

    metrics_rows: list[dict[str, Any]] = []
    try:
        auto_dir, wavelet_path, tie_metrics = _resolve_lfm_qc_sources(constraints_summary)
        wavelet_time_s, wavelet_values = load_wavelet_csv(wavelet_path)
        validate_wavelet_dt(wavelet_time_s, float(np.median(np.diff(sample_axis))))
        half_support_s = compute_wavelet_active_half_support_s(wavelet_time_s, wavelet_values)
        wavelet = grid.Wavelet(wavelet_values, wavelet_time_s, name=wavelet_path.stem)
        modeler = ConvModeler()
    except Exception as exc:
        for well_name, group in qc_df.groupby("well_name", sort=True):
            metrics_rows.append(
                {
                    "well_name": str(well_name),
                    "status": "failed",
                    "n_control_points": int(len(group)),
                    "source_values": ";".join(sorted(set(group["source"].astype(str)))),
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
        metrics_df = pd.DataFrame.from_records(metrics_rows)
        metrics_path = well_qc_dir / "well_qc_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
        return {
            "well_qc_dir": repo_relative_path(well_qc_dir, root=REPO_ROOT),
            "metrics": repo_relative_path(metrics_path, root=REPO_ROOT),
            "n_wells_qc": 0,
        }

    for well_name, group in qc_df.groupby("well_name", sort=True):
        group = group.sort_values(["twt_s", "md_m", "seismic_sample_index"]).reset_index(drop=True)
        safe = sanitize_filename(str(well_name))
        trace_path = trace_dir / f"well_qc_{safe}.csv"
        figure_path = figure_dir / f"well_qc_{safe}.png"
        base_row = {
            "well_name": str(well_name),
            "n_control_points": int(len(group)),
            "source_values": ";".join(
                sorted({str(value) for value in group.get("source", pd.Series(dtype=str)).dropna().unique()})
            ),
            "wavelet_file": repo_relative_path(wavelet_path, root=REPO_ROOT),
        }
        try:
            metric = _metric_row_for_well(tie_metrics, str(well_name))
            seismic_path = resolve_artifact_path(
                metric.get("seismic_trace_file"),
                root=REPO_ROOT,
                run_dir=auto_dir,
            )
            if seismic_path is None or not seismic_path.exists():
                raise FileNotFoundError(f"Missing fourth-step seismic trace for {well_name!r}.")
            seismic = load_saved_seismic_trace_csv(seismic_path)
            display_start = max(
                float(seismic.basis[0]),
                float(group["twt_s"].min()) - half_support_s,
            )
            display_end = min(
                float(seismic.basis[-1]),
                float(group["twt_s"].max()) + half_support_s,
            )
            display_mask = (seismic.basis >= display_start) & (seismic.basis <= display_end)
            display_twt = np.asarray(seismic.basis[display_mask], dtype=np.float64)
            seismic_raw = np.asarray(seismic.values[display_mask], dtype=np.float64)
            if display_twt.size < 8:
                raise ValueError(f"Too few samples in LFM QC display window: {display_twt.size}.")

            inline_values, xline_values = _lfm_qc_coordinates(
                group,
                metric,
                auto_dir=auto_dir,
                well_name=str(well_name),
                twt_s=display_twt,
            )

            display_indices = nearest_sample_indices(sample_axis, display_twt)
            lfm_ai_values = sample_volume_at_points(
                result.volume,
                result.geometry,
                inline_values,
                xline_values,
                display_indices,
            )
            if np.any(~np.isfinite(lfm_ai_values)) or np.any(lfm_ai_values <= 0.0):
                raise ValueError("LFM sampled AI contains non-finite or non-positive values in the QC window.")

            group_twt = group["twt_s"].to_numpy(dtype=np.float64)
            control_values = np.interp(
                display_twt,
                group_twt,
                group["ai"].to_numpy(dtype=np.float64),
                left=np.nan,
                right=np.nan,
            )
            reflectivity_values = _reflectivity_from_ai(lfm_ai_values)
            synthetic_raw = np.asarray(modeler(wavelet.values, reflectivity_values), dtype=np.float64)
            target_mask = (
                (display_twt >= float(group_twt.min()))
                & (display_twt <= float(group_twt.max()))
                & np.isfinite(seismic_raw)
                & np.isfinite(synthetic_raw)
            )
            if int(np.count_nonzero(target_mask)) < 8:
                raise ValueError("Too few valid target-window samples for LFM waveform QC.")
            observed_mean = float(np.mean(seismic_raw[target_mask]))
            observed_std = float(np.std(seismic_raw[target_mask]))
            if not np.isfinite(observed_std) or observed_std <= 0.0:
                raise ValueError("Observed seismic has zero standard deviation in the LFM QC target window.")
            observed = (seismic_raw - observed_mean) / observed_std
            denominator = max(float(np.dot(synthetic_raw[target_mask], synthetic_raw[target_mask])), 1e-12)
            scale = float(np.dot(observed[target_mask], synthetic_raw[target_mask]) / denominator)
            synthetic = scale * synthetic_raw

            synthetic_trace = grid.Seismic(synthetic, display_twt, "twt", name="LFM synthetic")
            observed_trace = grid.Seismic(observed, display_twt, "twt", name="Seismic normalized")
            xcorr_values = normalized_xcorr(observed_trace.values, synthetic_trace.values)
            xcorr_basis = synthetic_trace.sampling_rate * np.arange(
                -(synthetic_trace.size - 1),
                synthetic_trace.size,
            )
            xcorr = grid.XCorr(xcorr_values, xcorr_basis, "tlag", name="XCorr")
            dxcorr = dynamic_normalized_xcorr(observed_trace, synthetic_trace)
            control_trace = grid.Log(control_values, display_twt, "twt", name="Low-frequency control AI")
            sampled_trace = grid.Log(lfm_ai_values, display_twt, "twt", name="LFM sampled AI")
            reflectivity_trace = grid.Reflectivity(
                reflectivity_values,
                display_twt,
                "twt",
                name="LFM reflectivity",
            )

            trace_df = pd.DataFrame(
                {
                    "twt_s": display_twt,
                    "lfm_ai": control_values,
                    "lfm_ai_sampled": lfm_ai_values,
                    "reflectivity_lfm": reflectivity_values,
                    "seismic_raw": seismic_raw,
                    "seismic_normalized": observed,
                    "synthetic_unscaled": synthetic_raw,
                    "synthetic_scaled": synthetic,
                    "residual": observed - synthetic,
                    "valid_for_metrics": target_mask,
                    "inline_float": inline_values,
                    "xline_float": xline_values,
                }
            )
            trace_df.to_csv(trace_path, index=False, encoding="utf-8-sig")

            fig, _axes = plot_well_waveform_qc(
                [control_trace, sampled_trace],
                reflectivity_trace,
                synthetic_trace,
                observed_trace,
                xcorr,
                dxcorr,
                synthetic_ai=sampled_trace,
                figsize=(12.0, 7.5),
            )
            fig.suptitle(
                f"LFM well QC | {well_name} | corr={waveform_qc_metrics(observed, synthetic, target_mask)['corr']:.3f}"
            )
            fig.savefig(figure_path, dpi=180, bbox_inches="tight")
            plt.close(fig)

            ai_mask = (
                np.isfinite(group["ai"].to_numpy(dtype=np.float64))
                & np.isfinite(group["lfm_ai_sampled"].to_numpy(dtype=np.float64))
            )
            control_metric_trace = grid.Log(
                group["ai"].to_numpy(dtype=np.float64),
                group_twt,
                "twt",
                name="Low-frequency control AI",
            )
            sampled_metric_trace = grid.Log(
                group["lfm_ai_sampled"].to_numpy(dtype=np.float64),
                group_twt,
                "twt",
                name="LFM sampled AI",
            )
            metrics_rows.append(
                {
                    **base_row,
                    "status": "ok",
                    "trace_csv": repo_relative_path(trace_path, root=REPO_ROOT),
                    "figure": repo_relative_path(figure_path, root=REPO_ROOT),
                    "seismic_trace_file": repo_relative_path(seismic_path, root=REPO_ROOT),
                    "route": str(metric.get("route", "")),
                    "synthetic_scale": scale,
                    "xcorr_lag_s": float(xcorr.lag),
                    **impedance_qc_metrics(
                        model_ai=sampled_metric_trace,
                        low_ai=control_metric_trace,
                        mask=ai_mask,
                    ),
                    **{
                        f"waveform_{key}": value
                        for key, value in waveform_qc_metrics(observed, synthetic, target_mask).items()
                    },
                }
            )
        except Exception as exc:
            plt.close("all")
            metrics_rows.append(
                {
                    **base_row,
                    "status": "failed",
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )

    metrics_df = pd.DataFrame.from_records(metrics_rows)
    metrics_path = well_qc_dir / "well_qc_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    return {
        "well_qc_dir": repo_relative_path(well_qc_dir, root=REPO_ROOT),
        "metrics": repo_relative_path(metrics_path, root=REPO_ROOT),
        "n_wells_qc": int((metrics_df.get("status") == "ok").sum()) if not metrics_df.empty else 0,
    }


def _save_npz(result: Any, npz_file: Path, *, metadata_extra: dict[str, Any]) -> None:
    result.metadata.update(metadata_extra)
    npz_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_file,
        volume=result.volume.astype(np.float32),
        variance_volume=result.variance_volume.astype(np.float32),
        ilines=result.ilines,
        xlines=result.xlines,
        samples=result.samples,
        geometry_json=json.dumps(to_json_compatible(result.geometry), ensure_ascii=False),
        metadata_json=json.dumps(to_json_compatible(result.metadata), ensure_ascii=False),
        coverage_stats_json=json.dumps(to_json_compatible(result.coverage_stats), ensure_ascii=False),
    )


def _try_write_segy(
    result: Any,
    segy_file: Path,
    seismic_file: Path,
    seismic_type: str,
    seismic_cfg: dict[str, Any],
) -> str | None:
    if seismic_type.lower() != "segy":
        return "skipped_non_segy_source"
    try:
        import cigsegy

        keylocs = [
            int(seismic_cfg["iline_byte"]),
            int(seismic_cfg["xline_byte"]),
            int(seismic_cfg["istep"]),
            int(seismic_cfg["xstep"]),
        ]
        textual = build_segy_textual_header(
            "Time-domain AI low-frequency model",
            ["artifact=ai_lfm_time.npz", "source=lfm_precomputed.py"],
        )
        cigsegy.create_by_sharing_header(
            str(segy_file),
            str(seismic_file),
            np.ascontiguousarray(result.volume.astype(np.float32)),
            keylocs=keylocs,
            textual=textual,
        )
        return None
    except Exception as exc:
        return str(exc)


def _zgy_corners_from_survey(survey: Any, result: Any) -> tuple[tuple[float, float], ...]:
    il0 = float(result.ilines[0])
    iln = float(result.ilines[-1])
    xl0 = float(result.xlines[0])
    xln = float(result.xlines[-1])
    geometry = survey.line_geometry
    return (
        tuple(float(v) for v in geometry.line_to_coord(il0, xl0)),
        tuple(float(v) for v in geometry.line_to_coord(iln, xl0)),
        tuple(float(v) for v in geometry.line_to_coord(il0, xln)),
        tuple(float(v) for v in geometry.line_to_coord(iln, xln)),
    )


def _try_write_zgy(
    result: Any,
    zgy_file: Path,
    survey: Any,
    seismic_type: str,
    *,
    inline_chunk_size: int = 16,
) -> str | None:
    if seismic_type.lower() != "zgy":
        return "skipped_non_zgy_source"
    try:
        from pyzgy.write import SeismicWriter

        samples = np.asarray(result.samples, dtype=np.float64)
        if samples.size < 2:
            raise ValueError("ZGY export requires at least two samples.")
        sample_step_s = float(np.median(np.diff(samples)))
        if not np.allclose(np.diff(samples), sample_step_s, rtol=1e-6, atol=1e-9):
            raise ValueError("ZGY export requires a regular sample axis.")

        ilines = np.asarray(result.ilines, dtype=np.float64)
        xlines = np.asarray(result.xlines, dtype=np.float64)
        inline_inc = float(np.median(np.diff(ilines))) if ilines.size > 1 else 0.0
        xline_inc = float(np.median(np.diff(xlines))) if xlines.size > 1 else 0.0
        if ilines.size > 2 and not np.allclose(np.diff(ilines), inline_inc, rtol=0.0, atol=1e-8):
            raise ValueError("ZGY export requires a regular inline axis.")
        if xlines.size > 2 and not np.allclose(np.diff(xlines), xline_inc, rtol=0.0, atol=1e-8):
            raise ValueError("ZGY export requires a regular xline axis.")
        corners = _zgy_corners_from_survey(survey, result)

        zgy_file.parent.mkdir(parents=True, exist_ok=True)
        if zgy_file.exists():
            zgy_file.unlink()
        chunk = max(1, int(inline_chunk_size))
        volume = np.asarray(result.volume, dtype=np.float32)
        with SeismicWriter(
            zgy_file,
            tuple(int(v) for v in volume.shape),
            float(samples[0]) * 1000.0,
            sample_step_s * 1000.0,
            (float(ilines[0]), float(xlines[0])),
            (inline_inc, xline_inc),
            corners=corners,
        ) as writer:
            for il_start in range(0, volume.shape[0], chunk):
                il_end = min(volume.shape[0], il_start + chunk)
                writer.write_subvolume(volume[il_start:il_end], il_start, 0, 0)
        return None
    except Exception as exc:
        return str(exc)


def _load_constraints_control_points(
    constraints_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    control_path = constraints_dir / "lfm_control_points.csv"
    qc_path = constraints_dir / "well_constraint_qc.csv"
    summary_path = constraints_dir / "run_summary.json"
    if not control_path.exists():
        raise FileNotFoundError(f"Missing sixth-step LFM control points: {control_path}")
    control_df = pd.read_csv(control_path)
    required = {
        "well_name",
        "route",
        "source",
        "twt_s",
        "md_m",
        "x_m",
        "y_m",
        "inline_float",
        "xline_float",
        "zone_name",
        "u_in_zone",
        "ai",
        "weight",
        "flat_idx",
        "seismic_sample_index",
    }
    missing = required - set(control_df.columns)
    if missing:
        raise ValueError(f"lfm_control_points.csv is missing required columns: {sorted(missing)}")
    if control_df.empty:
        raise ValueError(f"Sixth-step LFM control table is empty: {control_path}")
    for col in ["twt_s", "inline_float", "xline_float", "u_in_zone", "ai", "weight"]:
        values = control_df[col].to_numpy(dtype=float)
        if np.any(~np.isfinite(values)):
            raise ValueError(f"lfm_control_points.csv column {col!r} contains non-finite values.")
    if np.any(control_df["ai"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("lfm_control_points.csv column 'ai' must be positive.")
    if np.any((control_df["u_in_zone"].to_numpy(dtype=float) < 0.0) | (control_df["u_in_zone"].to_numpy(dtype=float) > 1.0)):
        raise ValueError("lfm_control_points.csv column 'u_in_zone' must be within [0, 1].")

    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    qc_df = pd.read_csv(qc_path) if qc_path.exists() else pd.DataFrame()
    return control_df, qc_df, summary


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config, base_dir=REPO_ROOT)
    workflow = TimeWorkflowConfig.from_mapping(cfg)
    script_cfg = deep_merge_dict(DEFAULT_CONFIG, dict(cfg.get("lfm_precomputed") or {}))
    data_root = REPO_ROOT / workflow.data_root
    output_root = REPO_ROOT / workflow.output_root
    source_dirs = _resolve_source_dirs(script_cfg, output_root)

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"lfm_precomputed_{timestamp}"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    qc_dir = output_dir / "target_layer_qc"
    figures_dir = output_dir / "figures"
    well_qc_dir = output_dir / "well_qc"
    for directory in [output_dir, qc_dir, figures_dir, well_qc_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    seismic_cfg = workflow.seismic.as_dict()
    survey, seismic_file, seismic_type = _open_survey(seismic_cfg, data_root)
    geometry = survey.describe_geometry(domain="time")
    if str(geometry.get("sample_domain")).lower() != "time" or str(geometry.get("sample_unit")).lower() != "s":
        raise ValueError(f"Expected time-domain seismic geometry in seconds, got {geometry}")
    target_layer, horizon_metadata = _build_target_layer(script_cfg, geometry, qc_dir, data_root)
    sample_axis = survey.sample_axis("time").values.astype(np.float64)
    constraints_dir = source_dirs["well_constraints_dir"]
    control_df, qc_df, constraints_summary = _load_constraints_control_points(constraints_dir)
    control_path = output_dir / "lfm_control_points.csv"
    control_df.to_csv(control_path, index=False, encoding="utf-8-sig")
    if control_df.empty:
        raise ValueError("No LFM control points selected. Check sixth-step well_constraint_qc.csv for rejection reasons.")

    from cup.seismic.lfm_time import build_lfm_time_model_from_points

    result = build_lfm_time_model_from_points(
        target_layer=target_layer,
        control_points=_to_control_points(control_df),
        boundary_extension_samples=int(script_cfg["modeling"]["boundary_extension_samples"]),
        n_slices=int(script_cfg["modeling"]["n_slices"]),
        variogram=str(script_cfg["modeling"]["variogram"]),
        exact=bool(script_cfg["modeling"]["exact"]),
        nugget=float(script_cfg["modeling"]["nugget"]),
        post_slice_smoothing=bool(script_cfg["modeling"].get("post_slice_smoothing", False)),
    )
    result.metadata.update(
        {
            "control_source": "well_constraints",
            "well_constraints_dir": repo_relative_path(constraints_dir, root=REPO_ROOT),
            "frequency_bands": constraints_summary.get("frequency_bands"),
        }
    )
    metadata_extra = {
        "property_name": "AI",
        "target_layer": {
            "min_thickness": script_cfg["target_interval"].get("min_thickness"),
            "nearest_distance_limit": script_cfg["target_interval"].get("nearest_distance_limit"),
            "outlier_threshold": script_cfg["target_interval"].get("outlier_threshold"),
            "outlier_min_neighbor_count": script_cfg["target_interval"].get("outlier_min_neighbor_count", 2),
        },
        "horizons": horizon_metadata,
        "path_style": "repo_relative",
    }
    npz_file = output_dir / "ai_lfm_time.npz"
    _save_npz(result, npz_file, metadata_extra=metadata_extra)

    export_status = "disabled"
    if bool(script_cfg["export"].get("export_volume", True)):
        if seismic_type.lower() == "segy":
            export_status = _try_write_segy(
                result,
                output_dir / "ai_lfm_time.segy",
                seismic_file,
                seismic_type,
                seismic_cfg,
            )
            export_status = "written" if export_status is None else export_status
        elif seismic_type.lower() == "zgy":
            export_status = _try_write_zgy(
                result,
                output_dir / "ai_lfm_time.zgy",
                survey,
                seismic_type,
                inline_chunk_size=workflow.seismic.zgy_inline_chunk_size,
            )
            export_status = "written" if export_status is None else export_status
        else:
            export_status = f"unsupported_seismic_type:{seismic_type}"

    _plot_controls(control_df, figures_dir / "qc_control_points.png")
    _plot_lfm_result(result, figures_dir / "qc_ai_lfm_time.png")
    well_qc_summary = _write_lfm_well_qc(
        control_df,
        result,
        output_dir,
        constraints_summary=constraints_summary,
    )
    outputs = {
        "ai_lfm_time": repo_relative_path(npz_file, root=REPO_ROOT),
        "control_points": repo_relative_path(control_path, root=REPO_ROOT),
        "well_qc": well_qc_summary["well_qc_dir"],
        "well_qc_metrics": well_qc_summary["metrics"],
    }

    summary = {
        "source_dirs": {key: repo_relative_path(value, root=REPO_ROOT) for key, value in source_dirs.items()},
        "seismic_file": repo_relative_path(seismic_file, root=REPO_ROOT),
        "seismic_type": seismic_type,
        "well_constraints_summary": constraints_summary,
        "config": script_cfg,
        "selected_well_count": (
            int((qc_df["status"] == "selected").sum()) if "status" in qc_df.columns else int(control_df["well_name"].nunique())
        ),
        "control_point_count": int(len(control_df)),
        "outputs": outputs,
        "well_qc": well_qc_summary,
        "export_status": export_status,
        "coverage_stats": result.coverage_stats,
    }
    write_json(output_dir / "run_summary.json", summary)

    print("=== LFM Precomputed ===")
    print(f"Output: {output_dir}")
    print(f"Selected wells: {summary['selected_well_count']}")
    print(f"Control points: {summary['control_point_count']}")
    print(f"NPZ: {npz_file}")
    print(f"Well QC: {output_dir / 'well_qc'}")
    if export_status not in {"written", "disabled"}:
        print(f"Volume export skipped/failed: {export_status}")


if __name__ == "__main__":
    main()
