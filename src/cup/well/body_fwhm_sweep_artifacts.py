"""Tables, reusable arrays and figures for the GINN body FWHM sweep."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cup.utils.io import repo_relative_path, sanitize_filename, write_json
from cup.well.body_fwhm_sweep import BodyFwhmSweepResult, SCHEMA_VERSION, WellBodyFwhmSweep


_SUMMARY_METRICS = (
    "forward_corr",
    "forward_difference_rms_ratio",
    "gain_aligned_forward_nrmse",
    "native_residual_negative_curvature_r2",
    "model_residual_unsharp_r2",
    "native_residual_major_interval_width_p50_m",
    "native_residual_autocorrelation_half_width_m",
)


def _fwhm_key(value: float) -> str:
    return f"fwhm_{value:g}".replace(".", "p")


def _finite_scale(values: list[np.ndarray], *, quantile: float = 0.99) -> float:
    finite = [np.abs(item[np.isfinite(item)]) for item in values]
    finite = [item for item in finite if item.size]
    if not finite:
        return 1.0
    scale = float(np.quantile(np.concatenate(finite), quantile))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _normalized(values: np.ndarray, scale: float) -> np.ndarray:
    output = np.full(np.asarray(values).shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    output[finite] = np.asarray(values, dtype=np.float64)[finite] / float(scale)
    return output


def _plot_window_comparison(
    well: WellBodyFwhmSweep,
    *,
    top_m: float,
    bottom_m: float,
    event_top_m: float | None,
    event_bottom_m: float | None,
    output_path: Path,
    title: str,
    reference_fwhm_m: float,
) -> None:
    native_view = (
        (well.native_axis_m >= top_m)
        & (well.native_axis_m <= bottom_m)
        & well.native_target_support
    )
    model_view = (
        (well.model_axis_m >= top_m)
        & (well.model_axis_m <= bottom_m)
        & well.model_target_support
    )
    if not np.any(native_view) or not np.any(model_view):
        raise ValueError(f"{well.well_name}: comparison window has no native/model support.")
    body_scale = _finite_scale(
        [well.native_full_log_ai[native_view]]
        + [item.native_body_log_ai[native_view] for item in well.candidates]
    )
    body_values = np.concatenate(
        [well.native_full_log_ai[native_view]]
        + [item.native_body_log_ai[native_view] for item in well.candidates]
    )
    body_finite = body_values[np.isfinite(body_values)]
    body_min = float(np.quantile(body_finite, 0.01))
    body_max = float(np.quantile(body_finite, 0.99))
    body_pad = max(0.03 * (body_max - body_min), 1.0e-4 * body_scale)
    residual_limit = _finite_scale(
        [item.native_residual_log_ai[native_view] for item in well.candidates]
        + [
            item.curvature_fit_intercept
            + item.curvature_fit_gain * item.native_negative_curvature[native_view]
            for item in well.candidates
        ]
    )
    forward_scale = _finite_scale(
        [well.full_forward[model_view]]
        + [item.body_forward[model_view] for item in well.candidates],
        quantile=0.95,
    )
    seismic_scale = _finite_scale([well.real_seismic[model_view]], quantile=0.95)

    figure, axes = plt.subplots(
        len(well.candidates),
        4,
        figsize=(13.5, max(8.0, 2.0 * len(well.candidates))),
        squeeze=False,
        constrained_layout=True,
    )
    for row, candidate in enumerate(well.candidates):
        axes[row, 0].plot(
            _normalized(well.real_seismic, seismic_scale),
            well.model_axis_m,
            color="black",
            linewidth=1.0,
        )
        axes[row, 1].plot(
            well.native_full_log_ai,
            well.native_axis_m,
            color="0.25",
            linewidth=0.65,
            label="full well log",
        )
        axes[row, 1].plot(
            candidate.native_body_log_ai,
            well.native_axis_m,
            color="#d62728",
            linewidth=1.15,
            label=f"{candidate.fwhm_m:g} m body",
        )
        fitted_curvature = (
            candidate.curvature_fit_intercept
            + candidate.curvature_fit_gain * candidate.native_negative_curvature
        )
        axes[row, 2].plot(
            candidate.native_residual_log_ai,
            well.native_axis_m,
            color="#9467bd",
            linewidth=0.75,
            label="full − body",
        )
        axes[row, 2].plot(
            fitted_curvature,
            well.native_axis_m,
            color="0.2",
            linewidth=0.85,
            linestyle="--",
            label="fitted sharpening",
        )
        axes[row, 2].axvline(0.0, color="0.6", linewidth=0.5)
        axes[row, 3].plot(
            _normalized(well.full_forward, forward_scale),
            well.model_axis_m,
            color="black",
            linewidth=1.0,
            label="full forward",
        )
        axes[row, 3].plot(
            _normalized(candidate.body_forward, forward_scale),
            well.model_axis_m,
            color="#ff7f0e",
            linewidth=1.05,
            label="body forward",
        )
        axes[row, 0].set_xlim(-1.2, 1.2)
        axes[row, 1].set_xlim(body_min - body_pad, body_max + body_pad)
        axes[row, 2].set_xlim(-residual_limit, residual_limit)
        axes[row, 3].set_xlim(-1.2, 1.2)
        for column in range(4):
            panel = axes[row, column]
            panel.set_ylim(bottom_m, top_m)
            panel.grid(alpha=0.20)
            if event_top_m is not None and event_bottom_m is not None:
                panel.axhspan(event_top_m, event_bottom_m, color="#1f77b4", alpha=0.10)
        marker = " ← current" if candidate.fwhm_m == reference_fwhm_m else ""
        axes[row, 0].set_ylabel(f"F={candidate.fwhm_m:g} m{marker}\nTVDSS [m]")
    for column, name in enumerate(
        ("real seismic", "full / body log-AI", "residual / sharpening fit", "full / body forward")
    ):
        axes[0, column].set_title(name)
    axes[0, 1].legend(fontsize=7)
    axes[0, 2].legend(fontsize=7)
    axes[0, 3].legend(fontsize=7)
    figure.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_well_artifact(well: WellBodyFwhmSweep, path: Path) -> None:
    payload: dict[str, np.ndarray] = {
        "native_axis_m": well.native_axis_m,
        "native_full_log_ai": well.native_full_log_ai,
        "native_target_support": well.native_target_support,
        "model_axis_m": well.model_axis_m,
        "model_full_log_ai": well.model_full_log_ai,
        "model_target_support": well.model_target_support,
        "real_seismic": well.real_seismic,
        "full_forward": well.full_forward,
        "horizon_depth_m": np.asarray([item[0] for item in well.horizon_markers], dtype=np.float64),
        "horizon_name": np.asarray([item[1] for item in well.horizon_markers], dtype="U64"),
        "event_rank": np.asarray([item.event_rank for item in well.events], dtype=np.int32),
        "event_top_m": np.asarray([item.top_m for item in well.events], dtype=np.float64),
        "event_bottom_m": np.asarray([item.bottom_m for item in well.events], dtype=np.float64),
        "event_polarity": np.asarray([item.polarity for item in well.events], dtype=np.int8),
        "event_peak_abs": np.asarray([item.peak_abs for item in well.events], dtype=np.float64),
    }
    for candidate in well.candidates:
        prefix = _fwhm_key(candidate.fwhm_m)
        payload[f"{prefix}_native_body_log_ai"] = candidate.native_body_log_ai
        payload[f"{prefix}_native_residual_log_ai"] = candidate.native_residual_log_ai
        payload[f"{prefix}_native_negative_curvature"] = candidate.native_negative_curvature
        payload[f"{prefix}_model_body_log_ai"] = candidate.model_body_log_ai
        payload[f"{prefix}_model_residual_log_ai"] = candidate.model_residual_log_ai
        payload[f"{prefix}_model_sharpening_template"] = candidate.model_sharpening_template
        payload[f"{prefix}_body_forward"] = candidate.body_forward
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _summary_frame(candidate_metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fwhm_m, group in candidate_metrics.groupby("fwhm_m", sort=True):
        row: dict[str, Any] = {
            "candidate": f"F{float(fwhm_m):g}",
            "fwhm_m": float(fwhm_m),
            "well_count": int(group["well_name"].nunique()),
        }
        for metric in _SUMMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_p10"] = float(values.quantile(0.10))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_p90"] = float(values.quantile(0.90))
        records.append(row)
    return pd.DataFrame.from_records(records)


def _plot_summary(candidate_metrics: pd.DataFrame, output_path: Path) -> None:
    panels = (
        ("forward_corr", "Full/body forward correlation"),
        ("forward_difference_rms_ratio", "Forward difference RMS / full RMS"),
        ("native_residual_negative_curvature_r2", "Native residual explained by curvature"),
        ("model_residual_unsharp_r2", "5 m residual explained by unsharp body"),
        ("native_residual_major_interval_width_p50_m", "Major residual interval P50 [m]"),
        ("native_residual_autocorrelation_half_width_m", "Residual autocorrelation half-width [m]"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    for panel, (metric, title) in zip(axes.ravel(), panels):
        for _well_name, group in candidate_metrics.groupby("well_name", sort=False):
            ordered = group.sort_values("fwhm_m")
            panel.plot(
                ordered["fwhm_m"],
                ordered[metric],
                color="0.72",
                linewidth=0.8,
                marker="o",
                markersize=2.5,
            )
        median = candidate_metrics.groupby("fwhm_m", sort=True)[metric].median()
        panel.plot(
            median.index,
            median.values,
            color="#d62728",
            linewidth=2.0,
            marker="o",
            label="well median",
        )
        panel.set_title(title)
        panel.set_xlabel("Body smoothing FWHM [m]")
        panel.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("GINN V2 body/residual FWHM sweep")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_body_fwhm_sweep_artifacts(
    result: BodyFwhmSweepResult,
    *,
    output_dir: Path,
    repo_root: Path,
    resolved_config: Mapping[str, Any],
    inputs: Mapping[str, str],
    horizon_sources: list[dict[str, str]],
) -> dict[str, Any]:
    """Write one complete sweep result into a new output directory."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    wells_dir = output_dir / "wells"
    figures_dir.mkdir()
    wells_dir.mkdir()

    candidate_metrics = pd.DataFrame.from_records(result.candidate_metrics).sort_values(
        ["well_name", "fwhm_m"]
    )
    event_metrics = pd.DataFrame.from_records(result.event_metrics).sort_values(
        ["well_name", "event_rank", "fwhm_m"]
    )
    event_rows = [
        {
            "well_name": well.well_name,
            "event_rank": event.event_rank,
            "event_top_m": event.top_m,
            "event_bottom_m": event.bottom_m,
            "event_width_m": event.width_m,
            "event_polarity": event.polarity,
            "event_peak_abs": event.peak_abs,
        }
        for well in result.wells
        for event in well.events
    ]
    event_windows = pd.DataFrame.from_records(event_rows).sort_values(["well_name", "event_rank"])
    summary = _summary_frame(candidate_metrics)

    candidate_path = output_dir / "candidate_metrics.csv"
    event_metrics_path = output_dir / "event_metrics.csv"
    event_windows_path = output_dir / "event_windows.csv"
    summary_path = output_dir / "fwhm_summary.csv"
    candidate_metrics.to_csv(candidate_path, index=False)
    event_metrics.to_csv(event_metrics_path, index=False)
    event_windows.to_csv(event_windows_path, index=False)
    summary.to_csv(summary_path, index=False)

    summary_figure = figures_dir / "fwhm_summary.png"
    _plot_summary(candidate_metrics, summary_figure)
    well_artifacts: dict[str, str] = {}
    well_figures: dict[str, Any] = {}
    for well in result.wells:
        safe_name = sanitize_filename(well.well_name)
        artifact_path = wells_dir / f"{safe_name}.npz"
        _write_well_artifact(well, artifact_path)
        well_artifacts[well.well_name] = repo_relative_path(artifact_path, root=repo_root)
        well_figure_dir = figures_dir / safe_name
        target_top = float(well.horizon_markers[0][0])
        target_bottom = float(well.horizon_markers[-1][0])
        overview_path = well_figure_dir / "target_interval_sweep.png"
        _plot_window_comparison(
            well,
            top_m=target_top,
            bottom_m=target_bottom,
            event_top_m=None,
            event_bottom_m=None,
            output_path=overview_path,
            title=f"{well.well_name} | complete target interval",
            reference_fwhm_m=result.policy.reference_fwhm_m,
        )
        event_paths: list[str] = []
        for event in well.events:
            context = max(
                result.policy.event_context_min_m,
                result.policy.event_context_width_multiple * event.width_m,
            )
            path = well_figure_dir / f"event_{event.event_rank:02d}_sweep.png"
            _plot_window_comparison(
                well,
                top_m=event.top_m - context,
                bottom_m=event.bottom_m + context,
                event_top_m=event.top_m,
                event_bottom_m=event.bottom_m,
                output_path=path,
                title=(
                    f"{well.well_name} | fixed real-seismic event {event.event_rank} | "
                    f"{event.top_m:g}–{event.bottom_m:g} m"
                ),
                reference_fwhm_m=result.policy.reference_fwhm_m,
            )
            event_paths.append(repo_relative_path(path, root=repo_root))
        well_figures[well.well_name] = {
            "target_interval": repo_relative_path(overview_path, root=repo_root),
            "events": event_paths,
        }

    manifest = {
        "schema": SCHEMA_VERSION,
        "status": "completed",
        "sample_domain": "depth",
        "sample_unit": "m",
        "depth_basis": "tvdss",
        "inputs": dict(inputs),
        "horizon_sources": horizon_sources,
        "resolved_config": dict(resolved_config),
        "well_names": [well.well_name for well in result.wells],
        "fwhm_values_m": list(result.policy.fwhm_values_m),
        "reference_fwhm_m": result.policy.reference_fwhm_m,
        "tables": {
            "candidate_metrics": repo_relative_path(candidate_path, root=repo_root),
            "event_metrics": repo_relative_path(event_metrics_path, root=repo_root),
            "event_windows": repo_relative_path(event_windows_path, root=repo_root),
            "fwhm_summary": repo_relative_path(summary_path, root=repo_root),
        },
        "figures": {
            "summary": repo_relative_path(summary_figure, root=repo_root),
            "wells": well_figures,
        },
        "well_artifacts": well_artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


__all__ = ["write_body_fwhm_sweep_artifacts"]
