"""Shared well-tie artifact contracts, continuous log support and matching math."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd

from cup.utils.masks import true_runs
from cup.well.inventory import normalize_well_name
from cup.well.gaps import _gap_duration_s, fill_short_joint_gaps
from cup.well.las import StandardVpRhoLogs
from wtie.processing import grid

WELL_AUTO_TIE_SCHEMA_VERSION = "well_auto_tie_v5"


@dataclass(frozen=True)
class ContinuousTieLogResult:
    """Joint-valid Vp/Rho logs cropped to one continuous tie interval."""

    logs: grid.LogSet
    start_twt_s: float
    end_twt_s: float
    qc: dict[str, float | int | bool]


@dataclass(frozen=True)
class WellTiePlan:
    well_name: str
    route: str
    route_status: str
    wellbore_class_initial: str
    wellbore_class_qc: str
    has_time_depth: bool
    has_well_trace: bool
    has_well_tops: bool
    usable_p_sonic: bool
    usable_density: bool
    input_las: str
    time_depth_file: str
    well_trace_file: str
    surface_x: float | None
    surface_y: float | None
    kb_m: float | None
    reasons: str = ""
    well_role: str = "trusted"

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WellTieResult:
    well_name: str
    route: str
    tie_status: str
    initial_corr: float | None = None
    optimized_corr: float | None = None
    optimized_nmae: float | None = None
    best_table_shift_ms: float | None = None
    wavelet_file: str = ""
    optimized_tdt_file: str = ""
    qc_figure_dir: str = ""
    reasons: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def prepare_continuous_tie_logs(
    standard: StandardVpRhoLogs,
    table: grid.TimeDepthTable,
    *,
    window_start_s: float,
    window_end_s: float,
    max_short_gap_s: float,
    min_tie_samples: int,
    seismic_sample_interval_s: float,
) -> ContinuousTieLogResult:
    """Fill short gaps and crop logs to the longest joint-valid window run."""
    if not np.isfinite(window_start_s) or not np.isfinite(window_end_s):
        raise ValueError("Tie-window bounds must be finite.")
    if window_end_s <= window_start_s:
        raise ValueError("window_end_s must be greater than window_start_s.")
    if int(min_tie_samples) < 2:
        raise ValueError("min_tie_samples must be at least 2.")
    logs, filled = fill_short_joint_gaps(
        standard,
        table,
        max_short_gap_s=max_short_gap_s,
    )
    md = np.asarray(logs.basis, dtype=np.float64)
    vp = np.asarray(logs.Vp.values, dtype=np.float64)
    rho = np.asarray(logs.Rho.values, dtype=np.float64)
    twt = np.interp(md, table.md, table.twt, left=np.nan, right=np.nan)
    in_window = (
        np.isfinite(twt)
        & (twt >= float(window_start_s))
        & (twt <= float(window_end_s))
    )
    joint_valid = (
        in_window
        & np.isfinite(vp)
        & (vp > 0.0)
        & np.isfinite(rho)
        & (rho > 0.0)
    )
    runs = true_runs(joint_valid)
    if not runs:
        raise ValueError("No joint-valid Vp/Rho samples remain in the target tie window.")
    start, stop = max(runs, key=lambda item: item[1] - item[0])
    start_twt = float(twt[start])
    end_twt = float(twt[stop - 1])
    seismic_dt_s = float(seismic_sample_interval_s)
    if not np.isfinite(seismic_dt_s) or seismic_dt_s <= 0.0:
        raise ValueError("seismic_sample_interval_s must be positive and finite.")
    seismic_count = int(np.floor((end_twt - start_twt) / seismic_dt_s + 1e-9)) + 1
    if seismic_count < int(min_tie_samples):
        raise ValueError(
            "Longest continuous joint-valid tie segment is too short on the seismic time axis: "
            f"{seismic_count} < {int(min_tie_samples)} samples "
            f"(dt={seismic_dt_s:g}s, duration={end_twt - start_twt:g}s)."
        )

    original_joint = np.asarray(standard.observed_mask, dtype=bool)
    original_window_count = int(np.count_nonzero(in_window))
    observed_window_count = int(np.count_nonzero(in_window & original_joint))
    long_gap_runs = []
    for gap_start, gap_stop in true_runs(in_window & ~joint_valid):
        duration = _gap_duration_s(md, gap_start, gap_stop, table)
        if duration > float(max_short_gap_s) + 1e-12:
            long_gap_runs.append((gap_start, gap_stop, duration))

    cropped = grid.LogSet(
        {
            "Vp": grid.Log(
                vp[start:stop],
                md[start:stop],
                "md",
                name="Vp",
                unit="m/s",
                allow_nan=False,
            ),
            "Rho": grid.Log(
                rho[start:stop],
                md[start:stop],
                "md",
                name="Rho",
                unit="g/cm3",
                allow_nan=False,
            ),
        }
    )
    cropped.observed_mask = original_joint[start:stop].copy()
    cropped.filled_gap_mask = filled[start:stop].copy()
    qc: dict[str, float | int | bool] = {
        "joint_observed_fraction": (
            float(observed_window_count / original_window_count)
            if original_window_count
            else 0.0
        ),
        "short_gap_filled_samples": int(np.count_nonzero(filled & in_window)),
        "long_gap_count": int(len(long_gap_runs)),
        "longest_long_gap_s": float(
            max((item[2] for item in long_gap_runs), default=0.0)
        ),
        "continuous_tie_window_start_s": start_twt,
        "continuous_tie_window_end_s": end_twt,
        "continuous_tie_sample_count": seismic_count,
        "continuous_tie_log_sample_count": int(stop - start),
        "tie_window_clipped_for_log_gap": bool(
            start_twt > float(window_start_s) + 1e-9
            or end_twt < float(window_end_s) - 1e-9
        ),
    }
    return ContinuousTieLogResult(
        logs=cropped,
        start_twt_s=start_twt,
        end_twt_s=end_twt,
        qc=qc,
    )


def tie_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.strip().casefold() in {"", "nan", "none", "null"} else text


def well_rows_by_name(df: pd.DataFrame) -> dict[str, pd.Series]:
    if df.empty or "well_name" not in df.columns:
        return {}
    lookup: dict[str, pd.Series] = {}
    display_by_key: dict[str, str] = {}
    for _, row in df.iterrows():
        display = tie_text(row["well_name"])
        if not display:
            continue
        key = normalize_well_name(display)
        previous = display_by_key.get(key)
        if previous is not None:
            raise ValueError(
                f"Duplicate well_name after normalization in workflow CSV: {previous!r} and {display!r}."
            )
        lookup[key] = row
        display_by_key[key] = display
    return lookup


def regular_tie_dt(basis: np.ndarray, *, label: str) -> float:
    values = np.asarray(basis, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError(f"{label} basis has fewer than 2 samples.")
    diffs = np.diff(values)
    if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0.0):
        raise ValueError(f"{label} basis must be strictly increasing.")
    dt_s = float(np.median(diffs))
    if not np.allclose(diffs, dt_s, rtol=1e-4, atol=1e-9):
        raise ValueError(f"{label} basis is not regularly sampled.")
    return dt_s


def resample_tie_seismic_to_dt(seismic: Any, dt_s: float) -> Any:
    """把一个地震道重采样到目标采样间隔 ``dt_s``。

    第四步保存的工区地震道可能比子波提取器的采样粗（例如 4 ms 与 2 ms），
    而反射系数、子波与合成记录只定义在各自主轴上。匹配前把地震道搬到
    子波网格，后续的卷积与相关性才是在同一套采样上比较。
    """
    from wtie.optimize import tie as tie_ops

    target_dt_s = float(dt_s)
    if not np.isfinite(target_dt_s) or target_dt_s <= 0.0:
        raise ValueError("target dt_s must be finite and positive for seismic resampling.")
    resampled = tie_ops.resample_seismic(seismic, target_dt_s)
    actual_dt_s = regular_tie_dt(resampled.basis, label="resampled seismic trace")
    if not np.isclose(actual_dt_s, target_dt_s, rtol=1e-5, atol=1e-9):
        raise ValueError(
            f"resampled seismic dt {actual_dt_s:g}s does not match requested dt {target_dt_s:g}s."
        )
    return resampled


def scaled_synthetic_metrics(
    modeler: Any,
    wavelet: Any,
    reflectivity: Any,
    seismic: Any,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """计算归一化合成记录与地震道的匹配指标。

    返回 ``(seismic_norm, synthetic, corr, nmae, scale)``。
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Wavelet has even number of samples; appending one zero sample\.",
        )
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


__all__ = [
    "WELL_AUTO_TIE_SCHEMA_VERSION", "ContinuousTieLogResult", "WellTiePlan", "WellTieResult",
    "fill_short_joint_gaps", "prepare_continuous_tie_logs", "tie_text", "well_rows_by_name",
    "regular_tie_dt", "resample_tie_seismic_to_dt", "scaled_synthetic_metrics",
]
