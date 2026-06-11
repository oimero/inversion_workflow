"""Gap-aware well-log preparation using an MD-domain time-depth table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cup.utils.masks import true_runs
from cup.well.las import StandardVpRhoLogs
from wtie.processing import grid


@dataclass(frozen=True)
class ContinuousTieLogResult:
    """Joint-valid Vp/Rho logs cropped to one continuous tie interval."""

    logs: grid.LogSet
    start_twt_s: float
    end_twt_s: float
    qc: dict[str, float | int | bool]


def _gap_duration_s(
    md: np.ndarray,
    start: int,
    stop: int,
    table: grid.TimeDepthTable,
) -> float:
    left = max(0, start - 1)
    right = min(md.size - 1, stop)
    if not table.is_md_domain:
        raise ValueError("Gap-aware standard LAS processing requires an MD-domain TimeDepthTable.")
    if md[left] < table.md[0] or md[right] > table.md[-1]:
        return float("inf")
    return float(
        np.interp(md[right], table.md, table.twt)
        - np.interp(md[left], table.md, table.twt)
    )


def fill_short_joint_gaps(
    standard: StandardVpRhoLogs,
    table: grid.TimeDepthTable,
    *,
    max_short_gap_s: float,
) -> tuple[grid.LogSet, np.ndarray]:
    """Fill bounded joint-invalid runs whose mapped TWT duration is short."""
    if not table.is_md_domain:
        raise ValueError("Gap-aware standard LAS processing requires an MD-domain TimeDepthTable.")
    if not np.isfinite(max_short_gap_s) or max_short_gap_s < 0.0:
        raise ValueError("max_short_gap_s must be finite and non-negative.")
    md = np.asarray(standard.logs.basis, dtype=np.float64)
    vp = np.asarray(standard.logs.Vp.values, dtype=np.float64).copy()
    rho = np.asarray(standard.logs.Rho.values, dtype=np.float64).copy()
    joint_valid = np.isfinite(vp) & (vp > 0.0) & np.isfinite(rho) & (rho > 0.0)
    filled = np.zeros(md.shape, dtype=bool)
    for start, stop in true_runs(~joint_valid):
        if start == 0 or stop == md.size:
            continue
        duration = _gap_duration_s(md, start, stop, table)
        if duration > float(max_short_gap_s) + 1e-12:
            continue
        vp[start:stop] = np.interp(
            md[start:stop],
            md[[start - 1, stop]],
            vp[[start - 1, stop]],
        )
        rho[start:stop] = np.interp(
            md[start:stop],
            md[[start - 1, stop]],
            rho[[start - 1, stop]],
        )
        filled[start:stop] = True
    logs = grid.LogSet(
        {
            "Vp": grid.Log(vp, md, "md", name="Vp", unit="m/s", allow_nan=True),
            "Rho": grid.Log(rho, md, "md", name="Rho", unit="g/cm3", allow_nan=True),
        }
    )
    return logs, filled


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
