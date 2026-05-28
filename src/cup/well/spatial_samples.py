"""cup.well.spatial_samples: sample well trajectories on workflow axes.

This module turns a well trajectory and an MD-domain time-depth table into
explicit sample rows.  It does not read seismic data; trace assignment is handled
by ``cup.seismic.trace_sampling``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cup.well.trajectory import WellTrajectory
from wtie.processing import grid


SPATIAL_SAMPLE_COLUMNS = [
    "well_name",
    "sample_index",
    "twt_s",
    "md_m",
    "tvd_kb_m",
    "tvdss_m",
    "z_m",
    "x_m",
    "y_m",
]


@dataclass(frozen=True)
class WellSpatialSampleSet:
    """A well trajectory sampled on a TWT axis."""

    well_name: str
    rows: pd.DataFrame

    @property
    def sample_count(self) -> int:
        return int(len(self.rows))

    def to_dataframe(self) -> pd.DataFrame:
        return self.rows.copy()


def _validate_twt_axis(twt_axis: np.ndarray) -> np.ndarray:
    twt = np.asarray(twt_axis, dtype=np.float64)
    if twt.ndim != 1 or twt.size == 0:
        raise ValueError("twt_axis must be a non-empty 1D array.")
    if not np.all(np.isfinite(twt)):
        raise ValueError("twt_axis contains non-finite values.")
    if np.any(np.diff(twt) <= 0.0):
        raise ValueError("twt_axis must be strictly increasing.")
    return twt


def _validate_md_table(table: grid.TimeDepthTable) -> None:
    if not table.is_md_domain:
        raise ValueError("sample_trajectory_on_twt expects an MD-domain TimeDepthTable.")
    if table.size < 2:
        raise ValueError("Time-depth table must contain at least two samples.")


def sample_trajectory_on_twt(
    trajectory: WellTrajectory,
    table: grid.TimeDepthTable,
    twt_axis: np.ndarray,
) -> WellSpatialSampleSet:
    """Sample a trajectory at TWT positions using an MD-domain table."""
    _validate_md_table(table)
    twt = _validate_twt_axis(twt_axis)

    table_twt = np.asarray(table.twt, dtype=np.float64)
    table_md = np.asarray(table.md, dtype=np.float64)
    if float(twt[0]) < float(table_twt[0]) or float(twt[-1]) > float(table_twt[-1]):
        raise ValueError(
            f"TWT axis [{twt[0]}, {twt[-1]}] is outside table range [{table_twt[0]}, {table_twt[-1]}]."
        )

    md = np.interp(twt, table_twt, table_md)
    if not np.all(np.isfinite(md)):
        raise ValueError("Sampled MD values contain non-finite values.")
    if float(np.nanmin(md)) < float(trajectory.md_m[0]) or float(np.nanmax(md)) > float(trajectory.md_m[-1]):
        raise ValueError(
            f"Sampled MD range [{np.nanmin(md)}, {np.nanmax(md)}] is outside trajectory range "
            f"[{trajectory.md_m[0]}, {trajectory.md_m[-1]}]."
        )

    rows: list[dict[str, Any]] = []
    for index, (twt_s, md_m) in enumerate(zip(twt, md)):
        position = trajectory.position_at_md(float(md_m))
        rows.append(
            {
                "well_name": trajectory.well_name,
                "sample_index": index,
                "twt_s": float(twt_s),
                "md_m": float(position["md_m"]),
                "tvd_kb_m": float(position["tvd_kb_m"]),
                "tvdss_m": float(position["tvdss_m"]),
                "z_m": float(position["z_m"]),
                "x_m": float(position["x_m"]),
                "y_m": float(position["y_m"]),
            }
        )
    return WellSpatialSampleSet(
        well_name=trajectory.well_name,
        rows=pd.DataFrame.from_records(rows, columns=SPATIAL_SAMPLE_COLUMNS),
    )
