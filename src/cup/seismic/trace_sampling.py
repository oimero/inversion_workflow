"""cup.seismic.trace_sampling: trace plans and along-well seismic assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from cup.well.trajectory import WellSpatialSampleSet
from wtie.processing import grid


TRACE_SAMPLE_PLAN_COLUMNS = [
    "well_name",
    "sample_index",
    "twt_s",
    "md_m",
    "tvdss_m",
    "x_m",
    "y_m",
    "inline_float",
    "xline_float",
    "nearest_inline",
    "nearest_xline",
    "inline_index",
    "xline_index",
    "flat_idx",
    "survey_position",
]


@dataclass(frozen=True)
class TraceSamplePlan:
    """Nearest-trace sampling plan for a sampled well trajectory."""

    well_name: str
    rows: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        return self.rows.copy()

    @property
    def sample_count(self) -> int:
        return int(len(self.rows))

    def inside_fraction(self) -> float | None:
        if self.rows.empty:
            return None
        checked = self.rows["survey_position"].isin(["inside", "outside"])
        checked_count = int(checked.sum())
        if checked_count == 0:
            return None
        return float((self.rows.loc[checked, "survey_position"] == "inside").sum() / checked_count)

    def outside_fraction(self) -> float | None:
        inside = self.inside_fraction()
        return None if inside is None else float(1.0 - inside)

    def longest_inside_slice(self) -> slice:
        inside = self.rows["survey_position"].astype(str).eq("inside").to_numpy(dtype=bool)
        if not np.any(inside):
            return slice(0, 0)
        best_start = 0
        best_len = 0
        current_start = 0
        current_len = 0
        for index, value in enumerate(inside):
            if value:
                if current_len == 0:
                    current_start = index
                current_len += 1
                if current_len > best_len:
                    best_start = current_start
                    best_len = current_len
            else:
                current_len = 0
        return slice(best_start, best_start + best_len)

    def subset(self, sample_slice: slice) -> "TraceSamplePlan":
        rows = self.rows.iloc[sample_slice].reset_index(drop=True).copy()
        rows["sample_index"] = np.arange(len(rows), dtype=np.int64)
        return TraceSamplePlan(well_name=self.well_name, rows=rows)


def _trace_flat_index(survey: Any, inline_index: int, xline_index: int) -> int:
    try:
        if hasattr(survey, "trace_flat_index"):
            return int(survey.trace_flat_index(inline_index, xline_index))
        if hasattr(survey, "geom"):
            return int(survey.geom[int(inline_index), int(xline_index)])
        if hasattr(survey, "n_xlines"):
            return int(inline_index) * int(survey.n_xlines) + int(xline_index)
    except (AttributeError, TypeError, IndexError, ValueError) as exc:
        raise ValueError(f"Cannot resolve trace flat index for {(inline_index, xline_index)}: {exc}") from exc
    raise ValueError("survey does not expose trace flat-index lookup.")


def _nearest_indices_from_line(survey: Any, inline_float: float, xline_float: float) -> tuple[int, int, float, float]:
    geometry = survey.line_geometry
    nearest_inline = geometry.snap_inline(float(inline_float))
    nearest_xline = geometry.snap_xline(float(xline_float))
    inline_index_f, xline_index_f = geometry.line_to_index(nearest_inline, nearest_xline)
    return int(round(inline_index_f)), int(round(xline_index_f)), float(nearest_inline), float(nearest_xline)


def build_nearest_trace_sample_plan(samples: WellSpatialSampleSet, survey: Any) -> TraceSamplePlan:
    """Assign each spatial sample to its nearest valid seismic trace."""
    rows: list[dict[str, Any]] = []
    for _, sample in samples.rows.iterrows():
        base = {
            "well_name": str(sample["well_name"]),
            "sample_index": int(sample["sample_index"]),
            "twt_s": float(sample["twt_s"]),
            "md_m": float(sample["md_m"]),
            "tvdss_m": float(sample["tvdss_m"]),
            "x_m": float(sample["x_m"]),
            "y_m": float(sample["y_m"]),
        }
        try:
            inline_float, xline_float = survey.line_geometry.coord_to_line(float(sample["x_m"]), float(sample["y_m"]))
            inline_index, xline_index, nearest_inline, nearest_xline = _nearest_indices_from_line(
                survey, inline_float, xline_float
            )
            flat_idx = _trace_flat_index(survey, inline_index, xline_index)
            if flat_idx < 0:
                raise ValueError("nearest trace is missing")
            rows.append(
                {
                    **base,
                    "inline_float": float(inline_float),
                    "xline_float": float(xline_float),
                    "nearest_inline": nearest_inline,
                    "nearest_xline": nearest_xline,
                    "inline_index": inline_index,
                    "xline_index": xline_index,
                    "flat_idx": flat_idx,
                    "survey_position": "inside",
                }
            )
        except ValueError:
            rows.append(
                {
                    **base,
                    "inline_float": None,
                    "xline_float": None,
                    "nearest_inline": None,
                    "nearest_xline": None,
                    "inline_index": None,
                    "xline_index": None,
                    "flat_idx": None,
                    "survey_position": "outside",
                }
            )
    return TraceSamplePlan(
        well_name=samples.well_name,
        rows=pd.DataFrame.from_records(rows, columns=TRACE_SAMPLE_PLAN_COLUMNS),
    )


def _unique_trace_indices(rows: pd.DataFrame) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for _, row in rows.iterrows():
        if str(row["survey_position"]) != "inside":
            continue
        pairs.add((int(row["inline_index"]), int(row["xline_index"])))
    return sorted(pairs)


def assemble_nearest_trace_from_plan(
    plan: TraceSamplePlan,
    survey: Any,
    *,
    sample_start: float | None = None,
    sample_end: float | None = None,
    domain: str = "time",
) -> grid.Seismic:
    """Read unique nearest traces and assemble one along-trajectory seismic trace."""
    rows = plan.rows.reset_index(drop=True)
    if rows.empty:
        raise ValueError("Trace sample plan is empty.")
    if not rows["survey_position"].astype(str).eq("inside").all():
        raise ValueError("Trace sample plan contains non-inside samples; crop or reject before assembly.")

    indices = _unique_trace_indices(rows)
    if not indices:
        raise ValueError("Trace sample plan contains no inside traces.")

    trace_by_index = survey.read_traces_at_indices(indices, sample_start=sample_start, sample_end=sample_end, domain=domain)
    first_trace = next(iter(trace_by_index.values()))
    basis = np.asarray(first_trace.basis, dtype=np.float64)
    twt = rows["twt_s"].to_numpy(dtype=np.float64)
    if float(twt[0]) < float(basis[0]) - 1e-9 or float(twt[-1]) > float(basis[-1]) + 1e-9:
        raise ValueError(f"Plan TWT range [{twt[0]}, {twt[-1]}] is outside read trace range [{basis[0]}, {basis[-1]}].")

    values: list[float] = []
    for _, row in rows.iterrows():
        key = (int(row["inline_index"]), int(row["xline_index"]))
        trace = trace_by_index[key]
        values.append(float(np.interp(float(row["twt_s"]), trace.basis, trace.values)))
    return grid.Seismic(
        np.asarray(values, dtype=np.float64),
        twt,
        "twt" if domain == "time" else "md",
        name=f"Nearest trajectory seismic ({plan.well_name})",
    )


def trace_index_set(rows: Iterable[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in rows}
