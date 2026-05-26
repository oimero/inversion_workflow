"""Depth/time helpers for time-domain well tying."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import lasio
import numpy as np
import pandas as pd

from cup.petrel.load import read_petrel_checkshots_dataframe
from cup.well.assets import normalize_well_name
from wtie.processing import grid
from wtie.processing.logs import interpolate_nans


@dataclass(frozen=True)
class TargetTieWindow:
    """Target interval tie window in TWT seconds."""

    top_name: str
    bottom_name: str
    top_twt_s: float
    bottom_twt_s: float
    start_s: float
    end_s: float
    margin_top_s: float
    margin_bottom_s: float
    top_sample_method: str = ""
    bottom_sample_method: str = ""
    top_nearest_line_distance: float | None = None
    bottom_nearest_line_distance: float | None = None

    @property
    def duration_s(self) -> float:
        return float(self.end_s - self.start_s)


@dataclass(frozen=True)
class PreparedTieWindow:
    """A local TDT/logset package clipped to the actual tie window."""

    table: grid.TimeDepthTable
    logset_md: grid.LogSet
    table_rows: pd.DataFrame
    report: dict[str, Any]


def _finite_positive(values: np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= 0.0] = np.nan
    if np.all(np.isnan(arr)):
        raise ValueError(f"{label} contains no positive finite samples.")
    return arr


def _read_petrel_checkshot_dataframe(path: Path) -> pd.DataFrame:
    """Delegate to the unified Petrel checkshot parser in ``cup.petrel.load``."""
    return read_petrel_checkshots_dataframe(path)


def read_time_depth_table(path: str | Path, *, domain: Literal["md", "tvdss"] = "md") -> grid.TimeDepthTable:
    """Read a Petrel checkshot/time-depth table into a wtie table.

    Petrel exports in this project store picked TWT as negative milliseconds.
    The time-domain workflow uses positive seconds, so this reader applies
    ``abs(twt_ms) / 1000`` and sorts by the requested depth domain.
    """
    path = Path(path)
    df = _read_petrel_checkshot_dataframe(path)
    domain = str(domain).strip().lower()  # type: ignore[assignment]
    if domain not in {"md", "tvdss"}:
        raise ValueError(f"Unsupported TDT domain: {domain}.")

    depth_col = "md_m" if domain == "md" else "z_m"
    depth = df[depth_col].to_numpy(dtype=np.float64)
    if domain == "tvdss":
        depth = np.abs(depth)
    twt = np.abs(df["twt_ms"].to_numpy(dtype=np.float64)) / 1000.0

    finite = np.isfinite(depth) & np.isfinite(twt)
    depth = depth[finite]
    twt = twt[finite]
    if depth.size < 2:
        raise ValueError(f"Time-depth table has fewer than 2 valid samples: {path}")

    order = np.argsort(depth)
    depth = depth[order]
    twt = twt[order]

    # Keep the first sample for duplicate depths. Petrel exports are dense and
    # occasionally contain repeated values after rounding.
    unique_depth, unique_indices = np.unique(depth, return_index=True)
    depth = unique_depth
    twt = twt[unique_indices]
    if depth.size < 2:
        raise ValueError(f"Time-depth table has fewer than 2 unique depth samples: {path}")

    if np.any(np.diff(depth) < 0.0):
        raise ValueError(f"Time-depth table depth is not monotonic after sorting: {path}")

    # Some Petrel exports contain a short shallow turnaround or millisecond
    # rounding plateaus.  wtie requires strictly increasing TWT, so preserve the
    # physically usable monotonic segment instead of rejecting the whole well.
    start = int(np.nanargmin(twt))
    depth = depth[start:]
    twt = twt[start:]
    keep = np.zeros(twt.shape, dtype=bool)
    last_twt = -np.inf
    for index, value in enumerate(twt):
        if value > last_twt + 1e-9:
            keep[index] = True
            last_twt = float(value)
    depth = depth[keep]
    twt = twt[keep]
    if depth.size < 2:
        raise ValueError(f"Time-depth table has fewer than 2 strictly increasing TWT samples: {path}")
    if np.any(np.diff(twt) <= 0.0):
        raise ValueError(f"Time-depth table TWT is not strictly increasing after monotonic filtering: {path}")

    if domain == "md":
        return grid.TimeDepthTable(twt=twt, md=depth)
    return grid.TimeDepthTable(twt=twt, tvdss=depth)


def build_vp_rho_logset_from_preprocessed_las(path: str | Path) -> grid.LogSet:
    """Build MD-domain Vp/Rho LogSet from third-step standard LAS."""
    las = lasio.read(str(path))
    df = las.df()
    missing = [name for name in ("DT_USM", "RHO_GCC") if name not in df.columns]
    if missing:
        raise ValueError(f"Preprocessed LAS is missing required curves {missing}: {path}")

    md = np.asarray(df.index.to_numpy(dtype=np.float64), dtype=np.float64)
    dt_usm = _finite_positive(df["DT_USM"].to_numpy(dtype=np.float64), label="DT_USM")
    rho = _finite_positive(df["RHO_GCC"].to_numpy(dtype=np.float64), label="RHO_GCC")
    vp = 1_000_000.0 / dt_usm

    vp = interpolate_nans(vp, method="linear")
    rho = interpolate_nans(rho, method="linear")
    if np.any(~np.isfinite(vp)) or np.any(~np.isfinite(rho)):
        raise ValueError(f"Vp/Rho still contain non-finite samples after interpolation: {path}")

    return grid.LogSet(
        {
            "Vp": grid.Log(vp, md, "md", name="Vp", unit="m/s", allow_nan=False),
            "Rho": grid.Log(rho, md, "md", name="Rho", unit="g/cm3", allow_nan=False),
        }
    )


def validate_time_depth_table(
    table: grid.TimeDepthTable,
    log_basis_md: np.ndarray,
    *,
    min_overlap_samples: int = 64,
) -> dict[str, float | int]:
    """Validate MD overlap between a table and log basis."""
    if not table.is_md_domain:
        raise ValueError("vertical_with_tdt expects an MD-domain TimeDepthTable.")
    log_md = np.asarray(log_basis_md, dtype=np.float64)
    table_md = np.asarray(table.md, dtype=np.float64)
    overlap_min = max(float(log_md[0]), float(table_md[0]))
    overlap_max = min(float(log_md[-1]), float(table_md[-1]))
    if overlap_max <= overlap_min:
        raise ValueError(
            f"Log MD range [{log_md[0]}, {log_md[-1]}] does not overlap TDT MD range [{table_md[0]}, {table_md[-1]}]."
        )
    count = int(np.count_nonzero((log_md >= overlap_min) & (log_md <= overlap_max)))
    if count < int(min_overlap_samples):
        raise ValueError(f"Too few log samples overlap TDT: {count} < {min_overlap_samples}.")
    return {
        "overlap_md_min_m": overlap_min,
        "overlap_md_max_m": overlap_max,
        "overlap_log_sample_count": count,
    }


def tdt_overlaps_window(table: grid.TimeDepthTable, window: TargetTieWindow) -> bool:
    """Return whether an MD-domain TDT touches the target TWT window."""
    if not table.is_md_domain:
        raise ValueError("TDT/window overlap expects an MD-domain TimeDepthTable.")
    return min(float(table.twt[-1]), float(window.end_s)) > max(float(table.twt[0]), float(window.start_s))


def crop_logset_md(logset_md: grid.LogSet, md_min_m: float, md_max_m: float, *, min_samples: int = 2) -> grid.LogSet:
    """Crop an MD-domain LogSet by MD while preserving existing log names."""
    if not logset_md.is_md:
        raise ValueError("LogSet crop expects MD-domain logs.")
    md = np.asarray(logset_md.basis, dtype=np.float64)
    lo = float(min(md_min_m, md_max_m))
    hi = float(max(md_min_m, md_max_m))
    mask = np.isfinite(md) & (md >= lo) & (md <= hi)
    if int(np.count_nonzero(mask)) < int(min_samples):
        raise ValueError(f"Too few log samples in tie window: {int(np.count_nonzero(mask))} < {min_samples}.")
    new_basis = md[mask]
    logs: dict[str, grid.Log] = {}
    for key, log in logset_md.Logs.items():
        logs[key] = grid.Log(
            np.asarray(log.values, dtype=np.float64)[mask],
            new_basis,
            "md",
            name=log.name,
            unit=log.unit,
            allow_nan=log.allow_nan,
        )
    return grid.LogSet(logs)


def _source_at_twt(twt: np.ndarray, sources: np.ndarray, value: float) -> str:
    index = int(np.searchsorted(twt, float(value), side="left"))
    index = max(0, min(index, sources.size - 1))
    return str(sources[index])


def _rows_for_window(
    *,
    twt: np.ndarray,
    md: np.ndarray,
    source: np.ndarray,
    window: TargetTieWindow,
) -> pd.DataFrame:
    finite = np.isfinite(twt) & np.isfinite(md)
    twt = np.asarray(twt, dtype=np.float64)[finite]
    md = np.asarray(md, dtype=np.float64)[finite]
    source = np.asarray(source, dtype=object)[finite]
    if twt.size < 2:
        raise ValueError("Prepared TDT has fewer than 2 finite rows before clipping.")
    order = np.argsort(twt)
    twt = twt[order]
    md = md[order]
    source = source[order]
    _, unique_indices = np.unique(twt, return_index=True)
    twt = twt[unique_indices]
    md = md[unique_indices]
    source = source[unique_indices]

    actual_start = max(float(window.start_s), float(twt[0]))
    actual_end = min(float(window.end_s), float(twt[-1]))
    if actual_end <= actual_start:
        raise ValueError("Prepared TDT does not cover any part of the target tie window.")

    keep = (twt >= actual_start) & (twt <= actual_end)
    clipped_twt = twt[keep]
    clipped_md = md[keep]
    clipped_source = source[keep]
    boundary_rows: list[tuple[float, float, str]] = []
    for boundary in (actual_start, actual_end):
        if clipped_twt.size == 0 or not np.any(np.isclose(clipped_twt, boundary, rtol=0.0, atol=1e-9)):
            boundary_rows.append((boundary, float(np.interp(boundary, twt, md)), _source_at_twt(twt, source, boundary)))
    if boundary_rows:
        extra = pd.DataFrame.from_records(boundary_rows, columns=["twt_s", "md_m", "source"])
        rows = pd.concat(
            [
                pd.DataFrame({"twt_s": clipped_twt, "md_m": clipped_md, "source": clipped_source}),
                extra,
            ],
            ignore_index=True,
        )
    else:
        rows = pd.DataFrame({"twt_s": clipped_twt, "md_m": clipped_md, "source": clipped_source})
    rows = rows.sort_values("twt_s").drop_duplicates(subset=["twt_s"], keep="first").reset_index(drop=True)
    if len(rows) < 2:
        raise ValueError("Prepared TDT has fewer than 2 rows inside the target tie window.")
    return rows


def _log_basis_table_from_anchor(logset_md: grid.LogSet, *, anchor_md_m: float, anchor_twt_s: float) -> grid.TimeDepthTable:
    table = build_tdt_from_anchor(logset_md, anchor_md_m=anchor_md_m, anchor_twt_s=anchor_twt_s)
    return table


def _prepare_from_rows(
    *,
    rows: pd.DataFrame,
    logset_md: grid.LogSet,
    window: TargetTieWindow,
    min_tie_samples: int,
    report: dict[str, Any],
) -> PreparedTieWindow:
    table = grid.TimeDepthTable(twt=rows["twt_s"].to_numpy(dtype=np.float64), md=rows["md_m"].to_numpy(dtype=np.float64))
    md_min = float(np.interp(float(rows["twt_s"].iloc[0]), table.twt, table.md))
    md_max = float(np.interp(float(rows["twt_s"].iloc[-1]), table.twt, table.md))
    cropped_logset = crop_logset_md(logset_md, md_min, md_max, min_samples=min_tie_samples)
    report.update(
        {
            "target_top_name": window.top_name,
            "target_bottom_name": window.bottom_name,
            "target_top_twt_s": float(window.top_twt_s),
            "target_bottom_twt_s": float(window.bottom_twt_s),
            "target_window_start_s": float(window.start_s),
            "target_window_end_s": float(window.end_s),
            "tie_window_start_s": float(rows["twt_s"].iloc[0]),
            "tie_window_end_s": float(rows["twt_s"].iloc[-1]),
            "tie_window_duration_s": float(rows["twt_s"].iloc[-1] - rows["twt_s"].iloc[0]),
            "tie_window_md_min_m": float(cropped_logset.basis[0]),
            "tie_window_md_max_m": float(cropped_logset.basis[-1]),
            "tie_window_log_sample_count": int(cropped_logset.basis.size),
            "top_horizon_sample_method": window.top_sample_method,
            "bottom_horizon_sample_method": window.bottom_sample_method,
            "top_horizon_nearest_line_distance": window.top_nearest_line_distance,
            "bottom_horizon_nearest_line_distance": window.bottom_nearest_line_distance,
        }
    )
    return PreparedTieWindow(table=table, logset_md=cropped_logset, table_rows=rows, report=report)


def prepare_tdt_with_sonic_extension(
    *,
    raw_table: grid.TimeDepthTable,
    logset_md: grid.LogSet,
    window: TargetTieWindow,
    min_tie_samples: int = 64,
) -> PreparedTieWindow:
    """Clip an original TDT to a target window, extending from endpoints with sonic where needed."""
    if not raw_table.is_md_domain:
        raise ValueError("Sonic TDT extension expects an MD-domain TimeDepthTable.")
    if not logset_md.is_md:
        raise ValueError("Sonic TDT extension expects MD-domain logs.")
    if not tdt_overlaps_window(raw_table, window):
        raise ValueError("Raw TDT does not overlap the target tie window.")

    raw_min = float(raw_table.twt[0])
    raw_max = float(raw_table.twt[-1])
    raw_overlap = max(0.0, min(raw_max, float(window.end_s)) - max(raw_min, float(window.start_s)))
    raw_fraction = raw_overlap / max(float(window.duration_s), 1e-12)

    parts_twt: list[np.ndarray] = []
    parts_md: list[np.ndarray] = []
    parts_source: list[np.ndarray] = []
    top_extension_s = 0.0
    bottom_extension_s = 0.0

    if float(window.start_s) < raw_min:
        top_table = _log_basis_table_from_anchor(
            logset_md,
            anchor_md_m=float(raw_table.md[0]),
            anchor_twt_s=raw_min,
        )
        mask = (top_table.twt >= float(window.start_s)) & (top_table.twt < raw_min)
        if np.any(mask):
            parts_twt.append(top_table.twt[mask])
            parts_md.append(top_table.md[mask])
            parts_source.append(np.full(int(np.count_nonzero(mask)), "sonic_extension_top", dtype=object))
            top_extension_s = float(raw_min - max(float(window.start_s), float(top_table.twt[mask][0])))

    raw_mask = (raw_table.twt >= float(window.start_s)) & (raw_table.twt <= float(window.end_s))
    if raw_min <= float(window.start_s) <= raw_max:
        raw_mask[max(0, int(np.searchsorted(raw_table.twt, float(window.start_s), side="left")) - 1)] = True
    if raw_min <= float(window.end_s) <= raw_max:
        raw_mask[min(raw_table.twt.size - 1, int(np.searchsorted(raw_table.twt, float(window.end_s), side="right")))] = True
    if np.any(raw_mask):
        parts_twt.append(raw_table.twt[raw_mask])
        parts_md.append(raw_table.md[raw_mask])
        parts_source.append(np.full(int(np.count_nonzero(raw_mask)), "original_tdt", dtype=object))

    if float(window.end_s) > raw_max:
        bottom_table = _log_basis_table_from_anchor(
            logset_md,
            anchor_md_m=float(raw_table.md[-1]),
            anchor_twt_s=raw_max,
        )
        mask = (bottom_table.twt > raw_max) & (bottom_table.twt <= float(window.end_s))
        if np.any(mask):
            parts_twt.append(bottom_table.twt[mask])
            parts_md.append(bottom_table.md[mask])
            parts_source.append(np.full(int(np.count_nonzero(mask)), "sonic_extension_bottom", dtype=object))
            bottom_extension_s = float(min(float(window.end_s), float(bottom_table.twt[mask][-1])) - raw_max)

    if not parts_twt:
        raise ValueError("No TDT samples remain after target-window clipping.")
    rows = _rows_for_window(
        twt=np.concatenate(parts_twt),
        md=np.concatenate(parts_md),
        source=np.concatenate(parts_source),
        window=window,
    )

    has_extension = bool((rows["source"] != "original_tdt").any())
    if not has_extension and raw_fraction >= 0.999:
        support_class = "original_full_window"
    elif raw_fraction >= 0.5:
        support_class = "original_with_sonic_extension"
    else:
        support_class = "mostly_sonic_extended"
    clip_reasons = []
    if float(rows["twt_s"].iloc[0]) > float(window.start_s) + 1e-9:
        clip_reasons.append("log_or_tdt_top_limited")
    if float(rows["twt_s"].iloc[-1]) < float(window.end_s) - 1e-9:
        clip_reasons.append("log_or_tdt_bottom_limited")

    return _prepare_from_rows(
        rows=rows,
        logset_md=logset_md,
        window=window,
        min_tie_samples=min_tie_samples,
        report={
            "tdt_support_class": support_class,
            "original_tdt_window_fraction": float(raw_fraction),
            "original_tdt_twt_min_s": raw_min,
            "original_tdt_twt_max_s": raw_max,
            "sonic_extension_top_s": top_extension_s,
            "sonic_extension_bottom_s": bottom_extension_s,
            "window_clip_reason": ";".join(clip_reasons),
        },
    )


def prepare_anchor_tdt_for_window(
    *,
    table: grid.TimeDepthTable,
    logset_md: grid.LogSet,
    window: TargetTieWindow,
    min_tie_samples: int = 64,
    support_class: str = "anchor_integrated",
) -> PreparedTieWindow:
    """Clip an anchor-integrated TDT to the target tie window."""
    rows = _rows_for_window(
        twt=np.asarray(table.twt, dtype=np.float64),
        md=np.asarray(table.md, dtype=np.float64),
        source=np.full(int(table.size), "anchor_integrated", dtype=object),
        window=window,
    )
    clip_reasons = []
    if float(rows["twt_s"].iloc[0]) > float(window.start_s) + 1e-9:
        clip_reasons.append("log_top_limited")
    if float(rows["twt_s"].iloc[-1]) < float(window.end_s) - 1e-9:
        clip_reasons.append("log_bottom_limited")
    return _prepare_from_rows(
        rows=rows,
        logset_md=logset_md,
        window=window,
        min_tie_samples=min_tie_samples,
        report={
            "tdt_support_class": support_class,
            "original_tdt_window_fraction": 0.0,
            "original_tdt_twt_min_s": None,
            "original_tdt_twt_max_s": None,
            "sonic_extension_top_s": 0.0,
            "sonic_extension_bottom_s": 0.0,
            "window_clip_reason": ";".join(clip_reasons),
        },
    )


def build_tdt_from_anchor(
    logset_md: grid.LogSet,
    *,
    anchor_md_m: float,
    anchor_twt_s: float,
) -> grid.TimeDepthTable:
    """Build an MD-domain TDT by integrating Vp around one absolute anchor."""
    if not logset_md.is_md:
        raise ValueError("Anchor-based TDT construction expects MD-domain logs.")
    md = np.asarray(logset_md.basis, dtype=np.float64)
    vp = np.asarray(logset_md.vp, dtype=np.float64)
    finite = np.isfinite(md) & np.isfinite(vp) & (vp > 0.0)
    md = md[finite]
    vp = vp[finite]
    if md.size < 2:
        raise ValueError("Vp log has fewer than 2 finite positive samples.")

    order = np.argsort(md)
    md = md[order]
    vp = vp[order]
    unique_md, unique_indices = np.unique(md, return_index=True)
    md = unique_md
    vp = vp[unique_indices]
    if md.size < 2:
        raise ValueError("Vp log has fewer than 2 unique MD samples.")
    if not (float(md[0]) <= float(anchor_md_m) <= float(md[-1])):
        raise ValueError(f"Anchor MD {anchor_md_m} is outside log MD range [{md[0]}, {md[-1]}].")
    if not np.isfinite(anchor_twt_s) or float(anchor_twt_s) <= 0.0:
        raise ValueError(f"Anchor TWT must be a positive finite value in seconds, got {anchor_twt_s}.")

    slowness_spm = 1.0 / vp
    dmd = np.diff(md)
    if np.any(dmd <= 0.0):
        raise ValueError("Log MD basis must be strictly increasing after de-duplication.")
    incremental_twt = dmd * (slowness_spm[:-1] + slowness_spm[1:])
    relative_twt = np.concatenate(([0.0], np.cumsum(incremental_twt)))
    anchor_relative_twt = float(np.interp(float(anchor_md_m), md, relative_twt))
    twt = float(anchor_twt_s) + (relative_twt - anchor_relative_twt)

    valid = np.isfinite(twt) & (twt > 0.0)
    md = md[valid]
    twt = twt[valid]
    if md.size < 2:
        raise ValueError("Anchor-based TDT has fewer than 2 positive TWT samples.")
    if np.any(np.diff(twt) <= 0.0):
        raise ValueError("Anchor-based TDT is not strictly increasing in TWT.")
    return grid.TimeDepthTable(twt=twt, md=md)


def normalize_twt_seconds(value: float, *, unit: str = "auto") -> float:
    """Normalize Petrel horizon/checkshot time values to positive seconds."""
    raw = abs(float(value))
    if not np.isfinite(raw):
        raise ValueError(f"TWT value is not finite: {value}")
    unit_norm = str(unit or "auto").strip().casefold()
    if unit_norm in {"s", "sec", "second", "seconds"}:
        return raw
    if unit_norm in {"ms", "msec", "millisecond", "milliseconds"}:
        return raw / 1000.0
    if unit_norm == "auto":
        return raw / 1000.0 if raw > 20.0 else raw
    raise ValueError(f"Unsupported TWT unit: {unit}")


def find_well_top_md(well_tops_df: pd.DataFrame, *, well_name: str, surface: str) -> float:
    """Find one finite well-top MD using case-insensitive well/surface matching."""
    required = {"Well", "Surface", "MD"}
    missing = required.difference(well_tops_df.columns)
    if missing:
        raise ValueError(f"well_tops_df is missing required columns: {sorted(missing)}")
    well_key = normalize_well_name(well_name)
    surface_key = str(surface).strip().casefold()
    mask = well_tops_df["Well"].map(normalize_well_name).eq(well_key) & (
        well_tops_df["Surface"].astype(str).str.strip().str.casefold().eq(surface_key)
    )
    md_values = pd.to_numeric(well_tops_df.loc[mask, "MD"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    md_values = md_values[np.isfinite(md_values)]
    if md_values.size == 0:
        raise ValueError(f"No finite MD found for well top {surface!r} in well {well_name!r}.")
    if np.nanmax(md_values) - np.nanmin(md_values) > 0.01:
        raise ValueError(f"Multiple conflicting MD values found for well top {surface!r} in well {well_name!r}.")
    return float(md_values[0])


class HorizonGrid:
    """Small regular-grid interpolator for Petrel interpretation horizons."""

    def __init__(self, inline_axis: np.ndarray, xline_axis: np.ndarray, values: np.ndarray, *, name: str = ""):
        self.inline_axis = np.asarray(inline_axis, dtype=np.float64)
        self.xline_axis = np.asarray(xline_axis, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        self.name = name
        if self.values.shape != (self.inline_axis.size, self.xline_axis.size):
            raise ValueError("Horizon grid shape does not match inline/xline axes.")
        if self.inline_axis.size < 2 or self.xline_axis.size < 2:
            raise ValueError("Horizon grid requires at least 2 inline and 2 xline samples.")
        il_grid, xl_grid = np.meshgrid(self.inline_axis, self.xline_axis, indexing="ij")
        finite = np.isfinite(self.values)
        self._finite_inline = il_grid[finite]
        self._finite_xline = xl_grid[finite]
        self._finite_values = self.values[finite]
        if self._finite_values.size == 0:
            raise ValueError("Horizon grid contains no finite interpretation values.")

    @classmethod
    def from_petrel_dataframe(cls, df: pd.DataFrame, *, name: str = "") -> "HorizonGrid":
        required = {"inline", "xline", "interpretation"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"interpretation DataFrame is missing required columns: {sorted(missing)}")
        inline_axis = np.sort(pd.to_numeric(df["inline"], errors="coerce").dropna().unique().astype(np.float64))
        xline_axis = np.sort(pd.to_numeric(df["xline"], errors="coerce").dropna().unique().astype(np.float64))
        pivot = df.pivot_table(index="inline", columns="xline", values="interpretation", aggfunc="first")
        pivot = pivot.reindex(index=inline_axis, columns=xline_axis)
        return cls(inline_axis, xline_axis, pivot.to_numpy(dtype=np.float64), name=name)

    def sample_at_line(
        self,
        inline_float: float,
        xline_float: float,
        *,
        nearest_fallback_max_line_distance: float = 5.0,
    ) -> dict[str, float | str]:
        """Sample the horizon at floating line coordinates with audit metadata."""
        il = float(inline_float)
        xl = float(xline_float)
        if not (self.inline_axis[0] <= il <= self.inline_axis[-1]):
            raise ValueError(f"Inline {il} is outside horizon {self.name!r} range.")
        if not (self.xline_axis[0] <= xl <= self.xline_axis[-1]):
            raise ValueError(f"Xline {xl} is outside horizon {self.name!r} range.")

        i1 = int(np.searchsorted(self.inline_axis, il, side="right"))
        j1 = int(np.searchsorted(self.xline_axis, xl, side="right"))
        i0 = max(0, min(i1 - 1, self.inline_axis.size - 2))
        j0 = max(0, min(j1 - 1, self.xline_axis.size - 2))
        i1 = i0 + 1
        j1 = j0 + 1
        il0, il1 = self.inline_axis[i0], self.inline_axis[i1]
        xl0, xl1 = self.xline_axis[j0], self.xline_axis[j1]
        values = np.array(
            [
                [self.values[i0, j0], self.values[i0, j1]],
                [self.values[i1, j0], self.values[i1, j1]],
            ],
            dtype=np.float64,
        )
        if np.any(~np.isfinite(values)):
            distances = np.hypot(self._finite_inline - il, self._finite_xline - xl)
            nearest_index = int(np.nanargmin(distances))
            nearest_distance = float(distances[nearest_index])
            if nearest_distance <= float(nearest_fallback_max_line_distance):
                return {
                    "value": float(self._finite_values[nearest_index]),
                    "method": "nearest_valid_fallback",
                    "nearest_line_distance": nearest_distance,
                    "nearest_inline": float(self._finite_inline[nearest_index]),
                    "nearest_xline": float(self._finite_xline[nearest_index]),
                }
            raise ValueError(f"Horizon {self.name!r} has missing support around inline/xline {il}, {xl}.")
        wi = 0.0 if il1 == il0 else (il - il0) / (il1 - il0)
        wj = 0.0 if xl1 == xl0 else (xl - xl0) / (xl1 - xl0)
        value = float(
            values[0, 0] * (1.0 - wi) * (1.0 - wj)
            + values[1, 0] * wi * (1.0 - wj)
            + values[0, 1] * (1.0 - wi) * wj
            + values[1, 1] * wi * wj
        )
        return {
            "value": value,
            "method": "bilinear",
            "nearest_line_distance": 0.0,
            "nearest_inline": il,
            "nearest_xline": xl,
        }

    def value_at_line(
        self,
        inline_float: float,
        xline_float: float,
        *,
        nearest_fallback_max_line_distance: float = 5.0,
    ) -> float:
        """Return the sampled horizon value at floating line coordinates."""
        sample = self.sample_at_line(
            inline_float,
            xline_float,
            nearest_fallback_max_line_distance=nearest_fallback_max_line_distance,
        )
        return float(sample["value"])


def write_time_depth_table_csv(
    table: grid.TimeDepthTable,
    path: str | Path,
    *,
    sources: Sequence[str] | None = None,
) -> None:
    """Write a wtie TimeDepthTable to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_name = "md_m" if table.is_md_domain else "tvdss_m"
    rows = pd.DataFrame({"twt_s": table.twt, depth_name: table.depth})
    if sources is not None:
        if len(sources) != len(rows):
            raise ValueError("sources length must match the time-depth table length.")
        rows["source"] = list(sources)
    rows.to_csv(path, index=False)
