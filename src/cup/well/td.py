"""cup.well.td: 时深关系（Time-Depth, TD）构建与窗口裁剪。

本模块提供时间域井震标定所需的时深关系加载、构建、裁剪与校验工具，
以及目标层段窗口定义与锚点 TDT 构建。

边界说明
--------
- 本模块不负责 auto-tie 优化策略本身，仅提供数据准备层。
- Petrel checkshot 文本解析由 ``cup.petrel.load.import_petrel_checkshots_dataframe`` 完成；
  本模块只负责把解析表转换为项目内 ``TimeDepthTable``。
- TVDSS/MD 口径差异由调用方在进入本模块前统一。

核心公开对象
------------
1. TargetTieWindow / PreparedTieWindow: 标定窗口定义与裁剪结果。
2. load_petrel_time_depth_table: 读取 Petrel 时深表并归一化到正秒。
3. build_tdt_from_anchor: 基于单一锚点构建时深关系。
4. prepare_tdt_with_sonic_extension: 原始 TDT 裁剪 + 声波外推。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from cup.petrel.load import import_petrel_checkshots_dataframe
from cup.well.assets import normalize_well_name
from wtie.processing import grid


@dataclass(frozen=True)
class TargetTieWindow:
    """目标层段标定窗口，以 TWT 秒定义。"""

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
    """裁剪到标定窗口的本地 TDT 与 LogSet 组合。"""

    table: grid.TimeDepthTable
    logset_md: grid.LogSet
    table_rows: pd.DataFrame
    report: dict[str, Any]


def _read_petrel_checkshot_dataframe(path: Path) -> pd.DataFrame:
    """委托 Petrel 格式 Adapter 解析 checkshot 文本。"""
    return import_petrel_checkshots_dataframe(path)


def load_petrel_time_depth_table(path: str | Path, *, domain: Literal["md", "tvdss"] = "md") -> grid.TimeDepthTable:
    """读取 Petrel checkshot/时深表并转换为 wtie 时深表。

    本项目中的 Petrel 导出通常以负毫秒保存解释 TWT。时间域工作流使用正秒，
    因此本读取器会应用 ``abs(twt_ms) / 1000``，并按指定深度域排序。
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


def validate_time_depth_table(
    table: grid.TimeDepthTable,
    log_basis_md: np.ndarray,
    *,
    min_overlap_samples: int = 64,
) -> dict[str, float | int]:
    """校验时深表与测井采样轴之间的 MD 重叠范围。"""
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
    """判断 MD 域 TDT 是否接触目标 TWT 窗口。"""
    if not table.is_md_domain:
        raise ValueError("TDT/window overlap expects an MD-domain TimeDepthTable.")
    return min(float(table.twt[-1]), float(window.end_s)) > max(float(table.twt[0]), float(window.start_s))


def crop_logset_md(logset_md: grid.LogSet, md_min_m: float, md_max_m: float, *, min_samples: int = 2) -> grid.LogSet:
    """按 MD 裁剪 MD 域 LogSet，并保留原曲线名。"""
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
    """将原始 TDT 裁剪到目标窗口，不足部分用声波积分外推补齐。

    Parameters
    ----------
    raw_table : grid.TimeDepthTable
        原始 MD 域时深表。
    logset_md : grid.LogSet
        MD 域 Vp/Rho LogSet，用于声波外推。
    window : TargetTieWindow
        目标标定窗口。
    min_tie_samples : int, default=64
        最少标定采样点数。

    Returns
    -------
    PreparedTieWindow
        裁剪并外推后的标定窗口数据。
    """
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
    """将基于锚点积分的 TDT 裁剪到目标标定窗口。

    Parameters
    ----------
    table : grid.TimeDepthTable
        锚点积分得到的 MD 域时深表。
    logset_md : grid.LogSet
        MD 域 Vp/Rho LogSet，用于裁剪窗口深度。
    window : TargetTieWindow
        目标标定窗口。
    min_tie_samples : int, default=64
        最少标定采样点数。
    support_class : str, default="anchor_integrated"
        TDT 支撑类型标签。

    Returns
    -------
    PreparedTieWindow
        裁剪后的标定窗口数据。
    """
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
    """围绕一个绝对锚点积分 Vp，构建 MD 域 TDT。"""
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
    """将 Petrel 层位或 checkshot 时间值归一化为正秒。"""
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
    """按大小写不敏感的井名和层名匹配，查找唯一有限分层 MD。"""
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


def write_time_depth_table_csv(
    table: grid.TimeDepthTable,
    path: str | Path,
    *,
    sources: Sequence[str] | None = None,
) -> None:
    """将 wtie TimeDepthTable 写出为 CSV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_name = "md_m" if table.is_md_domain else "tvdss_m"
    rows = pd.DataFrame({"twt_s": table.twt, depth_name: table.depth})
    if sources is not None:
        if len(sources) != len(rows):
            raise ValueError("sources length must match the time-depth table length.")
        rows["source"] = list(sources)
    rows.to_csv(path, index=False)
