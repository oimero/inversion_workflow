"""井曲线处理工具。"""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, cast

import lasio
import numpy as np
import pandas as pd

from wtie.processing import grid

_REQUIRED_TOPS_COLUMNS = ("Well", "Surface", "MD")
_SENTINEL_VALUES = (-999.0, -999.25, -99999.0)
_DEFAULT_ANOMALY_VALUE = -999.25


def _get_unique_surface_md(
    well_tops_df: pd.DataFrame,
    well_name: str,
    surface_name: str,
) -> float:
    """返回指定井与层位对应的唯一 MD 深度（单位 m）。"""
    mask = (well_tops_df["Well"] == well_name) & (well_tops_df["Surface"] == surface_name)
    md_values = pd.to_numeric(well_tops_df.loc[mask, "MD"], errors="coerce").dropna()

    if md_values.empty:
        raise ValueError(f"井 {well_name} 未查询到层位 {surface_name} 的有效 MD。")

    unique_md = md_values.unique()
    if unique_md.size > 1:
        raise ValueError(f"井 {well_name} 的层位 {surface_name} 存在多个 MD 值: {unique_md.tolist()}。")

    return float(unique_md[0])


def clip_logsets_by_well_tops(
    well_tops_df: pd.DataFrame,
    logsets: Dict[str, grid.LogSet],
    top_surface_name: str,
    base_surface_name: str,
    extend: int = 0,
) -> Dict[str, grid.LogSet]:
    """按井分层在 MD 域裁剪井曲线集合（包含端点）。

    当层位区间超出曲线范围时，会自动钳制到曲线边界：
    - 起始层位浅于曲线起点时，使用曲线起点；
    - 终止层位深于曲线终点时，使用曲线终点。
    若设置 extend，会在裁剪区间两端各额外扩展对应数量的采样点。

    Parameters
    ----------
    well_tops_df : pd.DataFrame
            井分层表，至少包含 ``Well``、``Surface``、``MD`` 三列，MD 单位为 m。
    logsets : Dict[str, grid.LogSet]
            井曲线集合字典，键为井名。
    top_surface_name : str
            裁剪起始层位名称。
    base_surface_name : str
            裁剪终止层位名称。
    extend : int, default=0
        在 top_surface 上方、base_surface 下方额外扩展的采样点数。

    Returns
    -------
    Dict[str, grid.LogSet]
            按层位范围裁剪后的新字典，键与输入一致。

    Raises
    ------
    ValueError
            当井分层缺列、层位未查询到、层位 MD 反转，或输入 LogSet 非 MD 域时抛出。
    """
    missing = [col for col in _REQUIRED_TOPS_COLUMNS if col not in well_tops_df.columns]
    if missing:
        raise ValueError(f"well_tops_df 缺少必需列: {missing}")

    if isinstance(extend, bool) or not isinstance(extend, int) or extend < 0:
        raise ValueError(f"extend 必须是大于等于 0 的整数，当前为: {extend}")

    clipped_logsets: Dict[str, grid.LogSet] = {}

    for well_name, logset in logsets.items():
        if not logset.is_md:
            raise ValueError(f"井 {well_name} 的 LogSet 不是 MD 域，无法按 MD 裁剪。")

        top_md = _get_unique_surface_md(well_tops_df, well_name, top_surface_name)
        base_md = _get_unique_surface_md(well_tops_df, well_name, base_surface_name)

        if top_md > base_md:
            raise ValueError(
                f"井 {well_name} 层位深度反转: {top_surface_name}={top_md} m, {base_surface_name}={base_md} m。"
            )

        clip_start_md = max(top_md, float(logset.basis[0]))
        clip_end_md = min(base_md, float(logset.basis[-1]))

        if clip_start_md > clip_end_md:
            raise ValueError(
                f"井 {well_name} 的层位区间与日志 MD 范围无重叠: "
                f"[{top_md}, {base_md}] m vs [{float(logset.basis[0])}, {float(logset.basis[-1])}] m。"
            )

        basis = logset.basis
        idx_start = max(0, int((abs(basis - clip_start_md)).argmin()) - extend)
        idx_end = min(basis.size - 1, int((abs(basis - clip_end_md)).argmin()) + extend)
        slice_start_md = float(basis[idx_start])
        slice_end_md = float(basis[idx_end])

        clipped_logs: Dict[str, grid.Log] = {}
        for name, log in logset.Logs.items():
            sliced = log.time_slice(slice_start_md, slice_end_md)
            if not isinstance(sliced, grid.Log):
                raise TypeError(f"井 {well_name} 的曲线 {name} 裁剪后类型不是 Log。")
            clipped_logs[name] = cast(grid.Log, sliced)

        clipped_logsets[well_name] = grid.LogSet(clipped_logs)

    return clipped_logsets


def _normalize_curve_name_set(
    names: Optional[Sequence[str]],
    param_name: str,
) -> Optional[set[str]]:
    """标准化曲线名称参数，返回集合或 None。"""
    if names is None:
        return None

    if isinstance(names, str):
        result = {names}
    elif isinstance(names, Iterable):
        result = set(names)
    else:
        raise TypeError(f"{param_name} 必须是字符串序列或 None。")

    if any(not isinstance(name, str) for name in result):
        raise TypeError(f"{param_name} 中所有元素都必须是字符串。")
    if any(name.strip() == "" for name in result):
        raise ValueError(f"{param_name} 中不允许空字符串。")

    return result


def _replace_constant_runs_with_anomaly(
    values: np.ndarray,
    min_run_length: int,
    anomaly_value: float,
) -> tuple[np.ndarray, int, int]:
    """将连续相同值且长度达阈值的区间替换为异常值。"""
    cleaned = values.copy()
    # NaN 与约定哨兵值均视为缺失值，不参与连续相同值判定。
    missing_mask = np.isnan(cleaned) | np.isin(cleaned, _SENTINEL_VALUES)

    run_count = 0
    replaced_points = 0
    i = 0
    n = cleaned.size
    while i < n:
        if missing_mask[i]:
            i += 1
            continue

        j = i + 1
        while j < n and (not missing_mask[j]) and cleaned[j] == cleaned[i]:
            j += 1

        run_len = j - i
        if run_len >= min_run_length:
            cleaned[i:j] = anomaly_value
            run_count += 1
            replaced_points += run_len

        i = j

    return cleaned, run_count, replaced_points


def replace_constant_value_intervals_in_las(
    las_file_path: str,
    min_run_length: int,
    curve_names: Optional[Sequence[str]] = None,
    exclude_curve_names: Optional[Sequence[str]] = None,
    anomaly_value: float = _DEFAULT_ANOMALY_VALUE,
    write_fmt: str = "%.5f",
) -> Dict[str, Any]:
    """将 LAS 中连续相同值区间替换为异常值并写入 output 目录。

    判定规则：
    - 连续相同值采用严格相等判定；
    - 当连续长度大于等于 ``min_run_length`` 时，该区间全部替换为 ``anomaly_value``；
    - 缺失值（NaN、-999.0、-999.25、-99999）不参与连续区间判定。

    Parameters
    ----------
    las_file_path : str
        输入 LAS 文件路径。
    min_run_length : int
        连续相同值触发替换的最小长度（大于等于该值触发）。
    curve_names : Optional[Sequence[str]], default=None
        仅处理这些曲线名；若为 None，则默认处理全部曲线。
    exclude_curve_names : Optional[Sequence[str]], default=None
        从处理范围中排除这些曲线名。
    anomaly_value : float, default=-999.25
        用于替换目标区间的异常值。
    write_fmt : str, default="%.5f"
        LAS 写出数值格式。用于统一普通数值与 NULL 映射值的小数显示。

    Returns
    -------
    Dict[str, Any]
        简洁处理报告，包括输出路径与逐曲线替换统计。

    Raises
    ------
    ValueError
        当参数不合法、曲线名冲突、曲线名不存在时抛出。
    """
    input_path = Path(las_file_path)
    if not input_path.exists() or not input_path.is_file():
        raise ValueError(f"LAS 文件不存在: {las_file_path}")

    if isinstance(min_run_length, bool) or not isinstance(min_run_length, int) or min_run_length < 1:
        raise ValueError(f"min_run_length 必须是大于等于 1 的整数，当前为: {min_run_length}")

    if isinstance(anomaly_value, bool):
        raise ValueError(f"anomaly_value 必须是数值类型，当前为: {anomaly_value}")
    try:
        anomaly_value = float(anomaly_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"anomaly_value 必须是数值类型，当前为: {anomaly_value}") from exc
    if np.isnan(anomaly_value):
        raise ValueError("anomaly_value 不能为 NaN。")

    if not isinstance(write_fmt, str) or write_fmt.strip() == "":
        raise ValueError("write_fmt 必须是非空字符串，例如 '%.5f'。")
    try:
        _ = write_fmt % 1.2345
    except (TypeError, ValueError) as exc:
        raise ValueError(f"write_fmt 非法: {write_fmt}，请使用类似 '%.5f' 的格式字符串。") from exc

    include_set = _normalize_curve_name_set(curve_names, "curve_names")
    exclude_set = _normalize_curve_name_set(exclude_curve_names, "exclude_curve_names")

    if include_set and exclude_set:
        overlap = include_set & exclude_set
        if overlap:
            raise ValueError(f"curve_names 与 exclude_curve_names 存在冲突项: {sorted(overlap)}")

    las = lasio.read(str(input_path))
    available_curve_names = {str(curve.mnemonic) for curve in las.curves}

    if include_set:
        missing_include = sorted(include_set - available_curve_names)
        if missing_include:
            raise ValueError(f"curve_names 中存在未找到的曲线: {missing_include}")

    if exclude_set:
        missing_exclude = sorted(exclude_set - available_curve_names)
        if missing_exclude:
            raise ValueError(f"exclude_curve_names 中存在未找到的曲线: {missing_exclude}")

    target_curve_indices: list[int] = []
    for idx, curve in enumerate(las.curves):
        mnemonic = str(curve.mnemonic)
        if include_set is not None and mnemonic not in include_set:
            continue
        if exclude_set is not None and mnemonic in exclude_set:
            continue
        target_curve_indices.append(idx)

    per_curve_report: list[Dict[str, Any]] = []
    total_replaced_points = 0
    for idx in target_curve_indices:
        curve = las.curves[idx]
        numeric_values = pd.to_numeric(pd.Series(curve.data), errors="coerce").to_numpy(dtype=float)
        cleaned_values, run_count, replaced_points = _replace_constant_runs_with_anomaly(
            numeric_values,
            min_run_length=min_run_length,
            anomaly_value=anomaly_value,
        )

        if replaced_points > 0:
            curve.data = cleaned_values
            per_curve_report.append(
                {
                    "curve": str(curve.mnemonic),
                    "curve_index": idx,
                    "run_count": run_count,
                    "replaced_points": replaced_points,
                }
            )
            total_replaced_points += replaced_points

    output_dir = input_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name

    # 统一 NULL 字段显示格式，避免与数据区数值出现 -99999 / -99999.00000 混排。
    try:
        null_numeric_value = float(las.well["NULL"].value)
    except (KeyError, TypeError, ValueError):
        null_numeric_value = None
    if null_numeric_value is not None:
        las.well["NULL"].value = write_fmt % null_numeric_value

    las.write(str(output_path), fmt=write_fmt)

    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "anomaly_value": anomaly_value,
        "write_fmt": write_fmt,
        "target_curve_count": len(target_curve_indices),
        "curves_with_replacement": len(per_curve_report),
        "total_replaced_points": total_replaced_points,
        "curve_reports": per_curve_report,
    }
