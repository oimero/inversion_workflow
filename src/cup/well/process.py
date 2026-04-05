"""井曲线对象处理工具。"""

from typing import Any, Dict, Iterable, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd

from wtie.processing import grid

_REQUIRED_TOPS_COLUMNS = ("Well", "Surface", "MD")
_SENTINEL_VALUES = (-999.0, -999.25, -99999.0)
_DEFAULT_ANOMALY_VALUE = -999.25
LogsetInput = Union[grid.LogSet, Dict[str, grid.Log]]


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


def _normalize_logset_input(
    logset_input: LogsetInput,
    well_name: str,
    *,
    require_non_empty: bool = True,
) -> tuple[Dict[str, grid.Log], bool]:
    """将单井输入标准化为曲线字典，并返回是否原始为 LogSet。"""
    if isinstance(logset_input, grid.LogSet):
        logs = dict(logset_input.Logs)
        if require_non_empty and not logs:
            raise TypeError(f"井 {well_name} 的 LogSet 不能为空。")
        return logs, True

    if not isinstance(logset_input, dict):
        raise TypeError(f"井 {well_name} 的输入必须是 grid.LogSet 或 Dict[str, grid.Log]。")

    if require_non_empty and not logset_input:
        raise TypeError(f"井 {well_name} 的曲线映射必须是非空 Dict[str, grid.Log]。")

    if any(not isinstance(name, str) for name in logset_input.keys()):
        raise TypeError(f"井 {well_name} 的曲线名必须是字符串。")

    return cast(Dict[str, grid.Log], logset_input), False


def replace_constant_value_intervals_in_log_dicts(
    logsets: Dict[str, LogsetInput],
    min_run_length: int,
    curve_names: Optional[Sequence[str]] = None,
    exclude_curve_names: Optional[Sequence[str]] = None,
    anomaly_value: float = _DEFAULT_ANOMALY_VALUE,
) -> Dict[str, Any]:
    """批量替换 Dict[str, grid.Log] 中连续常值区间。

    判定规则：
    - 连续相同值采用严格相等判定；
    - 当连续长度大于等于 ``min_run_length`` 时，该区间全部替换为 ``anomaly_value``；
    - 缺失值（NaN、-999.0、-999.25、-99999）不参与连续区间判定。

    Parameters
    ----------
    logsets : Dict[str, Dict[str, grid.Log]]
            键为井名，值为该井曲线字典（曲线名 -> ``grid.Log``）。
    min_run_length : int
            连续相同值触发替换的最小长度（大于等于该值触发）。
    curve_names : Optional[Sequence[str]], default=None
            仅处理这些曲线名；若为 None，则处理每口井全部曲线。
    exclude_curve_names : Optional[Sequence[str]], default=None
            从处理范围中排除这些曲线名。
    anomaly_value : float, default=-999.25
            用于替换目标区间的异常值。

    Returns
    -------
    Dict[str, Any]
            处理结果，包含：
            - processed_logsets: Dict[str, Dict[str, grid.Log]]
            - anomaly_value: float
            - target_curve_count: int
            - curves_with_replacement: int
            - total_replaced_points: int
            - curve_reports: List[dict]

    Raises
    ------
    ValueError
            当参数不合法、曲线名冲突、井内曲线名不存在时抛出。
    TypeError
            当输入结构或曲线对象类型不符合约定时抛出。
    """
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

    include_set = _normalize_curve_name_set(curve_names, "curve_names")
    exclude_set = _normalize_curve_name_set(exclude_curve_names, "exclude_curve_names")

    if include_set and exclude_set:
        overlap = include_set & exclude_set
        if overlap:
            raise ValueError(f"curve_names 与 exclude_curve_names 存在冲突项: {sorted(overlap)}")

    processed_logsets: Dict[str, LogsetInput] = {}
    per_curve_report: list[Dict[str, Any]] = []
    total_target_curve_count = 0
    total_replaced_points = 0

    for well_name, logset_input in logsets.items():
        logs, input_is_logset = _normalize_logset_input(logset_input, well_name, require_non_empty=True)

        available_curve_names = set(logs.keys())
        if include_set:
            missing_include = sorted(include_set - available_curve_names)
            if missing_include:
                raise ValueError(f"井 {well_name} 的 curve_names 中存在未找到曲线: {missing_include}")

        if exclude_set:
            missing_exclude = sorted(exclude_set - available_curve_names)
            if missing_exclude:
                raise ValueError(f"井 {well_name} 的 exclude_curve_names 中存在未找到曲线: {missing_exclude}")

        processed_logs: Dict[str, grid.Log] = {}
        for curve_name, log in logs.items():
            if not isinstance(log, grid.Log):
                raise TypeError(f"井 {well_name} 的曲线 {curve_name} 不是 grid.Log。")

            if include_set is not None and curve_name not in include_set:
                processed_logs[curve_name] = log
                continue
            if exclude_set is not None and curve_name in exclude_set:
                processed_logs[curve_name] = log
                continue

            total_target_curve_count += 1

            numeric_values = pd.to_numeric(pd.Series(log.values), errors="coerce").to_numpy(dtype=float)
            cleaned_values, run_count, replaced_points = _replace_constant_runs_with_anomaly(
                numeric_values,
                min_run_length=min_run_length,
                anomaly_value=anomaly_value,
            )

            if replaced_points > 0:
                updated_log = cast(grid.Log, grid.update_trace_values(cleaned_values, log))
                if hasattr(log, "unit"):
                    setattr(updated_log, "unit", getattr(log, "unit"))
                processed_logs[curve_name] = updated_log

                per_curve_report.append(
                    {
                        "well": well_name,
                        "curve": curve_name,
                        "run_count": run_count,
                        "replaced_points": replaced_points,
                    }
                )
                total_replaced_points += replaced_points
            else:
                processed_logs[curve_name] = log

        if input_is_logset:
            processed_logsets[well_name] = grid.LogSet(processed_logs)
        else:
            processed_logsets[well_name] = processed_logs

    return {
        "processed_logsets": processed_logsets,
        "anomaly_value": anomaly_value,
        "target_curve_count": total_target_curve_count,
        "curves_with_replacement": len(per_curve_report),
        "total_replaced_points": total_replaced_points,
        "curve_reports": per_curve_report,
    }


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
    logsets: Dict[str, LogsetInput],
    top_surface_name: str,
    base_surface_name: str,
    extend: int = 0,
) -> Dict[str, Any]:
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
        Dict[str, Any]
            处理结果，包含：
            - processed_logsets: Dict[str, LogsetInput]
            - target_well_count: int
            - wells_with_fallback: int
            - surface_fallback_count: int
            - surface_fallback_reports: List[dict]

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

    clipped_logsets: Dict[str, LogsetInput] = {}
    surface_fallback_reports: list[Dict[str, Any]] = []

    for well_name, logset_input in logsets.items():
        logs, input_is_logset = _normalize_logset_input(logset_input, well_name, require_non_empty=True)
        if input_is_logset:
            logset = cast(grid.LogSet, logset_input)
        else:
            logset = grid.LogSet(logs)

        if not logset.is_md:
            raise ValueError(f"井 {well_name} 的 LogSet 不是 MD 域，无法按 MD 裁剪。")

        default_md = float(logset.basis[0])
        top_fallback = False
        base_fallback = False

        try:
            top_md = _get_unique_surface_md(well_tops_df, well_name, top_surface_name)
        except ValueError as exc:
            if "未查询到层位" not in str(exc):
                raise
            top_md = default_md
            top_fallback = True
            surface_fallback_reports.append(
                {
                    "well": well_name,
                    "surface": top_surface_name,
                    "fallback_md": top_md,
                    "reason": "surface_not_found",
                }
            )

        try:
            base_md = _get_unique_surface_md(well_tops_df, well_name, base_surface_name)
        except ValueError as exc:
            if "未查询到层位" not in str(exc):
                raise
            base_md = default_md
            base_fallback = True
            surface_fallback_reports.append(
                {
                    "well": well_name,
                    "surface": base_surface_name,
                    "fallback_md": base_md,
                    "reason": "surface_not_found",
                }
            )

        if top_md > base_md:
            if top_fallback or base_fallback:
                top_md, base_md = min(top_md, base_md), max(top_md, base_md)
            else:
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
        if idx_start == idx_end and basis.size > 1:
            idx_end = min(basis.size - 1, idx_end + 1)
        slice_start_md = float(basis[idx_start])
        slice_end_md = float(basis[idx_end])

        clipped_logs: Dict[str, grid.Log] = {}
        for name, log in logset.Logs.items():
            sliced = log.time_slice(slice_start_md, slice_end_md)
            if not isinstance(sliced, grid.Log):
                raise TypeError(f"井 {well_name} 的曲线 {name} 裁剪后类型不是 Log。")
            clipped_logs[name] = cast(grid.Log, sliced)

        if input_is_logset:
            clipped_logsets[well_name] = grid.LogSet(clipped_logs)
        else:
            clipped_logsets[well_name] = clipped_logs

    return {
        "processed_logsets": clipped_logsets,
        "target_well_count": len(logsets),
        "wells_with_fallback": len({report["well"] for report in surface_fallback_reports}),
        "surface_fallback_count": len(surface_fallback_reports),
        "surface_fallback_reports": surface_fallback_reports,
    }
