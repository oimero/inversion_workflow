"""井曲线处理工具。"""

from typing import Dict, cast

import pandas as pd

from wtie.processing import grid

_REQUIRED_TOPS_COLUMNS = ("Well", "Surface", "MD")


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
