"""井曲线处理工具。"""

from typing import Dict

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
) -> Dict[str, grid.LogSet]:
    """按井分层在 MD 域裁剪井曲线集合（包含端点）。

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

        clipped_logs = {name: log.time_slice(top_md, base_md) for name, log in logset.Logs.items()}
        clipped_logsets[well_name] = grid.LogSet(clipped_logs)

    return clipped_logsets
