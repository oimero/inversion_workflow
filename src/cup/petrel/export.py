"""Petrel数据导出工具。"""

from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import lasio
import numpy as np

from wtie.processing import grid


class ExportLogsetsSummary(TypedDict):
    """`export_logsets_to_las` 的结构化返回类型。"""

    exported_files: List[Path]
    skipped_wells: List[Dict[str, str]]
    skipped_curves: List[Dict[str, str]]


def _resolve_export_curve(logset: grid.LogSet, curve_name: str) -> grid.Log:
    """按曲线名从 LogSet 获取可导出曲线（含派生曲线）。"""
    if curve_name in logset.Logs:
        return logset.Logs[curve_name]

    if curve_name == "AI":
        return logset.AI

    if curve_name == "Vp_Vs_ratio":
        return logset.Vp_Vs_ratio

    raise KeyError(f"曲线不存在: {curve_name}")


def _build_las_from_logset(
    well_name: str,
    logset: grid.LogSet,
    selected_curve_names: List[str],
    null_value: float,
) -> lasio.LASFile:
    """将单井 LogSet 组装为 LASFile。"""
    las = lasio.LASFile()
    las.well["WELL"].value = well_name
    las.well["NULL"].value = float(null_value)

    basis = np.asarray(logset.basis, dtype=float)
    las.append_curve("DEPT", basis, unit="m", descr="Depth")

    for curve_name in selected_curve_names:
        log = _resolve_export_curve(logset, curve_name)
        values = np.asarray(log.values, dtype=float)
        unit = "" if log.unit is None else str(log.unit)
        las.append_curve(curve_name, values, unit=unit, descr=curve_name)

    return las


def export_logsets_to_las(
    logsets: Dict[str, grid.LogSet],
    output_dir: Path,
    curve_names: Optional[List[str]] = None,
    null_value: float = -999.25,
) -> ExportLogsetsSummary:
    """按井批量导出 LogSet 到 LAS 文件。

    Parameters
    ----------
    logsets : Dict[str, grid.LogSet]
            键为井名，值为对应 LogSet。
    output_dir : Path
            LAS 输出目录。
    curve_names : List[str], optional
            指定导出曲线列表，按传入顺序导出。
            若为 None，则导出该井 LogSet 中全部已有曲线。
            支持派生曲线名称: ``AI``、``Vp_Vs_ratio``。
    null_value : float, default -999.25
            导出 LAS 的 NULL 指示值。

    Returns
    -------
    ExportLogsetsSummary
            导出汇总信息，包含：
            - exported_files: List[Path]
            - skipped_wells: List[dict]
            - skipped_curves: List[dict]

    Notes
    -----
    - 全流程静默跳过，不因单井/单曲线失败中断。
    - 当指定曲线不存在时，记录到 skipped_curves。
    - 若某井最终无可导出曲线，记录到 skipped_wells。
    - 同名文件会覆盖。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files: List[Path] = []
    skipped_wells: List[Dict[str, str]] = []
    skipped_curves: List[Dict[str, str]] = []

    for well_name, logset in logsets.items():
        try:
            if curve_names is None:
                requested_curve_names = list(logset.Logs.keys())
            else:
                requested_curve_names = list(curve_names)

            available_curve_names: List[str] = []
            for curve_name in requested_curve_names:
                try:
                    _resolve_export_curve(logset, curve_name)
                    available_curve_names.append(curve_name)
                except Exception as exc:
                    skipped_curves.append({"well": well_name, "curve": curve_name, "reason": str(exc)})

            if not available_curve_names:
                skipped_wells.append({"well": well_name, "reason": "无可导出曲线"})
                continue

            las = _build_las_from_logset(
                well_name=well_name,
                logset=logset,
                selected_curve_names=available_curve_names,
                null_value=null_value,
            )

            output_file = output_dir / f"{well_name}.las"
            las.write(str(output_file), version=2.0, wrap=False)
            exported_files.append(output_file)

        except Exception as exc:
            skipped_wells.append({"well": well_name, "reason": str(exc)})

    return {
        "exported_files": exported_files,
        "skipped_wells": skipped_wells,
        "skipped_curves": skipped_curves,
    }
