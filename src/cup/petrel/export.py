"""cup.petrel.export: Petrel 相关文本导出工具。

本模块提供 Petrel checkshots 文本导出能力，用于将项目内部
``wtie.processing.grid`` 数据结构转换为 Petrel 常用交换格式。

边界说明
--------
- 本模块仅负责文件组织、字段映射与基础格式校验。
- 本模块不负责井曲线反演、时深关系优化或 Petrel 工程导入流程本身。
- 导出前的数据清洗、重采样与单位统一应由上游流程保证。
- LAS 导出属于井曲线 I/O，请使用 ``cup.well.las``。

核心公开对象
------------
1. export_vertical_tdt_to_petrel_checkshots: 导出直井 checkshots 文本。
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from wtie.processing import grid


def _normalize_optional_column(
    value: Optional[Union[float, np.ndarray]],
    size: int,
    name: str,
) -> Optional[np.ndarray]:
    """将可选标量/数组列标准化为长度一致的一维数组。"""
    if value is None:
        return None

    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(size, float(arr), dtype=float)

    arr = arr.reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} 长度不匹配: expect {size}, got {arr.size}")

    return arr


def export_vertical_tdt_to_petrel_checkshots(
    output_file: Path,
    tdt: grid.TimeDepthTable,
    well_name: str,
    kb: float,
    x: float,
    y: float,
    average_velocity: Optional[Union[float, np.ndarray]] = None,
    interval_velocity: Optional[Union[float, np.ndarray]] = None,
) -> Path:
    """导出直井时深关系到 Petrel checkshots 文本。

    该函数以 ``grid.TimeDepthTable`` 的 TVDSS 采样点为主轴，
    生成符合项目约定的 Petrel checkshots 纯文本文件。

    Parameters
    ----------
    output_file : Path
        输出文件路径。
    tdt : grid.TimeDepthTable
        时深关系表。当前函数仅支持 TVDSS 域表。
    well_name : str
        井名，会写入 ``Well name`` 列。
    kb : float
        Kelly Bushing 高程（m）。
    x : float
        井口 X 坐标（m）。直井情形下所有样点共用该值。
    y : float
        井口 Y 坐标（m）。直井情形下所有样点共用该值。
    average_velocity : float or np.ndarray, optional
        ``Average velocity`` 列。可传标量（全列常数）或与样点等长数组。
    interval_velocity : float or np.ndarray, optional
        ``Interval velocity`` 列。可传标量（全列常数）或与样点等长数组。

    Returns
    -------
    Path
        写出的文件路径。

    Raises
    ------
    ValueError
        当 ``tdt`` 为 MD 域、存在非有限值、内部 TWT 为负值或列长度不一致时抛出。

    Notes
    -----
    - ``Z`` 为负值，单位 m。
    - ``MD`` 为正值，单位 m，并按约定计算: ``MD = |Z| + kb``。
    - ``TWT`` 为负值，单位 ms（由内部正秒 ``tdt.twt`` 转换得到）。
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if tdt.is_md_domain:
        raise ValueError("仅支持 TVDSS 域 TimeDepthTable；当前输入为 MD 域。")

    z = -np.abs(np.asarray(tdt.tvdss, dtype=float))
    md = np.abs(z) + float(kb)
    twt_s = np.asarray(tdt.twt, dtype=float)
    twt_ms = -np.abs(twt_s) * 1000.0

    if not np.isfinite(z).all() or not np.isfinite(md).all() or not np.isfinite(twt_s).all():
        raise ValueError("Z/MD/TWT 存在非有限值，无法导出。")
    if (twt_s < 0.0).any():
        raise ValueError("内部 TWT 必须为非负值（s）。")

    n = z.size
    if md.size != n or twt_ms.size != n:
        raise ValueError("Z/MD/TWT 采样长度不一致。")

    avg_v = _normalize_optional_column(average_velocity, n, "average_velocity")
    int_v = _normalize_optional_column(interval_velocity, n, "interval_velocity")

    header = ["X", "Y", "Z", "MD", "TWT"]
    if avg_v is not None:
        header.append("Average velocity")
    if int_v is not None:
        header.append("Interval velocity")
    header.append("Well name")

    lines = [
        "VERSION 1",
        "BEGIN HEADER",
        *header,
        "END HEADER",
    ]

    for i in range(n):
        row = [
            f"{float(x):.2f}",
            f"{float(y):.2f}",
            f"{z[i]:.2f}",
            f"{md[i]:.2f}",
            f"{twt_ms[i]:.2f}",
        ]
        if avg_v is not None:
            row.append(f"{avg_v[i]:.6g}")
        if int_v is not None:
            row.append(f"{int_v[i]:.6g}")
        row.append(f'"{well_name}"')
        lines.append(" ".join(row))

    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_file
