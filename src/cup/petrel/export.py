"""Petrel数据导出工具。"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import lasio
import numpy as np

from wtie.processing import grid

WellExportInput = Union[grid.LogSet, Dict[str, grid.Log]]


def _extract_logs_mapping(well_data: WellExportInput) -> Mapping[str, grid.Log]:
    """从单井数据中提取曲线映射。"""
    if hasattr(well_data, "Logs"):
        logs = getattr(well_data, "Logs")
    elif isinstance(well_data, Mapping):
        logs = well_data
    else:
        logs = None

    if not isinstance(logs, Mapping) or not logs:
        raise KeyError("单井数据缺少有效的日志映射")

    for curve_name, curve in logs.items():
        if not isinstance(curve, grid.Log):
            raise TypeError(f"曲线 {curve_name} 不是 grid.Log")

    return logs


def _extract_basis(well_data: WellExportInput) -> np.ndarray:
    """从单井数据中提取深度基准。"""
    if hasattr(well_data, "basis"):
        basis = getattr(well_data, "basis")
    else:
        logs = _extract_logs_mapping(well_data)
        first_log = next(iter(logs.values()))
        basis = first_log.basis
        first_basis_type = first_log.basis_type
        for curve_name, log in logs.items():
            if not np.allclose(basis, log.basis):
                raise ValueError(f"曲线 {curve_name} 的 basis 与首条曲线不一致")
            if log.basis_type != first_basis_type:
                raise ValueError(f"曲线 {curve_name} 的 basis_type 与首条曲线不一致")

    if basis is None:
        raise KeyError("单井数据缺少 basis")

    return np.asarray(basis, dtype=float)


def _resolve_export_curve(well_data: WellExportInput, curve_name: str) -> grid.Log:
    """按曲线名获取可导出曲线。"""
    logs = _extract_logs_mapping(well_data)
    if curve_name in logs:
        return logs[curve_name]

    if hasattr(well_data, "AI") and curve_name == "AI":
        return getattr(well_data, "AI")

    if hasattr(well_data, "Vp_Vs_ratio") and curve_name == "Vp_Vs_ratio":
        return getattr(well_data, "Vp_Vs_ratio")

    raise KeyError(f"曲线不存在: {curve_name}")


def _extract_curve_values_and_unit(curve: grid.Log) -> tuple[np.ndarray, str]:
    """统一提取曲线数据与单位。"""
    values = np.asarray(curve.values, dtype=float)
    unit = "" if getattr(curve, "unit", None) is None else str(getattr(curve, "unit"))
    return values, unit


def _build_las_from_well_data(
    well_name: str,
    well_data: WellExportInput,
    selected_curve_names: List[str],
    null_value: float,
) -> lasio.LASFile:
    """将单井数据组装为 LASFile。"""
    las = lasio.LASFile()
    las.well["WELL"].value = well_name
    las.well["NULL"].value = float(null_value)

    basis = _extract_basis(well_data)
    las.append_curve("DEPT", basis, unit="m", descr="Depth")

    for curve_name in selected_curve_names:
        curve = _resolve_export_curve(well_data, curve_name)
        values, unit = _extract_curve_values_and_unit(curve)
        las.append_curve(curve_name, values, unit=unit, descr=curve_name)

    return las


def export_logsets_to_las(
    logsets: Dict[str, WellExportInput],
    output_dir: Path,
    curve_names: Optional[List[str]] = None,
    null_value: float = -999.25,
    write_fmt: str = "%.6f",
) -> Dict[str, Any]:
    """按井批量导出 LogSet/类 LogSet 字典到 LAS 文件。

    Parameters
    ----------
    logsets : Dict[str, WellExportInput]
            键为井名，值为对应 LogSet 或 ``Dict[str, grid.Log]``。
    output_dir : Path
            LAS 输出目录。
    curve_names : List[str], optional
            指定导出曲线列表，按传入顺序导出。
            若为 None，则导出该井全部已有曲线。
            对 LogSet 输入额外支持派生曲线名称: ``AI``、``Vp_Vs_ratio``。
    null_value : float, default -999.25
            导出 LAS 的 NULL 指示值。
    write_fmt : str, default "%.6f"
        LAS 写出数值格式。用于统一数据区与 NULL 字段的小数显示位数。

    Returns
    -------
    Dict[str, Any]
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

    if not isinstance(write_fmt, str) or write_fmt.strip() == "":
        raise ValueError("write_fmt 必须是非空字符串，例如 '%.5f'。")
    try:
        _ = write_fmt % 1.2345
    except (TypeError, ValueError) as exc:
        raise ValueError(f"write_fmt 非法: {write_fmt}，请使用类似 '%.5f' 的格式字符串。") from exc

    exported_files: List[Path] = []
    skipped_wells: List[Dict[str, str]] = []
    skipped_curves: List[Dict[str, str]] = []

    for well_name, well_data in logsets.items():
        try:
            logs_mapping = _extract_logs_mapping(well_data)
            if curve_names is None:
                requested_curve_names = list(logs_mapping.keys())
            else:
                requested_curve_names = list(curve_names)

            available_curve_names: List[str] = []
            for curve_name in requested_curve_names:
                try:
                    _resolve_export_curve(well_data, curve_name)
                    available_curve_names.append(curve_name)
                except Exception as exc:
                    skipped_curves.append({"well": well_name, "curve": curve_name, "reason": str(exc)})

            if not available_curve_names:
                skipped_wells.append({"well": well_name, "reason": "无可导出曲线"})
                continue

            las = _build_las_from_well_data(
                well_name=well_name,
                well_data=well_data,
                selected_curve_names=available_curve_names,
                null_value=null_value,
            )

            las.well["NULL"].value = write_fmt % float(null_value)

            output_file = output_dir / f"{well_name}.las"
            las.write(str(output_file), version=2.0, wrap=False, fmt=write_fmt)
            exported_files.append(output_file)

        except Exception as exc:
            skipped_wells.append({"well": well_name, "reason": str(exc)})

    return {
        "exported_files": exported_files,
        "skipped_wells": skipped_wells,
        "skipped_curves": skipped_curves,
    }


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

    该函数以 ``TimeDepthTable`` 的深度采样点为主轴，并按项目约定写出
    Petrel checkshots 所需字段。

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

    Notes
    -----
    - ``Z`` 为负值，单位 m。
    - ``MD`` 为正值，单位 m，并按约定计算: ``MD = |Z| + kb``。
    - ``TWT`` 为非负值，单位 ms（由 ``tdt.twt`` 的秒值转换得到）。
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if tdt.is_md_domain:
        raise ValueError("仅支持 TVDSS 域 TimeDepthTable；当前输入为 MD 域。")

    z = -np.abs(np.asarray(tdt.tvdss, dtype=float))
    md = np.abs(z) + float(kb)
    twt_ms = np.asarray(tdt.twt, dtype=float) * 1000.0

    if not np.isfinite(z).all() or not np.isfinite(md).all() or not np.isfinite(twt_ms).all():
        raise ValueError("Z/MD/TWT 存在非有限值，无法导出。")
    if (twt_ms < 0.0).any():
        raise ValueError("TWT 必须为非负值（ms）。")

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
