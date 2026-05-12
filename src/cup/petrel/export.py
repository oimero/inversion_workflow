"""cup.petrel.export: Petrel 相关文本与 LAS 导出工具。

本模块提供 LAS、checkshots 文本、两列 CSV 等导出能力，
用于将项目内部 ``wtie.processing.grid`` 数据结构转换为
Petrel 或下游解释流程常用的交换格式。

边界说明
--------
- 本模块仅负责文件组织、字段映射与基础格式校验。
- 本模块不负责井曲线反演、时深关系优化或 Petrel 工程导入流程本身。
- 导出前的数据清洗、重采样与单位统一应由上游流程保证。

核心公开对象
------------
1. export_logsets_to_las: 按井批量导出 LAS 文件。
2. export_vertical_tdt_to_petrel_checkshots: 导出直井 checkshots 文本。
"""

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import lasio
import numpy as np

from wtie.processing import grid

LogsetInput = Union[grid.LogSet, Dict[str, grid.Log]]


def _extract_logs_mapping(well_data: LogsetInput) -> Mapping[str, grid.Log]:
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


def _extract_basis(well_data: LogsetInput) -> np.ndarray:
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


def _ensure_md_domain(well_data: LogsetInput) -> None:
    """校验单井数据处于 MD 域。"""
    logs = _extract_logs_mapping(well_data)

    if hasattr(well_data, "is_md") and not bool(getattr(well_data, "is_md")):
        raise ValueError("仅支持导出 MD 域曲线到 LAS。")

    non_md_curves = [curve_name for curve_name, curve in logs.items() if not bool(getattr(curve, "is_md", False))]
    if non_md_curves:
        raise ValueError(f"仅支持导出 MD 域曲线到 LAS，以下曲线不是 MD 域: {non_md_curves}")


def _resolve_export_curve(well_data: LogsetInput, curve_name: str) -> grid.Log:
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
    well_data: LogsetInput,
    selected_curve_names: List[str],
    null_value: float,
) -> lasio.LASFile:
    """将单井数据组装为 LASFile。"""
    _ensure_md_domain(well_data)

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
    logsets: Dict[str, LogsetInput],
    output_dir: Path,
    curve_names: Optional[List[str]] = None,
    null_value: float = -999.25,
    write_fmt: str = "%.6f",
) -> Dict[str, Any]:
    """按井批量导出 LAS 文件。

    Parameters
    ----------
    logsets : Dict[str, LogsetInput]
        键为井名，值为对应 ``grid.LogSet`` 或 ``Dict[str, grid.Log]``。
    output_dir : Path
        LAS 输出目录。
    curve_names : List[str], optional
        指定导出曲线列表，按传入顺序导出。
        为 ``None`` 时导出该井全部已有曲线。
        对 ``grid.LogSet`` 输入额外支持派生曲线名 ``AI`` 与
        ``Vp_Vs_ratio``。
    null_value : float, default -999.25
        LAS 文件中的 NULL 指示值。
    write_fmt : str, default "%.6f"
        LAS 写出数值格式。用于统一数据区与 NULL 字段的小数显示位数。

    Returns
    -------
    Dict[str, Any]
        导出汇总信息字典，包含以下键：

        ``exported_files``
            成功写出的 LAS 文件路径列表。
        ``skipped_wells``
            被整井跳过的记录列表，每条记录至少包含 ``well`` 与 ``reason``。
        ``skipped_curves``
            被跳过的曲线记录列表，每条记录至少包含 ``well``、``curve``
            与 ``reason``。

    Raises
    ------
    ValueError
        当 ``write_fmt`` 不是合法的 ``printf`` 风格浮点格式串时抛出。

    Notes
    -----
    - 全流程采用“逐井容错”策略，不因单井或单曲线失败中断。
    - 导出前会显式校验输入曲线是否处于 MD 域。
    - 当指定曲线不存在时，会记录到 ``skipped_curves``。
    - 若某井最终无可导出曲线，会记录到 ``skipped_wells``。
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
        当 ``tdt`` 为 MD 域、存在非有限值、TWT 为负值或列长度不一致时抛出。

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


# def format_artifact_tag(value: Union[str, float, int]) -> str:
#     """将成果文件名中的标签值转为稳定字符串。

#     Parameters
#     ----------
#     value : str or float or int
#         待格式化的标签值。字符串会先执行 ``strip()``。

#     Returns
#     -------
#     str
#         可直接拼接到文件名中的稳定字符串。

#     Raises
#     ------
#     ValueError
#         当 ``value`` 是空白字符串时抛出。
#     """
#     if isinstance(value, str):
#         text = value.strip()
#         if not text:
#             raise ValueError("标签字符串不能为空。")
#         return text
#     return str(value)


# def _write_two_column_csv(
#     output_file: Path,
#     x_values: np.ndarray,
#     y_values: np.ndarray,
#     x_name: str,
#     y_name: str,
#     fmt: str = "%.9g",
# ) -> Path:
#     """写出两列数值 CSV。"""
#     output_file = Path(output_file)
#     output_file.parent.mkdir(parents=True, exist_ok=True)

#     x_arr = np.asarray(x_values, dtype=float).reshape(-1)
#     y_arr = np.asarray(y_values, dtype=float).reshape(-1)
#     if x_arr.size != y_arr.size:
#         raise ValueError(f"CSV 列长度不一致: {x_name}={x_arr.size}, {y_name}={y_arr.size}")

#     stacked = np.column_stack([x_arr, y_arr])
#     np.savetxt(
#         output_file,
#         stacked,
#         delimiter=",",
#         header=f"{x_name},{y_name}",
#         comments="",
#         fmt=fmt,
#     )
#     return output_file


# def export_twt_log_to_csv(
#     output_file: Path,
#     log: grid.Log,
#     value_name: str = "value",
#     fmt: str = "%.9g",
# ) -> Path:
#     """导出 TWT 域单条曲线到两列 CSV。

#     Parameters
#     ----------
#     output_file : Path
#         输出 CSV 路径。
#     log : grid.Log
#         待导出的单条曲线，要求处于 TWT 域。
#     value_name : str, default="value"
#         第二列列名。
#     fmt : str, default="%.9g"
#         数值写出格式。

#     Returns
#     -------
#     Path
#         写出的 CSV 文件路径。

#     Raises
#     ------
#     ValueError
#         当 ``log`` 不处于 TWT 域时抛出。
#     """
#     if not log.is_twt:
#         raise ValueError("仅支持导出 TWT 域曲线。")
#     return _write_two_column_csv(
#         output_file=output_file,
#         x_values=np.asarray(log.basis, dtype=float),
#         y_values=np.asarray(log.values, dtype=float),
#         x_name="twt_s",
#         y_name=value_name,
#         fmt=fmt,
#     )


# def export_wavelet_to_csv(
#     output_file: Path,
#     wavelet: Any,
#     fmt: str = "%.9g",
# ) -> Path:
#     """导出单道子波到两列 CSV。

#     Parameters
#     ----------
#     output_file : Path
#         输出 CSV 路径。
#     wavelet : Any
#         子波对象。需提供 ``basis/values`` 属性，或 ``t/y`` 属性。
#     fmt : str, default="%.9g"
#         数值写出格式。

#     Returns
#     -------
#     Path
#         写出的 CSV 文件路径。

#     Raises
#     ------
#     TypeError
#         当 ``wavelet`` 不提供受支持的时间轴与振幅属性组合时抛出。
#     """
#     if hasattr(wavelet, "basis") and hasattr(wavelet, "values"):
#         time_values = np.asarray(getattr(wavelet, "basis"), dtype=float)
#         amplitude_values = np.asarray(getattr(wavelet, "values"), dtype=float)
#     elif hasattr(wavelet, "t") and hasattr(wavelet, "y"):
#         time_values = np.asarray(getattr(wavelet, "t"), dtype=float)
#         amplitude_values = np.asarray(getattr(wavelet, "y"), dtype=float)
#     else:
#         raise TypeError("wavelet 必须提供 'basis/values' 或 't/y' 属性。")

#     return _write_two_column_csv(
#         output_file=output_file,
#         x_values=time_values,
#         y_values=amplitude_values,
#         x_name="time_s",
#         y_name="amplitude",
#         fmt=fmt,
#     )


# def export_vertical_wtie_artifacts(
#     output_dir: Path,
#     outputs: Any,
#     *,
#     well_name: str,
#     interpretation_offset: Union[str, float, int],
#     kb: float,
#     x: float,
#     y: float,
#     csv_fmt: str = "%.9g",
# ) -> Dict[str, Path]:
#     """导出直井自动井震标定的关键成果。

#     Parameters
#     ----------
#     output_dir : Path
#         成果输出目录。函数会自动创建 ``tdtable``、``ai``、``wavelet``
#         三个子目录。
#     outputs : Any
#         自动井震标定输出对象。需至少提供 ``table``、``logset_twt``、
#         ``wavelet`` 三个属性，其中 ``logset_twt`` 还需提供 ``AI`` 属性。
#     well_name : str
#         井名，用于文件命名与 checkshots 文本中的 ``Well name`` 列。
#     interpretation_offset : str or float or int
#         解释偏移标签，会参与输出文件名。
#     kb : float
#         Kelly Bushing 高程，单位 m。
#     x : float
#         井口 X 坐标，单位 m。
#     y : float
#         井口 Y 坐标，单位 m。
#     csv_fmt : str, default="%.9g"
#         AI 曲线与子波 CSV 的数值写出格式。

#     Returns
#     -------
#     Dict[str, Path]
#         成果路径字典，包含 ``tdtable_file``、``ai_file`` 与
#         ``wavelet_file``。

#     Raises
#     ------
#     AttributeError
#         当 ``outputs`` 缺少必需属性时抛出。

#     Notes
#     -----
#     导出内容包括：

#     - 优化后的 TVDSS-TWT 时深表（Petrel checkshots 文本）
#     - 优化后的 TWT 域 AI 曲线（CSV）
#     - 优化后的子波（CSV）
#     """
#     if not hasattr(outputs, "table"):
#         raise AttributeError("outputs 缺少 table 属性。")
#     if not hasattr(outputs, "logset_twt"):
#         raise AttributeError("outputs 缺少 logset_twt 属性。")
#     if not hasattr(outputs, "wavelet"):
#         raise AttributeError("outputs 缺少 wavelet 属性。")

#     tag = format_artifact_tag(interpretation_offset)
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     tdt_dir = output_dir / "tdtable"
#     ai_dir = output_dir / "ai"
#     wavelet_dir = output_dir / "wavelet"
#     tdt_dir.mkdir(parents=True, exist_ok=True)
#     ai_dir.mkdir(parents=True, exist_ok=True)
#     wavelet_dir.mkdir(parents=True, exist_ok=True)

#     tdt_file = tdt_dir / f"{well_name}_{tag}.txt"
#     ai_file = ai_dir / f"{well_name}_{tag}.csv"
#     wavelet_file = wavelet_dir / f"{well_name}_{tag}.csv"

#     tdt = getattr(outputs, "table")
#     logset_twt = getattr(outputs, "logset_twt")
#     wavelet = getattr(outputs, "wavelet")
#     if not hasattr(logset_twt, "AI"):
#         raise AttributeError("outputs.logset_twt 缺少 AI 属性。")
#     ai_log = getattr(logset_twt, "AI")

#     export_vertical_tdt_to_petrel_checkshots(
#         output_file=tdt_file,
#         tdt=tdt,
#         well_name=well_name,
#         kb=kb,
#         x=x,
#         y=y,
#     )
#     export_twt_log_to_csv(
#         output_file=ai_file,
#         log=ai_log,
#         value_name="ai",
#         fmt=csv_fmt,
#     )
#     export_wavelet_to_csv(
#         output_file=wavelet_file,
#         wavelet=wavelet,
#         fmt=csv_fmt,
#     )

#     return {
#         "tdtable_file": tdt_file,
#         "ai_file": ai_file,
#         "wavelet_file": wavelet_file,
#     }
