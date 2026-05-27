"""cup.well.las: LAS 文件扫描、筛选导出与井曲线 LogSet 读取。

本模块提供 LAS 文件头的轻量扫描、曲线存在性检查，以及将筛选后的曲线
导出为标准 LAS 文件的工具；同时保留旧深度域从原始 LAS 直接抽取
Vp/Rho/Vs 的兼容入口。

边界说明
--------
- 本模块不负责曲线分类与主曲线选择，这些由 ``cup.well.curves`` 处理。
- 导出时不修改曲线数值，输入清洗应在上游完成。
- ``old_*`` 入口只服务旧深度域原始 LAS 直读流程，时间域主链应读取
  含 ``DT_USM`` 与 ``RHO_GCC`` 的标准 LAS。

核心公开对象
------------
1. scan_las_curves: 扫描 LAS 文件头与曲线列表。
2. export_selected_curves_to_las: 将筛选后的曲线集合导出为 LAS。
3. load_vp_rho_logset_from_standard_las: 从标准 LAS 构建 Vp/Rho LogSet。
4. old_load_vp_rho_logset_from_las: 从原始 LAS 构建旧深度域 Vp/Rho LogSet。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import lasio
import numpy as np
import pandas as pd

from cup.well.curves import CurveInfo, exact_mnemonic, normalize_mnemonic
from cup.well.mnemonics import _RHO_MNEMONICS, _VP_MNEMONICS, _VS_MNEMONICS
from wtie.processing import grid
from wtie.processing.logs import interpolate_nans

_SENTINEL_VALUES = (-999.0, -999.25, -99999)


@dataclass(frozen=True)
class LasHeader:
    """Lightweight LAS header summary."""

    well_name: str
    index_mnemonic: str
    index_unit: str
    start: float | None
    stop: float | None
    step: float | None
    null_value: float | None
    curve_count: int
    estimated_sample_count: int | None
    index_is_monotonic: bool | None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_raw_mnemonic(name: object) -> str:
    """规范化旧式原始 LAS 曲线简称的大小写与空白。"""
    return str(name).strip().upper()


def _matches_mnemonic_with_optional_suffix(column_name: str, base_mnemonic: str) -> bool:
    """判断列名是否匹配“基础缩写 + 可选下划线后缀”规则。"""
    col_norm = _normalize_raw_mnemonic(column_name)
    base_norm = _normalize_raw_mnemonic(base_mnemonic)
    return col_norm == base_norm or col_norm.startswith(f"{base_norm}_")


def _select_curve_mnemonic(
    las_df: pd.DataFrame,
    candidate_mnemonics: Tuple[str, ...],
    property_name: str,
    curve_mnemonic: Optional[str] = None,
) -> str:
    """从旧式候选简称中选择唯一可用曲线。"""
    columns = [str(c) for c in las_df.columns]
    norm_to_original = {_normalize_raw_mnemonic(c): c for c in columns}

    if curve_mnemonic is not None:
        norm_user = _normalize_raw_mnemonic(curve_mnemonic)
        if norm_user not in norm_to_original:
            raise ValueError(f"指定的 {property_name} 曲线简称不存在: {curve_mnemonic}. 可用曲线: {columns}")
        return norm_to_original[norm_user]

    matched = [
        col
        for col in columns
        if any(_matches_mnemonic_with_optional_suffix(col, candidate) for candidate in candidate_mnemonics)
    ]
    if len(matched) == 0:
        raise ValueError(
            f"未找到 {property_name} 曲线。候选简称: {list(candidate_mnemonics)}. 请检查是否存在其他可用简称？"
        )
    if len(matched) > 1:
        raise ValueError(
            f"检测到多个 {property_name} 候选曲线: {matched}. 请通过 curve_mnemonic 显式指定要使用的简称。"
        )
    return matched[0]


def _get_curve_unit(las: lasio.LASFile, selected_mnemonic: str) -> str:
    """获取 LAS 曲线的单位字符串。"""
    norm_selected = _normalize_raw_mnemonic(selected_mnemonic)
    for curve in las.curves:
        if _normalize_raw_mnemonic(curve.mnemonic) == norm_selected:
            return str(curve.unit or "")
    return ""


def _replace_sentinel_values(values: object) -> np.ndarray:
    """将旧式 LAS 异常占位值替换为 NaN。"""
    out = np.asarray(values, dtype=float).copy()
    for sentinel in _SENTINEL_VALUES:
        out[np.isclose(out, sentinel, equal_nan=False)] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def _convert_velocity_input_to_mps(values: object, unit: str, property_name: str) -> np.ndarray:
    """将速度或时差曲线转换为 m/s。"""
    curve_values = _replace_sentinel_values(values)
    curve_values[curve_values <= 0] = np.nan

    unit_norm = str(unit).strip().lower().replace(" ", "")
    if unit_norm in {"us/ft", "μs/ft", "µs/ft"}:
        velocity = 0.3048 * 1e6 / curve_values
    elif unit_norm in {"us/m", "μs/m", "µs/m"}:
        velocity = 1e6 / curve_values
    elif unit_norm in {"m/s", "mps", "m/sec", "meter/s", "meters/s"}:
        velocity = curve_values
    else:
        raise ValueError(f"{property_name} 曲线单位不受支持: '{unit}'. 当前仅支持 us/ft、us/m 或 m/s。")

    if np.all(np.isnan(velocity)):
        raise ValueError(f"{property_name} 曲线在异常值处理与单位转换后全部为 NaN。")
    return velocity


def _convert_density_to_g_cm3(density_values: object, unit: str) -> np.ndarray:
    """将密度曲线转换为 g/cm3。"""
    density = _replace_sentinel_values(density_values)
    unit_norm = str(unit).strip().lower().replace(" ", "")
    if unit_norm in {"g/cm3", "g/cc", "g/cm^3"}:
        density_g_cm3 = density
    elif unit_norm in {"kg/m3", "kg/m^3"}:
        density_g_cm3 = density / 1000.0
    else:
        raise ValueError(f"Rho 曲线单位不受支持: '{unit}'. 当前仅支持 g/cm3、g/cc 或 kg/m3。")

    if np.all(np.isnan(density_g_cm3)):
        raise ValueError("Rho 曲线在异常值处理与单位转换后全部为 NaN。")
    return density_g_cm3


def _finite_positive(values: np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= 0.0] = np.nan
    if np.all(np.isnan(arr)):
        raise ValueError(f"{label} contains no positive finite samples.")
    return arr


def old_extract_vp_log_from_las(
    las_file: lasio.LASFile,
    unit: str,
    curve_mnemonic: Optional[str] = None,
) -> grid.Log:
    """旧深度域兼容入口：从原始 LAS 文件中提取纵波速度曲线（Vp）。"""
    las_df = las_file.df()
    selected = _select_curve_mnemonic(las_df, _VP_MNEMONICS, "Vp", curve_mnemonic)
    vp = _convert_velocity_input_to_mps(las_df.loc[:, selected].to_numpy(), unit, "Vp")
    vp = interpolate_nans(vp, method="linear")
    return grid.Log(vp, las_df.index.values, "md", name="Vp", unit="m/s", allow_nan=False)


def old_extract_vs_log_from_las(
    las_file: lasio.LASFile,
    unit: str,
    curve_mnemonic: Optional[str] = None,
) -> grid.Log:
    """旧深度域兼容入口：从原始 LAS 文件中提取横波速度曲线（Vs）。"""
    las_df = las_file.df()
    selected = _select_curve_mnemonic(las_df, _VS_MNEMONICS, "Vs", curve_mnemonic)
    vs = _convert_velocity_input_to_mps(las_df.loc[:, selected].to_numpy(), unit, "Vs")
    vs = interpolate_nans(vs, method="linear")
    return grid.Log(vs, las_df.index.values, "md", name="Vs", unit="m/s", allow_nan=False)


def old_extract_rho_log_from_las(
    las_file: lasio.LASFile,
    unit: str,
    curve_mnemonic: Optional[str] = None,
) -> grid.Log:
    """旧深度域兼容入口：从原始 LAS 文件中提取密度曲线（Rho）。"""
    las_df = las_file.df()
    selected = _select_curve_mnemonic(las_df, _RHO_MNEMONICS, "Rho", curve_mnemonic)
    rho = _convert_density_to_g_cm3(las_df.loc[:, selected].to_numpy(), unit)
    rho = interpolate_nans(rho, method="linear")
    return grid.Log(rho, las_df.index.values, "md", name="Rho", unit="g/cm3", allow_nan=False)


def old_extract_any_log_from_las(las_file: lasio.LASFile, curve_mnemonic: str) -> grid.Log:
    """旧式原始 LAS 直读入口：提取任意单条曲线。"""
    curve_mnemonic = str(curve_mnemonic).strip()
    if not curve_mnemonic:
        raise ValueError("curve_mnemonic 不能为空。")

    las_df = las_file.df()
    columns = [str(c) for c in las_df.columns]
    norm_to_original = {_normalize_raw_mnemonic(c): c for c in columns}
    norm_user = _normalize_raw_mnemonic(curve_mnemonic)
    if norm_user not in norm_to_original:
        raise ValueError(f"指定曲线简称不存在: {curve_mnemonic}. 可用曲线: {columns}")

    selected = norm_to_original[norm_user]
    values = _replace_sentinel_values(las_df.loc[:, selected].to_numpy())
    if np.all(np.isnan(values)):
        raise ValueError(f"{selected} 曲线在异常值处理后全部为 NaN。")
    unit_from_las = _get_curve_unit(las_file, selected)
    return grid.Log(values, las_df.index.values, "md", name=curve_mnemonic, unit=unit_from_las, allow_nan=True)


def old_load_vp_rho_logset_from_las(
    las_file_path: Path,
    vp_mnemonic: Optional[str] = None,
    rho_mnemonic: Optional[str] = None,
    vp_unit: Optional[str] = "us/m",
    rho_unit: Optional[str] = "g/cm3",
) -> grid.LogSet:
    """旧深度域兼容入口：从原始 LAS 文件路径读取 Vp/Rho 并组装为 ``grid.LogSet``。"""
    las_file_path = Path(las_file_path)
    if not las_file_path.exists():
        raise FileNotFoundError(f"LAS 文件不存在: {las_file_path}")

    las_file = lasio.read(las_file_path)
    vp_log = old_extract_vp_log_from_las(
        las_file, curve_mnemonic=vp_mnemonic, unit=vp_unit if vp_unit is not None else "us/m"
    )
    rho_log = old_extract_rho_log_from_las(
        las_file, curve_mnemonic=rho_mnemonic, unit=rho_unit if rho_unit is not None else "g/cm3"
    )
    return grid.LogSet({"Vp": vp_log, "Rho": rho_log})


def load_vp_rho_logset_from_standard_las(path: str | Path) -> grid.LogSet:
    """从标准 LAS 构建 MD 域 Vp/Rho LogSet。

    标准 LAS 指包含第三步标准曲线 ``DT_USM`` 与 ``RHO_GCC`` 的 LAS，
    不要求文件一定由 ``scripts/log_preprocess.py`` 生成。
    """
    las = lasio.read(str(path))
    df = las.df()
    missing = [name for name in ("DT_USM", "RHO_GCC") if name not in df.columns]
    if missing:
        raise ValueError(f"Standard LAS is missing required curves {missing}: {path}")

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


def _header_value(las: lasio.LASFile, section: str, mnemonic: str) -> Any:
    try:
        return getattr(las, section)[mnemonic].value
    except (KeyError, AttributeError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _estimated_sample_count(start: float | None, stop: float | None, step: float | None) -> int | None:
    if start is None or stop is None or step is None or step == 0.0:
        return None
    count = int(abs((stop - start) / step)) + 1
    return count if count > 0 else None


def _index_is_monotonic(start: float | None, stop: float | None, step: float | None) -> bool | None:
    if start is None or stop is None or step is None or step == 0.0:
        return None
    return (stop - start) * step >= 0.0


def _build_las_header(las: lasio.LASFile, *, fallback_well_name: str) -> LasHeader:
    index_curve = las.curves[0] if las.curves else None
    start = _optional_float(_header_value(las, "well", "STRT"))
    stop = _optional_float(_header_value(las, "well", "STOP"))
    step = _optional_float(_header_value(las, "well", "STEP"))
    null_value = _optional_float(_header_value(las, "well", "NULL"))
    well_name = str(_header_value(las, "well", "WELL") or fallback_well_name).strip()
    return LasHeader(
        well_name=well_name,
        index_mnemonic=str(index_curve.mnemonic if index_curve is not None else "DEPT"),
        index_unit=str(index_curve.unit or "") if index_curve is not None else "",
        start=start,
        stop=stop,
        step=step,
        null_value=null_value,
        curve_count=len(las.curves),
        estimated_sample_count=_estimated_sample_count(start, stop, step),
        index_is_monotonic=_index_is_monotonic(start, stop, step),
    )


def scan_las_header(path: Path) -> LasHeader:
    """不加载数据样点，仅读取 LAS 元数据。"""
    las = lasio.read(str(path), ignore_data=True)
    return _build_las_header(las, fallback_well_name=Path(path).stem)


def scan_las_curves(path: Path) -> tuple[LasHeader, list[CurveInfo]]:
    """不加载数据样点，仅读取 LAS 曲线头信息。"""
    las = lasio.read(str(path), ignore_data=True)
    header = _build_las_header(las, fallback_well_name=Path(path).stem)
    curves = [
        CurveInfo(
            mnemonic=str(curve.mnemonic),
            unit=str(curve.unit or ""),
            description=str(curve.descr or ""),
            index=index,
        )
        for index, curve in enumerate(las.curves)
    ]
    return header, curves


def _resolve_curve_index(
    curve_index: dict[str, int], normalized_index: dict[str, list[int]], mnemonic: str
) -> int | None:
    exact = exact_mnemonic(mnemonic)
    if exact in curve_index:
        return curve_index[exact]
    candidates = normalized_index.get(normalize_mnemonic(mnemonic), [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return candidates[0]
    return None


def _validate_write_format(write_fmt: str) -> None:
    if not isinstance(write_fmt, str) or not write_fmt.strip():
        raise ValueError("write_fmt must be a non-empty printf-style float format.")
    try:
        _ = write_fmt % 1.2345
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid write_fmt: {write_fmt}") from exc


def export_selected_curves_to_las(
    source_las: Path,
    output_las: Path,
    selected_mnemonics: Sequence[str],
    *,
    null_value: float = -999.25,
    write_fmt: str = "%.6f",
) -> tuple[Path, list[dict[str, str]], list[str]]:
    """导出 LAS 索引曲线和选中的原始曲线。"""
    _validate_write_format(write_fmt)
    output_las = Path(output_las)
    output_las.parent.mkdir(parents=True, exist_ok=True)

    las = lasio.read(str(source_las))
    if not las.curves:
        raise ValueError(f"LAS file has no curves: {source_las}")
    data = np.asarray(las.data)
    if data.ndim != 2 or data.shape[1] != len(las.curves):
        raise ValueError(f"LAS data shape does not match curve headers: {source_las}")

    curve_index = {exact_mnemonic(curve.mnemonic): index for index, curve in enumerate(las.curves)}
    normalized_index: dict[str, list[int]] = {}
    for index, curve in enumerate(las.curves):
        normalized_index.setdefault(normalize_mnemonic(curve.mnemonic), []).append(index)
    requested = list(dict.fromkeys(str(mnemonic) for mnemonic in selected_mnemonics))

    out = lasio.LASFile()
    for item in las.well:
        out.well[item.mnemonic] = item
    out.well["NULL"].value = float(null_value)
    source_well_name = str(_header_value(las, "well", "WELL") or Path(source_las).stem).strip()
    out.well["WELL"].value = source_well_name

    index_curve = las.curves[0]
    out.append_curve(
        str(index_curve.mnemonic),
        data[:, 0],
        unit=str(index_curve.unit or ""),
        descr=str(index_curve.descr or ""),
    )

    skipped: list[dict[str, str]] = []
    exported_mnemonics: list[str] = []
    for mnemonic in requested:
        index = _resolve_curve_index(curve_index, normalized_index, mnemonic)
        if index is None:
            skipped.append({"curve": mnemonic, "reason": "selected_curve_missing_in_las"})
            continue
        if index == 0:
            continue
        curve = las.curves[index]
        try:
            out.append_curve(
                str(curve.mnemonic),
                data[:, index],
                unit=str(curve.unit or ""),
                descr=str(curve.descr or ""),
            )
            exported_mnemonics.append(str(curve.mnemonic))
        except Exception as exc:
            skipped.append({"curve": mnemonic, "reason": str(exc)})

    if len(out.curves) <= 1:
        raise ValueError(f"No selected curves were exported from {source_las}")

    out.write(str(output_las), version=2.0, wrap=False, fmt=write_fmt)
    return output_las, skipped, exported_mnemonics
