"""cup.well.las: LAS 文件通用 I/O、标准曲线读取与工作流导出。

本模块分三层：通用 LAS 物理 I/O、项目标准曲线读取、工作流专用 LAS
导出。同时保留旧深度域从原始 LAS 直接抽取 Vp/Rho/Vs 的兼容入口。

边界说明
--------
- 本模块不负责曲线分类与主曲线选择，这些由 ``cup.well.curves`` 处理。
- 通用读取不做单位转换、不做曲线语义推断、不做标准命名。
- ``old_*`` 入口只服务旧深度域原始 LAS 直读流程，时间域主链应读取
  含 ``DT_USM`` 与 ``RHO_GCC`` 的标准 LAS。

核心公开对象
------------
1. scan_las_curves / scan_las_header: 扫描 LAS 元数据。
2. read_las_curve / read_las_curves: 按用户指定 mnemonic 读取通用曲线。
3. load_vp_rho_logset_from_standard_las: 从标准 LAS 构建 Vp/Rho LogSet。
4. export_selected_curves_to_las / export_logsets_to_las: LAS 导出。
5. old_load_vp_rho_logset_from_las: 旧深度域 Vp/Rho 兼容 Adapter。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import lasio
import numpy as np

from cup.well.curves import CurveInfo, exact_mnemonic, normalize_mnemonic
from cup.well.mnemonics import _RHO_MNEMONICS, _VP_MNEMONICS, _VS_MNEMONICS
from wtie.processing import grid
from wtie.processing.logs import interpolate_nans

_SENTINEL_VALUES = (-999.0, -999.25, -9999.0, -99999.0)
LogsetInput = grid.LogSet | dict[str, grid.Log]
MATCH_POLICIES = {"exact", "normalized", "exact_then_normalized"}
NULL_POLICIES = {"las_only", "common_sentinels", "las_and_common_sentinels", "none"}


@dataclass(frozen=True)
class LasHeader:
    """轻量 LAS 文件头摘要。"""

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


@dataclass(frozen=True)
class LasCurveLookup:
    """LAS 曲线索引查找表。"""

    exact_index: dict[str, int]
    normalized_index: dict[str, list[int]]


@dataclass(frozen=True)
class ResolvedLasCurve:
    """已解析的 LAS 曲线位置与头信息。"""

    index: int
    mnemonic: str
    unit: str
    description: str


def _replace_sentinel_values(values: object, *, null_value: float | None = None, null_policy: str = "common_sentinels") -> np.ndarray:
    """将 LAS 空值和常见异常占位值替换为 NaN。"""
    policy = str(null_policy).strip()
    if policy not in NULL_POLICIES:
        raise ValueError(f"Unsupported null_policy: {null_policy}. Expected one of {sorted(NULL_POLICIES)}.")
    out = np.asarray(values, dtype=float).copy()
    sentinels: list[float] = []
    if policy in {"las_only", "las_and_common_sentinels"} and null_value is not None and np.isfinite(null_value):
        sentinels.append(float(null_value))
    if policy in {"common_sentinels", "las_and_common_sentinels"}:
        sentinels.extend(float(item) for item in _SENTINEL_VALUES)
    for sentinel in sentinels:
        out[np.isclose(out, sentinel, rtol=0.0, atol=1e-8, equal_nan=False)] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def _normalize_unit(unit: object) -> str:
    """规范化单位写法，以便兼容 ``μ`` / ``µ`` / ``u`` 等常见变体。"""
    return str(unit or "").strip().lower().replace(" ", "").replace("μ", "u").replace("µ", "u")


def build_las_curve_lookup(las: lasio.LASFile) -> LasCurveLookup:
    """构建 LAS 曲线 exact/normalized mnemonic 查找表。

    Parameters
    ----------
    las : lasio.LASFile
        已打开的 LAS 文件对象。

    Returns
    -------
    LasCurveLookup
        含 ``exact_index`` 与 ``normalized_index`` 的查找表。
    """
    exact_index: dict[str, int] = {}
    normalized_index: dict[str, list[int]] = {}
    for index, curve in enumerate(las.curves):
        exact = exact_mnemonic(curve.mnemonic)
        if exact in exact_index:
            raise ValueError(f"Duplicate exact LAS mnemonic found: {curve.mnemonic}")
        exact_index[exact] = index
        normalized_index.setdefault(normalize_mnemonic(curve.mnemonic), []).append(index)
    return LasCurveLookup(exact_index=exact_index, normalized_index=normalized_index)


def resolve_las_curve_index(
    las: lasio.LASFile,
    mnemonic: str,
    *,
    match_policy: str = "exact_then_normalized",
    lookup: LasCurveLookup | None = None,
    source: str | Path | None = None,
) -> int | None:
    """按 exact/normalized mnemonic 规则解析 LAS 曲线 index。

    Parameters
    ----------
    las : lasio.LASFile
        已打开的 LAS 文件对象。
    mnemonic : str
        目标曲线简称。
    match_policy : str, default="exact_then_normalized"
        匹配策略：``"exact"``、``"normalized"`` 或 ``"exact_then_normalized"``。
    lookup : LasCurveLookup | None, optional
        预构建的查找表，为 None 时从 ``las`` 重新构建。
    source : str | Path | None, optional
        数据来源标识，仅用于错误信息。

    Returns
    -------
    int | None
        匹配到的曲线 index，未找到时返回 None。

    Raises
    ------
    ValueError
        当 normalized 匹配到多条曲线（歧义）时。
    """
    policy = str(match_policy).strip()
    if policy not in MATCH_POLICIES:
        raise ValueError(f"Unsupported match_policy: {match_policy}. Expected one of {sorted(MATCH_POLICIES)}.")

    lookup = build_las_curve_lookup(las) if lookup is None else lookup
    requested_exact = exact_mnemonic(mnemonic)
    if policy in {"exact", "exact_then_normalized"} and requested_exact in lookup.exact_index:
        return lookup.exact_index[requested_exact]
    if policy == "exact":
        return None

    candidates = lookup.normalized_index.get(normalize_mnemonic(mnemonic), [])
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = [str(las.curves[index].mnemonic) for index in candidates]
        source_text = f" in {source}" if source is not None else ""
        raise ValueError(
            f"Ambiguous LAS mnemonic {mnemonic!r}{source_text}: normalized match found multiple curves {names}. "
            "Use an exact mnemonic such as a LASIO ':1' suffix."
        )
    return None


def _resolved_las_curve(
    las: lasio.LASFile,
    mnemonic: str,
    *,
    match_policy: str,
    lookup: LasCurveLookup | None = None,
    source: str | Path | None = None,
) -> ResolvedLasCurve | None:
    index = resolve_las_curve_index(
        las,
        mnemonic,
        match_policy=match_policy,
        lookup=lookup,
        source=source,
    )
    if index is None:
        return None
    curve = las.curves[index]
    return ResolvedLasCurve(
        index=index,
        mnemonic=str(curve.mnemonic),
        unit=str(curve.unit or ""),
        description=str(curve.descr or ""),
    )


def _las_null_value(las: lasio.LASFile) -> float | None:
    return _optional_float(_header_value(las, "well", "NULL"))


def _las_data_array(las: lasio.LASFile, *, source: str | Path | None = None) -> np.ndarray:
    if not las.curves:
        raise ValueError(f"LAS file has no curves: {source}")
    data = np.asarray(las.data)
    if data.ndim != 2 or data.shape[1] != len(las.curves):
        raise ValueError(f"LAS data shape does not match curve headers: {source}")
    return data


def _read_las_curve_from_lasio(
    las: lasio.LASFile,
    mnemonic: str,
    *,
    match_policy: str = "exact_then_normalized",
    null_policy: str = "las_and_common_sentinels",
    allow_all_nan: bool = False,
    lookup: LasCurveLookup | None = None,
    source: str | Path | None = None,
) -> grid.Log:
    resolved = _resolved_las_curve(
        las,
        mnemonic,
        match_policy=match_policy,
        lookup=lookup,
        source=source,
    )
    if resolved is None:
        available = [str(curve.mnemonic) for curve in las.curves]
        source_text = f" in {source}" if source is not None else ""
        raise ValueError(f"LAS mnemonic {mnemonic!r} not found{source_text}. Available curves: {available}")
    if resolved.index == 0:
        raise ValueError(f"Requested LAS mnemonic {mnemonic!r} resolves to the index curve, not a log curve.")

    data = _las_data_array(las, source=source)
    basis = np.asarray(data[:, 0], dtype=np.float64)
    values = _replace_sentinel_values(
        data[:, resolved.index],
        null_value=_las_null_value(las),
        null_policy=null_policy,
    )
    if not allow_all_nan and np.all(np.isnan(values)):
        source_text = f" in {source}" if source is not None else ""
        raise ValueError(f"LAS curve {resolved.mnemonic!r}{source_text} contains no finite samples after null handling.")
    return grid.Log(values, basis, "md", name=resolved.mnemonic, unit=resolved.unit, allow_nan=True)


def read_las_curve(
    path: str | Path,
    mnemonic: str,
    *,
    match_policy: str = "exact_then_normalized",
    null_policy: str = "las_and_common_sentinels",
    allow_all_nan: bool = False,
) -> grid.Log:
    """从 LAS 文件按指定 mnemonic 读取单条曲线为 ``grid.Log``。

    Parameters
    ----------
    path : str | Path
        LAS 文件路径。
    mnemonic : str
        目标曲线简称。
    match_policy : str, default="exact_then_normalized"
        匹配策略。
    null_policy : str, default="las_and_common_sentinels"
        空值处理策略。
    allow_all_nan : bool, default=False
        是否允许全 NaN 曲线。

    Returns
    -------
    grid.Log
        MD 域曲线对象。
    """
    path = Path(path)
    las = lasio.read(str(path))
    return _read_las_curve_from_lasio(
        las,
        mnemonic,
        match_policy=match_policy,
        null_policy=null_policy,
        allow_all_nan=allow_all_nan,
        source=path,
    )


def read_las_curves(
    path: str | Path,
    mnemonics: Sequence[str],
    *,
    match_policy: str = "exact_then_normalized",
    null_policy: str = "las_and_common_sentinels",
    allow_all_nan: bool = False,
) -> dict[str, grid.Log]:
    """从 LAS 文件批量读取曲线，返回 ``{请求 mnemonic: grid.Log}``。

    Parameters
    ----------
    path : str | Path
        LAS 文件路径。
    mnemonics : Sequence[str]
        目标曲线简称列表。
    match_policy : str, default="exact_then_normalized"
        匹配策略。
    null_policy : str, default="las_and_common_sentinels"
        空值处理策略。
    allow_all_nan : bool, default=False
        是否允许全 NaN 曲线。

    Returns
    -------
    dict[str, grid.Log]
        键为请求的 mnemonic、值为 MD 域曲线。
    """
    path = Path(path)
    las = lasio.read(str(path))
    lookup = build_las_curve_lookup(las)
    logs: dict[str, grid.Log] = {}
    for mnemonic in mnemonics:
        logs[str(mnemonic)] = _read_las_curve_from_lasio(
            las,
            str(mnemonic),
            match_policy=match_policy,
            null_policy=null_policy,
            allow_all_nan=allow_all_nan,
            lookup=lookup,
            source=path,
        )
    return logs


def _convert_velocity_input_to_mps(values: object, unit: str, property_name: str) -> np.ndarray:
    """将速度或时差曲线转换为 m/s。"""
    curve_values = _replace_sentinel_values(values)
    curve_values[curve_values <= 0] = np.nan

    unit_norm = _normalize_unit(unit)
    if unit_norm == "us/ft":
        velocity = 0.3048 * 1e6 / curve_values
    elif unit_norm == "us/m":
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
    unit_norm = _normalize_unit(unit)
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


def _read_legacy_candidate_log(
    las_file: lasio.LASFile,
    candidate_mnemonics: Tuple[str, ...],
    property_name: str,
    curve_mnemonic: Optional[str] = None,
) -> grid.Log:
    if curve_mnemonic is not None:
        return _read_las_curve_from_lasio(las_file, curve_mnemonic, source=f"legacy {property_name} reader")

    lookup = build_las_curve_lookup(las_file)
    matched: dict[int, grid.Log] = {}
    for candidate in candidate_mnemonics:
        resolved = resolve_las_curve_index(
            las_file,
            candidate,
            match_policy="exact_then_normalized",
            lookup=lookup,
            source=f"legacy {property_name} reader",
        )
        if resolved is not None and resolved != 0:
            matched[resolved] = _read_las_curve_from_lasio(
                las_file,
                str(las_file.curves[resolved].mnemonic),
                match_policy="exact",
                lookup=lookup,
                source=f"legacy {property_name} reader",
            )

    if not matched:
        raise ValueError(
            f"未找到 {property_name} 曲线。候选简称: {list(candidate_mnemonics)}. 请检查是否存在其他可用简称？"
        )
    if len(matched) > 1:
        names = [log.name for log in matched.values()]
        raise ValueError(
            f"检测到多个 {property_name} 候选曲线: {names}. 请通过 curve_mnemonic 显式指定要使用的简称。"
        )
    return next(iter(matched.values()))


def old_extract_vp_log_from_las(
    las_file: lasio.LASFile,
    unit: str,
    curve_mnemonic: Optional[str] = None,
) -> grid.Log:
    """旧深度域兼容入口：从原始 LAS 文件中提取纵波速度曲线（Vp）。"""
    source_log = _read_legacy_candidate_log(las_file, _VP_MNEMONICS, "Vp", curve_mnemonic)
    vp = _convert_velocity_input_to_mps(source_log.values, unit, "Vp")
    vp = interpolate_nans(vp, method="linear")
    return grid.Log(vp, source_log.basis, "md", name="Vp", unit="m/s", allow_nan=False)


def old_extract_vs_log_from_las(
    las_file: lasio.LASFile,
    unit: str,
    curve_mnemonic: Optional[str] = None,
) -> grid.Log:
    """旧深度域兼容入口：从原始 LAS 文件中提取横波速度曲线（Vs）。"""
    source_log = _read_legacy_candidate_log(las_file, _VS_MNEMONICS, "Vs", curve_mnemonic)
    vs = _convert_velocity_input_to_mps(source_log.values, unit, "Vs")
    vs = interpolate_nans(vs, method="linear")
    return grid.Log(vs, source_log.basis, "md", name="Vs", unit="m/s", allow_nan=False)


def old_extract_rho_log_from_las(
    las_file: lasio.LASFile,
    unit: str,
    curve_mnemonic: Optional[str] = None,
) -> grid.Log:
    """旧深度域兼容入口：从原始 LAS 文件中提取密度曲线（Rho）。"""
    source_log = _read_legacy_candidate_log(las_file, _RHO_MNEMONICS, "Rho", curve_mnemonic)
    rho = _convert_density_to_g_cm3(source_log.values, unit)
    rho = interpolate_nans(rho, method="linear")
    return grid.Log(rho, source_log.basis, "md", name="Rho", unit="g/cm3", allow_nan=False)


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

    Parameters
    ----------
    path : str | Path
        标准 LAS 文件路径，必须包含 ``DT_USM`` 与 ``RHO_GCC`` 曲线。

    Returns
    -------
    grid.LogSet
        含 ``Vp`` (m/s) 与 ``Rho`` (g/cm3) 的 MD 域 LogSet。

    Raises
    ------
    ValueError
        缺少必需曲线、采样轴不一致或插值后仍含非有限值时。
    """
    try:
        curves = read_las_curves(path, ["DT_USM", "RHO_GCC"])
    except ValueError as exc:
        raise ValueError(f"Standard LAS is missing required curves ['DT_USM', 'RHO_GCC']: {path}") from exc

    dt_log = curves["DT_USM"]
    rho_log = curves["RHO_GCC"]
    if not np.allclose(dt_log.basis, rho_log.basis, equal_nan=False):
        raise ValueError(f"DT_USM and RHO_GCC basis do not match: {path}")

    md = np.asarray(dt_log.basis, dtype=np.float64)
    dt_usm = _finite_positive(dt_log.values, label="DT_USM")
    rho = _finite_positive(rho_log.values, label="RHO_GCC")
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
    """不加载数据样点，仅读取 LAS 元数据。

    Parameters
    ----------
    path : Path
        LAS 文件路径。

    Returns
    -------
    LasHeader
        LAS 文件头摘要。
    """
    las = lasio.read(str(path), ignore_data=True)
    return _build_las_header(las, fallback_well_name=Path(path).stem)


def scan_las_curves(path: Path) -> tuple[LasHeader, list[CurveInfo]]:
    """不加载数据样点，仅读取 LAS 曲线头信息。

    Parameters
    ----------
    path : Path
        LAS 文件路径。

    Returns
    -------
    tuple[LasHeader, list[CurveInfo]]
        LAS 文件头摘要与曲线头信息列表。
    """
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


def _validate_write_format(write_fmt: str) -> None:
    if not isinstance(write_fmt, str) or not write_fmt.strip():
        raise ValueError("write_fmt must be a non-empty printf-style float format.")
    try:
        _ = write_fmt % 1.2345
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid write_fmt: {write_fmt}") from exc


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
    selected_curve_names: list[str],
    null_value: float,
) -> lasio.LASFile:
    """将单井 LogSet/Log 映射组装为 LASFile。"""
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
    logsets: dict[str, LogsetInput],
    output_dir: Path,
    curve_names: list[str] | None = None,
    null_value: float = -999.25,
    write_fmt: str = "%.6f",
) -> dict[str, Any]:
    """按井批量导出 MD 域 LogSet/Log 映射到 LAS 文件。

    Parameters
    ----------
    logsets : dict[str, LogsetInput]
        键为井名，值为 ``grid.LogSet`` 或 ``dict[str, grid.Log]``。
    output_dir : Path
        输出目录。
    curve_names : list[str] | None, optional
        要导出的曲线名列表，为 None 时导出全部。
    null_value : float, default=-999.25
        LAS 文件缺失值。
    write_fmt : str, default="%.6f"
        数值写入格式。

    Returns
    -------
    dict[str, Any]
        含 ``exported_files``、``skipped_wells``、``skipped_curves`` 的结果字典。
    """
    _validate_write_format(write_fmt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files: list[Path] = []
    skipped_wells: list[dict[str, str]] = []
    skipped_curves: list[dict[str, str]] = []

    for well_name, well_data in logsets.items():
        try:
            logs_mapping = _extract_logs_mapping(well_data)
            requested_curve_names = list(logs_mapping.keys()) if curve_names is None else list(curve_names)

            available_curve_names: list[str] = []
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


def export_selected_curves_to_las(
    source_las: Path,
    output_las: Path,
    selected_mnemonics: Sequence[str],
    *,
    null_value: float = -999.25,
    write_fmt: str = "%.6f",
) -> tuple[Path, list[dict[str, str]], list[str]]:
    """导出 LAS 索引曲线和选中的原始曲线。

    Parameters
    ----------
    source_las : Path
        源 LAS 文件路径。
    output_las : Path
        目标 LAS 文件路径。
    selected_mnemonics : Sequence[str]
        要导出的曲线 mnemonic 列表。
    null_value : float, default=-999.25
        LAS 文件缺失值。
    write_fmt : str, default="%.6f"
        数值写入格式。

    Returns
    -------
    tuple[Path, list[dict[str, str]], list[str]]
        ``(输出路径, 跳过曲线列表, 实际导出 mnemonic 列表)``。
    """
    _validate_write_format(write_fmt)
    output_las = Path(output_las)
    output_las.parent.mkdir(parents=True, exist_ok=True)

    las = lasio.read(str(source_las))
    if not las.curves:
        raise ValueError(f"LAS file has no curves: {source_las}")
    data = np.asarray(las.data)
    if data.ndim != 2 or data.shape[1] != len(las.curves):
        raise ValueError(f"LAS data shape does not match curve headers: {source_las}")

    lookup = build_las_curve_lookup(las)
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
        try:
            resolved = _resolved_las_curve(
                las,
                mnemonic,
                match_policy="exact_then_normalized",
                lookup=lookup,
                source=source_las,
            )
        except Exception as exc:
            skipped.append({"curve": mnemonic, "reason": str(exc)})
            continue
        if resolved is None:
            skipped.append({"curve": mnemonic, "reason": "selected_curve_missing_in_las"})
            continue
        if resolved.index == 0:
            continue
        try:
            out.append_curve(
                resolved.mnemonic,
                data[:, resolved.index],
                unit=resolved.unit,
                descr=resolved.description,
            )
            exported_mnemonics.append(resolved.mnemonic)
        except Exception as exc:
            skipped.append({"curve": mnemonic, "reason": str(exc)})

    if len(out.curves) <= 1:
        raise ValueError(f"No selected curves were exported from {source_las}")

    out.write(str(output_las), version=2.0, wrap=False, fmt=write_fmt)
    return output_las, skipped, exported_mnemonics
