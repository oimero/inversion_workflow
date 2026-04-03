"""Petrel数据加载工具。"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import lasio
import numpy as np
import pandas as pd

from cup.well.mnemonics import _RHO_MNEMONICS, _VP_MNEMONICS, _VS_MNEMONICS
from wtie.processing import grid
from wtie.processing.logs import interpolate_nans

_SENTINEL_VALUES = (-999.0, -999.25, -99999)
_INTERPRETATION_LINE_PATTERN = re.compile(
    r"^\s*INLINE\s*:\s*([+-]?\d+)\s+XLINE\s*:\s*([+-]?\d+)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
    flags=re.IGNORECASE,
)


def _normalize_mnemonic(name: str) -> str:
    return str(name).strip().upper()


def _select_curve_mnemonic(
    las_df: pd.DataFrame,
    candidate_mnemonics: Tuple[str, ...],
    property_name: str,
    curve_mnemonic: Optional[str] = None,
) -> str:
    columns = [str(c) for c in las_df.columns]
    norm_to_original = {_normalize_mnemonic(c): c for c in columns}

    if curve_mnemonic is not None:
        norm_user = _normalize_mnemonic(curve_mnemonic)
        if norm_user not in norm_to_original:
            raise ValueError(f"指定的 {property_name} 曲线简称不存在: {curve_mnemonic}. 可用曲线: {columns}")
        return norm_to_original[norm_user]

    wanted = {_normalize_mnemonic(m) for m in candidate_mnemonics}
    matched = [col for col in columns if _normalize_mnemonic(col) in wanted]

    if len(matched) == 0:
        raise ValueError(
            f"未找到 {property_name} 曲线。候选简称: {list(candidate_mnemonics)}. 请检查是否存在其他可用简称？"
        )

    if len(matched) > 1:
        raise ValueError(
            f"检测到多个 {property_name} 候选曲线: {matched}. 请通过 curve_mnemonic 显式指定要使用的简称。"
        )

    return matched[0]


def _get_curve_unit(las_file: lasio.LASFile, selected_mnemonic: str) -> str:
    norm_selected = _normalize_mnemonic(selected_mnemonic)
    for curve in las_file.curves:
        if _normalize_mnemonic(curve.mnemonic) == norm_selected:
            return str(curve.unit or "")
    return ""


def _resolve_input_unit(
    las_file: lasio.LASFile,
    selected_mnemonic: str,
    property_name: str,
    provided_unit: Optional[str],
) -> str:
    if provided_unit is not None:
        return str(provided_unit)

    unit_from_las = _get_curve_unit(las_file, selected_mnemonic).strip()
    if not unit_from_las:
        raise ValueError(f"{property_name} 曲线未读取到单位信息。请通过 unit 参数显式指定输入单位。")

    return unit_from_las


def _replace_sentinel_values(values: object) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for sentinel in _SENTINEL_VALUES:
        out[np.isclose(out, sentinel, equal_nan=False)] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def _convert_sonic_to_velocity_mps(sonic_values: object, unit: str, property_name: str) -> np.ndarray:
    sonic = _replace_sentinel_values(sonic_values)
    sonic[sonic <= 0] = np.nan

    unit_norm = str(unit).strip().lower().replace(" ", "")
    if unit_norm in {"us/ft", "μs/ft", "µs/ft"}:
        velocity = 0.3048 * 1e6 / sonic
    elif unit_norm in {"us/m", "μs/m", "µs/m"}:
        velocity = 1e6 / sonic
    else:
        raise ValueError(f"{property_name} 曲线单位不受支持: '{unit}'. 当前仅支持 us/ft 或 us/m。")

    if np.all(np.isnan(velocity)):
        raise ValueError(f"{property_name} 曲线在异常值处理与单位转换后全部为 NaN。")

    return velocity


def _convert_density_to_g_cm3(density_values: object, unit: str) -> np.ndarray:
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


def extract_vp_log_from_las(
    las_file: lasio.LASFile,
    curve_mnemonic: Optional[str] = None,
    unit: Optional[str] = None,
) -> grid.Log:
    """
    从 LAS 文件中提取纵波速度曲线（Vp），输出单位统一为 m/s。

    Parameters
    ----------
    las_file : lasio.LASFile
        已加载的 LAS 文件对象。
    curve_mnemonic : str, optional
        指定要使用的曲线简称。若未指定且匹配到多个候选，会报错。
    unit : str, optional
        输入曲线单位。若提供，则完全覆盖 LAS 中的单位信息；
        若不提供，则尝试从 LAS 读取单位，读取失败时会报错。

    Returns
    -------
    grid.Log
        纵波速度曲线，单位为 m/s。

    Raises
    ------
    ValueError
        曲线缺失、候选歧义、单位缺失或单位不受支持时抛出。
    """
    las_df = las_file.df()
    selected = _select_curve_mnemonic(las_df, _VP_MNEMONICS, "Vp", curve_mnemonic)
    resolved_unit = _resolve_input_unit(las_file, selected, "Vp", unit)
    vp = _convert_sonic_to_velocity_mps(las_df.loc[:, selected].to_numpy(), resolved_unit, "Vp")
    vp = interpolate_nans(vp, method="linear")
    return grid.Log(vp, las_df.index.values, "md", name="Vp", unit="m/s", allow_nan=False)


def extract_vs_log_from_las(
    las_file: lasio.LASFile,
    curve_mnemonic: Optional[str] = None,
    unit: Optional[str] = None,
) -> grid.Log:
    """
    从 LAS 文件中提取横波速度曲线（Vs），输出单位统一为 m/s。

    Parameters
    ----------
    las_file : lasio.LASFile
        已加载的 LAS 文件对象。
    curve_mnemonic : str, optional
        指定要使用的曲线简称。若未指定且匹配到多个候选，会报错。
    unit : str, optional
        输入曲线单位。若提供，则完全覆盖 LAS 中的单位信息；
        若不提供，则尝试从 LAS 读取单位，读取失败时会报错。

    Returns
    -------
    grid.Log
        横波速度曲线，单位为 m/s。

    Raises
    ------
    ValueError
        曲线缺失、候选歧义、单位缺失或单位不受支持时抛出。
    """
    las_df = las_file.df()
    selected = _select_curve_mnemonic(las_df, _VS_MNEMONICS, "Vs", curve_mnemonic)
    resolved_unit = _resolve_input_unit(las_file, selected, "Vs", unit)
    vs = _convert_sonic_to_velocity_mps(las_df.loc[:, selected].to_numpy(), resolved_unit, "Vs")
    vs = interpolate_nans(vs, method="linear")
    return grid.Log(vs, las_df.index.values, "md", name="Vs", unit="m/s", allow_nan=False)


def extract_rho_log_from_las(
    las_file: lasio.LASFile,
    curve_mnemonic: Optional[str] = None,
    unit: Optional[str] = None,
) -> grid.Log:
    """
    从 LAS 文件中提取密度曲线（Rho），输出单位统一为 g/cm3。

    Parameters
    ----------
    las_file : lasio.LASFile
        已加载的 LAS 文件对象。
    curve_mnemonic : str, optional
        指定要使用的曲线简称。若未指定且匹配到多个候选，会报错。
    unit : str, optional
        输入曲线单位。若提供，则完全覆盖 LAS 中的单位信息；
        若不提供，则尝试从 LAS 读取单位，读取失败时会报错。

    Returns
    -------
    grid.Log
        密度曲线，单位为 g/cm3。

    Raises
    ------
    ValueError
        曲线缺失、候选歧义、单位缺失或单位不受支持时抛出。
    """
    las_df = las_file.df()
    selected = _select_curve_mnemonic(las_df, _RHO_MNEMONICS, "Rho", curve_mnemonic)
    resolved_unit = _resolve_input_unit(las_file, selected, "Rho", unit)
    rho = _convert_density_to_g_cm3(las_df.loc[:, selected].to_numpy(), resolved_unit)
    rho = interpolate_nans(rho, method="linear")
    return grid.Log(rho, las_df.index.values, "md", name="Rho", unit="g/cm3", allow_nan=False)


def extract_any_log_from_las(las_file: lasio.LASFile, curve_mnemonic: str) -> grid.Log:
    """
    从 LAS 文件中提取任意单条曲线。

    Parameters
    ----------
    las_file : lasio.LASFile
        已加载的 LAS 文件对象。
    curve_mnemonic : str
        目标曲线简称，大小写不敏感。

    Returns
    -------
    grid.Log
        提取后的曲线。仅做异常值替换，不做插值；允许包含 NaN。

    Raises
    ------
    ValueError
        当 curve_mnemonic 为空、曲线不存在或异常值处理后全部为 NaN 时抛出。
    """
    curve_mnemonic = str(curve_mnemonic).strip()
    if not curve_mnemonic:
        raise ValueError("curve_mnemonic 不能为空。")

    las_df = las_file.df()
    columns = [str(c) for c in las_df.columns]
    norm_to_original = {_normalize_mnemonic(c): c for c in columns}

    norm_user = _normalize_mnemonic(curve_mnemonic)
    if norm_user not in norm_to_original:
        raise ValueError(f"指定曲线简称不存在: {curve_mnemonic}. 可用曲线: {columns}")

    selected = norm_to_original[norm_user]
    values = _replace_sentinel_values(las_df.loc[:, selected].to_numpy())
    if np.all(np.isnan(values)):
        raise ValueError(f"{selected} 曲线在异常值处理后全部为 NaN。")

    return grid.Log(values, las_df.index.values, "md", name=curve_mnemonic, unit="", allow_nan=True)


def load_vp_rho_logset_from_las(
    las_file_path: Path,
    vp_mnemonic: Optional[str] = None,
    rho_mnemonic: Optional[str] = None,
    vp_unit: Optional[str] = None,
    rho_unit: Optional[str] = None,
) -> grid.LogSet:
    """
    从 LAS 文件路径读取 Vp 与 Rho 曲线并组装为 LogSet。

    Parameters
    ----------
    las_file_path : Path
        LAS 文件路径。
    vp_mnemonic : str, optional
        指定 Vp 使用的曲线简称。未指定时按候选简称自动匹配，若匹配到多个则报错。
    rho_mnemonic : str, optional
        指定 Rho 使用的曲线简称。未指定时按候选简称自动匹配，若匹配到多个则报错。
    vp_unit : str, optional
        Vp 输入单位。若提供，则完全覆盖 LAS 中的单位信息；
        若不提供，则尝试从 LAS 读取单位，读取失败时会报错。
    rho_unit : str, optional
        Rho 输入单位。若提供，则完全覆盖 LAS 中的单位信息；
        若不提供，则尝试从 LAS 读取单位，读取失败时会报错。

    Returns
    -------
    grid.LogSet
        至少包含 Vp 与 Rho 两条曲线的 LogSet。

    Raises
    ------
    ValueError
        曲线不存在、候选歧义或单位不受支持时抛出。
    FileNotFoundError
        输入 LAS 文件路径不存在时抛出。
    """
    las_file_path = Path(las_file_path)
    if not las_file_path.exists():
        raise FileNotFoundError(f"LAS 文件不存在: {las_file_path}")

    las_file = lasio.read(las_file_path)
    vp_log = extract_vp_log_from_las(las_file, curve_mnemonic=vp_mnemonic, unit=vp_unit)
    rho_log = extract_rho_log_from_las(las_file, curve_mnemonic=rho_mnemonic, unit=rho_unit)

    return grid.LogSet({"Vp": vp_log, "Rho": rho_log})


def _read_text_lines_with_fallback(file_path: Path, encodings: Optional[List[str]] = None) -> List[str]:
    """
    使用编码回退策略读取文本文件行。

    Parameters
    ----------
    file_path : Path
        目标文件路径。
    encodings : List[str], optional
        按顺序尝试的候选编码列表。

    Returns
    -------
    List[str]
        解码后的文本行列表。

    Raises
    ------
    UnicodeDecodeError
        当所有候选编码均解码失败时抛出。
    """
    if encodings is None:
        encodings = ["utf-8", "utf-8-sig", "gb18030", "cp1252", "latin-1"]

    last_error: Optional[UnicodeDecodeError] = None
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.readlines()
        except UnicodeDecodeError as err:
            last_error = err

    if last_error is not None:
        raise last_error

    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Failed to decode file: {file_path}")


def _split_petrel_data_line(line: str) -> List[str]:
    """
    按空白符拆分 Petrel 数据行，同时保留双引号包裹字段的完整性。

    Notes
    -----
    Petrel 导出内容中，未加引号的 token 里可能包含单引号（例如经纬度
    DMS 文本）。因此这里只将双引号视为字段包裹符。
    """
    tokens = re.findall(r'"[^"]*"|\S+', line)
    return [t[1:-1] if len(t) >= 2 and t[0] == '"' and t[-1] == '"' else t for t in tokens]


def import_well_heads_petrel(well_heads_file: Path) -> pd.DataFrame:
    """
    导入 Petrel 井头文件，并仅保留所需列。

    Parameters
    ----------
    well_heads_file : Path
        Petrel 导出的井头文本文件路径。

    Returns
    -------
    pd.DataFrame
        仅包含以下列的 DataFrame：
        ['Name', 'Surface X', 'Surface Y', 'Well datum name',
         'Well datum value', 'Bottom hole X', 'Bottom hole Y']

    Raises
    ------
    ValueError
        当文件头中缺少必需列时抛出。
    """
    required_columns = [
        "Name",
        "Surface X",
        "Surface Y",
        "Well datum name",
        "Well datum value",
        "Bottom hole X",
        "Bottom hole Y",
    ]

    lines = _read_text_lines_with_fallback(well_heads_file)

    begin_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        token = line.strip()
        if token == "BEGIN HEADER":
            begin_idx = i
        elif token == "END HEADER":
            end_idx = i
            break

    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        raise ValueError(f"Invalid Petrel well heads header block: {well_heads_file}")

    header_columns = [line.strip() for line in lines[begin_idx + 1 : end_idx] if line.strip()]
    col_to_index = {name: idx for idx, name in enumerate(header_columns)}

    missing = [c for c in required_columns if c not in col_to_index]
    if missing:
        raise ValueError(f"Required columns missing in Petrel header: {missing}")

    records = []
    for raw_line in lines[end_idx + 1 :]:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Keep double-quoted fields intact, but ignore single quotes in DMS strings.
        tokens = _split_petrel_data_line(line)
        if len(tokens) < len(header_columns):
            continue

        record = {col: tokens[col_to_index[col]] for col in required_columns}
        records.append(record)

    df = pd.DataFrame.from_records(records, columns=required_columns)

    numeric_cols = ["Surface X", "Surface Y", "Well datum value", "Bottom hole X", "Bottom hole Y"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def import_well_tops_petrel(well_tops_file: Path) -> pd.DataFrame:
    """
    导入 Petrel 分层文件，并仅保留所需列。

    函数会先读取 ``BEGIN HEADER`` 与 ``END HEADER`` 之间的表头定义，
    然后使用感知引号的分词方式解析数据行。

    Parameters
    ----------
    well_tops_file : Path
        Petrel 导出的分层文本文件路径。

    Returns
    -------
    pd.DataFrame
        仅包含以下列的 DataFrame：
        ['Well', 'Surface', 'X', 'Y', 'Z', 'MD', 'PVD auto']

    Raises
    ------
    ValueError
        当文件头中缺少必需列时抛出。
    """
    required_columns = ["Well", "Surface", "X", "Y", "Z", "MD", "PVD auto"]

    lines = _read_text_lines_with_fallback(well_tops_file)

    begin_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        token = line.strip()
        if token == "BEGIN HEADER":
            begin_idx = i
        elif token == "END HEADER":
            end_idx = i
            break

    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        raise ValueError(f"Invalid Petrel well tops header block: {well_tops_file}")

    header_columns = [line.strip() for line in lines[begin_idx + 1 : end_idx] if line.strip()]
    col_to_index = {name: idx for idx, name in enumerate(header_columns)}

    missing = [c for c in required_columns if c not in col_to_index]
    if missing:
        raise ValueError(f"Required columns missing in Petrel header: {missing}")

    records = []
    for raw_line in lines[end_idx + 1 :]:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Keep double-quoted fields intact, but ignore single quotes in DMS strings.
        tokens = _split_petrel_data_line(line)
        if len(tokens) < len(header_columns):
            continue

        record = {col: tokens[col_to_index[col]] for col in required_columns}
        records.append(record)

    df = pd.DataFrame.from_records(records, columns=required_columns)

    numeric_cols = ["X", "Y", "Z", "MD", "PVD auto"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def import_interpretation_petrel(interpretation_file: Path) -> pd.DataFrame:
    """
    导入 Petrel 层位解释文本，返回固定五列 DataFrame。

    输入行格式示例::

        INLINE :   1501 XLINE :   4199    167767.05023   7264082.47336      5466.60010

    Returns
    -------
    pd.DataFrame
        列名固定为 ['inline', 'xline', 'x', 'y', 'interpretation']。
        按 (inline, xline) 去重，仅保留首次出现记录。

    Raises
    ------
    ValueError
        当任一数据行不符合固定格式时抛出。
    """
    lines = _read_text_lines_with_fallback(Path(interpretation_file))
    records = []

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        match = _INTERPRETATION_LINE_PATTERN.match(line)
        if match is None:
            raise ValueError(f"Invalid interpretation line at {line_no}: {raw_line.rstrip()}")

        records.append(
            {
                "inline": int(match.group(1)),
                "xline": int(match.group(2)),
                "x": float(match.group(3)),
                "y": float(match.group(4)),
                "interpretation": float(match.group(5)),
            }
        )

    df = pd.DataFrame.from_records(records, columns=["inline", "xline", "x", "y", "interpretation"])
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["inline", "xline"], keep="first", ignore_index=True)
    df = df.astype(
        {
            "inline": "int64",
            "xline": "int64",
            "x": "float64",
            "y": "float64",
            "interpretation": "float64",
        }
    )
    return df
