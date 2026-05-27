"""cup.petrel.load: Petrel 相关文本与地震数据加载工具。

本模块负责读取 SEG-Y/ZGY 地震体，以及 Petrel 导出的
checkshots / well heads / well tops / interpretation 文本，
并转换为项目内部 ``wtie.processing.grid`` 对象或 ``pandas.DataFrame``。

边界说明
--------
- 本模块只负责格式解析、基本单位归一化与必要的轻量校验。
- 本模块不负责复杂质量控制、空间插值或地震-测井联合反演。
- 当输入文件本身存在格式噪声时，本模块会尽量汇报上下文，但不会替代上游修数。

核心公开对象
------------
1. import_seismic: 读取 3D SEG-Y 或 ZGY 地震体。
2. read_petrel_checkshots_dataframe: 将 Petrel checkshot 文本解析为表格。
3. import_well_heads_petrel / import_well_tops_petrel / import_interpretation_petrel:
   读取 Petrel 文本结果。
4. old_import_checkshots_petrel: 旧式 checkshot 到 TimeDepthTable 入口。
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from wtie.processing import grid

_INTERPRETATION_LINE_PATTERN = re.compile(
    r"^\s*INLINE\s*:\s*([+-]?\d+)\s+XLINE\s*:\s*([+-]?\d+)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
    flags=re.IGNORECASE,
)


def import_seismic(
    seismic_file: Path,
    seismic_type: str = "segy",
    iline: Optional[int] = None,
    xline: Optional[int] = None,
    istep: Optional[int] = None,
    xstep: Optional[int] = None,
) -> np.ndarray:
    """读取整块地震体并返回 ``(n_il, n_xl, n_samples)`` 数据体。

    Parameters
    ----------
    seismic_file : Path
        地震文件路径。
    seismic_type : str, optional
        地震类型，支持 ``"segy"`` 和 ``"zgy"``。
    iline, xline, istep, xstep : int | None, optional
        SEG-Y 读取参数。配置优先；为 ``None`` 时由底层库自动推断。

    Returns
    -------
    np.ndarray
        三维地震体，``dtype=float32``。

    Raises
    ------
    ValueError
        当 ``seismic_type`` 不受支持，或读出的数据不是三维体时抛出。
    """
    seismic_type_lower = str(seismic_type).lower()

    if seismic_type_lower == "segy":
        import cigsegy

        segy_kwargs = {}
        if iline is not None:
            segy_kwargs["iline"] = int(iline)
        if xline is not None:
            segy_kwargs["xline"] = int(xline)
        if istep is not None:
            segy_kwargs["istep"] = int(istep)
        if xstep is not None:
            segy_kwargs["xstep"] = int(xstep)

        volume = cigsegy.fromfile(
            str(seismic_file),
            **segy_kwargs,
        )
        volume = np.asarray(volume, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(f"Only 3D post-stack SEG-Y is supported, got ndim={volume.ndim}")
        return volume

    if seismic_type_lower == "zgy":
        from pyzgy.read import SeismicReader

        with SeismicReader(str(seismic_file)) as reader:
            volume = reader.read_volume()
        volume = np.asarray(volume, dtype=np.float32)
        if volume.ndim != 3:
            raise ValueError(f"Only 3D ZGY volume is supported, got ndim={volume.ndim}")
        return volume

    raise ValueError(f"Unsupported seismic_type: {seismic_type}. Expect 'segy' or 'zgy'.")


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


def _warn_if_petrel_rows_skipped(
    skipped_rows: List[Tuple[int, int, str]],
    *,
    file_path: Path,
    record_type: str,
    expected_column_count: int,
) -> None:
    """若存在因列数不足被跳过的行，则汇总后发出警告。"""
    if not skipped_rows:
        return

    details = "\n".join(
        f"line {line_no}: got {token_count} < {expected_column_count} tokens: {line}"
        for line_no, token_count, line in skipped_rows
    )
    warnings.warn(
        f"Malformed Petrel {record_type} rows in {file_path}:\n{details}",
        UserWarning,
        stacklevel=2,
    )


def read_petrel_checkshots_dataframe(path: Path) -> pd.DataFrame:
    """读取 Petrel checkshot / 时深表文本并返回标准列名 DataFrame。

    这是项目内 Petrel checkshot 文本的统一公共解析器。它处理
    ``BEGIN HEADER`` / ``END HEADER`` 块、带引号字段，以及不同 Petrel
    版本中的列名别名，例如 ``TWT picked`` / ``Well``。

    Parameters
    ----------
    path : Path
        Petrel checkshot 文本路径。

    Returns
    -------
    pd.DataFrame
        包含 ``x_m, y_m, z_m, md_m, twt_ms, well_name``，可选包含
        ``average_velocity, interval_velocity``。数值保留 Petrel 原始口径，
        因此 ``twt_ms`` 可能为负值。

    Raises
    ------
    ValueError
        缺少必需列或头块格式错误时抛出。
    """
    path = Path(path)

    _COLUMN_ALIASES: dict[str, str] = {
        "X": "x_m",
        "Y": "y_m",
        "Z": "z_m",
        "MD": "md_m",
        "TWT": "twt_ms",
        "TWT picked": "twt_ms",
        "TWT_PICKED": "twt_ms",
        "Well name": "well_name",
        "Well": "well_name",
        "Average velocity": "average_velocity",
        "Interval velocity": "interval_velocity",
    }
    _REQUIRED_PETREL_NAMES = ["X", "Y", "Z", "MD"]
    _TWT_NAMES = {"TWT", "TWT picked", "TWT_PICKED"}
    _WELL_NAMES = {"Well name", "Well"}
    _OPTIONAL_PETREL_NAMES = ["Average velocity", "Interval velocity"]

    lines = _read_text_lines_with_fallback(path)

    begin_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        token = line.strip()
        if token == "BEGIN HEADER":
            begin_idx = i
        elif token == "END HEADER":
            end_idx = i
            break

    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        raise ValueError(f"Invalid Petrel checkshots header block: {path}")

    header_columns = [line.strip() for line in lines[begin_idx + 1 : end_idx] if line.strip()]
    col_to_index = {name: idx for idx, name in enumerate(header_columns)}

    missing_req = sorted(c for c in _REQUIRED_PETREL_NAMES if c not in col_to_index)
    if missing_req:
        raise ValueError(f"Required columns missing in Petrel header of {path}: {missing_req}")
    if not (_TWT_NAMES & set(header_columns)):
        raise ValueError(f"No TWT column found in Petrel header of {path}. Expected one of: {sorted(_TWT_NAMES)}")
    if not (_WELL_NAMES & set(header_columns)):
        raise ValueError(f"No well name column found in Petrel header of {path}. Expected one of: {sorted(_WELL_NAMES)}")

    selected_petrel_names = list(_REQUIRED_PETREL_NAMES)
    for names in (_TWT_NAMES, _WELL_NAMES):
        for name in sorted(names):
            if name in col_to_index:
                selected_petrel_names.append(name)
                break
    for name in _OPTIONAL_PETREL_NAMES:
        if name in col_to_index:
            selected_petrel_names.append(name)

    records: list[dict[str, object]] = []
    skipped_rows: list[tuple[int, int, str]] = []
    for line_no, raw_line in enumerate(lines[end_idx + 1 :], start=end_idx + 2):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        tokens = _split_petrel_data_line(line)
        if len(tokens) < len(header_columns):
            skipped_rows.append((line_no, len(tokens), line))
            continue

        record: dict[str, object] = {}
        for petrel_name in selected_petrel_names:
            record[petrel_name] = tokens[col_to_index[petrel_name]]
        records.append(record)

    _warn_if_petrel_rows_skipped(
        skipped_rows,
        file_path=path,
        record_type="checkshots",
        expected_column_count=len(header_columns),
    )

    df = pd.DataFrame.from_records(records, columns=selected_petrel_names)

    numeric_cols = [c for c in ["X", "Y", "Z", "MD", *selected_petrel_names] if c in df.columns and c not in _WELL_NAMES]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(columns=_COLUMN_ALIASES)
    return df[[c for c in ["x_m", "y_m", "z_m", "md_m", "twt_ms", "well_name", "average_velocity", "interval_velocity"] if c in df.columns]]


def _parse_checkshots_petrel_dataframe(checkshots_file: Path) -> pd.DataFrame:
    """旧兼容包装：读取 Petrel checkshot 文本并应用旧单位约定。

    本函数仅为 ``old_import_checkshots_petrel`` 保留。新调用方应直接使用
    ``read_petrel_checkshots_dataframe``。
    """
    df = read_petrel_checkshots_dataframe(checkshots_file)
    df["z_m"] = np.abs(df["z_m"].to_numpy(dtype=np.float64))
    df["twt_s"] = np.abs(df["twt_ms"].to_numpy(dtype=np.float64)) / 1000.0
    return df


def _monotonic_checkshot_arrays(
    df: pd.DataFrame,
    *,
    depth_domain: str,
    source: Path,
) -> tuple[np.ndarray, np.ndarray]:
    depth_col = "md_m" if depth_domain == "md" else "z_m"
    depth = df[depth_col].to_numpy(dtype=float, copy=False)
    twt = df["twt_s"].to_numpy(dtype=float, copy=False)

    finite = np.isfinite(depth) & np.isfinite(twt)
    depth = depth[finite]
    twt = twt[finite]
    if depth.size < 2:
        raise ValueError(f"Petrel checkshots has fewer than 2 valid samples: {source}")

    order = np.argsort(depth)
    depth = depth[order]
    twt = twt[order]

    unique_depth, unique_indices = np.unique(depth, return_index=True)
    depth = unique_depth
    twt = twt[unique_indices]
    if depth.size < 2:
        raise ValueError(f"Petrel checkshots has fewer than 2 unique depth samples: {source}")

    start = int(np.nanargmin(twt))
    depth = depth[start:]
    twt = twt[start:]
    keep = np.zeros(twt.shape, dtype=bool)
    last_twt = -np.inf
    for index, value in enumerate(twt):
        if value > last_twt + 1e-9:
            keep[index] = True
            last_twt = float(value)
    depth = depth[keep]
    twt = twt[keep]
    if depth.size < 2:
        raise ValueError(f"Petrel checkshots has fewer than 2 strictly increasing TWT samples: {source}")
    return depth, twt


def old_import_checkshots_petrel(checkshots_file: Path, depth_domain: str = "md") -> grid.TimeDepthTable:
    """旧式入口：导入 Petrel checkshots/时深表文本并转换为 TimeDepthTable。

    新工作流应优先使用 ``cup.well.td.load_petrel_time_depth_table``。

    Parameters
    ----------
    checkshots_file : Path
        Petrel 导出的 checkshots 文本路径。
    depth_domain : str, default="md"
        深度域类型，支持 ``"md"`` 与 ``"tvdss"``。

    Returns
    -------
    grid.TimeDepthTable
        统一单位后的时深关系对象。``TWT`` 单位固定为 s，深度单位固定为 m。

    Raises
    ------
    ValueError
        当 ``depth_domain`` 非法时抛出。

    Warns
    -----
    UserWarning
        当存在列数不足而被跳过的数据行时发出警告，警告中会附带行号、
        token 数与原始行文本。
    """
    depth_domain = str(depth_domain).strip().lower()
    if depth_domain not in {"md", "tvdss"}:
        raise ValueError(f"Unsupported depth_domain: {depth_domain}. Expect 'md' or 'tvdss'.")

    checkshots_file = Path(checkshots_file)
    df = _parse_checkshots_petrel_dataframe(checkshots_file)
    depth, twt = _monotonic_checkshot_arrays(df, depth_domain=depth_domain, source=checkshots_file)
    if depth_domain == "md":
        return grid.TimeDepthTable(twt=twt, md=depth)

    return grid.TimeDepthTable(twt=twt, tvdss=depth)


def import_well_heads_petrel(well_heads_file: Path) -> pd.DataFrame:
    """导入 Petrel 井头文件，并仅保留项目所需列。

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

    Warns
    -----
    UserWarning
        当存在列数不足而被跳过的数据行时发出警告，警告中会附带行号、
        token 数与原始行文本。
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
    skipped_rows: List[Tuple[int, int, str]] = []
    for line_no, raw_line in enumerate(lines[end_idx + 1 :], start=end_idx + 2):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Keep double-quoted fields intact, but ignore single quotes in DMS strings.
        tokens = _split_petrel_data_line(line)
        if len(tokens) < len(header_columns):
            skipped_rows.append((line_no, len(tokens), line))
            continue

        record = {col: tokens[col_to_index[col]] for col in required_columns}
        records.append(record)

    _warn_if_petrel_rows_skipped(
        skipped_rows,
        file_path=well_heads_file,
        record_type="well head",
        expected_column_count=len(header_columns),
    )

    df = pd.DataFrame.from_records(records, columns=required_columns)

    numeric_cols = ["Surface X", "Surface Y", "Well datum value", "Bottom hole X", "Bottom hole Y"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def import_well_tops_petrel(well_tops_file: Path) -> pd.DataFrame:
    """导入 Petrel 分层文件，并仅保留项目所需列。

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

    Warns
    -----
    UserWarning
        当存在列数不足而被跳过的数据行时发出警告，警告中会附带行号、
        token 数与原始行文本。
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
    skipped_rows: List[Tuple[int, int, str]] = []
    for line_no, raw_line in enumerate(lines[end_idx + 1 :], start=end_idx + 2):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Keep double-quoted fields intact, but ignore single quotes in DMS strings.
        tokens = _split_petrel_data_line(line)
        if len(tokens) < len(header_columns):
            skipped_rows.append((line_no, len(tokens), line))
            continue

        record = {col: tokens[col_to_index[col]] for col in required_columns}
        records.append(record)

    _warn_if_petrel_rows_skipped(
        skipped_rows,
        file_path=well_tops_file,
        record_type="well tops",
        expected_column_count=len(header_columns),
    )

    df = pd.DataFrame.from_records(records, columns=required_columns)

    numeric_cols = ["X", "Y", "Z", "MD", "PVD auto"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Z"] = np.abs(df["Z"])
    df["PVD auto"] = np.abs(df["PVD auto"])

    return df


def import_interpretation_petrel(interpretation_file: Path) -> pd.DataFrame:
    """导入 Petrel 层位解释文本。

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
