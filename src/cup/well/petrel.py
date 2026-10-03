"""Petrel well-head, well-tops, and checkshot readers."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from cup.utils.text import read_text_lines, split_export_fields


def _warn_if_petrel_rows_skipped(
    skipped_rows: list[tuple[int, int, str]],
    *,
    file_path: Path,
    record_type: str,
    expected_column_count: int,
    stacklevel: int = 2,
) -> None:
    """Warn with the original context when short Petrel rows are skipped."""

    if not skipped_rows:
        return

    details = "\n".join(
        f"line {line_no}: got {token_count} < {expected_column_count} tokens: {line}"
        for line_no, token_count, line in skipped_rows
    )
    warnings.warn(
        f"Malformed Petrel {record_type} rows in {file_path}:\n{details}",
        UserWarning,
        stacklevel=stacklevel,
    )


def _read_petrel_header(
    path: Path,
    *,
    record_type: str,
) -> tuple[list[str], int, list[str], dict[str, int]]:
    """Read one Petrel header block while preserving reader-specific errors."""

    lines = read_text_lines(path)
    begin_idx: int | None = None
    end_idx: int | None = None
    for index, line in enumerate(lines):
        token = line.strip()
        if token == "BEGIN HEADER":
            begin_idx = index
        elif token == "END HEADER":
            end_idx = index
            break

    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        raise ValueError(f"Invalid Petrel {record_type} header block: {path}")

    header_columns = [line.strip() for line in lines[begin_idx + 1 : end_idx] if line.strip()]
    column_indices = {name: index for index, name in enumerate(header_columns)}
    return lines, end_idx, header_columns, column_indices


def _read_petrel_rows(
    *,
    lines: list[str],
    end_idx: int,
    header_columns: list[str],
    column_indices: dict[str, int],
    selected_columns: list[str],
    file_path: Path,
    record_type: str,
) -> list[dict[str, str]]:
    """Tokenize data rows, skip short rows, and report them for one reader."""

    records: list[dict[str, str]] = []
    skipped_rows: list[tuple[int, int, str]] = []
    for line_no, raw_line in enumerate(lines[end_idx + 1 :], start=end_idx + 2):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = split_export_fields(line)
        if len(tokens) < len(header_columns):
            skipped_rows.append((line_no, len(tokens), line))
            continue
        records.append({name: tokens[column_indices[name]] for name in selected_columns})

    _warn_if_petrel_rows_skipped(
        skipped_rows,
        file_path=file_path,
        record_type=record_type,
        expected_column_count=len(header_columns),
        stacklevel=3,
    )
    return records


def import_petrel_checkshots_dataframe(path: Path) -> pd.DataFrame:
    """Read a Petrel checkshot file into the project's standard columns.

    The returned ``twt_ms`` retains the Petrel sign and millisecond units.
    Domain conversion belongs to :mod:`cup.well.trajectory`.
    """

    path = Path(path)
    column_aliases: dict[str, str] = {
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
    required_petrel_names = ["X", "Y", "Z", "MD"]
    twt_names = {"TWT", "TWT picked", "TWT_PICKED"}
    well_names = {"Well name", "Well"}
    optional_petrel_names = ["Average velocity", "Interval velocity"]

    lines, end_idx, header_columns, column_indices = _read_petrel_header(
        path,
        record_type="checkshots",
    )
    missing_required = sorted(name for name in required_petrel_names if name not in column_indices)
    if missing_required:
        raise ValueError(f"Required columns missing in Petrel header of {path}: {missing_required}")
    if not (twt_names & set(header_columns)):
        raise ValueError(f"No TWT column found in Petrel header of {path}. Expected one of: {sorted(twt_names)}")
    if not (well_names & set(header_columns)):
        raise ValueError(f"No well name column found in Petrel header of {path}. Expected one of: {sorted(well_names)}")

    selected_names = list(required_petrel_names)
    for names in (twt_names, well_names):
        for name in sorted(names):
            if name in column_indices:
                selected_names.append(name)
                break
    for name in optional_petrel_names:
        if name in column_indices:
            selected_names.append(name)

    records = _read_petrel_rows(
        lines=lines,
        end_idx=end_idx,
        header_columns=header_columns,
        column_indices=column_indices,
        selected_columns=selected_names,
        file_path=path,
        record_type="checkshots",
    )
    frame = pd.DataFrame.from_records(records, columns=selected_names)
    numeric_columns = [
        name
        for name in ["X", "Y", "Z", "MD", *selected_names]
        if name in frame.columns and name not in well_names
    ]
    for name in numeric_columns:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.rename(columns=column_aliases)
    return frame[
        [
            name
            for name in [
                "x_m",
                "y_m",
                "z_m",
                "md_m",
                "twt_ms",
                "well_name",
                "average_velocity",
                "interval_velocity",
            ]
            if name in frame.columns
        ]
    ]


def import_well_heads_petrel(well_heads_file: Path) -> pd.DataFrame:
    """Read a Petrel well-head export into the workflow columns."""

    required_columns = [
        "Name",
        "Surface X",
        "Surface Y",
        "Well datum name",
        "Well datum value",
        "Bottom hole X",
        "Bottom hole Y",
    ]
    path = Path(well_heads_file)
    lines, end_idx, header_columns, column_indices = _read_petrel_header(
        path,
        record_type="well heads",
    )
    missing = [name for name in required_columns if name not in column_indices]
    if missing:
        raise ValueError(f"Required columns missing in Petrel header: {missing}")

    records = _read_petrel_rows(
        lines=lines,
        end_idx=end_idx,
        header_columns=header_columns,
        column_indices=column_indices,
        selected_columns=required_columns,
        file_path=path,
        record_type="well head",
    )
    frame = pd.DataFrame.from_records(records, columns=required_columns)
    for name in ["Surface X", "Surface Y", "Well datum value", "Bottom hole X", "Bottom hole Y"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def import_well_tops_petrel(well_tops_file: Path) -> pd.DataFrame:
    """Read a Petrel well-tops export into the workflow columns."""

    required_columns = ["Well", "Surface", "X", "Y", "Z", "MD", "PVD auto"]
    path = Path(well_tops_file)
    lines, end_idx, header_columns, column_indices = _read_petrel_header(
        path,
        record_type="well tops",
    )
    missing = [name for name in required_columns if name not in column_indices]
    if missing:
        raise ValueError(f"Required columns missing in Petrel header: {missing}")

    records = _read_petrel_rows(
        lines=lines,
        end_idx=end_idx,
        header_columns=header_columns,
        column_indices=column_indices,
        selected_columns=required_columns,
        file_path=path,
        record_type="well tops",
    )
    frame = pd.DataFrame.from_records(records, columns=required_columns)
    for name in ["X", "Y", "Z", "MD", "PVD auto"]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["Z"] = np.abs(frame["Z"])
    frame["PVD auto"] = np.abs(frame["PVD auto"])
    return frame


__all__ = [
    "import_petrel_checkshots_dataframe",
    "import_well_heads_petrel",
    "import_well_tops_petrel",
]
