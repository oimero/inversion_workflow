"""Utilities for loading well and seismic data."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lasio
import numpy as np
import pandas as pd
import pyzgy

from wtie.processing import grid
from wtie.processing.logs import interpolate_nans


def _read_text_lines_with_fallback(file_path: Path, encodings: Optional[List[str]] = None) -> List[str]:
    """
    Read text file lines with encoding fallbacks.

    Parameters
    ----------
    file_path : Path
        Target file path.
    encodings : List[str], optional
        Candidate encodings to try in order.

    Returns
    -------
    List[str]
        File lines decoded to Python strings.

    Raises
    ------
    UnicodeDecodeError
        If all candidate encodings fail.
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
    Split a Petrel data line by whitespace while preserving double-quoted fields.

    Notes
    -----
    Petrel exports may contain single quotes in unquoted tokens (for example
    latitude/longitude DMS text). We therefore only treat double quotes as
    field wrappers.
    """
    tokens = re.findall(r'"[^"]*"|\S+', line)
    return [t[1:-1] if len(t) >= 2 and t[0] == '"' and t[-1] == '"' else t for t in tokens]


def import_well_heads_petrel(well_heads_file: Path) -> pd.DataFrame:
    """
    Import Petrel well heads file and keep only required columns.

    Parameters
    ----------
    well_heads_file : Path
        Path to a Petrel exported well heads text file.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly these columns:
        ['Name', 'Surface X', 'Surface Y', 'Well datum name',
         'Well datum value', 'Bottom hole X', 'Bottom hole Y']

    Raises
    ------
    ValueError
        If required columns are missing in file header.
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
    Import Petrel well tops file and keep only required columns.

    The function reads header definitions between ``BEGIN HEADER`` and
    ``END HEADER`` then parses data lines with quote-aware tokenization.

    Parameters
    ----------
    well_tops_file : Path
        Path to a Petrel exported well tops text file.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly these columns:
        ['Well', 'Surface', 'X', 'Y', 'Z', 'MD', 'PVD auto']

    Raises
    ------
    ValueError
        If required columns are missing in file header.
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

