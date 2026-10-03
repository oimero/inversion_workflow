"""Shared text decoding and export-field tokenization primitives."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence


DEFAULT_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "gb18030", "cp1252", "latin-1")


def read_text_lines(
    file_path: str | Path,
    encodings: Sequence[str] | None = None,
) -> list[str]:
    """Read text lines with the project's encoding fallback order."""

    path = Path(file_path)
    candidates = DEFAULT_TEXT_ENCODINGS if encodings is None else tuple(encodings)
    last_error: UnicodeDecodeError | None = None
    for encoding in candidates:
        try:
            with path.open("r", encoding=encoding) as handle:
                return handle.readlines()
        except UnicodeDecodeError as error:
            last_error = error

    if last_error is not None:
        raise last_error
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Failed to decode file: {path}")


def split_export_fields(line: str) -> list[str]:
    """Split whitespace-delimited export fields while preserving double quotes.

    Single quotes remain ordinary token characters because Petrel exports can
    contain them inside unquoted coordinate or DMS fields.
    """

    tokens = re.findall(r'"[^"]*"|\S+', line)
    return [token[1:-1] if len(token) >= 2 and token[0] == '"' and token[-1] == '"' else token for token in tokens]


__all__ = ["DEFAULT_TEXT_ENCODINGS", "read_text_lines", "split_export_fields"]
