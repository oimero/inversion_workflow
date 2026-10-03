"""Read Petrel horizon interpretation exports."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from cup.utils.text import read_text_lines


_INTERPRETATION_LINE_PATTERN = re.compile(
    r"^\s*INLINE\s*:\s*([+-]?\d+)\s+XLINE\s*:\s*([+-]?\d+)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$",
    flags=re.IGNORECASE,
)


def import_interpretation_petrel(interpretation_file: Path) -> pd.DataFrame:
    """Read Petrel ``INLINE : ... XLINE : ...`` interpretation rows."""

    lines = read_text_lines(Path(interpretation_file))
    records: list[dict[str, object]] = []
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

    frame = pd.DataFrame.from_records(
        records,
        columns=["inline", "xline", "x", "y", "interpretation"],
    )
    if frame.empty:
        return frame
    frame = frame.drop_duplicates(subset=["inline", "xline"], keep="first", ignore_index=True)
    return frame.astype(
        {
            "inline": "int64",
            "xline": "int64",
            "x": "float64",
            "y": "float64",
            "interpretation": "float64",
        }
    )


__all__ = ["import_interpretation_petrel"]
