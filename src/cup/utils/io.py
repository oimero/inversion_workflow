"""Shared infrastructure and I/O helpers.

Functions in this module are project-agnostic and do not depend on any
geophysical libraries beyond standard Python + matplotlib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml


# ── Path resolution ──


def resolve_relative_path(relative: str | Path, *, root: Path) -> Path:
    """Return an absolute path.  If *relative* is already absolute it is
    returned unchanged; otherwise it is resolved under *root*."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return (root / p).resolve()


# ── Config loading ──


def load_yaml_config(config_path: str | Path, *, base_dir: Path | None = None) -> dict[str, Any]:
    """Load a YAML config file, resolving relative paths against *base_dir*."""
    path = Path(config_path)
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


# ── String sanitisation ──


def sanitize_filename(name: str) -> str:
    """Replace characters that are unsafe in file names with underscores."""
    bad = {"/", "\\", " ", ":", "*", "?", "\"", "<", ">", "|"}
    return "".join("_" if c in bad else c for c in name)


# ── Matplotlib ──


def save_mpl_figure(path: Path, *, dpi: int = 180) -> None:
    """Save the current matplotlib figure to *path*, closing it afterwards."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
    except RuntimeError:
        pass
    plt.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── SEG-Y ──


def build_segy_textual_header(title: str, lines: list[str] | None = None) -> str:
    """Build a 3200-byte SEG-Y textual header from a title and extra lines."""
    all_lines = [title] + (lines or [])
    rows = [f"C{i:>2d} {text}"[:80].ljust(80) for i, text in enumerate(all_lines, start=1)]
    rows.extend([f"C{i:>2d}".ljust(80) for i in range(len(rows) + 1, 41)])
    textual = "".join(rows)
    if len(textual) != 3200:
        raise ValueError(f"Expected 3200-char textual header, got {len(textual)}")
    return textual
