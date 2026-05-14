"""Shared infrastructure and I/O helpers.

Functions in this module are project-agnostic and do not depend on any
geophysical libraries beyond standard Python.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── Path resolution ──


def resolve_relative_path(relative: str | Path, *, root: Path) -> Path:
    """Return an absolute path.  If *relative* is already absolute it is
    returned unchanged; otherwise it is resolved under *root*."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def repo_relative_path(path: str | Path, *, root: Path) -> str:
    """Return a portable POSIX-style path relative to *root*.

    Repo artifacts should store paths with this helper instead of embedding
    workstation-specific absolute paths.
    """
    root = Path(root).resolve()
    p = Path(path)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (root / p).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside repository root and cannot be stored portably: {resolved}") from exc


def resolve_repo_metadata_path(value: str | Path, *, root: Path) -> Path:
    """Resolve a repo-relative path stored in artifact metadata.

    Absolute metadata paths are intentionally rejected so stale workstation
    paths fail loudly instead of being guessed.
    """
    p = Path(value)
    if p.is_absolute():
        raise ValueError(
            f"Artifact metadata contains an absolute path, which is not portable: {p}. "
            "Regenerate the artifact with repo-relative metadata."
        )
    resolved = (Path(root).resolve() / p).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Artifact metadata path does not exist: {value} -> {resolved}")
    return resolved


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
    bad = {"/", "\\", " ", ":", "*", "?", '"', "<", ">", "|"}
    return "".join("_" if c in bad else c for c in name)


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


# ── JSON serialization ──


def to_json_compatible(value: Any) -> Any:
    """Recursively convert *value* to JSON-serialisable types.

    Handles ``Path``, ``numpy`` scalars and arrays, and containers.
    Non-finite floats are converted to ``null``.
    """
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return to_json_compatible(value.item())
        return [to_json_compatible(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_json_compatible(v) for k, v in value.items()}
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* to *path* as JSON with UTF-8 encoding and 2-space indent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(to_json_compatible(payload), fp, ensure_ascii=False, indent=2)
